from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class MixtureHead(nn.Module):
    def __init__(self, input_dim, num_mixtures):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.output = nn.Linear(input_dim, 3 * num_mixtures)

    def forward(self, x):
        logits, mus, raw_sigmas = torch.split(
            self.output(x), self.num_mixtures, dim=-1
        )
        return torch.softmax(logits, dim=-1), mus, F.softplus(raw_sigmas) + 1e-4


class SnapshotMDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_mixtures):
        super().__init__()
        layers = []
        width_in = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(width_in, hidden_dim), nn.ReLU()])
            width_in = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.head = MixtureHead(hidden_dim, num_mixtures)

    def forward(self, x):
        if x.ndim == 3:
            x = x[:, -1, :]
        return self.head(self.encoder(x))


class RecurrentMDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_mixtures):
        super().__init__()
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = MixtureHead(hidden_dim, num_mixtures)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        return self.head(encoded[:, -1, :])


class RecurrentPointModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        encoded, _ = self.encoder(x)
        return self.output(encoded[:, -1, :]).squeeze(-1)


def build_density_model(spec, input_dim, num_mixtures):
    family = spec["family"]
    if family == "snapshot":
        return SnapshotMDN(
            input_dim,
            int(spec["hidden_dim"]),
            int(spec["num_layers"]),
            num_mixtures,
        )
    if family == "recurrent":
        return RecurrentMDN(
            input_dim,
            int(spec["hidden_dim"]),
            int(spec["num_layers"]),
            num_mixtures,
        )
    raise ValueError(f"Unknown model family: {family}")


def mdn_nll(pis, mus, sigmas, y):
    y = y.reshape(-1, 1)
    z = (y - mus) / sigmas
    log_components = (
        -0.5 * np.log(2.0 * np.pi) - torch.log(sigmas) - 0.5 * z * z
    )
    return -torch.logsumexp(torch.log(pis + 1e-12) + log_components, dim=1).mean()


def parameter_count(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _epoch_nll(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pis, mus, sigmas = model(xb)
            loss = mdn_nll(pis, mus, sigmas, yb)
            total += float(loss.item()) * len(xb)
            count += len(xb)
    return total / count


def train_density_model(
    X_train,
    y_train,
    X_val,
    y_val,
    spec,
    num_mixtures,
    seed,
    checkpoint_path,
    max_epochs,
    min_epochs,
    patience,
    batch_size=128,
    min_delta=1e-4,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_density_model(spec, X_train.shape[2], num_mixtures).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(np.asarray(X_train, dtype=np.float32)),
            torch.from_numpy(np.asarray(y_train, dtype=np.float32)),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(np.asarray(X_val, dtype=np.float32)),
            torch.from_numpy(np.asarray(y_val, dtype=np.float32)),
        ),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")
    stale = 0
    started = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pis, mus, sigmas = model(xb)
            loss = mdn_nll(pis, mus, sigmas, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss.item()) * len(xb)
            train_count += len(xb)

        train_nll = train_total / train_count
        val_nll = _epoch_nll(model, val_loader, device)
        if not np.isfinite(train_nll) or not np.isfinite(val_nll):
            raise FloatingPointError("Non-finite density loss.")
        history.append(
            {"epoch": epoch, "train_nll": train_nll, "val_nll": val_nll}
        )

        if val_nll < best_val - min_delta:
            best_val = val_nll
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "spec": spec,
                    "num_mixtures": int(num_mixtures),
                    "input_dim": int(X_train.shape[2]),
                    "seed": int(seed),
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            break

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    details = {
        "best_val_nll_scaled": best_val,
        "epochs_ran": len(history),
        "optimizer_steps": len(train_loader) * len(history),
        "parameter_count": parameter_count(model),
        "wall_seconds": time.time() - started,
        "history": history,
    }
    checkpoint_path.with_suffix(".json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )
    return model, details


def load_density_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_density_model(
        payload["spec"], payload["input_dim"], payload["num_mixtures"]
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def predict_density(model, X, batch_size=2048):
    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(
        torch.from_numpy(np.array(X, dtype=np.float32, copy=True)),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    pis_all = []
    mus_all = []
    sigmas_all = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            pis, mus, sigmas = model(xb.to(device, non_blocking=True))
            pis_all.append(pis.cpu().numpy())
            mus_all.append(mus.cpu().numpy())
            sigmas_all.append(sigmas.cpu().numpy())
    return (
        np.concatenate(pis_all),
        np.concatenate(mus_all),
        np.concatenate(sigmas_all),
    )


def _epoch_mse(model, loader, device):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            loss = F.mse_loss(model(xb), yb)
            total += float(loss.item()) * len(xb)
            count += len(xb)
    return total / count


def train_point_model(
    X_train,
    y_train,
    X_val,
    y_val,
    spec,
    seed,
    checkpoint_path,
    max_epochs,
    min_epochs,
    patience,
    batch_size=2048,
    min_delta=1e-4,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentPointModel(
        X_train.shape[2], int(spec["hidden_dim"]), int(spec["num_layers"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(np.asarray(X_train, dtype=np.float32)),
            torch.from_numpy(np.asarray(y_train, dtype=np.float32)),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(np.asarray(X_val, dtype=np.float32)),
            torch.from_numpy(np.asarray(y_val, dtype=np.float32)),
        ),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    stale = 0
    history = []
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss.item()) * len(xb)
            train_count += len(xb)
        train_mse = train_total / train_count
        val_mse = _epoch_mse(model, val_loader, device)
        if not np.isfinite(train_mse) or not np.isfinite(val_mse):
            raise FloatingPointError("Non-finite point-model loss.")
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
        if val_mse < best_val - min_delta:
            best_val = val_mse
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "spec": spec,
                    "input_dim": int(X_train.shape[2]),
                    "seed": int(seed),
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            break
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    details = {
        "best_val_mse_scaled": best_val,
        "epochs_ran": len(history),
        "optimizer_steps": len(train_loader) * len(history),
        "parameter_count": parameter_count(model),
        "wall_seconds": time.time() - started,
        "history": history,
    }
    checkpoint_path.with_suffix(".json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )
    return model, details


def load_point_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    spec = payload["spec"]
    model = RecurrentPointModel(
        payload["input_dim"], int(spec["hidden_dim"]), int(spec["num_layers"])
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def predict_point(model, X, batch_size=2048):
    device = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(
        torch.from_numpy(np.array(X, dtype=np.float32, copy=True)),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    predictions = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            predictions.append(model(xb.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(predictions)
