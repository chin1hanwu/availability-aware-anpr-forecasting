from __future__ import annotations

import gc
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT_DIR = ROOT / "outputs" / "paper_results"
RESULT_DIR = OUTPUT_DIR / "computational_efficiency"
WARMUP_CALLS = 100
TIMED_CALLS = 1000


def load_density_cpu(path: Path):
    from forecast_models import build_density_model

    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_density_model(
        payload["spec"], payload["input_dim"], payload["num_mixtures"]
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def load_point_cpu(path: Path):
    from forecast_models import RecurrentPointModel

    payload = torch.load(path, map_location="cpu", weights_only=False)
    spec = payload["spec"]
    model = RecurrentPointModel(
        payload["input_dim"], int(spec["hidden_dim"]), int(spec["num_layers"])
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def parameter_count(model) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def tree_complexity(models) -> tuple[int, int]:
    nodes = 0
    leaves = 0
    for model in models:
        for stage in model._predictors:
            for tree in stage:
                nodes += len(tree.nodes)
                leaves += int(np.asarray(tree.nodes["is_leaf"]).sum())
    return nodes, leaves


def training_time_summary(checkpoints) -> tuple[float | None, float | None]:
    values = []
    for checkpoint in checkpoints:
        sidecar = Path(checkpoint).with_suffix(".json")
        if sidecar.exists():
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            if "wall_seconds" in payload:
                values.append(float(payload["wall_seconds"]))
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def benchmark(callable_):
    for _ in range(WARMUP_CALLS):
        callable_()

    samples = np.empty(TIMED_CALLS, dtype=float)
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for index in range(TIMED_CALLS):
            started = time.perf_counter_ns()
            callable_()
            samples[index] = (time.perf_counter_ns() - started) / 1e6
    finally:
        if gc_enabled:
            gc.enable()

    return {
        "latency_median_ms": float(np.median(samples)),
        "latency_p95_ms": float(np.percentile(samples, 95)),
        "latency_mean_ms": float(np.mean(samples)),
        "latency_std_ms": float(np.std(samples, ddof=1)),
    }


def main():
    lock = json.loads(
        (OUTPUT_DIR / "freeze" / "selection_lock.json").read_text(encoding="utf-8")
    )
    X_raw = np.load(OUTPUT_DIR / "g0" / "val_X_raw.npy", mmap_mode="r")
    X_scaled = np.load(OUTPUT_DIR / "g0" / "val_X_scaled.npy", mmap_mode="r")
    input_dim = int(X_scaled.shape[2])

    records = []

    fixed = json.loads(
        (OUTPUT_DIR / "ap02" / "fixed_references" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    ridge_path = OUTPUT_DIR / fixed["ridge"]["model"]
    ridge = joblib.load(ridge_path)
    ridge_input = np.asarray(X_scaled[0:1, -1, :])
    ridge_record = {
        "model": "Ridge Gaussian",
        "history_periods": 1,
        "parameter_count": int(np.size(ridge.coef_) + 1),
        "tree_nodes": None,
        "tree_leaves": None,
        "artifact_kib": ridge_path.stat().st_size / 1024.0,
        "training_wall_mean_s": None,
        "training_wall_sd_s": None,
    }
    ridge_record.update(benchmark(lambda: ridge.predict(ridge_input)))
    records.append(ridge_record)

    for key, label in (
        ("snapshot", "Snapshot quantile GBDT"),
        ("lagged", "Explicit-lag quantile GBDT"),
    ):
        tree = lock["tree_models"][key]
        model_path = OUTPUT_DIR / tree["models"]
        models = joblib.load(model_path)
        history = int(tree["spec"]["history_length"])
        model_input = np.asarray(X_raw[0:1, -history:, :]).reshape(1, -1)
        nodes, leaves = tree_complexity(models)

        def predict_quantiles(models=models, model_input=model_input):
            return tuple(model.predict(model_input) for model in models)

        record = {
            "model": label,
            "history_periods": history,
            "parameter_count": None,
            "tree_nodes": nodes,
            "tree_leaves": leaves,
            "artifact_kib": model_path.stat().st_size / 1024.0,
            "training_wall_mean_s": None,
            "training_wall_sd_s": None,
        }
        record.update(benchmark(predict_quantiles))
        records.append(record)

    final_models = lock["final_models"]
    for family, label in (
        ("snapshot", "Snapshot MDN, K=2"),
        ("recurrent", "Recurrent MDN, K=2"),
    ):
        model_info = final_models[family]
        checkpoints = [OUTPUT_DIR / path for path in model_info["checkpoints"].values()]
        model = load_density_cpu(checkpoints[0])
        history = int(model_info["spec"]["history_length"])
        model_input = torch.from_numpy(
            np.array(X_scaled[0:1, -history:, :], dtype=np.float32, copy=True)
        )
        train_mean, train_sd = training_time_summary(checkpoints)

        def predict_density(model=model, model_input=model_input):
            with torch.inference_mode():
                return model(model_input)

        record = {
            "model": label,
            "history_periods": history,
            "parameter_count": parameter_count(model),
            "tree_nodes": None,
            "tree_leaves": None,
            "artifact_kib": checkpoints[0].stat().st_size / 1024.0,
            "training_wall_mean_s": train_mean,
            "training_wall_sd_s": train_sd,
        }
        record.update(benchmark(predict_density))
        records.append(record)

    point_info = final_models["lstm_mse"]
    point_checkpoints = [
        OUTPUT_DIR / item["checkpoint"]
        for item in point_info["checkpoints"].values()
    ]
    point_model = load_point_cpu(point_checkpoints[0])
    point_history = int(point_info["spec"]["history_length"])
    point_input = torch.from_numpy(
        np.array(X_scaled[0:1, -point_history:, :], dtype=np.float32, copy=True)
    )
    point_train_mean, point_train_sd = training_time_summary(point_checkpoints)

    def predict_point():
        with torch.inference_mode():
            return point_model(point_input)

    point_record = {
        "model": "LSTM-MSE Gaussian",
        "history_periods": point_history,
        "parameter_count": parameter_count(point_model),
        "tree_nodes": None,
        "tree_leaves": None,
        "artifact_kib": point_checkpoints[0].stat().st_size / 1024.0,
        "training_wall_mean_s": point_train_mean,
        "training_wall_sd_s": point_train_sd,
    }
    point_record.update(benchmark(predict_point))
    records.append(point_record)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": lock["protocol_version"],
        "input_split": "validation",
        "timed_scope": "prepared single-update-state model input to native model output",
        "model_loading_included": False,
        "feature_construction_included": False,
        "warmup_calls": WARMUP_CALLS,
        "timed_calls": TIMED_CALLS,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "scikit_learn": sklearn.__version__,
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "models": records}
    (RESULT_DIR / "inference_latency.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    frame = pd.DataFrame(records)
    frame.to_csv(RESULT_DIR / "inference_latency.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
