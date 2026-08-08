from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from data_processing import (
    compute_rolling_features,
    create_trajectory_indexed_dataset,
    load_and_preprocess_data,
    load_matched_data,
    split_time_series_data_by_period,
)
from probabilistic_metrics import (
    density_metrics,
    gaussian_crps,
    interval_score,
    ks_statistic,
    moving_block_period_indices,
    negative_mass,
    normal_interval,
    pinball_loss,
    period_mean_series,
    select_block_length,
    truncated_mixture_crps,
    truncated_mixture_nll,
    truncated_mixture_pit,
    truncated_mixture_quantile,
)
from forecast_models import (
    load_density_model,
    predict_density,
    predict_point,
    train_density_model,
    train_point_model,
)


ROOT = Path(__file__).resolve().parent
BASE_CONFIG_PATH = ROOT / "study_config.yaml"
EXPERIMENT_CONFIG_PATH = ROOT / "experiment_config.yaml"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_configs():
    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    with open(EXPERIMENT_CONFIG_PATH, "r", encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    output_dir = (ROOT / experiment["output_dir"]).resolve()
    return base, experiment, output_dir


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def append_access_event(output_dir, stage, detail):
    path = Path(output_dir) / "test_access_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"timestamp_utc": utc_now(), "stage": stage, "detail": detail}
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def prepare_data():
    base, experiment, output_dir = read_configs()
    g0_dir = output_dir / "g0"
    g0_dir.mkdir(parents=True, exist_ok=True)

    X, y, features, _, metadata = load_and_preprocess_data(
        base, return_sample_metadata=True
    )
    split_cfg = base["data_split"]
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        metadata_splits,
        boundaries,
    ) = split_time_series_data_by_period(
        X,
        y,
        metadata,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
    )

    expected_cases = experiment["expected_split"]["case_counts"]
    expected_periods = experiment["expected_split"]["period_counts"]
    if boundaries["case_counts"] != expected_cases:
        raise RuntimeError(
            f"Case counts differ from protocol: {boundaries['case_counts']}"
        )
    if boundaries["period_counts"] != expected_periods:
        raise RuntimeError(
            f"Period counts differ from protocol: {boundaries['period_counts']}"
        )

    x_scaler = MinMaxScaler(feature_range=(0, 1))
    x_scaler.fit(X_train.reshape(-1, X_train.shape[2]))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler.fit(y_train.reshape(-1, 1))
    joblib.dump(x_scaler, g0_dir / "x_scaler.joblib")
    joblib.dump(y_scaler, g0_dir / "y_scaler.joblib")

    arrays = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
    files = []
    for split, (X_split, y_split) in arrays.items():
        X_scaled = x_scaler.transform(
            X_split.reshape(-1, X_split.shape[2])
        ).reshape(X_split.shape).astype(np.float32)
        y_scaled = y_scaler.transform(y_split.reshape(-1, 1)).reshape(-1).astype(
            np.float32
        )
        split_files = {
            f"{split}_X_raw.npy": np.asarray(X_split, dtype=np.float32),
            f"{split}_X_scaled.npy": X_scaled,
            f"{split}_y_raw.npy": np.asarray(y_split, dtype=np.float32),
            f"{split}_y_scaled.npy": y_scaled,
        }
        for name, values in split_files.items():
            path = g0_dir / name
            np.save(path, values)
            files.append(path)
        metadata_path = g0_dir / f"{split}_metadata.csv"
        metadata_splits[split].to_csv(
            metadata_path, index=False, date_format="%Y-%m-%d %H:%M:%S"
        )
        files.append(metadata_path)

    files.extend([g0_dir / "x_scaler.joblib", g0_dir / "y_scaler.joblib"])
    manifest = {
        "protocol_version": experiment["protocol_version"],
        "created_utc": utc_now(),
        "features": features,
        "split": boundaries,
        "training_defaults": {
            "travel_time_median": float(np.median(y_train)),
            "travel_time_variance": float(np.var(y_train, ddof=1)),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
        },
        "files": {
            str(path.relative_to(output_dir)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        },
    }
    write_json(g0_dir / "split_manifest.json", manifest)
    append_access_event(
        output_dir,
        "G0_PREPARE",
        "Test arrays were materialized only to enforce the complete-period split; no test metric or model selection was computed.",
    )
    print(json.dumps(boundaries, indent=2))


def load_split(output_dir, split):
    g0_dir = Path(output_dir) / "g0"
    return {
        "X_raw": np.load(g0_dir / f"{split}_X_raw.npy", mmap_mode="c"),
        "X_scaled": np.load(g0_dir / f"{split}_X_scaled.npy", mmap_mode="c"),
        "y_raw": np.load(g0_dir / f"{split}_y_raw.npy", mmap_mode="c"),
        "y_scaled": np.load(g0_dir / f"{split}_y_scaled.npy", mmap_mode="c"),
        "metadata": pd.read_csv(
            g0_dir / f"{split}_metadata.csv", parse_dates=["prediction_time"]
        ),
    }


def development_data(output_dir):
    return load_split(output_dir, "train"), load_split(output_dir, "val")


def float_slug(value):
    return f"{float(value):.0e}".replace("+", "").replace("-0", "m").replace("-", "m")


def model_specs(experiment):
    grid = experiment["model_grid"]
    batch_size = int(experiment["training"]["batch_size"])
    protocol_slug = str(
        experiment.get("training_protocol_version", experiment["protocol_version"])
    ).replace(".", "")
    specs = []
    for capacity, lr, wd in itertools.product(
        grid["snapshot_capacities"],
        grid["learning_rates"],
        grid["weight_decays"],
    ):
        spec = {
            "family": "snapshot",
            "history_length": 1,
            **capacity,
            "learning_rate": float(lr),
            "weight_decay": float(wd),
        }
        spec["config_id"] = (
            f"snapshot_h{capacity['hidden_dim']}_l{capacity['num_layers']}"
            f"_lr{float_slug(lr)}_wd{float_slug(wd)}_bs{batch_size}_p{protocol_slug}"
        )
        specs.append(spec)
    for history, capacity, lr, wd in itertools.product(
        grid["history_lengths"],
        grid["recurrent_capacities"],
        grid["learning_rates"],
        grid["weight_decays"],
    ):
        spec = {
            "family": "recurrent",
            "history_length": int(history),
            **capacity,
            "learning_rate": float(lr),
            "weight_decay": float(wd),
        }
        spec["config_id"] = (
            f"recurrent_T{history}_h{capacity['hidden_dim']}_l{capacity['num_layers']}"
            f"_lr{float_slug(lr)}_wd{float_slug(wd)}_bs{batch_size}_p{protocol_slug}"
        )
        specs.append(spec)
    return specs


def raw_mixture_parameters(pis, mus_scaled, sigmas_scaled, y_scaler):
    mus = y_scaler.inverse_transform(mus_scaled.reshape(-1, 1)).reshape(
        mus_scaled.shape
    )
    sigmas = sigmas_scaled / float(y_scaler.scale_[0])
    return pis, mus, sigmas


def run_density_trial(
    output_dir,
    stage,
    spec,
    seed,
    train,
    val,
    y_scaler,
    training_cfg,
    num_mixtures=3,
):
    trial_dir = Path(output_dir) / "ap02" / stage
    key = f"{spec['config_id']}_seed{seed}"
    manifest_path = trial_dir / "trials" / f"{key}.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    history = int(spec["history_length"])
    X_train = train["X_scaled"][:, -history:, :]
    X_val = val["X_scaled"][:, -history:, :]
    checkpoint_path = trial_dir / "checkpoints" / f"{key}.pth"
    model, details = train_density_model(
        X_train,
        train["y_scaled"],
        X_val,
        val["y_scaled"],
        spec,
        num_mixtures,
        seed,
        checkpoint_path,
        max_epochs=int(training_cfg["max_epochs"]),
        min_epochs=int(training_cfg["min_epochs"]),
        patience=int(training_cfg["patience"]),
        batch_size=int(training_cfg["batch_size"]),
        min_delta=float(training_cfg["min_delta"]),
    )
    pis, mus_scaled, sigmas_scaled = predict_density(model, X_val)
    pis, mus, sigmas = raw_mixture_parameters(
        pis, mus_scaled, sigmas_scaled, y_scaler
    )
    metrics = density_metrics(pis, mus, sigmas, val["y_raw"])
    prediction_path = trial_dir / "validation_predictions" / f"{key}.npz"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_path,
        pis=pis.astype(np.float32),
        mus=mus.astype(np.float32),
        sigmas=sigmas.astype(np.float32),
        crps=metrics["crps"].astype(np.float32),
        nll=metrics["nll"].astype(np.float32),
        pit=metrics["pit"].astype(np.float32),
    )
    result = {
        "stage": stage,
        "config_id": spec["config_id"],
        "family": spec["family"],
        "seed": int(seed),
        "num_mixtures": int(num_mixtures),
        "spec": spec,
        "checkpoint": str(checkpoint_path.relative_to(output_dir)),
        "validation_predictions": str(prediction_path.relative_to(output_dir)),
        "val_crps_case": float(np.mean(metrics["crps"])),
        "val_crps_period": float(
            period_mean_series(metrics["crps"], val["metadata"])["value"].mean()
        ),
        "val_nll_case": float(np.mean(metrics["nll"])),
        "val_pit_ks": ks_statistic(metrics["pit"]),
        **{key: value for key, value in details.items() if key != "history"},
        "completed_utc": utc_now(),
    }
    write_json(manifest_path, result)
    return result


def collect_trial_manifests(path):
    rows = []
    for manifest in sorted(Path(path).glob("*.json")):
        rows.append(json.loads(manifest.read_text(encoding="utf-8")))
    return rows


def stage_training_config(experiment, stage):
    cfg = dict(experiment["training"][stage])
    cfg["batch_size"] = experiment["training"]["batch_size"]
    cfg["min_delta"] = experiment["training"]["min_delta"]
    return cfg


def run_stage_a(limit=None, family=None):
    _, experiment, output_dir = read_configs()
    train, val = development_data(output_dir)
    y_scaler = joblib.load(output_dir / "g0" / "y_scaler.joblib")
    all_specs = model_specs(experiment)
    specs = [spec for spec in all_specs if family in (None, spec["family"])]
    seeds = [int(x) for x in experiment["seeds"]["development"]]
    training_cfg = stage_training_config(experiment, "stage_a")
    completed_now = 0
    for spec in specs:
        for seed in seeds:
            manifest = (
                output_dir
                / "ap02"
                / "stage_a"
                / "trials"
                / f"{spec['config_id']}_seed{seed}.json"
            )
            if manifest.exists():
                continue
            print(f"[stage-a] {spec['config_id']} seed={seed}", flush=True)
            run_density_trial(
                output_dir,
                "stage_a",
                spec,
                seed,
                train,
                val,
                y_scaler,
                training_cfg,
                num_mixtures=int(experiment["model_grid"]["stage_a_mixtures"]),
            )
            completed_now += 1
            if limit is not None and completed_now >= limit:
                break
        if limit is not None and completed_now >= limit:
            break

    rows = collect_trial_manifests(
        output_dir / "ap02" / "stage_a" / "trials"
    )
    valid_ids = {spec["config_id"] for spec in all_specs}
    rows = [
        row
        for row in rows
        if row["config_id"] in valid_ids and int(row["seed"]) in seeds
    ]
    if rows:
        pd.DataFrame(rows).drop(columns=["spec"], errors="ignore").to_csv(
            output_dir / "ap02" / "stage_a" / "validation_trials.csv",
            index=False,
        )
    expected = len(all_specs) * len(seeds)
    print(f"[stage-a] completed {len(rows)}/{expected}")
    if len(rows) == expected:
        frame = pd.DataFrame(rows)
        aggregate = (
            frame.groupby(["family", "config_id"], as_index=False)
            .agg(
                mean_val_crps=("val_crps_case", "mean"),
                mean_val_nll=("val_nll_case", "mean"),
                mean_val_pit_ks=("val_pit_ks", "mean"),
                parameter_count=("parameter_count", "first"),
            )
        )
        finalists = []
        spec_by_id = {spec["config_id"]: spec for spec in all_specs}
        for family, group in aggregate.groupby("family"):
            group = group.sort_values(
                ["mean_val_crps", "parameter_count", "config_id"]
            ).head(2)
            for config_id in group["config_id"]:
                finalists.append(spec_by_id[config_id])
        write_json(
            output_dir / "ap02" / "stage_a" / "stage_b_finalists.json",
            {"created_utc": utc_now(), "finalists": finalists},
        )


def bootstrap_case_mean_se(scores, metadata, period_indices, blocks):
    period_order = period_indices["update_period_id"].to_numpy()
    frame = pd.DataFrame(
        {
            "update_period_id": metadata["update_period_id"].to_numpy(),
            "score": np.asarray(scores, dtype=float),
        }
    )
    grouped = frame.groupby("update_period_id", sort=False)["score"].agg(["sum", "count"])
    sums = grouped.loc[period_order, "sum"].to_numpy()
    counts = grouped.loc[period_order, "count"].to_numpy()
    values = np.sum(sums[blocks], axis=1) / np.sum(counts[blocks], axis=1)
    return float(np.std(values, ddof=1))


def run_stage_b(limit=None):
    _, experiment, output_dir = read_configs()
    finalists_path = output_dir / "ap02" / "stage_a" / "stage_b_finalists.json"
    if not finalists_path.exists():
        raise RuntimeError("Stage A is incomplete.")
    finalists = json.loads(finalists_path.read_text(encoding="utf-8"))["finalists"]
    train, val = development_data(output_dir)
    y_scaler = joblib.load(output_dir / "g0" / "y_scaler.joblib")
    seeds = [int(x) for x in experiment["seeds"]["final"]]
    training_cfg = stage_training_config(experiment, "stage_b")
    completed_now = 0
    for spec in finalists:
        for seed in seeds:
            manifest = (
                output_dir
                / "ap02"
                / "stage_b"
                / "trials"
                / f"{spec['config_id']}_seed{seed}.json"
            )
            if manifest.exists():
                continue
            print(f"[stage-b] {spec['config_id']} seed={seed}", flush=True)
            run_density_trial(
                output_dir,
                "stage_b",
                spec,
                seed,
                train,
                val,
                y_scaler,
                training_cfg,
                num_mixtures=int(experiment["model_grid"]["stage_a_mixtures"]),
            )
            completed_now += 1
            if limit is not None and completed_now >= limit:
                break
        if limit is not None and completed_now >= limit:
            break

    rows = collect_trial_manifests(output_dir / "ap02" / "stage_b" / "trials")
    valid_ids = {spec["config_id"] for spec in finalists}
    rows = [
        row
        for row in rows
        if row["config_id"] in valid_ids and int(row["seed"]) in seeds
    ]
    if rows:
        pd.DataFrame(rows).drop(columns=["spec"], errors="ignore").to_csv(
            output_dir / "ap02" / "stage_b" / "validation_trials.csv",
            index=False,
        )
    expected = len(finalists) * len(seeds)
    print(f"[stage-b] completed {len(rows)}/{expected}")
    if len(rows) != expected:
        return

    period_sequences = {}
    scores_by_config = {}
    for spec in finalists:
        seed_scores = []
        for seed in seeds:
            pred_path = (
                output_dir
                / "ap02"
                / "stage_b"
                / "validation_predictions"
                / f"{spec['config_id']}_seed{seed}.npz"
            )
            with np.load(pred_path) as prediction:
                seed_scores.append(prediction["crps"].astype(float))
        mean_scores = np.mean(seed_scores, axis=0)
        scores_by_config[spec["config_id"]] = mean_scores
        period_sequences[spec["config_id"]] = period_mean_series(
            mean_scores, val["metadata"]
        )

    fixed_predictions = output_dir / "ap02" / "fixed_references" / "validation_predictions.npz"
    if not fixed_predictions.exists():
        raise RuntimeError("Fixed-reference validation predictions are required for AP-09.")
    with np.load(fixed_predictions) as prediction:
        period_sequences["naive_gaussian"] = period_mean_series(
            prediction["naive_crps"].astype(float), val["metadata"]
        )
        period_sequences["ridge_gaussian"] = period_mean_series(
            prediction["ridge_crps"].astype(float), val["metadata"]
        )
    ap03_predictions = output_dir / "ap03" / "validation_predictions.npz"
    if not ap03_predictions.exists():
        raise RuntimeError("AP-03 validation predictions are required for AP-09.")
    with np.load(ap03_predictions) as prediction:
        quantile_scores = prediction["raw_crps"].astype(float)
    period_sequences["snapshot_quantile_induced_gaussian"] = period_mean_series(
        quantile_scores, val["metadata"]
    )

    block_length, diagnostics = select_block_length(
        period_sequences, experiment["ap09"]["block_candidates"]
    )
    period_index, blocks = moving_block_period_indices(
        val["metadata"],
        block_length,
        int(experiment["ap09"]["validation_repeats"]),
        seed=9042,
    )
    ap09_dir = output_dir / "ap09"
    ap09_dir.mkdir(parents=True, exist_ok=True)
    np.save(ap09_dir / "validation_bootstrap_indices.npy", blocks)
    pd.DataFrame(diagnostics).to_csv(ap09_dir / "validation_acf.csv", index=False)
    write_json(
        ap09_dir / "block_length_selection.json",
        {
            "created_utc": utc_now(),
            "block_length_periods": block_length,
            "candidate_lengths": experiment["ap09"]["block_candidates"],
            "diagnostics": diagnostics,
        },
    )

    frame = pd.DataFrame(rows)
    curve_rows = []
    for row in rows:
        sidecar = (output_dir / row["checkpoint"]).with_suffix(".json")
        details = json.loads(sidecar.read_text(encoding="utf-8"))
        for epoch in details["history"]:
            curve_rows.append(
                {
                    "family": row["family"],
                    "config_id": row["config_id"],
                    "seed": row["seed"],
                    **epoch,
                }
            )
    pd.DataFrame(curve_rows).to_csv(
        output_dir / "ap02" / "training_curves.csv", index=False
    )
    aggregate = (
        frame.groupby(["family", "config_id"], as_index=False)
        .agg(
            mean_val_crps=("val_crps_case", "mean"),
            mean_val_nll=("val_nll_case", "mean"),
            mean_val_pit_ks=("val_pit_ks", "mean"),
            parameter_count=("parameter_count", "first"),
        )
    )
    selected = {}
    specs_by_id = {spec["config_id"]: spec for spec in finalists}
    for family, group in aggregate.groupby("family"):
        group = group.sort_values("mean_val_crps")
        best = group.iloc[0]
        best_scores = scores_by_config[best["config_id"]]
        one_se = bootstrap_case_mean_se(
            best_scores, val["metadata"], period_index, blocks
        )
        eligible = group[
            group["mean_val_crps"] <= float(best["mean_val_crps"]) + one_se
        ].sort_values(["parameter_count", "mean_val_crps", "config_id"])
        winner = eligible.iloc[0]
        spec = specs_by_id[winner["config_id"]]
        selected[family] = {
            "spec": spec,
            "mean_val_crps": float(winner["mean_val_crps"]),
            "mean_val_nll": float(winner["mean_val_nll"]),
            "mean_val_pit_ks": float(winner["mean_val_pit_ks"]),
            "one_se": one_se,
            "parameter_count": int(winner["parameter_count"]),
        }
    write_json(
        output_dir / "ap02" / "selected_models.json",
        {
            "created_utc": utc_now(),
            "block_length_periods": block_length,
            "selected": selected,
        },
    )
    split_manifest = json.loads(
        (output_dir / "g0" / "split_manifest.json").read_text(encoding="utf-8")
    )
    audit_rows = []
    for family, selection in selected.items():
        spec = selection["spec"]
        for seed in seeds:
            row = next(
                item
                for item in rows
                if item["config_id"] == spec["config_id"]
                and int(item["seed"]) == seed
            )
            checkpoint = output_dir / row["checkpoint"]
            audit_rows.append(
                {
                    "family": family,
                    "config_id": spec["config_id"],
                    "seed": seed,
                    "history_length": int(spec["history_length"]),
                    "parameter_count": int(row["parameter_count"]),
                    "epochs_ran": int(row["epochs_ran"]),
                    "optimizer_steps": int(row["optimizer_steps"]),
                    "train_X_sha256": split_manifest["files"][
                        str(Path("g0") / "train_X_scaled.npy")
                    ]["sha256"],
                    "validation_X_sha256": split_manifest["files"][
                        str(Path("g0") / "val_X_scaled.npy")
                    ]["sha256"],
                    "feature_order": json.dumps(split_manifest["features"]),
                    "max_validation_prediction_time": str(
                        val["metadata"]["prediction_time"].max()
                    ),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "test_data_used": False,
                }
            )
    pd.DataFrame(audit_rows).to_csv(
        output_dir / "ap02" / "fairness_audit.csv", index=False
    )
    write_json(
        output_dir / "ap02" / "manifest.json",
        {
            "created_utc": utc_now(),
            "protocol_version": experiment["protocol_version"],
            "training_protocol_version": experiment.get("training_protocol_version"),
            "families": ["snapshot", "recurrent"],
            "stage_a_trials": len(model_specs(experiment))
            * len(experiment["seeds"]["development"]),
            "stage_b_trials": len(finalists) * len(seeds),
            "selected_models": selected,
            "tcn_status": "excluded categorically before test opening",
        },
    )
    write_json(
        output_dir / "ap09" / "manifest.json",
        {
            "created_utc": utc_now(),
            "sampling_unit": "unique update period",
            "validation_repeats": int(experiment["ap09"]["validation_repeats"]),
            "test_repeats": int(experiment["ap09"]["test_repeats"]),
            "block_length_periods": block_length,
            "blocks_cross_time_gaps": False,
        },
    )


def run_k_scan(limit=None):
    _, experiment, output_dir = read_configs()
    selected_path = output_dir / "ap02" / "selected_models.json"
    block_path = output_dir / "ap09" / "block_length_selection.json"
    if not selected_path.exists() or not block_path.exists():
        raise RuntimeError("Stage B selection and AP-09 block length are required.")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    base_spec = selected["selected"]["recurrent"]["spec"]
    block_length = int(
        json.loads(block_path.read_text(encoding="utf-8"))["block_length_periods"]
    )
    train, val = development_data(output_dir)
    y_scaler = joblib.load(output_dir / "g0" / "y_scaler.joblib")
    seeds = [int(x) for x in experiment["seeds"]["final"]]
    training_cfg = stage_training_config(experiment, "stage_b")
    k_values = [int(x) for x in experiment["model_grid"]["k_scan"]]
    completed_now = 0
    for k in k_values:
        spec = dict(base_spec)
        spec["encoder_config_id"] = base_spec["config_id"]
        spec["config_id"] = f"{base_spec['config_id']}_K{k}"
        for seed in seeds:
            manifest = (
                output_dir
                / "ap02"
                / "ap04_k_scan"
                / "trials"
                / f"{spec['config_id']}_seed{seed}.json"
            )
            if manifest.exists():
                continue
            print(f"[ap04] K={k} seed={seed}", flush=True)
            if k == int(experiment["model_grid"]["stage_a_mixtures"]):
                source_path = (
                    output_dir
                    / "ap02"
                    / "stage_b"
                    / "trials"
                    / f"{base_spec['config_id']}_seed{seed}.json"
                )
                if not source_path.exists():
                    raise RuntimeError(f"Missing reusable Stage-B K=3 trial: {source_path}")
                source = json.loads(source_path.read_text(encoding="utf-8"))
                reused = {
                    **source,
                    "stage": "ap04_k_scan",
                    "config_id": spec["config_id"],
                    "num_mixtures": k,
                    "spec": spec,
                    "reused_from": str(source_path.relative_to(output_dir)),
                    "completed_utc": utc_now(),
                }
                write_json(manifest, reused)
            else:
                run_density_trial(
                    output_dir,
                    "ap04_k_scan",
                    spec,
                    seed,
                    train,
                    val,
                    y_scaler,
                    training_cfg,
                    num_mixtures=k,
                )
            completed_now += 1
            if limit is not None and completed_now >= limit:
                break
        if limit is not None and completed_now >= limit:
            break

    rows = collect_trial_manifests(
        output_dir / "ap02" / "ap04_k_scan" / "trials"
    )
    valid_ids = {f"{base_spec['config_id']}_K{k}" for k in k_values}
    rows = [
        row
        for row in rows
        if row["config_id"] in valid_ids and int(row["seed"]) in seeds
    ]
    ap04_dir = output_dir / "ap04"
    ap04_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).drop(columns=["spec"], errors="ignore").to_csv(
            ap04_dir / "validation_k_scan_by_seed.csv", index=False
        )
    expected = len(k_values) * len(seeds)
    print(f"[ap04] completed {len(rows)}/{expected}")
    if len(rows) != expected:
        return

    frame = pd.DataFrame(rows)
    aggregate = (
        frame.groupby("num_mixtures", as_index=False)
        .agg(
            mean_val_crps=("val_crps_case", "mean"),
            mean_val_nll=("val_nll_case", "mean"),
            mean_val_pit_ks=("val_pit_ks", "mean"),
            parameter_count=("parameter_count", "first"),
        )
        .rename(columns={"num_mixtures": "K"})
        .sort_values("K")
    )
    aggregate.to_csv(ap04_dir / "validation_k_scan.csv", index=False)
    epsilon = float(experiment["selection"]["epsilon_cal"])
    min_ks = float(aggregate["mean_val_pit_ks"].min())
    feasible = aggregate[aggregate["mean_val_pit_ks"] <= min_ks + epsilon]
    best = feasible.sort_values("mean_val_crps").iloc[0]
    best_k = int(best["K"])
    seed_scores = []
    best_rows = {
        int(row["seed"]): row
        for row in rows
        if int(row["num_mixtures"]) == best_k
    }
    for seed in seeds:
        path = output_dir / best_rows[seed]["validation_predictions"]
        with np.load(path) as prediction:
            seed_scores.append(prediction["crps"].astype(float))
    mean_scores = np.mean(seed_scores, axis=0)
    period_index, blocks = moving_block_period_indices(
        val["metadata"],
        block_length,
        int(experiment["ap09"]["validation_repeats"]),
        seed=9042,
    )
    one_se = bootstrap_case_mean_se(
        mean_scores, val["metadata"], period_index, blocks
    )
    within = feasible[
        feasible["mean_val_crps"] <= float(best["mean_val_crps"]) + one_se
    ].sort_values(["K", "mean_val_nll"])
    selected_k = int(within.iloc[0]["K"])
    write_json(
        ap04_dir / "selected_k.json",
        {
            "created_utc": utc_now(),
            "encoder_spec": base_spec,
            "K_best_crps": best_k,
            "K_star": selected_k,
            "one_se": one_se,
            "epsilon_cal": epsilon,
            "h3_activated": selected_k > 2,
        },
    )
    write_json(
        ap04_dir / "selection_rule.json",
        {
            "created_utc": utc_now(),
            "calibration_feasible_rule": "mean PIT KS <= minimum mean PIT KS + epsilon_cal",
            "epsilon_cal": epsilon,
            "one_se_rule": "smallest K within one block-bootstrap SE of feasible best CRPS",
            "block_length_periods": block_length,
        },
    )
    audit = aggregate.copy()
    audit["calibration_feasible"] = audit["mean_val_pit_ks"] <= min_ks + epsilon
    audit["selected_K_star"] = audit["K"] == selected_k
    audit["H3_reference_K2"] = audit["K"] == 2
    audit.to_csv(ap04_dir / "complexity_audit.csv", index=False)
    write_json(
        ap04_dir / "manifest.json",
        {
            "created_utc": utc_now(),
            "K_values": k_values,
            "seeds": seeds,
            "K3_reused_from_stage_b": True,
            "K_star": selected_k,
            "H3_activated": selected_k > 2,
        },
    )


def run_final_models(limit=None):
    _, experiment, output_dir = read_configs()
    selected_path = output_dir / "ap02" / "selected_models.json"
    selected_k_path = output_dir / "ap04" / "selected_k.json"
    if not selected_path.exists() or not selected_k_path.exists():
        raise RuntimeError("AP-02 model selection and AP-04 K selection are required.")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))["selected"]
    selected_k = int(
        json.loads(selected_k_path.read_text(encoding="utf-8"))["K_star"]
    )
    train, val = development_data(output_dir)
    y_scaler = joblib.load(output_dir / "g0" / "y_scaler.joblib")
    seeds = [int(x) for x in experiment["seeds"]["final"]]
    training_cfg = stage_training_config(experiment, "stage_b")
    completed_now = 0

    snapshot_spec = selected["snapshot"]["spec"]
    snapshot_checkpoints = {}
    if selected_k == int(experiment["model_grid"]["stage_a_mixtures"]):
        for seed in seeds:
            path = (
                output_dir
                / "ap02"
                / "stage_b"
                / "checkpoints"
                / f"{snapshot_spec['config_id']}_seed{seed}.pth"
            )
            if not path.exists():
                raise RuntimeError(f"Missing selected Snapshot checkpoint: {path}")
            snapshot_checkpoints[str(seed)] = str(path.relative_to(output_dir))
    else:
        final_spec = dict(snapshot_spec)
        final_spec["encoder_config_id"] = snapshot_spec["config_id"]
        final_spec["config_id"] = f"{snapshot_spec['config_id']}_K{selected_k}"
        for seed in seeds:
            manifest = (
                output_dir
                / "ap02"
                / "final_heads"
                / "trials"
                / f"{final_spec['config_id']}_seed{seed}.json"
            )
            if not manifest.exists():
                print(f"[final-heads] snapshot K={selected_k} seed={seed}", flush=True)
                run_density_trial(
                    output_dir,
                    "final_heads",
                    final_spec,
                    seed,
                    train,
                    val,
                    y_scaler,
                    training_cfg,
                    num_mixtures=selected_k,
                )
                completed_now += 1
                if limit is not None and completed_now >= limit:
                    return
            row = json.loads(manifest.read_text(encoding="utf-8"))
            snapshot_checkpoints[str(seed)] = row["checkpoint"]

    recurrent_spec = selected["recurrent"]["spec"]
    recurrent_checkpoints = {}
    recurrent_k2_checkpoints = {}
    for k, destination in (
        (selected_k, recurrent_checkpoints),
        (2, recurrent_k2_checkpoints),
    ):
        config_id = f"{recurrent_spec['config_id']}_K{k}"
        for seed in seeds:
            if k == int(experiment["model_grid"]["stage_a_mixtures"]):
                path = (
                    output_dir
                    / "ap02"
                    / "stage_b"
                    / "checkpoints"
                    / f"{recurrent_spec['config_id']}_seed{seed}.pth"
                )
            else:
                path = (
                    output_dir
                    / "ap02"
                    / "ap04_k_scan"
                    / "checkpoints"
                    / f"{config_id}_seed{seed}.pth"
                )
            if not path.exists():
                raise RuntimeError(f"Missing recurrent K={k} checkpoint: {path}")
            destination[str(seed)] = str(path.relative_to(output_dir))

    point_dir = output_dir / "ap02" / "final_models" / "lstm_mse"
    point_rows = []
    history = int(recurrent_spec["history_length"])
    for seed in seeds:
        manifest_path = point_dir / "trials" / f"seed{seed}.json"
        if manifest_path.exists():
            point_rows.append(json.loads(manifest_path.read_text(encoding="utf-8")))
            continue
        checkpoint_path = point_dir / "checkpoints" / f"seed{seed}.pth"
        print(f"[final-models] LSTM-MSE seed={seed}", flush=True)
        model, details = train_point_model(
            train["X_scaled"][:, -history:, :],
            train["y_scaled"],
            val["X_scaled"][:, -history:, :],
            val["y_scaled"],
            recurrent_spec,
            seed,
            checkpoint_path,
            max_epochs=int(training_cfg["max_epochs"]),
            min_epochs=int(training_cfg["min_epochs"]),
            patience=int(training_cfg["patience"]),
            batch_size=int(training_cfg["batch_size"]),
            min_delta=float(training_cfg["min_delta"]),
        )
        train_scaled = predict_point(
            model, train["X_scaled"][:, -history:, :]
        )
        train_mu = y_scaler.inverse_transform(train_scaled[:, None])[:, 0]
        residual_sigma = float(
            np.std(np.asarray(train["y_raw"]) - train_mu, ddof=1)
        )
        val_scaled = predict_point(model, val["X_scaled"][:, -history:, :])
        val_mu = y_scaler.inverse_transform(val_scaled[:, None])[:, 0]
        val_metrics = density_metrics(
            np.ones((len(val_mu), 1)),
            val_mu[:, None],
            np.full((len(val_mu), 1), residual_sigma),
            val["y_raw"],
        )
        row = {
            "seed": seed,
            "spec": recurrent_spec,
            "checkpoint": str(checkpoint_path.relative_to(output_dir)),
            "residual_sigma_training_seconds": residual_sigma,
            "val_crps_case": float(np.mean(val_metrics["crps"])),
            "val_nll_case": float(np.mean(val_metrics["nll"])),
            **{key: value for key, value in details.items() if key != "history"},
            "completed_utc": utc_now(),
        }
        write_json(manifest_path, row)
        point_rows.append(row)
        completed_now += 1
        if limit is not None and completed_now >= limit:
            return

    if len(point_rows) != len(seeds):
        return
    point_checkpoints = {
        str(row["seed"]): {
            "checkpoint": row["checkpoint"],
            "residual_sigma_training_seconds": row[
                "residual_sigma_training_seconds"
            ],
        }
        for row in point_rows
    }
    write_json(
        output_dir / "ap02" / "final_models.json",
        {
            "created_utc": utc_now(),
            "K_star": selected_k,
            "snapshot": {
                "spec": snapshot_spec,
                "checkpoints": snapshot_checkpoints,
            },
            "recurrent": {
                "spec": recurrent_spec,
                "checkpoints": recurrent_checkpoints,
                "K2_checkpoints": recurrent_k2_checkpoints,
            },
            "lstm_mse": {
                "spec": recurrent_spec,
                "checkpoints": point_checkpoints,
            },
        },
    )


def run_fixed_references():
    _, _, output_dir = read_configs()
    train, val = development_data(output_dir)
    reference_dir = output_dir / "ap02" / "fixed_references"
    reference_dir.mkdir(parents=True, exist_ok=True)
    train_y = np.asarray(train["y_raw"], dtype=float)
    val_y = np.asarray(val["y_raw"], dtype=float)
    naive_mu = float(np.mean(train_y))
    naive_sigma = float(np.std(train_y, ddof=1))
    naive = density_metrics(
        np.ones((len(val_y), 1)),
        np.full((len(val_y), 1), naive_mu),
        np.full((len(val_y), 1), naive_sigma),
        val_y,
    )
    ridge = Ridge(alpha=1.0)
    X_train = np.asarray(train["X_scaled"][:, -1, :])
    X_val = np.asarray(val["X_scaled"][:, -1, :])
    ridge.fit(X_train, train_y)
    train_mu = ridge.predict(X_train)
    ridge_sigma = float(np.std(train_y - train_mu, ddof=1))
    val_mu = ridge.predict(X_val)
    ridge_metrics = density_metrics(
        np.ones((len(val_y), 1)),
        val_mu[:, None],
        np.full((len(val_y), 1), ridge_sigma),
        val_y,
    )
    joblib.dump(ridge, reference_dir / "ridge.joblib")
    np.savez_compressed(
        reference_dir / "validation_predictions.npz",
        naive_crps=naive["crps"].astype(np.float32),
        ridge_crps=ridge_metrics["crps"].astype(np.float32),
        ridge_mu=val_mu.astype(np.float32),
    )
    write_json(
        reference_dir / "manifest.json",
        {
            "created_utc": utc_now(),
            "naive": {
                "mu_training_seconds": naive_mu,
                "sigma_training_seconds": naive_sigma,
                "val_crps_case": float(np.mean(naive["crps"])),
                "val_nll_case": float(np.mean(naive["nll"])),
            },
            "ridge": {
                "alpha": 1.0,
                "input": "scaled snapshot",
                "residual_sigma_training_seconds": ridge_sigma,
                "val_crps_case": float(np.mean(ridge_metrics["crps"])),
                "val_nll_case": float(np.mean(ridge_metrics["nll"])),
                "model": str((reference_dir / "ridge.joblib").relative_to(output_dir)),
            },
        },
    )


def depth_slug(value):
    return "none" if value is None else str(int(value))


def quantile_tree_specs(experiment, kind):
    cfg = experiment["gbdt"]
    specs = []
    if kind == "snapshot":
        values = itertools.product(
            cfg["max_depth"],
            cfg["learning_rate"],
            cfg["max_iter"],
            cfg["l2_regularization"],
        )
        for depth, lr, iterations, l2 in values:
            specs.append(
                {
                    "kind": kind,
                    "history_length": 1,
                    "max_depth": depth,
                    "learning_rate": float(lr),
                    "max_iter": int(iterations),
                    "l2_regularization": float(l2),
                    "config_id": (
                        f"snapshot_d{depth_slug(depth)}_lr{float_slug(lr)}"
                        f"_i{iterations}_l2{float_slug(l2)}"
                    ),
                }
            )
    elif kind == "lagged":
        values = itertools.product(
            experiment["model_grid"]["history_lengths"],
            [3, 5],
            cfg["learning_rate"],
            cfg["max_iter"],
        )
        for history, depth, lr, iterations in values:
            specs.append(
                {
                    "kind": kind,
                    "history_length": int(history),
                    "max_depth": int(depth),
                    "learning_rate": float(lr),
                    "max_iter": int(iterations),
                    "l2_regularization": 1.0,
                    "config_id": (
                        f"lagged_T{history}_d{depth}_lr{float_slug(lr)}_i{iterations}"
                    ),
                }
            )
    else:
        raise ValueError(kind)
    return specs


def fit_quantile_models(X_train, y_train, spec, experiment):
    models = []
    for quantile in experiment["gbdt"]["quantiles"]:
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=float(quantile),
            max_depth=spec["max_depth"],
            learning_rate=spec["learning_rate"],
            max_iter=spec["max_iter"],
            l2_regularization=spec["l2_regularization"],
            min_samples_leaf=int(experiment["gbdt"]["min_samples_leaf"]),
            max_bins=255,
            early_stopping=False,
            random_state=0,
        )
        model.fit(X_train, y_train)
        models.append(model)
    return models


def run_quantile_trees(kind, limit=None):
    _, experiment, output_dir = read_configs()
    train, val = development_data(output_dir)
    specs = quantile_tree_specs(experiment, kind)
    tree_dir = output_dir / ("ap03" if kind == "snapshot" else "ap02") / f"{kind}_quantile_gbdt"
    trial_dir = tree_dir / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)
    completed_now = 0
    for spec in specs:
        manifest_path = trial_dir / f"{spec['config_id']}.json"
        if manifest_path.exists():
            continue
        history = int(spec["history_length"])
        X_train = np.asarray(train["X_raw"][:, -history:, :]).reshape(
            len(train["y_raw"]), -1
        )
        X_val = np.asarray(val["X_raw"][:, -history:, :]).reshape(
            len(val["y_raw"]), -1
        )
        print(f"[{kind}-gbdt] {spec['config_id']}", flush=True)
        models = fit_quantile_models(X_train, train["y_raw"], spec, experiment)
        predictions = np.column_stack([model.predict(X_val) for model in models])
        losses = np.column_stack(
            [
                pinball_loss(val["y_raw"], predictions[:, index], quantile)
                for index, quantile in enumerate(experiment["gbdt"]["quantiles"])
            ]
        )
        prediction_path = tree_dir / "validation_predictions" / f"{spec['config_id']}.npz"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_path,
            quantiles=predictions.astype(np.float32),
            pinball=losses.astype(np.float32),
        )
        result = {
            "config_id": spec["config_id"],
            "spec": spec,
            "mean_pinball": float(np.mean(losses)),
            "pinball_q10": float(np.mean(losses[:, 0])),
            "pinball_q50": float(np.mean(losses[:, 1])),
            "pinball_q90": float(np.mean(losses[:, 2])),
            "crossing_rate": float(
                np.mean(
                    (predictions[:, 0] > predictions[:, 1])
                    | (predictions[:, 1] > predictions[:, 2])
                )
            ),
            "validation_predictions": str(prediction_path.relative_to(output_dir)),
            "completed_utc": utc_now(),
        }
        write_json(manifest_path, result)
        completed_now += 1
        if limit is not None and completed_now >= limit:
            break

    rows = collect_trial_manifests(trial_dir)
    valid_ids = {spec["config_id"] for spec in specs}
    rows = [row for row in rows if row["config_id"] in valid_ids]
    if rows:
        pd.DataFrame(rows).drop(columns=["spec"], errors="ignore").to_csv(
            tree_dir / "validation_grid.csv", index=False
        )
    print(f"[{kind}-gbdt] completed {len(rows)}/{len(specs)}")
    if len(rows) != len(specs):
        return
    frame = pd.DataFrame(rows).sort_values("mean_pinball")
    best_id = frame.iloc[0]["config_id"]
    spec_by_id = {spec["config_id"]: spec for spec in specs}
    best_spec = spec_by_id[best_id]
    history = int(best_spec["history_length"])
    X_train = np.asarray(train["X_raw"][:, -history:, :]).reshape(
        len(train["y_raw"]), -1
    )
    models = fit_quantile_models(X_train, train["y_raw"], best_spec, experiment)
    model_path = tree_dir / "selected_models.joblib"
    joblib.dump(models, model_path)
    write_json(
        tree_dir / "selected_model.json",
        {
            "created_utc": utc_now(),
            "spec": best_spec,
            "mean_pinball": float(frame.iloc[0]["mean_pinball"]),
            "models": str(model_path.relative_to(output_dir)),
        },
    )


def quantiles_to_gaussian(predictions):
    predictions = np.asarray(predictions, dtype=float)
    raw_crossing = (predictions[:, 0] > predictions[:, 1]) | (
        predictions[:, 1] > predictions[:, 2]
    )
    lower = np.minimum(predictions[:, 0], predictions[:, 2])
    upper = np.maximum(predictions[:, 0], predictions[:, 2])
    mu = np.clip(predictions[:, 1], lower, upper)
    sigma = np.maximum((upper - lower) / 2.563103, 1e-3)
    return mu, sigma, raw_crossing


def negative_mass_summary(values, model, split):
    values = np.asarray(values, dtype=float)
    row = {
        "model": model,
        "split": split,
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }
    for threshold in (1e-6, 1e-4, 1e-3, 1e-2, 0.05):
        row[f"share_gt_{float_slug(threshold)}"] = float(np.mean(values > threshold))
    return row


def run_ap03_development():
    _, experiment, output_dir = read_configs()
    tree_dir = output_dir / "ap03" / "snapshot_quantile_gbdt"
    selected_path = tree_dir / "selected_model.json"
    if not selected_path.exists():
        raise RuntimeError("AP-03 snapshot quantile GBDT selection is required.")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    models = joblib.load(output_dir / selected["models"])
    train, val = development_data(output_dir)
    ap03_dir = output_dir / "ap03"
    ap03_dir.mkdir(parents=True, exist_ok=True)

    numerical_rows = []
    mass_rows = []
    val_payload = None
    for split_name, split in (("train", train), ("validation", val)):
        X = np.asarray(split["X_raw"][:, -1, :])
        predictions = np.column_stack([model.predict(X) for model in models])
        mu, sigma, raw_crossing = quantiles_to_gaussian(predictions)
        pis = np.ones((len(mu), 1), dtype=float)
        mus = mu[:, None]
        sigmas = sigma[:, None]
        coarse = truncated_mixture_crps(
            pis, mus, sigmas, split["y_raw"], points=2001
        )
        fine = truncated_mixture_crps(
            pis, mus, sigmas, split["y_raw"], points=4001
        )
        difference = np.abs(coarse - fine)
        numerical_rows.append(
            {
                "split": split_name,
                "n_cases": len(mu),
                "mean_abs_difference_seconds": float(np.mean(difference)),
                "max_abs_difference_seconds": float(np.max(difference)),
                "all_finite_2001": bool(np.isfinite(coarse).all()),
                "all_finite_4001": bool(np.isfinite(fine).all()),
            }
        )
        masses = negative_mass(pis, mus, sigmas)
        mass_rows.append(
            negative_mass_summary(
                masses, "snapshot_quantile_induced_gaussian", split_name
            )
        )
        if split_name == "validation":
            raw = density_metrics(pis, mus, sigmas, split["y_raw"])
            truncated_lo = truncated_mixture_quantile(pis, mus, sigmas, 0.05)
            truncated_hi = truncated_mixture_quantile(pis, mus, sigmas, 0.95)
            truncated_pit = truncated_mixture_pit(
                pis, mus, sigmas, split["y_raw"]
            )
            pinball = np.column_stack(
                [
                    pinball_loss(split["y_raw"], predictions[:, index], quantile)
                    for index, quantile in enumerate(experiment["gbdt"]["quantiles"])
                ]
            )
            val_payload = {
                "quantiles": predictions.astype(np.float32),
                "pinball": pinball.astype(np.float32),
                "mu": mu.astype(np.float32),
                "sigma": sigma.astype(np.float32),
                "raw_crossing": raw_crossing.astype(np.uint8),
                "raw_crps": raw["crps"].astype(np.float32),
                "raw_nll": raw["nll"].astype(np.float32),
                "raw_pit": raw["pit"].astype(np.float32),
                "raw_lo90": raw["lo90"].astype(np.float32),
                "raw_hi90": raw["hi90"].astype(np.float32),
                "negative_mass": masses.astype(np.float32),
                "truncated_crps": coarse.astype(np.float32),
                "truncated_nll": truncated_mixture_nll(
                    pis, mus, sigmas, split["y_raw"]
                ).astype(np.float32),
                "truncated_pit": truncated_pit.astype(np.float32),
                "truncated_lo90": truncated_lo.astype(np.float32),
                "truncated_hi90": truncated_hi.astype(np.float32),
            }

    numerical = pd.DataFrame(numerical_rows)
    numerical.to_csv(ap03_dir / "truncated_crps_grid_check.csv", index=False)
    if (
        not numerical["all_finite_2001"].all()
        or not numerical["all_finite_4001"].all()
        or (numerical["mean_abs_difference_seconds"] > 0.01).any()
    ):
        raise RuntimeError("AP-03 truncated CRPS integration check failed.")

    np.savez_compressed(ap03_dir / "validation_predictions.npz", **val_payload)
    pd.DataFrame(mass_rows).to_csv(
        ap03_dir / "validation_negative_mass_summary.csv", index=False
    )
    y_val = np.asarray(val["y_raw"], dtype=float)
    quantiles = val_payload["quantiles"]
    native = {
        "model": "snapshot_quantile_gbdt_native",
        "mean_pinball": float(np.mean(val_payload["pinball"])),
        "pinball_q10": float(np.mean(val_payload["pinball"][:, 0])),
        "pinball_q50": float(np.mean(val_payload["pinball"][:, 1])),
        "pinball_q90": float(np.mean(val_payload["pinball"][:, 2])),
        "calibration_q10": float(np.mean(y_val <= quantiles[:, 0]) - 0.10),
        "calibration_q50": float(np.mean(y_val <= quantiles[:, 1]) - 0.50),
        "calibration_q90": float(np.mean(y_val <= quantiles[:, 2]) - 0.90),
        "raw_crossing_rate": float(np.mean(val_payload["raw_crossing"])),
        "coverage80": float(
            np.mean((y_val >= quantiles[:, 0]) & (y_val <= quantiles[:, 2]))
        ),
        "width80": float(np.mean(quantiles[:, 2] - quantiles[:, 0])),
        "nll": None,
    }
    density_rows = []
    for variant in ("raw", "truncated"):
        lo = val_payload[f"{variant}_lo90"]
        hi = val_payload[f"{variant}_hi90"]
        pit = val_payload[f"{variant}_pit"]
        density_rows.append(
            {
                "model": "snapshot_quantile_induced_gaussian",
                "variant": variant,
                "mean_crps": float(np.mean(val_payload[f"{variant}_crps"])),
                "mean_nll": float(np.mean(val_payload[f"{variant}_nll"])),
                "pit_ks": ks_statistic(pit),
                "coverage90": float(np.mean((y_val >= lo) & (y_val <= hi))),
                "width90": float(np.mean(hi - lo)),
                "interval_score90": float(
                    np.mean(interval_score(y_val, lo, hi, alpha=0.1))
                ),
            }
        )
    write_json(ap03_dir / "native_quantile_validation.json", native)
    pd.DataFrame(density_rows).to_csv(
        ap03_dir / "induced_density_validation.csv", index=False
    )
    write_json(
        ap03_dir / "interval_definition.json",
        {
            "created_utc": utc_now(),
            "service_interval": "raw_quantile_induced_gaussian_90pct",
            "selection_status": "predeclared working interval; truncation is diagnostic",
            "sigma_min_seconds": 0.001,
            "truncated_crps_grid_points": 2001,
        },
    )
    write_json(
        ap03_dir / "manifest.json",
        {
            "created_utc": utc_now(),
            "selected_quantile_tree": selected,
            "native_quantiles": experiment["gbdt"]["quantiles"],
            "working_density": "quantile-induced Gaussian",
            "positive_support_comparison": "zero truncation",
            "truncated_crps_grid_points": 2001,
            "grid_check_passed": True,
        },
    )


def period_sensing_state(split):
    metadata = split["metadata"]
    snapshot = np.asarray(split["X_raw"][:, -1, :])
    cases = metadata[["case_id", "update_period_id", "prediction_time"]].copy()
    cases["R_comp"] = snapshot[:, 2]
    cases["rho_t"] = snapshot[:, 3] / 300.0
    variation = cases.groupby("update_period_id")[["R_comp", "rho_t"]].nunique()
    if (variation.to_numpy() > 1).any():
        raise RuntimeError("Sensing state varies within an update period.")
    periods = (
        cases.groupby("update_period_id", sort=False, as_index=False)
        .agg(
            prediction_time=("prediction_time", "first"),
            R_comp=("R_comp", "first"),
            rho_t=("rho_t", "first"),
            n_cases=("case_id", "size"),
        )
    )
    return cases, periods


def weighted_interval_metrics(y, lo, hi, period_ids, mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {
            f"{metric}_{weight}": float("nan")
            for metric in (
                "coverage",
                "width",
                "interval_score",
                "lower_miss",
                "upper_miss",
            )
            for weight in ("case", "period")
        }
    values = pd.DataFrame(
        {
            "update_period_id": np.asarray(period_ids)[mask],
            "coverage": ((y[mask] >= lo[mask]) & (y[mask] <= hi[mask])).astype(float),
            "width": hi[mask] - lo[mask],
            "interval_score": interval_score(y[mask], lo[mask], hi[mask], alpha=0.1),
            "lower_miss": (y[mask] < lo[mask]).astype(float),
            "upper_miss": (y[mask] > hi[mask]).astype(float),
        }
    )
    period_values = values.groupby("update_period_id", sort=False).mean(numeric_only=True)
    result = {}
    for metric in (
        "coverage",
        "width",
        "interval_score",
        "lower_miss",
        "upper_miss",
    ):
        result[f"{metric}_case"] = float(values[metric].mean())
        result[f"{metric}_period"] = float(period_values[metric].mean())
    return result


def service_rule_metrics(
    split, component_lo, component_hi, fallback_lo, fallback_hi, broadcast_periods
):
    y = np.asarray(split["y_raw"], dtype=float)
    period_ids = split["metadata"]["update_period_id"].to_numpy()
    period_mode = pd.Series(
        np.asarray(broadcast_periods, dtype=bool),
        index=np.asarray(broadcast_periods.index),
    )
    broadcast = pd.Series(period_ids).map(period_mode).to_numpy(dtype=bool)
    combined_lo = np.where(broadcast, component_lo, fallback_lo)
    combined_hi = np.where(broadcast, component_hi, fallback_hi)
    result = {
        "publication_rate_period": float(np.mean(broadcast_periods)),
        "trajectory_exposure_share": float(np.mean(broadcast)),
        "n_broadcast_periods": int(np.sum(broadcast_periods)),
        "n_fallback_periods": int(len(broadcast_periods) - np.sum(broadcast_periods)),
    }
    for name, lo, hi, mask in (
        ("broadcast", component_lo, component_hi, broadcast),
        ("fallback", fallback_lo, fallback_hi, ~broadcast),
        ("combined", combined_lo, combined_hi, np.ones(len(y), dtype=bool)),
    ):
        for key, value in weighted_interval_metrics(
            y, np.asarray(lo), np.asarray(hi), period_ids, mask
        ).items():
            result[f"{name}_{key}"] = value
    result["combined_width_share_of_update_period"] = (
        result["combined_width_period"] / 300.0
    )
    result["combined_width_reduction_from_fallback"] = float(
        np.mean(fallback_hi - fallback_lo) - result["combined_width_period"]
    )
    return result


def run_ap06_development():
    _, experiment, output_dir = read_configs()
    predictions_path = output_dir / "ap03" / "validation_predictions.npz"
    block_path = output_dir / "ap09" / "block_length_selection.json"
    if not predictions_path.exists() or not block_path.exists():
        raise RuntimeError("AP-03 interval definition and AP-09 block length are required.")
    train, val = development_data(output_dir)
    with np.load(predictions_path) as payload:
        component_lo = payload["raw_lo90"].astype(float)
        component_hi = payload["raw_hi90"].astype(float)
    train_y = np.asarray(train["y_raw"], dtype=float)
    fallback_mu = float(np.mean(train_y))
    fallback_sigma = float(np.std(train_y, ddof=1))
    fallback_lo_scalar, fallback_hi_scalar = normal_interval(
        fallback_mu, fallback_sigma, 0.05, 0.95
    )
    fallback_lo = np.full(len(val["y_raw"]), float(fallback_lo_scalar))
    fallback_hi = np.full(len(val["y_raw"]), float(fallback_hi_scalar))
    _, periods = period_sensing_state(val)
    periods = periods.set_index("update_period_id", drop=False)
    block_length = int(
        json.loads(block_path.read_text(encoding="utf-8"))["block_length_periods"]
    )
    n_min = max(2 * block_length, int(np.ceil(0.10 * len(periods))))
    rows = []
    for q_R, q_rho in itertools.product(
        experiment["ap06"]["q_R"], experiment["ap06"]["q_rho"]
    ):
        R_low = float(np.quantile(periods["R_comp"], q_R, method="linear"))
        rho_high = float(np.quantile(periods["rho_t"], q_rho, method="linear"))
        broadcast = (periods["R_comp"] >= R_low) & (periods["rho_t"] <= rho_high)
        metrics = service_rule_metrics(
            val, component_lo, component_hi, fallback_lo, fallback_hi, broadcast
        )
        rows.append(
            {
                "q_R": float(q_R),
                "q_rho": float(q_rho),
                "R_low": R_low,
                "rho_high": rho_high,
                "eligible_count": bool(
                    metrics["n_broadcast_periods"] >= n_min
                    and metrics["n_fallback_periods"] >= n_min
                ),
                **metrics,
            }
        )
    grid = pd.DataFrame(rows)
    pareto = []
    for index, row in grid.iterrows():
        dominates = (
            (grid["publication_rate_period"] >= row["publication_rate_period"])
            & (grid["broadcast_coverage_period"] >= row["broadcast_coverage_period"])
            & (grid["combined_interval_score_period"] <= row["combined_interval_score_period"])
            & (
                (grid["publication_rate_period"] > row["publication_rate_period"])
                | (grid["broadcast_coverage_period"] > row["broadcast_coverage_period"])
                | (grid["combined_interval_score_period"] < row["combined_interval_score_period"])
            )
        )
        pareto.append(not bool(dominates.drop(index).any()))
    grid["pareto"] = pareto

    strategy_names = ["reliability_first", "balanced", "information_first"]
    selected_policies = {}
    for name, epsilon in zip(strategy_names, experiment["ap06"]["risk_tolerances"]):
        feasible = grid[
            grid["eligible_count"]
            & (
                grid["broadcast_coverage_period"]
                >= float(experiment["ap06"]["nominal_coverage"]) - float(epsilon)
            )
        ]
        if feasible.empty:
            selected_policies[name] = {
                "mode": "fallback_only",
                "epsilon": float(epsilon),
            }
            continue
        winner = feasible.sort_values(
            [
                "publication_rate_period",
                "combined_interval_score_period",
                "broadcast_width_period",
                "q_R",
                "q_rho",
            ],
            ascending=[False, True, True, False, True],
        ).iloc[0]
        selected_policies[name] = {
            "mode": "joint_gate",
            "epsilon": float(epsilon),
            "q_R": float(winner["q_R"]),
            "q_rho": float(winner["q_rho"]),
            "R_low": float(winner["R_low"]),
            "rho_high": float(winner["rho_high"]),
            "validation_publication_rate_period": float(
                winner["publication_rate_period"]
            ),
            "validation_broadcast_coverage_period": float(
                winner["broadcast_coverage_period"]
            ),
        }

    anchor = grid[(grid["q_R"] == 0.25) & (grid["q_rho"] == 0.75)].iloc[0]
    anchor_rate = float(anchor["publication_rate_period"])
    rho_candidates = []
    for q_rho in experiment["ap06"]["q_rho"]:
        threshold = float(np.quantile(periods["rho_t"], q_rho, method="linear"))
        broadcast = periods["rho_t"] <= threshold
        rho_candidates.append((abs(float(broadcast.mean()) - anchor_rate), q_rho, threshold))
    _, rho_q, rho_threshold = sorted(rho_candidates, key=lambda x: (x[0], x[1]))[0]
    R_candidates = []
    for q_R in experiment["ap06"]["q_R"]:
        threshold = float(np.quantile(periods["R_comp"], q_R, method="linear"))
        broadcast = periods["R_comp"] >= threshold
        R_candidates.append((abs(float(broadcast.mean()) - anchor_rate), -q_R, threshold))
    _, negative_R_q, R_threshold = sorted(R_candidates, key=lambda x: (x[0], x[1]))[0]
    comparison_rules = {
        "anchor": {
            "mode": "joint_gate",
            "q_R": 0.25,
            "q_rho": 0.75,
            "R_low": float(anchor["R_low"]),
            "rho_high": float(anchor["rho_high"]),
            "validation_publication_rate_period": anchor_rate,
        },
        "no_gate": {"mode": "broadcast_all"},
        "rho_only": {
            "mode": "rho_only",
            "q_rho": float(rho_q),
            "rho_high": float(rho_threshold),
        },
        "R_comp_only": {
            "mode": "R_comp_only",
            "q_R": float(-negative_R_q),
            "R_low": float(R_threshold),
        },
    }
    delay_path = output_dir / "ap01" / "delay_per_period_development.csv"
    if not delay_path.exists():
        raise RuntimeError("AP-01 validation delay diagnostics are required.")
    delay = pd.read_csv(delay_path, parse_dates=["prediction_time"])
    delay = delay[delay["split"] == "validation"].set_index("update_period_id")
    periods["label_age_seconds"] = delay.reindex(periods.index)[
        "label_age_seconds"
    ]
    age_values = periods["label_age_seconds"].dropna().to_numpy(dtype=float)
    age_candidates = []
    for q_age in experiment["ap06"]["q_rho"]:
        threshold = float(np.quantile(age_values, q_age, method="linear"))
        broadcast = periods["label_age_seconds"] <= threshold
        age_candidates.append(
            (
                abs(float(broadcast.mean()) - anchor_rate),
                float(q_age),
                threshold,
                float(broadcast.mean()),
            )
        )
    _, q_age, age_threshold, age_rate = sorted(
        age_candidates, key=lambda value: (value[0], value[1])
    )[0]
    age_R_candidates = []
    for q_age, q_R in itertools.product(
        experiment["ap06"]["q_rho"], experiment["ap06"]["q_R"]
    ):
        age_threshold_candidate = float(
            np.quantile(age_values, q_age, method="linear")
        )
        R_threshold_candidate = float(
            np.quantile(periods["R_comp"], q_R, method="linear")
        )
        broadcast = (
            periods["label_age_seconds"] <= age_threshold_candidate
        ) & (periods["R_comp"] >= R_threshold_candidate)
        age_R_candidates.append(
            (
                abs(float(broadcast.mean()) - anchor_rate),
                float(q_age),
                -float(q_R),
                age_threshold_candidate,
                R_threshold_candidate,
                float(broadcast.mean()),
            )
        )
    (
        _,
        age_R_q_age,
        negative_age_R_q_R,
        age_R_threshold,
        age_R_R_threshold,
        age_R_rate,
    ) = sorted(age_R_candidates, key=lambda value: (value[0], value[1], value[2]))[0]
    ap01_rules = {
        **comparison_rules,
        "label_age_only": {
            "mode": "label_age_only",
            "q_age": q_age,
            "label_age_high": age_threshold,
            "validation_publication_rate_period": age_rate,
        },
        "label_age_R_comp": {
            "mode": "label_age_R_comp",
            "q_age": age_R_q_age,
            "q_R": -negative_age_R_q_R,
            "label_age_high": age_R_threshold,
            "R_low": age_R_R_threshold,
            "validation_publication_rate_period": age_R_rate,
        },
    }
    ap06_dir = output_dir / "ap06"
    ap06_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(ap06_dir / "validation_threshold_grid.csv", index=False)
    grid[grid["pareto"]].to_csv(ap06_dir / "validation_pareto.csv", index=False)
    write_json(
        ap06_dir / "selected_policies.json",
        {
            "created_utc": utc_now(),
            "nominal_coverage": float(experiment["ap06"]["nominal_coverage"]),
            "n_min_periods_per_mode": n_min,
            "block_length_periods": block_length,
            "fallback": {
                "mu": fallback_mu,
                "sigma": fallback_sigma,
                "lo90": float(fallback_lo_scalar),
                "hi90": float(fallback_hi_scalar),
            },
            "selected_policies": selected_policies,
            "comparison_rules": comparison_rules,
        },
    )
    write_json(
        output_dir / "ap01" / "service_rules.json",
        {
            "created_utc": utc_now(),
            **ap01_rules,
            "oracle_state_used_for_gating": False,
        },
    )
    write_json(
        ap06_dir / "manifest.json",
        {
            "created_utc": utc_now(),
            "threshold_candidates": len(grid),
            "selection_data": "unique validation periods",
            "nominal_coverage": float(experiment["ap06"]["nominal_coverage"]),
            "n_min_periods_per_mode": n_min,
            "test_grid_prohibited": True,
        },
    )


def freeze_selection():
    base, experiment, output_dir = read_configs()
    required = [
        output_dir / "g0" / "split_manifest.json",
        output_dir / "ap02" / "stage_a" / "stage_b_finalists.json",
        output_dir / "ap02" / "selected_models.json",
        output_dir / "ap02" / "final_models.json",
        output_dir / "ap02" / "fixed_references" / "manifest.json",
        output_dir / "ap02" / "fairness_audit.csv",
        output_dir / "ap02" / "manifest.json",
        output_dir / "ap02" / "lagged_quantile_gbdt" / "selected_model.json",
        output_dir / "ap03" / "snapshot_quantile_gbdt" / "selected_model.json",
        output_dir / "ap03" / "interval_definition.json",
        output_dir / "ap03" / "manifest.json",
        output_dir / "ap04" / "selected_k.json",
        output_dir / "ap04" / "manifest.json",
        output_dir / "ap06" / "selected_policies.json",
        output_dir / "ap06" / "manifest.json",
        output_dir / "ap09" / "block_length_selection.json",
        output_dir / "ap09" / "manifest.json",
        output_dir / "ap01" / "validation_variants.csv",
        output_dir / "ap01" / "service_rules.json",
        output_dir / "ap01" / "manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Selection cannot be frozen; missing artifacts: {missing}")
    access_path = output_dir / "test_access_log.jsonl"
    access_events = []
    if access_path.exists():
        access_events = [
            json.loads(line)
            for line in access_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    scoring_events = [
        event for event in access_events if str(event.get("stage", "")).startswith("TEST_")
    ]
    if scoring_events:
        raise RuntimeError("Experiment test scoring occurred before selection freeze.")

    selected = json.loads(
        (output_dir / "ap02" / "selected_models.json").read_text(encoding="utf-8")
    )
    final_models = json.loads(
        (output_dir / "ap02" / "final_models.json").read_text(encoding="utf-8")
    )
    selected_k = json.loads(
        (output_dir / "ap04" / "selected_k.json").read_text(encoding="utf-8")
    )
    policies = json.loads(
        (output_dir / "ap06" / "selected_policies.json").read_text(encoding="utf-8")
    )
    block = json.loads(
        (output_dir / "ap09" / "block_length_selection.json").read_text(
            encoding="utf-8"
        )
    )
    ap01_service_rules = json.loads(
        (output_dir / "ap01" / "service_rules.json").read_text(encoding="utf-8")
    )
    ap01_service_rules.pop("created_utc", None)
    ap01_service_rules.pop("oracle_state_used_for_gating", None)
    tree_models = {
        "snapshot": json.loads(
            (
                output_dir
                / "ap03"
                / "snapshot_quantile_gbdt"
                / "selected_model.json"
            ).read_text(encoding="utf-8")
        ),
        "lagged": json.loads(
            (
                output_dir
                / "ap02"
                / "lagged_quantile_gbdt"
                / "selected_model.json"
            ).read_text(encoding="utf-8")
        ),
    }
    roots = [
        output_dir / "g0",
        output_dir / "ap01",
        output_dir / "ap02" / "stage_a",
        output_dir / "ap02" / "stage_b",
        output_dir / "ap02" / "ap04_k_scan",
        output_dir / "ap02" / "final_heads",
        output_dir / "ap02" / "final_models",
        output_dir / "ap02" / "fixed_references",
        output_dir / "ap02" / "lagged_quantile_gbdt",
        output_dir / "ap03",
        output_dir / "ap04",
        output_dir / "ap06",
        output_dir / "ap09",
    ]
    artifact_paths = []
    for root in roots:
        if root.exists():
            artifact_paths.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and "tcn" not in path.name.lower()
            )
    source_paths = [
        BASE_CONFIG_PATH,
        EXPERIMENT_CONFIG_PATH,
        ROOT / "data_processing.py",
        ROOT / "experiment_runner.py",
        ROOT / "frozen_evaluation.py",
        ROOT / "forecast_models.py",
        ROOT / "probabilistic_metrics.py",
        ROOT / "docs" / "experiment_protocol.md",
        Path(base["data"]["plate1_path"]),
        Path(base["data"]["plate2_path"]),
    ]
    all_paths = sorted(
        {path.resolve() for path in artifact_paths + source_paths}, key=str
    )
    hashes = {
        str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in all_paths
    }
    excluded_tcn = []
    for path in sorted((output_dir / "ap02" / "stage_a" / "trials").glob("*tcn*.json")):
        excluded_tcn.append(
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": sha256_file(path),
                "reason": "Model family removed categorically in protocol 1.5 before test opening.",
            }
        )
    freeze_dir = output_dir / "freeze"
    freeze_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        freeze_dir / "selection_lock.json",
        {
            "created_utc": utc_now(),
            "protocol_version": experiment["protocol_version"],
            "training_protocol_version": experiment.get("training_protocol_version"),
            "test_status": "historically viewed holdout; no experiment scoring before this lock",
            "test_scoring_events_before_freeze": scoring_events,
            "selected_models": selected,
            "final_models": final_models,
            "selected_k": selected_k,
            "tree_models": tree_models,
            "service_policies": policies,
            "ap01_service_rules": ap01_service_rules,
            "block_length": block,
            "excluded_tcn_trials": excluded_tcn,
            "hashes": hashes,
        },
    )


def exclude_tail_periods(metadata, count=6):
    ordered = metadata["update_period_id"].drop_duplicates().to_numpy()
    excluded = set(ordered[-count:]) if len(ordered) >= count else set(ordered)
    return ~metadata["update_period_id"].isin(excluded).to_numpy()


def lookahead_snapshots(base, output_dir, lookahead_seconds):
    manifest = json.loads(
        (Path(output_dir) / "g0" / "split_manifest.json").read_text(encoding="utf-8")
    )
    defaults = manifest["training_defaults"]
    t_up, _, merged = load_matched_data(base)
    feature_df = compute_rolling_features(
        t_up,
        merged,
        int(base["data"]["delta_obs"]),
        int(base["data"]["window_width"]),
        mu_default=float(defaults["travel_time_median"]),
        s2_default=float(defaults["travel_time_variance"]),
        completion_lookahead_seconds=int(lookahead_seconds),
    )
    X, _, metadata = create_trajectory_indexed_dataset(
        merged,
        feature_df,
        int(base["data"]["delta_obs"]),
        int(base["model"]["sequence_length"]),
        return_metadata=True,
    )
    boundaries = manifest["split"]
    train_end = int(boundaries["train_end"])
    val_end = int(boundaries["val_end"])
    return {
        "train": X.numpy()[:train_end, -1, :],
        "val": X.numpy()[train_end:val_end, -1, :],
        "metadata_train": metadata.iloc[:train_end].reset_index(drop=True),
        "metadata_val": metadata.iloc[train_end:val_end].reset_index(drop=True),
    }


def delay_period_diagnostics(base, metadata, snapshots):
    _, _, merged = load_matched_data(base)
    up_times = merged["timestamp_up"].to_numpy(dtype="datetime64[ns]")
    down_times = merged["timestamp_down"].to_numpy(dtype="datetime64[ns]")
    window = np.timedelta64(int(base["data"]["window_width"]), "m")
    horizon = np.timedelta64(1800, "s")
    rows = []
    frame = metadata.copy()
    frame["N_up"] = snapshots[:, 0]
    frame["N_complete"] = snapshots[:, 1]
    frame["R_comp"] = snapshots[:, 2]
    frame["rho_t"] = snapshots[:, 3] / 300.0
    period_state = frame.groupby("update_period_id", sort=False).first().reset_index()
    for row in period_state.itertuples(index=False):
        t = np.datetime64(pd.Timestamp(row.prediction_time), "ns")
        start = np.searchsorted(up_times, t - window, side="right")
        end = np.searchsorted(up_times, t, side="right")
        candidate_down = down_times[start:end]
        candidate_up = up_times[start:end]
        n_eventual = int(end - start)
        completed = candidate_down <= t
        completed_30 = candidate_down <= t + horizon
        label_age = (
            float((t - candidate_up[completed].max()) / np.timedelta64(1, "s"))
            if completed.any()
            else float("nan")
        )
        n_up = float(row.N_up)
        n_complete = float(row.N_complete)
        q_match = n_eventual / max(n_up, 1.0)
        c_delay = n_complete / max(n_eventual, 1.0)
        rows.append(
            {
                "update_period_id": int(row.update_period_id),
                "prediction_time": row.prediction_time,
                "N_up": n_up,
                "N_complete": n_complete,
                "N_eventual": n_eventual,
                "Q_match": q_match,
                "C_delay": c_delay,
                "R_comp": float(row.R_comp),
                "rho_t": float(row.rho_t),
                "inflight_share": 1.0 - c_delay,
                "label_age_seconds": label_age,
                "h1800_saturation": float(completed_30.sum() / max(n_eventual, 1)),
            }
        )
    return pd.DataFrame(rows)


def run_ap01_development(limit=None):
    base, experiment, output_dir = read_configs()
    selected_tree_path = (
        output_dir / "ap03" / "snapshot_quantile_gbdt" / "selected_model.json"
    )
    if not selected_tree_path.exists():
        raise RuntimeError("AP-03 snapshot quantile GBDT selection is required.")
    selected_tree = json.loads(selected_tree_path.read_text(encoding="utf-8"))
    tree_spec = selected_tree["spec"]
    train, val = development_data(output_dir)
    train_mask = exclude_tail_periods(train["metadata"], count=6)
    val_mask = exclude_tail_periods(val["metadata"], count=6)
    feature_sets = {
        "full_h0": list(range(14)),
        "no_R": [index for index in range(14) if index != 2],
        "no_rho_carrier": [index for index in range(14) if index != 3],
        "no_completion_block": [index for index in range(14) if index not in (1, 2)],
        "no_moment_block": [index for index in range(14) if index not in (3, 4)],
        "base": [0] + list(range(5, 14)),
    }
    variants = {}
    train_snapshot = np.asarray(train["X_raw"][:, -1, :])
    val_snapshot = np.asarray(val["X_raw"][:, -1, :])
    for name, indices in feature_sets.items():
        variants[name] = (
            train_snapshot[train_mask][:, indices],
            val_snapshot[val_mask][:, indices],
        )
    for seconds in experiment["ap01"]["lookahead_seconds"]:
        seconds = int(seconds)
        if seconds == 0:
            continue
        lookahead = lookahead_snapshots(base, output_dir, seconds)
        if not lookahead["metadata_train"]["case_id"].equals(
            train["metadata"]["case_id"]
        ):
            raise RuntimeError("AP-01 train case alignment changed.")
        if not lookahead["metadata_val"]["case_id"].equals(
            val["metadata"]["case_id"]
        ):
            raise RuntimeError("AP-01 validation case alignment changed.")
        variants[f"full_h{seconds}"] = (
            lookahead["train"][train_mask],
            lookahead["val"][val_mask],
        )

    ap01_dir = output_dir / "ap01"
    trial_dir = ap01_dir / "development_trials"
    trial_dir.mkdir(parents=True, exist_ok=True)
    completed_now = 0
    rows = []
    for name, (X_train, X_val) in variants.items():
        manifest_path = trial_dir / f"{name}.json"
        if manifest_path.exists():
            rows.append(json.loads(manifest_path.read_text(encoding="utf-8")))
            continue
        print(f"[ap01] {name}", flush=True)
        models = fit_quantile_models(
            X_train, train["y_raw"][train_mask], tree_spec, experiment
        )
        predictions = np.column_stack([model.predict(X_val) for model in models])
        losses = np.column_stack(
            [
                pinball_loss(
                    val["y_raw"][val_mask], predictions[:, index], quantile
                )
                for index, quantile in enumerate(experiment["gbdt"]["quantiles"])
            ]
        )
        model_path = ap01_dir / "models" / f"{name}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(models, model_path)
        prediction_path = ap01_dir / "validation_predictions" / f"{name}.npz"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_path,
            quantiles=predictions.astype(np.float32),
            pinball=losses.astype(np.float32),
        )
        result = {
            "variant": name,
            "mean_pinball": float(np.mean(losses)),
            "pinball_q10": float(np.mean(losses[:, 0])),
            "pinball_q50": float(np.mean(losses[:, 1])),
            "pinball_q90": float(np.mean(losses[:, 2])),
            "models": str(model_path.relative_to(output_dir)),
            "validation_predictions": str(prediction_path.relative_to(output_dir)),
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "completed_utc": utc_now(),
        }
        write_json(manifest_path, result)
        rows.append(result)
        completed_now += 1
        if limit is not None and completed_now >= limit:
            break
    pd.DataFrame(rows).to_csv(ap01_dir / "validation_variants.csv", index=False)

    train_diagnostics = delay_period_diagnostics(
        base, train["metadata"], train_snapshot
    )
    train_diagnostics["split"] = "train"
    val_diagnostics = delay_period_diagnostics(base, val["metadata"], val_snapshot)
    val_diagnostics["split"] = "validation"
    period_diagnostics = pd.concat(
        [train_diagnostics, val_diagnostics], ignore_index=True
    )
    period_diagnostics.to_csv(ap01_dir / "delay_per_period_development.csv", index=False)
    if not (
        (period_diagnostics["N_complete"] <= period_diagnostics["N_eventual"]).all()
        and (period_diagnostics["N_eventual"] <= period_diagnostics["N_up"]).all()
    ):
        raise RuntimeError("AP-01 development delay count identity failed.")
    decomposition_error = np.abs(
        period_diagnostics["Q_match"] * period_diagnostics["C_delay"]
        - period_diagnostics["R_comp"]
    )
    if float(decomposition_error.max()) > 1e-6:
        raise RuntimeError("AP-01 development R_comp decomposition failed.")
    distribution_rows = []
    for split, split_data in period_diagnostics.groupby("split", sort=False):
        for variable in (
            "label_age_seconds",
            "inflight_share",
            "Q_match",
            "C_delay",
            "R_comp",
            "rho_t",
        ):
            values = split_data[variable].dropna().to_numpy(dtype=float)
            distribution_rows.append(
                {
                    "split": split,
                    "variable": variable,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "p05": float(np.quantile(values, 0.05)),
                    "p25": float(np.quantile(values, 0.25)),
                    "p75": float(np.quantile(values, 0.75)),
                    "p95": float(np.quantile(values, 0.95)),
                }
            )
    pd.DataFrame(distribution_rows).to_csv(
        ap01_dir / "delay_distribution_summary.csv", index=False
    )
    correlation_rows = []
    for split, split_data in period_diagnostics.groupby("split", sort=False):
        correlation_rows.append(
            {
                "split": split,
                "rho_label_age_pearson": float(
                    split_data[["rho_t", "label_age_seconds"]].corr(
                        method="pearson"
                    ).iloc[0, 1]
                ),
                "rho_label_age_spearman": float(
                    split_data[["rho_t", "label_age_seconds"]].corr(
                        method="spearman"
                    ).iloc[0, 1]
                ),
            }
        )
    pd.DataFrame(correlation_rows).to_csv(
        ap01_dir / "delay_correlations.csv", index=False
    )
    summary_rows = []
    for split, split_data, y_values in (
        ("train", train_diagnostics, train["y_raw"]),
        ("validation", val_diagnostics, val["y_raw"]),
    ):
        summary_rows.append(
            {
                "split": split,
                "n_cases": len(y_values),
                "n_periods": len(split_data),
                "travel_time_mean_seconds": float(np.mean(y_values)),
                "travel_time_median_seconds": float(np.median(y_values)),
                "travel_time_p95_seconds": float(np.quantile(y_values, 0.95)),
                "p_travel_time_gt_300": float(np.mean(y_values > 300.0)),
                "label_age_median_seconds": float(
                    split_data["label_age_seconds"].median()
                ),
                "inflight_share_mean": float(split_data["inflight_share"].mean()),
                "h1800_saturation_mean": float(
                    split_data["h1800_saturation"].mean()
                ),
            }
        )
    pd.DataFrame(summary_rows).to_csv(ap01_dir / "delay_summary.csv", index=False)
    write_json(
        ap01_dir / "manifest.json",
        {
            "created_utc": utc_now(),
            "tail_periods_excluded_per_split": 6,
            "lookahead_seconds": experiment["ap01"]["lookahead_seconds"],
            "prediction_variants": sorted(variants),
            "selection_data": "train and validation only",
            "oracle_used_for_deployment": False,
        },
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "stage-a",
            "stage-b",
            "k-scan",
            "trees",
            "ap01",
            "ap03",
            "ap06",
            "final-models",
            "freeze",
            "fixed-references",
        ],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--family", choices=["snapshot", "recurrent"], default=None
    )
    parser.add_argument("--kind", choices=["snapshot", "lagged"], default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "prepare":
        prepare_data()
    elif args.command == "stage-a":
        run_stage_a(limit=args.limit, family=args.family)
    elif args.command == "stage-b":
        run_stage_b(limit=args.limit)
    elif args.command == "k-scan":
        run_k_scan(limit=args.limit)
    elif args.command == "trees":
        if args.kind is None:
            raise SystemExit("--kind is required for trees")
        run_quantile_trees(args.kind, limit=args.limit)
    elif args.command == "ap01":
        run_ap01_development(limit=args.limit)
    elif args.command == "ap03":
        run_ap03_development()
    elif args.command == "ap06":
        run_ap06_development()
    elif args.command == "final-models":
        run_final_models(limit=args.limit)
    elif args.command == "freeze":
        freeze_selection()
    elif args.command == "fixed-references":
        run_fixed_references()


if __name__ == "__main__":
    main()
