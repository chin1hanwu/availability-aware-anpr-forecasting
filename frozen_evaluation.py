from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from data_processing import (
    compute_rolling_features,
    create_trajectory_indexed_dataset,
    load_matched_data,
)
from probabilistic_metrics import (
    density_metrics,
    interval_score,
    ks_statistic,
    moving_block_period_indices,
    negative_mass,
    pinball_loss,
    truncated_mixture_crps,
    truncated_mixture_nll,
    truncated_mixture_pit,
    truncated_mixture_quantile,
)
from forecast_models import (
    load_density_model,
    load_point_model,
    predict_density,
    predict_point,
)


ROOT = Path(__file__).resolve().parent
BASE_CONFIG_PATH = ROOT / "study_config.yaml"
EXPERIMENT_CONFIG_PATH = ROOT / "experiment_config.yaml"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


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


def read_context():
    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    with open(EXPERIMENT_CONFIG_PATH, "r", encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    output_dir = (ROOT / experiment["output_dir"]).resolve()
    lock_path = output_dir / "freeze" / "selection_lock.json"
    if not lock_path.exists():
        raise RuntimeError("selection_lock.json is required before test scoring.")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["protocol_version"] != experiment["protocol_version"]:
        raise RuntimeError("Protocol version differs from the selection lock.")
    return base, experiment, output_dir, lock


def verify_test_cache(output_dir, lock):
    paths = [
        output_dir / "g0" / "test_X_raw.npy",
        output_dir / "g0" / "test_X_scaled.npy",
        output_dir / "g0" / "test_y_raw.npy",
        output_dir / "g0" / "test_y_scaled.npy",
        output_dir / "g0" / "test_metadata.csv",
    ]
    for path in paths:
        key = str(path.relative_to(ROOT))
        expected = lock["hashes"].get(key)
        if expected is None or sha256_file(path) != expected["sha256"]:
            raise RuntimeError(f"Frozen test cache hash mismatch: {path}")


def verify_frozen_artifacts(lock):
    for name, expected in lock["hashes"].items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists() or sha256_file(path) != expected["sha256"]:
            raise RuntimeError(f"Frozen artifact hash mismatch: {path}")


def append_access_event(output_dir, stage, detail):
    path = output_dir / "test_access_log.jsonl"
    event = {"timestamp_utc": utc_now(), "stage": stage, "detail": detail}
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def load_test(output_dir):
    g0 = output_dir / "g0"
    return {
        "X_raw": np.load(g0 / "test_X_raw.npy", mmap_mode="c"),
        "X_scaled": np.load(g0 / "test_X_scaled.npy", mmap_mode="c"),
        "y_raw": np.load(g0 / "test_y_raw.npy", mmap_mode="c"),
        "y_scaled": np.load(g0 / "test_y_scaled.npy", mmap_mode="c"),
        "metadata": pd.read_csv(
            g0 / "test_metadata.csv", parse_dates=["prediction_time"]
        ),
    }


def raw_mixture_parameters(pis, mus_scaled, sigmas_scaled, y_scaler):
    mus = y_scaler.inverse_transform(mus_scaled.reshape(-1, 1)).reshape(
        mus_scaled.shape
    )
    sigmas = sigmas_scaled / float(y_scaler.scale_[0])
    return pis, mus, sigmas


def quantiles_to_gaussian(predictions):
    predictions = np.asarray(predictions, dtype=float)
    crossing = (predictions[:, 0] > predictions[:, 1]) | (
        predictions[:, 1] > predictions[:, 2]
    )
    lower = np.minimum(predictions[:, 0], predictions[:, 2])
    upper = np.maximum(predictions[:, 0], predictions[:, 2])
    mu = np.clip(predictions[:, 1], lower, upper)
    sigma = np.maximum((upper - lower) / 2.563103, 1e-3)
    return mu, sigma, crossing


def prediction_record(model, seed, pis, mus, sigmas, y, density_source):
    metrics = density_metrics(pis, mus, sigmas, y)
    return {
        "model": model,
        "seed": seed,
        "density_source": density_source,
        "pis": np.asarray(pis, dtype=float),
        "mus": np.asarray(mus, dtype=float),
        "sigmas": np.asarray(sigmas, dtype=float),
        "metrics": metrics,
    }


def save_prediction(output_dir, record):
    seed = "deterministic" if record["seed"] is None else f"seed{record['seed']}"
    path = output_dir / "test" / "predictions" / f"{record['model']}_{seed}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pis=record["pis"].astype(np.float32),
        mus=record["mus"].astype(np.float32),
        sigmas=record["sigmas"].astype(np.float32),
        **{
            key: np.asarray(value).astype(np.float32)
            for key, value in record["metrics"].items()
        },
    )
    return path


def evaluate_predictions(test, output_dir, lock):
    y = np.asarray(test["y_raw"], dtype=float)
    y_scaler = joblib.load(output_dir / "g0" / "y_scaler.joblib")
    final_models = lock["final_models"]
    records = []

    fixed = json.loads(
        (output_dir / "ap02" / "fixed_references" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    naive_mu = float(fixed["naive"]["mu_training_seconds"])
    naive_sigma = float(fixed["naive"]["sigma_training_seconds"])
    records.append(
        prediction_record(
            "naive_gaussian",
            None,
            np.ones((len(y), 1)),
            np.full((len(y), 1), naive_mu),
            np.full((len(y), 1), naive_sigma),
            y,
            "training_marginal_gaussian",
        )
    )
    ridge = joblib.load(output_dir / fixed["ridge"]["model"])
    ridge_mu = ridge.predict(np.asarray(test["X_scaled"][:, -1, :]))
    ridge_sigma = float(fixed["ridge"]["residual_sigma_training_seconds"])
    records.append(
        prediction_record(
            "ridge_gaussian",
            None,
            np.ones((len(y), 1)),
            ridge_mu[:, None],
            np.full((len(y), 1), ridge_sigma),
            y,
            "training_residual_gaussian",
        )
    )

    native_quantiles = {}
    for kind, model_name in (
        ("snapshot", "snapshot_quantile"),
        ("lagged", "lagged_quantile"),
    ):
        tree = lock["tree_models"][kind]
        models = joblib.load(output_dir / tree["models"])
        history = int(tree["spec"]["history_length"])
        X = np.asarray(test["X_raw"][:, -history:, :]).reshape(len(y), -1)
        quantiles = np.column_stack([model.predict(X) for model in models])
        native_quantiles[model_name] = quantiles
        mu, sigma, _ = quantiles_to_gaussian(quantiles)
        records.append(
            prediction_record(
                f"{model_name}_induced_gaussian",
                None,
                np.ones((len(y), 1)),
                mu[:, None],
                sigma[:, None],
                y,
                "quantile_postprocessing",
            )
        )

    for family in ("snapshot", "recurrent"):
        spec = final_models[family]["spec"]
        history = int(spec["history_length"])
        for seed, checkpoint in final_models[family]["checkpoints"].items():
            model, _ = load_density_model(output_dir / checkpoint)
            pis, mus_scaled, sigmas_scaled = predict_density(
                model, test["X_scaled"][:, -history:, :]
            )
            pis, mus, sigmas = raw_mixture_parameters(
                pis, mus_scaled, sigmas_scaled, y_scaler
            )
            records.append(
                prediction_record(
                    f"{family}_mdn",
                    int(seed),
                    pis,
                    mus,
                    sigmas,
                    y,
                    "direct_likelihood",
                )
            )

    for seed, point in final_models["lstm_mse"]["checkpoints"].items():
        spec = final_models["lstm_mse"]["spec"]
        history = int(spec["history_length"])
        model, _ = load_point_model(output_dir / point["checkpoint"])
        mu_scaled = predict_point(model, test["X_scaled"][:, -history:, :])
        mu = y_scaler.inverse_transform(mu_scaled[:, None])[:, 0]
        sigma = float(point["residual_sigma_training_seconds"])
        records.append(
            prediction_record(
                "lstm_mse_gaussian",
                int(seed),
                np.ones((len(y), 1)),
                mu[:, None],
                np.full((len(y), 1), sigma),
                y,
                "training_residual_gaussian",
            )
        )
    for record in records:
        save_prediction(output_dir, record)
    return records, native_quantiles


def metric_mean(values, period_ids):
    values = np.asarray(values, dtype=float)
    frame = pd.DataFrame({"period": period_ids, "value": values})
    return float(values.mean()), float(frame.groupby("period")["value"].mean().mean())


def summarize_density_records(records, test, output_dir):
    period_ids = test["metadata"]["update_period_id"].to_numpy()
    rows = []
    for record in records:
        row = {
            "model": record["model"],
            "seed": record["seed"],
            "density_source": record["density_source"],
            "pit_ks": ks_statistic(record["metrics"]["pit"]),
        }
        for metric in ("crps", "nll", "covered90", "width90", "interval_score90"):
            case_mean, period_mean = metric_mean(
                record["metrics"][metric], period_ids
            )
            row[f"{metric}_case"] = case_mean
            row[f"{metric}_period"] = period_mean
        rows.append(row)
    frame = pd.DataFrame(rows)
    ap02 = output_dir / "ap02"
    frame.to_csv(ap02 / "metrics_by_seed.csv", index=False)
    metrics = [
        column
        for column in frame.columns
        if column not in ("model", "seed", "density_source")
    ]
    aggregate_rows = []
    for model, group in frame.groupby("model", sort=False):
        row = {
            "model": model,
            "density_source": group["density_source"].iloc[0],
            "n_seeds": int(group["seed"].notna().sum()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_seed_sd"] = (
                float(group[metric].std(ddof=1)) if len(group) > 1 else float("nan")
            )
        aggregate_rows.append(row)
    pd.DataFrame(aggregate_rows).to_csv(
        ap02 / "metrics_aggregate.csv", index=False
    )
    return frame


def summarize_native_quantiles(native_quantiles, test, experiment, output_dir):
    y = np.asarray(test["y_raw"], dtype=float)
    rows = []
    per_case = test["metadata"][["case_id", "update_period_id"]].copy()
    for model, predictions in native_quantiles.items():
        losses = np.column_stack(
            [
                pinball_loss(y, predictions[:, index], quantile)
                for index, quantile in enumerate(experiment["gbdt"]["quantiles"])
            ]
        )
        crossing = (predictions[:, 0] > predictions[:, 1]) | (
            predictions[:, 1] > predictions[:, 2]
        )
        rows.append(
            {
                "model": model,
                "mean_pinball": float(losses.mean()),
                "pinball_q10": float(losses[:, 0].mean()),
                "pinball_q50": float(losses[:, 1].mean()),
                "pinball_q90": float(losses[:, 2].mean()),
                "calibration_q10": float(np.mean(y <= predictions[:, 0]) - 0.10),
                "calibration_q50": float(np.mean(y <= predictions[:, 1]) - 0.50),
                "calibration_q90": float(np.mean(y <= predictions[:, 2]) - 0.90),
                "raw_crossing_rate": float(crossing.mean()),
                "coverage80": float(
                    np.mean((y >= predictions[:, 0]) & (y <= predictions[:, 2]))
                ),
                "width80": float(np.mean(predictions[:, 2] - predictions[:, 0])),
                "nll": float("nan"),
            }
        )
        per_case[f"{model}_mean_pinball"] = losses.mean(axis=1)
    ap03 = output_dir / "ap03"
    pd.DataFrame(rows).to_csv(ap03 / "native_quantile_test.csv", index=False)
    per_case.to_csv(ap03 / "quantile_per_case.csv", index=False)
    return per_case


def negative_mass_summary(values, model, seed):
    values = np.asarray(values, dtype=float)
    row = {
        "model": model,
        "seed": seed,
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }
    for label, threshold in (
        ("1e-6", 1e-6),
        ("1e-4", 1e-4),
        ("1e-3", 1e-3),
        ("1e-2", 1e-2),
        ("0.05", 0.05),
    ):
        row[f"share_gt_{label}"] = float(np.mean(values > threshold))
    return row


def run_ap03_test(records, test, output_dir):
    y = np.asarray(test["y_raw"], dtype=float)
    metadata = test["metadata"]
    metric_rows = []
    mass_rows = []
    mass_frames = []
    for record in records:
        masses = negative_mass(record["pis"], record["mus"], record["sigmas"])
        mass_rows.append(negative_mass_summary(masses, record["model"], record["seed"]))
        mass_frame = metadata[["case_id", "update_period_id"]].copy()
        mass_frame["model"] = record["model"]
        mass_frame["seed"] = record["seed"]
        mass_frame["negative_mass"] = masses
        mass_frames.append(mass_frame)

        truncated_lo = truncated_mixture_quantile(
            record["pis"], record["mus"], record["sigmas"], 0.05
        )
        truncated_hi = truncated_mixture_quantile(
            record["pis"], record["mus"], record["sigmas"], 0.95
        )
        truncated = {
            "crps": truncated_mixture_crps(
                record["pis"], record["mus"], record["sigmas"], y, points=2001
            ),
            "nll": truncated_mixture_nll(
                record["pis"], record["mus"], record["sigmas"], y
            ),
            "pit": truncated_mixture_pit(
                record["pis"], record["mus"], record["sigmas"], y
            ),
            "lo90": truncated_lo,
            "hi90": truncated_hi,
            "covered90": ((y >= truncated_lo) & (y <= truncated_hi)).astype(float),
            "width90": truncated_hi - truncated_lo,
            "interval_score90": interval_score(
                y, truncated_lo, truncated_hi, alpha=0.1
            ),
        }
        record["negative_mass"] = masses
        record["truncated_metrics"] = truncated
        for variant, metrics in (
            ("raw", record["metrics"]),
            ("truncated", truncated),
        ):
            metric_rows.append(
                {
                    "model": record["model"],
                    "seed": record["seed"],
                    "density_source": record["density_source"],
                    "variant": variant,
                    "crps_case": float(np.mean(metrics["crps"])),
                    "nll_case": float(np.mean(metrics["nll"])),
                    "pit_ks": ks_statistic(metrics["pit"]),
                    "coverage90_case": float(np.mean(metrics["covered90"])),
                    "width90_case": float(np.mean(metrics["width90"])),
                    "interval_score90_case": float(
                        np.mean(metrics["interval_score90"])
                    ),
                }
            )
    ap03 = output_dir / "ap03"
    pd.DataFrame(metric_rows).to_csv(
        ap03 / "metrics_by_model_seed.csv", index=False
    )
    pd.DataFrame(mass_rows).to_csv(
        ap03 / "negative_mass_summary.csv", index=False
    )
    pd.concat(mass_frames, ignore_index=True).to_csv(
        ap03 / "negative_mass_per_case.csv", index=False
    )


def grouped_period_arrays(values, metadata, period_order, mask=None):
    values = np.asarray(values, dtype=float)
    if mask is None:
        mask = np.ones(len(values), dtype=bool)
    frame = pd.DataFrame(
        {
            "update_period_id": metadata["update_period_id"].to_numpy()[mask],
            "value": values[mask],
        }
    )
    grouped = frame.groupby("update_period_id")["value"].agg(["sum", "count", "mean"])
    grouped = grouped.reindex(period_order, fill_value=0.0)
    return (
        grouped["sum"].to_numpy(dtype=float),
        grouped["count"].to_numpy(dtype=float),
        grouped["mean"].to_numpy(dtype=float),
    )


def bootstrap_estimates(values, metadata, period_index, blocks, mask=None):
    period_order = period_index["update_period_id"].to_numpy()
    sums, counts, means = grouped_period_arrays(values, metadata, period_order, mask)
    sampled_counts = np.sum(counts[blocks], axis=1)
    case_values = np.sum(sums[blocks], axis=1) / sampled_counts
    sampled_present = counts[blocks] > 0
    period_values = np.sum(means[blocks] * sampled_present, axis=1) / np.sum(
        sampled_present, axis=1
    )
    return case_values, period_values


def percentile_interval(values):
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def mean_record_metrics(records, model, metric, variant="raw"):
    selected = [record for record in records if record["model"] == model]
    if not selected:
        raise KeyError(model)
    key = "metrics" if variant == "raw" else "truncated_metrics"
    return np.mean([record[key][metric] for record in selected], axis=0)


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def run_ap09_test(records, native_per_case, test, experiment, output_dir, lock):
    block_length = int(lock["block_length"]["block_length_periods"])
    period_index, blocks = moving_block_period_indices(
        test["metadata"],
        block_length,
        int(experiment["ap09"]["test_repeats"]),
        seed=19042,
    )
    ap09 = output_dir / "ap09"
    ap09.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ap09 / "bootstrap_indices.npz",
        update_period_id=period_index["update_period_id"].to_numpy(),
        indices=blocks,
    )
    rows = []
    models = list(dict.fromkeys(record["model"] for record in records))
    for model in models:
        model_records = [record for record in records if record["model"] == model]
        for metric in ("crps", "nll", "covered90", "width90", "interval_score90"):
            values = np.mean(
                [record["metrics"][metric] for record in model_records], axis=0
            )
            case_boot, period_boot = bootstrap_estimates(
                values, test["metadata"], period_index, blocks
            )
            case_ci = percentile_interval(case_boot)
            period_ci = percentile_interval(period_boot)
            seed_points = [
                float(np.mean(record["metrics"][metric])) for record in model_records
            ]
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "case_estimate": float(np.mean(values)),
                    "case_ci_low": case_ci[0],
                    "case_ci_high": case_ci[1],
                    "period_estimate": float(
                        pd.DataFrame(
                            {
                                "period": test["metadata"]["update_period_id"],
                                "value": values,
                            }
                        )
                        .groupby("period")["value"]
                        .mean()
                        .mean()
                    ),
                    "period_ci_low": period_ci[0],
                    "period_ci_high": period_ci[1],
                    "seed_sd": (
                        float(np.std(seed_points, ddof=1))
                        if len(seed_points) > 1
                        else float("nan")
                    ),
                    "bootstrap_repeats": len(blocks),
                    "block_length_periods": block_length,
                }
            )
    for model in ("snapshot_quantile", "lagged_quantile"):
        values = native_per_case[f"{model}_mean_pinball"].to_numpy(dtype=float)
        case_boot, period_boot = bootstrap_estimates(
            values, test["metadata"], period_index, blocks
        )
        case_ci = percentile_interval(case_boot)
        period_ci = percentile_interval(period_boot)
        rows.append(
            {
                "model": model,
                "metric": "mean_pinball",
                "case_estimate": float(np.mean(values)),
                "case_ci_low": case_ci[0],
                "case_ci_high": case_ci[1],
                "period_estimate": float(
                    pd.DataFrame(
                        {
                            "period": test["metadata"]["update_period_id"],
                            "value": values,
                        }
                    )
                    .groupby("period")["value"]
                    .mean()
                    .mean()
                ),
                "period_ci_low": period_ci[0],
                "period_ci_high": period_ci[1],
                "seed_sd": float("nan"),
                "bootstrap_repeats": len(blocks),
                "block_length_periods": block_length,
            }
        )
    pd.DataFrame(rows).to_csv(ap09 / "metric_intervals.csv", index=False)

    comparisons = [
        ("snapshot_mdn", "recurrent_mdn"),
        ("lagged_quantile_induced_gaussian", "recurrent_mdn"),
        ("lstm_mse_gaussian", "recurrent_mdn"),
    ]
    effect_rows = []
    for comparator, reference in comparisons:
        difference = mean_record_metrics(records, reference, "crps") - mean_record_metrics(
            records, comparator, "crps"
        )
        case_boot, period_boot = bootstrap_estimates(
            difference, test["metadata"], period_index, blocks
        )
        point = float(np.mean(difference))
        centered = case_boot - float(np.mean(case_boot))
        p_value = float(
            (1 + np.sum(np.abs(centered) >= abs(point))) / (len(centered) + 1)
        )
        case_ci = percentile_interval(case_boot)
        period_ci = percentile_interval(period_boot)
        effect_rows.append(
            {
                "family": "AP02_confirmatory",
                "reference": reference,
                "comparator": comparator,
                "effect": "reference_minus_comparator_crps",
                "case_effect": point,
                "case_ci_low": case_ci[0],
                "case_ci_high": case_ci[1],
                "period_effect": float(np.mean(period_boot)),
                "period_ci_low": period_ci[0],
                "period_ci_high": period_ci[1],
                "p_value": p_value,
            }
        )
    adjusted = holm_adjust([row["p_value"] for row in effect_rows])
    for row, value in zip(effect_rows, adjusted):
        row["p_holm"] = float(value)
    pd.DataFrame(effect_rows).to_csv(ap09 / "paired_effects.csv", index=False)
    pd.DataFrame(effect_rows)[
        ["family", "reference", "comparator", "p_value", "p_holm"]
    ].to_csv(ap09 / "holm_families.csv", index=False)
    return period_index, blocks


def exclude_tail_periods(metadata, count=6):
    ordered = metadata["update_period_id"].drop_duplicates().to_numpy()
    excluded = set(ordered[-count:])
    return ~metadata["update_period_id"].isin(excluded).to_numpy()


def lookahead_test_snapshot(base, output_dir, lookahead_seconds):
    split_manifest = json.loads(
        (output_dir / "g0" / "split_manifest.json").read_text(encoding="utf-8")
    )
    defaults = split_manifest["training_defaults"]
    t_up, _, merged = load_matched_data(base)
    features = compute_rolling_features(
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
        features,
        int(base["data"]["delta_obs"]),
        int(base["model"]["sequence_length"]),
        return_metadata=True,
    )
    val_end = int(split_manifest["split"]["val_end"])
    return (
        X.numpy()[val_end:, -1, :],
        metadata.iloc[val_end:].reset_index(drop=True),
    )


def test_delay_diagnostics(base, test):
    _, _, merged = load_matched_data(base)
    merged = merged.sort_values("timestamp_up").reset_index(drop=True)
    up_times = merged["timestamp_up"].to_numpy(dtype="datetime64[ns]")
    down_times = merged["timestamp_down"].to_numpy(dtype="datetime64[ns]")
    window = np.timedelta64(int(base["data"]["window_width"]), "m")
    horizon = np.timedelta64(1800, "s")
    snapshot = np.asarray(test["X_raw"][:, -1, :])
    frame = test["metadata"].copy()
    frame["N_up"] = snapshot[:, 0]
    frame["N_complete"] = snapshot[:, 1]
    frame["R_comp"] = snapshot[:, 2]
    frame["rho_t"] = snapshot[:, 3] / 300.0
    period_state = frame.groupby("update_period_id", sort=False).first().reset_index()
    rows = []
    for row in period_state.itertuples(index=False):
        time = np.datetime64(pd.Timestamp(row.prediction_time), "ns")
        start = np.searchsorted(up_times, time - window, side="right")
        end = np.searchsorted(up_times, time, side="right")
        candidate_up = up_times[start:end]
        candidate_down = down_times[start:end]
        completed = candidate_down <= time
        n_eventual = int(end - start)
        n_up = float(row.N_up)
        n_complete = float(row.N_complete)
        q_match = n_eventual / max(n_up, 1.0)
        c_delay = n_complete / max(n_eventual, 1.0)
        label_age = (
            float((time - candidate_up[completed].max()) / np.timedelta64(1, "s"))
            if completed.any()
            else float("nan")
        )
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
                "h1800_saturation": float(
                    np.sum(candidate_down <= time + horizon) / max(n_eventual, 1)
                ),
            }
        )
    diagnostics = pd.DataFrame(rows)
    if not (
        (diagnostics["N_complete"] <= diagnostics["N_eventual"]).all()
        and (diagnostics["N_eventual"] <= diagnostics["N_up"]).all()
    ):
        raise RuntimeError("AP-01 delay count identity failed on test periods.")
    identity = diagnostics["Q_match"] * diagnostics["C_delay"]
    if float(np.max(np.abs(identity - diagnostics["R_comp"]))) > 1e-6:
        raise RuntimeError("AP-01 R_comp decomposition failed on test periods.")
    return diagnostics


def run_ap01_test(base, experiment, output_dir, test, lock):
    mask = exclude_tail_periods(test["metadata"], count=6)
    metadata = test["metadata"].loc[mask].reset_index(drop=True)
    y = np.asarray(test["y_raw"])[mask]
    full_snapshot = np.asarray(test["X_raw"][:, -1, :])
    feature_sets = {
        "full_h0": list(range(14)),
        "no_R": [index for index in range(14) if index != 2],
        "no_rho_carrier": [index for index in range(14) if index != 3],
        "no_completion_block": [index for index in range(14) if index not in (1, 2)],
        "no_moment_block": [index for index in range(14) if index not in (3, 4)],
        "base": [0] + list(range(5, 14)),
    }
    variants = {
        name: full_snapshot[mask][:, indices] for name, indices in feature_sets.items()
    }
    for seconds in experiment["ap01"]["lookahead_seconds"]:
        seconds = int(seconds)
        if seconds == 0:
            continue
        snapshot, lookahead_metadata = lookahead_test_snapshot(
            base, output_dir, seconds
        )
        if not lookahead_metadata["case_id"].equals(test["metadata"]["case_id"]):
            raise RuntimeError("AP-01 test case alignment changed.")
        variants[f"full_h{seconds}"] = snapshot[mask]

    per_case = metadata[["case_id", "update_period_id", "prediction_time"]].copy()
    score_by_variant = {}
    rows = []
    for name, X in variants.items():
        model_path = output_dir / "ap01" / "models" / f"{name}.joblib"
        if not model_path.exists():
            raise RuntimeError(f"Missing frozen AP-01 model: {model_path}")
        models = joblib.load(model_path)
        quantiles = np.column_stack([model.predict(X) for model in models])
        losses = np.column_stack(
            [
                pinball_loss(y, quantiles[:, index], quantile)
                for index, quantile in enumerate(experiment["gbdt"]["quantiles"])
            ]
        )
        mu, sigma, crossing = quantiles_to_gaussian(quantiles)
        density = density_metrics(
            np.ones((len(y), 1)), mu[:, None], sigma[:, None], y
        )
        score_by_variant[name] = density["crps"]
        per_case[f"{name}_crps"] = density["crps"]
        per_case[f"{name}_mean_pinball"] = losses.mean(axis=1)
        rows.append(
            {
                "variant": name,
                "n_cases": len(y),
                "n_periods": metadata["update_period_id"].nunique(),
                "mean_crps": float(np.mean(density["crps"])),
                "mean_pinball": float(np.mean(losses)),
                "raw_crossing_rate": float(np.mean(crossing)),
                "coverage80": float(
                    np.mean((y >= quantiles[:, 0]) & (y <= quantiles[:, 2]))
                ),
                "width80": float(np.mean(quantiles[:, 2] - quantiles[:, 0])),
            }
        )
    ap01 = output_dir / "ap01"
    per_case.to_csv(ap01 / "per_case_scores.csv", index=False)
    pd.DataFrame(rows).to_csv(ap01 / "test_variants.csv", index=False)
    period_scores = per_case.groupby("update_period_id", as_index=False).mean(
        numeric_only=True
    )
    period_scores.to_csv(ap01 / "per_period_scores.csv", index=False)

    block_length = int(lock["block_length"]["block_length_periods"])
    period_index, blocks = moving_block_period_indices(
        metadata,
        block_length,
        int(experiment["ap09"]["test_repeats"]),
        seed=29042,
    )
    comparisons = [
        "full_h1800",
        "no_rho_carrier",
        "no_completion_block",
        "base",
    ]
    effect_rows = []
    for variant in comparisons:
        difference = score_by_variant[variant] - score_by_variant["full_h0"]
        case_boot, period_boot = bootstrap_estimates(
            difference, metadata, period_index, blocks
        )
        point = float(np.mean(difference))
        centered = case_boot - float(np.mean(case_boot))
        p_value = float(
            (1 + np.sum(np.abs(centered) >= abs(point))) / (len(centered) + 1)
        )
        case_ci = percentile_interval(case_boot)
        period_ci = percentile_interval(period_boot)
        effect_rows.append(
            {
                "reference": "full_h0",
                "variant": variant,
                "effect": "variant_minus_full_crps",
                "case_effect": point,
                "case_ci_low": case_ci[0],
                "case_ci_high": case_ci[1],
                "period_effect": float(np.mean(period_boot)),
                "period_ci_low": period_ci[0],
                "period_ci_high": period_ci[1],
                "p_value": p_value,
            }
        )
    adjusted = holm_adjust([row["p_value"] for row in effect_rows])
    for row, value in zip(effect_rows, adjusted):
        row["p_holm"] = float(value)
    pd.DataFrame(effect_rows).to_csv(ap01 / "effects.csv", index=False)

    diagnostics = test_delay_diagnostics(base, test)
    diagnostics.to_csv(ap01 / "delay_per_period_test.csv", index=False)
    summary_path = ap01 / "delay_summary.csv"
    summary = pd.read_csv(summary_path)
    test_summary = pd.DataFrame(
        [
            {
                "split": "test",
                "n_cases": len(test["y_raw"]),
                "n_periods": len(diagnostics),
                "travel_time_mean_seconds": float(np.mean(test["y_raw"])),
                "travel_time_median_seconds": float(np.median(test["y_raw"])),
                "travel_time_p95_seconds": float(np.quantile(test["y_raw"], 0.95)),
                "p_travel_time_gt_300": float(np.mean(test["y_raw"] > 300.0)),
                "label_age_median_seconds": float(
                    diagnostics["label_age_seconds"].median()
                ),
                "inflight_share_mean": float(diagnostics["inflight_share"].mean()),
                "h1800_saturation_mean": float(
                    diagnostics["h1800_saturation"].mean()
                ),
            }
        ]
    )
    pd.concat([summary[summary["split"] != "test"], test_summary], ignore_index=True).to_csv(
        summary_path, index=False
    )
    return diagnostics


def period_sensing_state(test):
    snapshot = np.asarray(test["X_raw"][:, -1, :])
    cases = test["metadata"][["case_id", "update_period_id", "prediction_time"]].copy()
    cases["R_comp"] = snapshot[:, 2]
    cases["rho_t"] = snapshot[:, 3] / 300.0
    variation = cases.groupby("update_period_id")[["R_comp", "rho_t"]].nunique()
    if (variation.to_numpy() > 1).any():
        raise RuntimeError("Test sensing state varies within an update period.")
    periods = (
        cases.groupby("update_period_id", sort=False, as_index=False)
        .agg(
            prediction_time=("prediction_time", "first"),
            R_comp=("R_comp", "first"),
            rho_t=("rho_t", "first"),
            n_cases=("case_id", "size"),
        )
        .set_index("update_period_id", drop=False)
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
            "period": np.asarray(period_ids)[mask],
            "coverage": ((y[mask] >= lo[mask]) & (y[mask] <= hi[mask])).astype(float),
            "width": hi[mask] - lo[mask],
            "interval_score": interval_score(y[mask], lo[mask], hi[mask], 0.1),
            "lower_miss": (y[mask] < lo[mask]).astype(float),
            "upper_miss": (y[mask] > hi[mask]).astype(float),
        }
    )
    period_values = values.groupby("period").mean(numeric_only=True)
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


def service_rule_metrics(test, component_lo, component_hi, fallback_lo, fallback_hi, modes):
    y = np.asarray(test["y_raw"], dtype=float)
    period_ids = test["metadata"]["update_period_id"].to_numpy()
    mode_by_period = pd.Series(np.asarray(modes, dtype=bool), index=modes.index)
    broadcast = pd.Series(period_ids).map(mode_by_period).to_numpy(dtype=bool)
    combined_lo = np.where(broadcast, component_lo, fallback_lo)
    combined_hi = np.where(broadcast, component_hi, fallback_hi)
    result = {
        "publication_rate_period": float(np.mean(modes)),
        "trajectory_exposure_share": float(np.mean(broadcast)),
        "n_broadcast_periods": int(np.sum(modes)),
        "n_fallback_periods": int(len(modes) - np.sum(modes)),
    }
    for name, lo, hi, mask in (
        ("broadcast", component_lo, component_hi, broadcast),
        ("fallback", fallback_lo, fallback_hi, ~broadcast),
        ("combined", combined_lo, combined_hi, np.ones(len(y), dtype=bool)),
    ):
        for key, value in weighted_interval_metrics(y, lo, hi, period_ids, mask).items():
            result[f"{name}_{key}"] = value
    result["combined_width_share_of_update_period"] = (
        result["combined_width_period"] / 300.0
    )
    result["combined_width_reduction_from_fallback"] = float(
        np.mean(fallback_hi - fallback_lo) - result["combined_width_period"]
    )
    return result, broadcast, combined_lo, combined_hi


def apply_frozen_rule(periods, rule):
    mode = rule["mode"]
    if mode == "fallback_only":
        return pd.Series(False, index=periods.index)
    if mode == "broadcast_all":
        return pd.Series(True, index=periods.index)
    if mode == "joint_gate":
        return (periods["R_comp"] >= float(rule["R_low"])) & (
            periods["rho_t"] <= float(rule["rho_high"])
        )
    if mode == "rho_only":
        return periods["rho_t"] <= float(rule["rho_high"])
    if mode == "R_comp_only":
        return periods["R_comp"] >= float(rule["R_low"])
    if mode == "label_age_only":
        return periods["label_age_seconds"] <= float(rule["label_age_high"])
    if mode == "label_age_R_comp":
        return (periods["label_age_seconds"] <= float(rule["label_age_high"])) & (
            periods["R_comp"] >= float(rule["R_low"])
        )
    raise ValueError(mode)


def aggregate_service_seed_rows(frame):
    metrics = [
        column for column in frame.columns if column not in ("model", "seed", "rule")
    ]
    rows = []
    for (model, rule), group in frame.groupby(["model", "rule"], sort=False):
        row = {
            "model": model,
            "rule": rule,
            "n_seeds": int(group["seed"].notna().sum()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_seed_sd"] = (
                float(group[metric].std(ddof=1)) if len(group) > 1 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run_ap06_and_ap10_test(
    records, test, output_dir, lock, diagnostics, period_index, blocks
):
    _, periods = period_sensing_state(test)
    periods = periods.join(
        diagnostics.set_index("update_period_id")[["label_age_seconds"]], how="left"
    )
    policies = lock["service_policies"]
    fallback = policies["fallback"]
    fallback_lo = np.full(len(test["y_raw"]), float(fallback["lo90"]))
    fallback_hi = np.full(len(test["y_raw"]), float(fallback["hi90"]))
    primary = next(
        record
        for record in records
        if record["model"] == "snapshot_quantile_induced_gaussian"
    )
    component_lo = primary["metrics"]["lo90"]
    component_hi = primary["metrics"]["hi90"]
    rules = {
        **policies["selected_policies"],
        **policies["comparison_rules"],
        **lock.get("ap01_service_rules", {}),
    }
    policy_rows = []
    mode_columns = {}
    combined_by_rule = {}
    for name, rule in rules.items():
        modes = apply_frozen_rule(periods, rule)
        metrics, broadcast, combined_lo, combined_hi = service_rule_metrics(
            test,
            component_lo,
            component_hi,
            fallback_lo,
            fallback_hi,
            modes,
        )
        policy_rows.append({"policy": name, "mode": rule["mode"], **metrics})
        mode_columns[name] = (modes, broadcast)
        combined_by_rule[name] = (combined_lo, combined_hi)
    ap06 = output_dir / "ap06"
    pd.DataFrame(policy_rows).to_csv(ap06 / "test_service_by_policy.csv", index=False)

    period_output = periods.reset_index(drop=True).copy()
    for name, (modes, _) in mode_columns.items():
        period_output[f"broadcast_{name}"] = modes.to_numpy(dtype=np.uint8)
    period_output.to_csv(ap06 / "per_period_service.csv", index=False)

    anchor_modes = mode_columns["anchor"][0]
    service_rows = []
    for record in records:
        metrics, _, _, _ = service_rule_metrics(
            test,
            record["metrics"]["lo90"],
            record["metrics"]["hi90"],
            fallback_lo,
            fallback_hi,
            anchor_modes,
        )
        service_rows.append(
            {
                "model": record["model"],
                "seed": record["seed"],
                "rule": "anchor",
                **metrics,
            }
        )
    service_by_seed = pd.DataFrame(service_rows)
    ap10 = output_dir / "ap10"
    ap10.mkdir(parents=True, exist_ok=True)
    service_by_seed.to_csv(ap10 / "service_kpi_by_seed.csv", index=False)
    aggregate_service_seed_rows(service_by_seed).to_csv(
        ap10 / "service_kpi_aggregate.csv", index=False
    )

    cases, _ = period_sensing_state(test)
    per_case = cases.copy()
    per_case["y_seconds"] = np.asarray(test["y_raw"])
    per_case["component_lo90"] = component_lo
    per_case["component_hi90"] = component_hi
    per_case["fallback_lo90"] = fallback_lo
    per_case["fallback_hi90"] = fallback_hi
    for name, (_, broadcast) in mode_columns.items():
        per_case[f"broadcast_{name}"] = broadcast.astype(np.uint8)
    anchor_lo, anchor_hi = combined_by_rule["anchor"]
    per_case["anchor_service_lo90"] = anchor_lo
    per_case["anchor_service_hi90"] = anchor_hi
    per_case.to_csv(ap10 / "service_kpi_per_case.csv", index=False)
    period_output.to_csv(ap10 / "service_kpi_per_period.csv", index=False)
    write_json(
        ap10 / "service_kpi_thresholds.json",
        {
            "created_utc": utc_now(),
            "protocol_version": lock["protocol_version"],
            "case_count": len(test["y_raw"]),
            "unique_period_count": int(periods.index.nunique()),
            "threshold_source": "unique validation periods",
            "policies": rules,
            "fallback": fallback,
            "primary_component": "snapshot_quantile_induced_gaussian",
            "publication_weight": "equal weight per unique update period",
            "trajectory_exposure_weight": "equal weight per trajectory",
            "coverage_width_weights": ["trajectory", "equal-period"],
        },
    )

    effect_rows = []
    anchor_score = interval_score(
        np.asarray(test["y_raw"]), anchor_lo, anchor_hi, alpha=0.1
    )
    for name, (lo, hi) in combined_by_rule.items():
        if name == "anchor":
            continue
        difference = interval_score(
            np.asarray(test["y_raw"]), lo, hi, alpha=0.1
        ) - anchor_score
        case_boot, period_boot = bootstrap_estimates(
            difference, test["metadata"], period_index, blocks
        )
        case_ci = percentile_interval(case_boot)
        period_ci = percentile_interval(period_boot)
        effect_rows.append(
            {
                "policy": name,
                "reference": "anchor",
                "effect": "policy_minus_anchor_interval_score",
                "case_effect": float(np.mean(difference)),
                "case_ci_low": case_ci[0],
                "case_ci_high": case_ci[1],
                "period_effect": float(np.mean(period_boot)),
                "period_ci_low": period_ci[0],
                "period_ci_high": period_ci[1],
            }
        )
    pd.DataFrame(effect_rows).to_csv(ap06 / "policy_effects.csv", index=False)


def run_ap03_paired_effects(records, test, period_index, blocks, output_dir):
    rows = []
    models = list(dict.fromkeys(record["model"] for record in records))
    for model in models:
        model_records = [record for record in records if record["model"] == model]
        for metric in ("crps", "nll"):
            raw = np.mean(
                [record["metrics"][metric] for record in model_records], axis=0
            )
            truncated = np.mean(
                [record["truncated_metrics"][metric] for record in model_records],
                axis=0,
            )
            difference = truncated - raw
            case_boot, period_boot = bootstrap_estimates(
                difference, test["metadata"], period_index, blocks
            )
            point = float(np.mean(difference))
            centered = case_boot - float(np.mean(case_boot))
            p_value = float(
                (1 + np.sum(np.abs(centered) >= abs(point))) / (len(centered) + 1)
            )
            case_ci = percentile_interval(case_boot)
            period_ci = percentile_interval(period_boot)
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "effect": "truncated_minus_raw",
                    "case_effect": point,
                    "case_ci_low": case_ci[0],
                    "case_ci_high": case_ci[1],
                    "period_effect": float(np.mean(period_boot)),
                    "period_ci_low": period_ci[0],
                    "period_ci_high": period_ci[1],
                    "p_value": p_value,
                }
            )
    adjusted = holm_adjust([row["p_value"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["p_holm"] = float(value)
    pd.DataFrame(rows).to_csv(output_dir / "ap03" / "paired_effects.csv", index=False)


def run_ap04_test(test, output_dir, lock, records, period_index, blocks):
    selected_k = int(lock["selected_k"]["K_star"])
    ap04 = output_dir / "ap04"
    if selected_k <= 2:
        pd.DataFrame(
            [
                {
                    "K_star": selected_k,
                    "status": "not activated by the validation rule",
                }
            ]
        ).to_csv(ap04 / "h3_test_effects.csv", index=False)
        return
    y_scaler = joblib.load(output_dir / "g0" / "y_scaler.joblib")
    spec = lock["final_models"]["recurrent"]["spec"]
    history = int(spec["history_length"])
    k2_scores = []
    for seed, checkpoint in lock["final_models"]["recurrent"][
        "K2_checkpoints"
    ].items():
        model, _ = load_density_model(output_dir / checkpoint)
        pis, mus_scaled, sigmas_scaled = predict_density(
            model, test["X_scaled"][:, -history:, :]
        )
        pis, mus, sigmas = raw_mixture_parameters(
            pis, mus_scaled, sigmas_scaled, y_scaler
        )
        k2_scores.append(
            density_metrics(pis, mus, sigmas, test["y_raw"])["crps"]
        )
    k2 = np.mean(k2_scores, axis=0)
    kstar = mean_record_metrics(records, "recurrent_mdn", "crps")
    difference = kstar - k2
    rho = np.asarray(test["X_raw"][:, -1, 3], dtype=float) / 300.0
    rho_high = float(
        lock["service_policies"]["comparison_rules"]["anchor"]["rho_high"]
    )
    high = rho >= rho_high
    low = ~high
    high_case, high_period = bootstrap_estimates(
        difference, test["metadata"], period_index, blocks, mask=high
    )
    low_case, low_period = bootstrap_estimates(
        difference, test["metadata"], period_index, blocks, mask=low
    )
    interaction_case = high_case - low_case
    interaction_period = high_period - low_period
    rows = []
    for stratum, mask, case_boot, period_boot in (
        ("high_rho", high, high_case, high_period),
        ("low_rho", low, low_case, low_period),
    ):
        case_ci = percentile_interval(case_boot)
        period_ci = percentile_interval(period_boot)
        rows.append(
            {
                "K_star": selected_k,
                "reference_K": 2,
                "stratum": stratum,
                "n_cases": int(np.sum(mask)),
                "effect": "K_star_minus_K2_crps",
                "case_effect": float(np.mean(difference[mask])),
                "case_ci_low": case_ci[0],
                "case_ci_high": case_ci[1],
                "period_effect": float(np.mean(period_boot)),
                "period_ci_low": period_ci[0],
                "period_ci_high": period_ci[1],
            }
        )
    interaction_case_ci = percentile_interval(interaction_case)
    interaction_period_ci = percentile_interval(interaction_period)
    rows.append(
        {
            "K_star": selected_k,
            "reference_K": 2,
            "stratum": "high_minus_low_interaction",
            "n_cases": len(difference),
            "effect": "difference_in_K_effect",
            "case_effect": float(np.mean(interaction_case)),
            "case_ci_low": interaction_case_ci[0],
            "case_ci_high": interaction_case_ci[1],
            "period_effect": float(np.mean(interaction_period)),
            "period_ci_low": interaction_period_ci[0],
            "period_ci_high": interaction_period_ci[1],
        }
    )
    pd.DataFrame(rows).to_csv(ap04 / "h3_test_effects.csv", index=False)


def write_test_manifest(output_dir, lock):
    roots = [
        output_dir / "test",
        output_dir / "ap01",
        output_dir / "ap02",
        output_dir / "ap03",
        output_dir / "ap04",
        output_dir / "ap06",
        output_dir / "ap09",
        output_dir / "ap10",
    ]
    files = []
    for root in roots:
        if root.exists():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and "tcn" not in path.name.lower()
            )
    files = sorted(set(files), key=str)
    write_json(
        output_dir / "test" / "manifest.json",
        {
            "completed_utc": utc_now(),
            "protocol_version": lock["protocol_version"],
            "selection_lock_sha256": sha256_file(
                output_dir / "freeze" / "selection_lock.json"
            ),
            "files": {
                str(path.relative_to(output_dir)): {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in files
                if path != output_dir / "test" / "manifest.json"
            },
        },
    )


def main():
    base, experiment, output_dir, lock = read_context()
    verify_frozen_artifacts(lock)
    verify_test_cache(output_dir, lock)
    append_access_event(
        output_dir,
        "TEST_SCORE_BEGIN",
        "Single frozen batch for AP-01/02/03/04/06/09 and AP-10.",
    )
    test = load_test(output_dir)
    expected = experiment["expected_split"]
    if len(test["y_raw"]) != int(expected["case_counts"]["test"]):
        raise RuntimeError("Test case count differs from the frozen protocol.")
    if test["metadata"]["update_period_id"].nunique() != int(
        expected["period_counts"]["test"]
    ):
        raise RuntimeError("Test period count differs from the frozen protocol.")

    records, native_quantiles = evaluate_predictions(test, output_dir, lock)
    summarize_density_records(records, test, output_dir)
    native_per_case = summarize_native_quantiles(
        native_quantiles, test, experiment, output_dir
    )
    run_ap03_test(records, test, output_dir)
    period_index, blocks = run_ap09_test(
        records, native_per_case, test, experiment, output_dir, lock
    )
    run_ap03_paired_effects(records, test, period_index, blocks, output_dir)
    run_ap04_test(test, output_dir, lock, records, period_index, blocks)
    diagnostics = run_ap01_test(base, experiment, output_dir, test, lock)
    run_ap06_and_ap10_test(
        records, test, output_dir, lock, diagnostics, period_index, blocks
    )
    write_test_manifest(output_dir, lock)
    append_access_event(
        output_dir,
        "TEST_SCORE_COMPLETE",
        "Frozen batch completed without reopening model or policy selection.",
    )


if __name__ == "__main__":
    main()
