from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm


def normal_pdf(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def gaussian_crps(mu, sigma, y):
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
    z = (np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)) / sigma
    return sigma * (
        z * (2.0 * ndtr(z) - 1.0)
        + 2.0 * normal_pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )


def gaussian_nll(mu, sigma, y):
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
    z = (np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)) / sigma
    return np.log(sigma) + 0.5 * np.log(2.0 * np.pi) + 0.5 * z * z


def gaussian_pit(mu, sigma, y):
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
    return np.clip(ndtr((np.asarray(y) - np.asarray(mu)) / sigma), 1e-9, 1 - 1e-9)


def _a_term(a, s):
    s = np.maximum(np.asarray(s, dtype=float), 1e-8)
    z = np.asarray(a, dtype=float) / s
    return 2.0 * s * normal_pdf(z) + np.asarray(a) * (2.0 * ndtr(z) - 1.0)


def mixture_crps(pis, mus, sigmas, y):
    pis = np.asarray(pis, dtype=float)
    mus = np.asarray(mus, dtype=float)
    sigmas = np.maximum(np.asarray(sigmas, dtype=float), 1e-8)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    term1 = np.sum(pis * _a_term(y - mus, sigmas), axis=1)
    pair = np.zeros(len(y), dtype=float)
    for a in range(pis.shape[1]):
        for b in range(pis.shape[1]):
            pair += (
                pis[:, a]
                * pis[:, b]
                * _a_term(
                    mus[:, a] - mus[:, b],
                    np.sqrt(sigmas[:, a] ** 2 + sigmas[:, b] ** 2),
                )
            )
    return term1 - 0.5 * pair


def mixture_nll(pis, mus, sigmas, y):
    pis = np.asarray(pis, dtype=float)
    mus = np.asarray(mus, dtype=float)
    sigmas = np.maximum(np.asarray(sigmas, dtype=float), 1e-8)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    z = (y - mus) / sigmas
    log_terms = (
        np.log(np.maximum(pis, 1e-15))
        - np.log(sigmas)
        - 0.5 * np.log(2.0 * np.pi)
        - 0.5 * z * z
    )
    max_log = np.max(log_terms, axis=1, keepdims=True)
    return -(
        max_log[:, 0]
        + np.log(np.sum(np.exp(log_terms - max_log), axis=1))
    )


def mixture_cdf(pis, mus, sigmas, y):
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    sigmas = np.maximum(np.asarray(sigmas, dtype=float), 1e-8)
    return np.sum(np.asarray(pis) * ndtr((y - np.asarray(mus)) / sigmas), axis=1)


def mixture_quantile(pis, mus, sigmas, probability, iterations=60):
    pis = np.asarray(pis, dtype=float)
    mus = np.asarray(mus, dtype=float)
    sigmas = np.maximum(np.asarray(sigmas, dtype=float), 1e-8)
    probability = np.broadcast_to(np.asarray(probability, dtype=float), (len(pis),))
    lo = np.min(mus - 10.0 * sigmas, axis=1)
    hi = np.max(mus + 10.0 * sigmas, axis=1)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        cdf = mixture_cdf(pis, mus, sigmas, mid)
        lo = np.where(cdf < probability, mid, lo)
        hi = np.where(cdf >= probability, mid, hi)
    return 0.5 * (lo + hi)


def negative_mass(pis, mus, sigmas):
    return mixture_cdf(pis, mus, sigmas, np.zeros(len(pis)))


def truncated_mixture_nll(pis, mus, sigmas, y):
    mass = negative_mass(pis, mus, sigmas)
    return mixture_nll(pis, mus, sigmas, y) + np.log(np.maximum(1.0 - mass, 1e-15))


def truncated_mixture_pit(pis, mus, sigmas, y):
    mass = negative_mass(pis, mus, sigmas)
    z = np.maximum(1.0 - mass, 1e-15)
    return np.clip((mixture_cdf(pis, mus, sigmas, y) - mass) / z, 1e-9, 1 - 1e-9)


def truncated_mixture_quantile(pis, mus, sigmas, probability):
    mass = negative_mass(pis, mus, sigmas)
    target = mass + np.asarray(probability) * (1.0 - mass)
    return np.maximum(mixture_quantile(pis, mus, sigmas, target), 0.0)


def truncated_mixture_crps(pis, mus, sigmas, y, points=2001, chunk_size=128):
    pis = np.asarray(pis, dtype=float)
    mus = np.asarray(mus, dtype=float)
    sigmas = np.maximum(np.asarray(sigmas, dtype=float), 1e-8)
    y = np.asarray(y, dtype=float)
    result = np.empty(len(y), dtype=float)
    left_points = (int(points) + 1) // 2
    right_points = int(points) - left_points + 1
    left_unit = np.linspace(0.0, 1.0, left_points)
    right_unit = np.linspace(0.0, 1.0, right_points)
    for start in range(0, len(y), int(chunk_size)):
        stop = min(start + int(chunk_size), len(y))
        chunk_y = y[start:stop]
        chunk_pis = pis[start:stop]
        chunk_mus = mus[start:stop]
        chunk_sigmas = sigmas[start:stop]
        upper = np.maximum(
            np.maximum(chunk_y, np.max(chunk_mus + 10.0 * chunk_sigmas, axis=1)),
            1.0,
        )
        mass = negative_mass(chunk_pis, chunk_mus, chunk_sigmas)
        scale = np.maximum(1.0 - mass[:, None], 1e-15)
        split = np.maximum(chunk_y, 0.0)
        left_grid = split[:, None] * left_unit[None, :]
        right_grid = split[:, None] + (
            upper - split
        )[:, None] * right_unit[None, :]

        def truncated_cdf(grid):
            z = (
                grid[:, :, None] - chunk_mus[:, None, :]
            ) / chunk_sigmas[:, None, :]
            cdf = np.sum(chunk_pis[:, None, :] * ndtr(z), axis=2)
            return np.clip((cdf - mass[:, None]) / scale, 0.0, 1.0)

        left_cdf = truncated_cdf(left_grid)
        right_cdf = truncated_cdf(right_grid)
        result[start:stop] = (
            np.maximum(-chunk_y, 0.0)
            + np.trapz(left_cdf**2, left_grid, axis=1)
            + np.trapz((1.0 - right_cdf) ** 2, right_grid, axis=1)
        )
    return result


def pinball_loss(y, prediction, quantile):
    error = np.asarray(y, dtype=float) - np.asarray(prediction, dtype=float)
    return np.maximum(quantile * error, (quantile - 1.0) * error)


def interval_score(y, lo, hi, alpha=0.1):
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return (
        hi
        - lo
        + (2.0 / alpha) * (lo - y) * (y < lo)
        + (2.0 / alpha) * (y - hi) * (y > hi)
    )


def ks_statistic(values):
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) == 0:
        return float("nan")
    ranks = np.arange(1, len(values) + 1, dtype=float)
    return float(
        max(
            np.max(ranks / len(values) - values),
            np.max(values - (ranks - 1.0) / len(values)),
        )
    )


def density_metrics(pis, mus, sigmas, y):
    crps = mixture_crps(pis, mus, sigmas, y)
    nll = mixture_nll(pis, mus, sigmas, y)
    pit = np.clip(mixture_cdf(pis, mus, sigmas, y), 1e-9, 1 - 1e-9)
    lo = mixture_quantile(pis, mus, sigmas, 0.05)
    hi = mixture_quantile(pis, mus, sigmas, 0.95)
    covered = ((np.asarray(y) >= lo) & (np.asarray(y) <= hi)).astype(float)
    return {
        "crps": crps,
        "nll": nll,
        "pit": pit,
        "lo90": lo,
        "hi90": hi,
        "covered90": covered,
        "width90": hi - lo,
        "interval_score90": interval_score(y, lo, hi, alpha=0.1),
    }


def aggregate_case_and_period(values, period_ids):
    values = np.asarray(values, dtype=float)
    frame = pd.DataFrame(
        {"update_period_id": np.asarray(period_ids), "value": values}
    )
    return {
        "case_mean": float(np.mean(values)),
        "period_mean": float(
            frame.groupby("update_period_id", sort=False)["value"].mean().mean()
        ),
    }


def period_mean_series(values, metadata):
    frame = metadata[["update_period_id", "prediction_time"]].copy()
    frame["value"] = np.asarray(values, dtype=float)
    return (
        frame.groupby("update_period_id", sort=False, as_index=False)
        .agg(prediction_time=("prediction_time", "first"), value=("value", "mean"))
    )


def acf(values, max_lag):
    values = np.asarray(values, dtype=float)
    values = values - np.mean(values)
    denominator = float(np.dot(values, values))
    if denominator == 0:
        return np.zeros(max_lag + 1)
    result = np.ones(max_lag + 1)
    for lag in range(1, max_lag + 1):
        result[lag] = float(np.dot(values[:-lag], values[lag:]) / denominator)
    return result


def segmented_acf(values, times, max_lag):
    values = np.asarray(values, dtype=float)
    times = pd.to_datetime(times).reset_index(drop=True)
    centered = values - np.mean(values)
    denominator = float(np.dot(centered, centered))
    if denominator == 0:
        return np.zeros(max_lag + 1)
    breaks = np.flatnonzero(
        times.diff().fillna(pd.Timedelta(minutes=5)) != pd.Timedelta(minutes=5)
    )
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(values)]
    result = np.ones(max_lag + 1)
    for lag in range(1, max_lag + 1):
        numerator = 0.0
        for start, end in zip(starts, ends):
            if end - start > lag:
                segment = centered[start:end]
                numerator += float(np.dot(segment[:-lag], segment[lag:]))
        result[lag] = numerator / denominator
    return result


def select_block_length(period_sequences, candidates):
    candidates = sorted(int(x) for x in candidates)
    max_lag = max(candidates) - 1
    required = 30
    diagnostics = []
    for name, values in period_sequences.items():
        if isinstance(values, pd.DataFrame):
            series = values["value"].to_numpy(dtype=float)
            series_acf = segmented_acf(
                series, values["prediction_time"], max_lag + 6
            )
        else:
            series = np.asarray(values, dtype=float)
            series_acf = acf(series, max_lag + 6)
        bound = 1.96 / np.sqrt(len(series))
        decorrelation = None
        for lag in range(1, max_lag + 1):
            if np.all(np.abs(series_acf[lag : lag + 6]) <= bound):
                decorrelation = lag
                break
        if decorrelation is None:
            decorrelation = max(candidates)
        required = max(required, decorrelation + 1)
        diagnostics.append(
            {"model": name, "decorrelation_lag": decorrelation, "bound": bound}
        )
    eligible = [length for length in candidates if length >= required]
    if not eligible:
        raise ValueError(f"No block candidate is at least the required length {required}.")
    return min(eligible), diagnostics


def moving_block_period_indices(metadata, block_length, repeats, seed):
    periods = (
        metadata[["update_period_id", "prediction_time"]]
        .drop_duplicates("update_period_id")
        .sort_values("prediction_time")
        .reset_index(drop=True)
    )
    times = pd.to_datetime(periods["prediction_time"])
    breaks = np.flatnonzero(times.diff().fillna(pd.Timedelta(minutes=5)) != pd.Timedelta(minutes=5))
    starts = np.r_[0, breaks]
    ends = np.r_[breaks, len(periods)]
    valid_starts = []
    for start, end in zip(starts, ends):
        if end - start >= block_length:
            valid_starts.extend(range(int(start), int(end - block_length + 1)))
    if not valid_starts:
        raise ValueError("No contiguous segment can supply one full moving block.")
    rng = np.random.default_rng(seed)
    out = np.empty((repeats, len(periods)), dtype=np.int32)
    valid_starts = np.asarray(valid_starts, dtype=int)
    for repeat in range(repeats):
        sampled = []
        while len(sampled) < len(periods):
            start = int(rng.choice(valid_starts))
            sampled.extend(range(start, start + block_length))
        out[repeat] = np.asarray(sampled[: len(periods)], dtype=np.int32)
    return periods, out


def normal_interval(mu, sigma, lower=0.05, upper=0.95):
    sigma = np.asarray(sigma, dtype=float)
    return (
        np.asarray(mu) + norm.ppf(lower) * sigma,
        np.asarray(mu) + norm.ppf(upper) * sigma,
    )
