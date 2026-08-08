import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import os
import warnings

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv('LSTM_MDN_DATA_DIR', SCRIPT_DIR / 'data')).resolve()

def _resolve_data_path(file_path):
    path_obj = Path(file_path)
    if path_obj.is_absolute():
        return path_obj
    return DATA_DIR / path_obj

def time_features(timestamp):
    """Extract cyclical time features from timestamp"""
    dt_object = timestamp
    # seconds since midnight
    seconds_since_midnight = dt_object.hour * 3600 + dt_object.minute * 60 + dt_object.second

    # Paper Eq. (14) ToD(t) = (sin(2pi*s/86400), cos(2pi*s/86400))
    # We also keep Dow(t) as one-hot or cyclical if needed.
    # Here we stick to the 6-dim features for compatibility with existing code structure if kept,
    # but the paper specifies 2-dim ToD + 7-dim DoW.
    # Let's implementation strictly following paper for ToD,
    # but for simplicity we might keep the 6-dim detailed version if it doesn't hurt,
    # OR strictly follow paper. Let's follow paper more closely but keep it simple.
    # The paper uses: sin(2pi*s/86400), cos(2pi*s/86400)

    tod_sin = np.sin(2 * np.pi * seconds_since_midnight / 86400)
    tod_cos = np.cos(2 * np.pi * seconds_since_midnight / 86400)

    # DoW: 0=Monday, 6=Sunday
    dow = dt_object.dayofweek
    dow_onehot = np.zeros(7)
    dow_onehot[dow] = 1

    return np.concatenate([[tod_sin, tod_cos], dow_onehot])

def compute_rolling_features(
    up_df,
    merged_df,
    delta_obs_min,
    window_width_min,
    mu_default=None,
    s2_default=None,
    completion_lookahead_seconds=0,
):
    """
    Compute rolling window features N_up, N_match, R_comp, bar_y, s_y^2
    as defined in Eq. (11)-(13) and (14)-(15).

    Time is discretized at t_n = n * delta_obs.
    """
    # 1. Define the time grid
    # Start from slight before the first data point to cover everything
    start_time = up_df['timestamp_up'].min().floor(f'{delta_obs_min}min')
    end_time = merged_df['timestamp_down'].max().ceil(f'{delta_obs_min}min')

    time_grid = pd.date_range(start=start_time, end=end_time, freq=f'{delta_obs_min}min')

    features_list = []

    # Pre-calculate useful columns
    # We need to vectorized this for speed.
    # iterating over time_grid might be slow if grid is large.
    # but for typical traffic data (months), it's manageable (e.g. 1 month = 8640 points for 5min)

    # N_up is computed from upstream detections only, which avoids conditioning on
    # future downstream matching outcomes.
    up_times_all = up_df['timestamp_up'].values

    up_times = merged_df['timestamp_up'].values
    down_times = merged_df['timestamp_down'].values
    travel_times = merged_df['travel_time'].values

    # Convert parameters to nanoseconds for comparison with numpy datetime64
    window_ns = np.timedelta64(window_width_min, 'm')

    # The caller must provide training-segment defaults to avoid leakage.
    if mu_default is None or s2_default is None:
        raise ValueError(
            "mu_default and s2_default must be provided from the training segment "
            "to satisfy leakage-control requirements."
        )

    # Use a sliding window approach?
    # Since events are sorted, we can use searchsorted.

    # Sort upstream times for N_up counting.
    up_times_all = np.sort(up_times_all)

    # Sort matched trajectories by upstream time (should be already sorted, but ensure it)
    sort_idx = np.argsort(up_times)
    up_times = up_times[sort_idx]
    down_times = down_times[sort_idx]
    travel_times = travel_times[sort_idx]

    for t in time_grid:
        t_ns = t.to_datetime64()
        t_minus_W = t_ns - window_ns

        # 1. Count upstream entries in (t-W, t] from upstream detections.
        start_idx_up = np.searchsorted(up_times_all, t_minus_W, side='right')
        end_idx_up = np.searchsorted(up_times_all, t_ns, side='right')
        N_up = int(end_idx_up - start_idx_up)

        # 2. Identify matched trajectories whose upstream entry is in (t-W, t].
        start_idx = np.searchsorted(up_times, t_minus_W, side='right')
        end_idx = np.searchsorted(up_times, t_ns, side='right')

        # Candidate set for this window
        window_indices = np.arange(start_idx, end_idx)

        if len(window_indices) == 0:
            N_match = 0
            R_comp = 0.0
            bar_y = mu_default
            s_y2 = s2_default
        else:
            # Check completions available by t plus the diagnostic look-ahead.
            # Note: down_times are not necessarily sorted, so we must check boolean
            candidates_down_times = down_times[window_indices]
            candidates_travel_times = travel_times[window_indices]

            # Completed set W_t(W)
            completion_cutoff = t_ns + np.timedelta64(
                int(completion_lookahead_seconds), 's'
            )
            is_completed = candidates_down_times <= completion_cutoff
            completed_travel_times = candidates_travel_times[is_completed]

            N_match = len(completed_travel_times)
            R_comp = N_match / max(N_up, 1.0)

            if N_match > 0:
                bar_y = np.mean(completed_travel_times)
                s_y2 = np.var(completed_travel_times, ddof=1) if N_match > 1 else 0.0 # Eq 13 divides by max(|W|-1, 1)
            else:
                bar_y = mu_default
                s_y2 = s2_default

        # Extract Time-of-Day and Day-of-Week
        # t is a pd.Timestamp
        curr_tod_dow = time_features(t)

        # Feature vector: [N_up, N_match, R_comp, bar_y, s_y2, ToD..., DoW...]
        # Basic features (5 dims) + Time features (9 dims: 2 ToD + 7 DoW) = 14 dims
        feat = np.concatenate([[N_up, N_match, R_comp, bar_y, s_y2], curr_tod_dow])
        features_list.append(feat)

    feature_df = pd.DataFrame(features_list, index=time_grid)
    feature_cols = ['N_up', 'N_match', 'R_comp', 'bar_y', 's_y2'] + \
                   [f'time_feat_{i}' for i in range(len(features_list[0])-5)]
    feature_df.columns = feature_cols

    return feature_df

def create_trajectory_indexed_dataset(
    merged_df,
    feature_df,
    delta_obs_min,
    sequence_length,
    return_metadata=False,
):
    """
    Align each trajectory i with its prediction index t(i) and history x_{1:t(i)}.
    t(i) = max {t_n : t_n <= tau_i^u}
    """
    X_list = []
    y_list = []
    valid_indices = []

    # Delta time delta object
    delta = pd.Timedelta(minutes=delta_obs_min)

    # Iterate over all matched trajectories
    # This loop can be optimized, but let's be explicit first.

    # Ensure features are sorted by time (index)
    feature_df = feature_df.sort_index()
    time_index = feature_df.index

    # For each vehicle, find t(i)
    # t(i) is effectively floor(tau_i^u / delta) * delta
    # We can compute this vectorized

    filtered_df, filtered_t_i = _aligned_sample_frame(
        merged_df,
        time_index[0],
        delta_obs_min,
        sequence_length,
    )

    if len(filtered_df) == 0:
        if return_metadata:
            return None, None, None
        return None, None

    # We need to map t_i to integer index in feature_df
    # Since feature_df is regular frequency, we can calculate index directly
    # or use searchsorted (safer)

    # Get numeric codes for timestamps ensures faster alignment
    feature_times_numeric = time_index.values.astype(np.int64)
    target_times_numeric = filtered_t_i.values.astype(np.int64)

    # Find positions
    # searchsorted usually returns where to insert, here we want exact match (or floor match which we already did)
    # Since we floored t_i, it should match one of the grid points exactly.
    # Note: searchsorted returns indices in feature_df
    feature_indices = np.searchsorted(feature_times_numeric, target_times_numeric)

    # Collect sequences
    # X shape: (N, seq_len, n_features)
    # We can iterate or try to construct a 3D tensor directly if memory allows.

    feature_values = feature_df.values

    # Vectorized construction of sequences
    # Construct an index matrix of shape (N, seq_len)
    # For a vehicle with feature_index k, we need indices [k-seq_len+1, ..., k]
    SEQ = sequence_length
    num_samples = len(feature_indices)

    # shape (N, SEQ)
    # Broadcast subtraction: feature_indices[:, None] - np.arange(SEQ-1, -1, -1)
    idx_matrix = feature_indices[:, None] - np.arange(SEQ-1, -1, -1)

    # 2026-02-17: Ensure no index is out of bounds (negative)
    # Although we filtered by min_time_required, double check
    if idx_matrix.min() < 0:
        warnings.warn("Found negative indices in sequence construction. Filtering invalid rows.")
        valid_rows = idx_matrix.min(axis=1) >= 0
        idx_matrix = idx_matrix[valid_rows]
        filtered_df = filtered_df.iloc[valid_rows]
        filtered_t_i = filtered_t_i.iloc[valid_rows]

    X = feature_values[idx_matrix] # (N, SEQ, Feat)
    y = filtered_df['travel_time'].values

    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()

    if not return_metadata:
        return X_tensor, y_tensor

    period_codes, _ = pd.factorize(pd.to_datetime(filtered_t_i), sort=False)
    metadata = pd.DataFrame(
        {
            "case_id": np.arange(len(filtered_df), dtype=np.int64),
            "update_period_id": period_codes.astype(np.int64),
            "prediction_time": pd.to_datetime(filtered_t_i).to_numpy(),
        }
    )
    return X_tensor, y_tensor, metadata


def _aligned_sample_frame(merged_df, grid_start, delta_obs_min, sequence_length):
    """Return the target rows that have a complete feature history."""
    delta = pd.Timedelta(minutes=delta_obs_min)
    tau_u = pd.to_datetime(merged_df['timestamp_up'])
    prediction_times = tau_u.dt.floor(f'{delta_obs_min}min')
    min_time_required = pd.Timestamp(grid_start) + (sequence_length - 1) * delta
    valid_mask = prediction_times >= min_time_required
    return merged_df.loc[valid_mask].copy(), prediction_times.loc[valid_mask].copy()


def complete_period_split_boundaries(metadata, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Choose case boundaries nearest the requested ratios without splitting periods."""
    if len(metadata) == 0:
        raise ValueError("Cannot split empty metadata.")
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.")
    required = {"case_id", "update_period_id", "prediction_time"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Metadata must contain {sorted(required)}.")

    ordered = metadata.reset_index(drop=True)
    times = pd.to_datetime(ordered["prediction_time"])
    if not times.is_monotonic_increasing:
        raise ValueError("Sample metadata must be sorted by prediction_time.")

    period_ids = ordered["update_period_id"].to_numpy()
    seen = set()
    previous = None
    for period_id in period_ids:
        if period_id != previous:
            if period_id in seen:
                raise ValueError("Each update period must form one contiguous block.")
            seen.add(period_id)
            previous = period_id

    counts = ordered.groupby("update_period_id", sort=False).size().to_numpy(dtype=int)
    cumulative = np.cumsum(counts)
    if len(cumulative) < 3:
        raise ValueError("At least three update periods are required.")

    n = len(ordered)
    train_target = n * float(train_ratio)
    val_target = n * float(train_ratio + val_ratio)
    train_candidates = cumulative[:-2]
    train_end = int(train_candidates[np.argmin(np.abs(train_candidates - train_target))])
    val_candidates = cumulative[(cumulative > train_end) & (cumulative < n)]
    if len(val_candidates) == 0:
        raise ValueError("No complete validation boundary is available.")
    val_end = int(val_candidates[np.argmin(np.abs(val_candidates - val_target))])

    split_periods = {
        "train": ordered.iloc[:train_end]["update_period_id"].nunique(),
        "val": ordered.iloc[train_end:val_end]["update_period_id"].nunique(),
        "test": ordered.iloc[val_end:]["update_period_id"].nunique(),
    }
    return {
        "train_end": train_end,
        "val_end": val_end,
        "n_cases": n,
        "case_counts": {
            "train": train_end,
            "val": val_end - train_end,
            "test": n - val_end,
        },
        "period_counts": split_periods,
    }


def split_time_series_data_by_period(
    X,
    y,
    metadata,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
):
    """Split aligned arrays at complete update-period boundaries."""
    if not (len(X) == len(y) == len(metadata)):
        raise ValueError("X, y, and metadata must have identical lengths.")
    boundaries = complete_period_split_boundaries(
        metadata,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    train_end = boundaries["train_end"]
    val_end = boundaries["val_end"]
    metadata = metadata.reset_index(drop=True)
    arrays = (
        X[:train_end],
        X[train_end:val_end],
        X[val_end:],
        y[:train_end],
        y[train_end:val_end],
        y[val_end:],
    )
    metadata_splits = {
        "train": metadata.iloc[:train_end].reset_index(drop=True),
        "val": metadata.iloc[train_end:val_end].reset_index(drop=True),
        "test": metadata.iloc[val_end:].reset_index(drop=True),
    }

    period_sets = {
        name: set(frame["update_period_id"].tolist())
        for name, frame in metadata_splits.items()
    }
    if period_sets["train"] & period_sets["val"]:
        raise ValueError("Train and validation periods overlap.")
    if period_sets["train"] & period_sets["test"]:
        raise ValueError("Train and test periods overlap.")
    if period_sets["val"] & period_sets["test"]:
        raise ValueError("Validation and test periods overlap.")
    return (*arrays, metadata_splits, boundaries)

def load_matched_data(config):
    """Load detections and construct the filtered one-to-one matched trajectories."""
    plate1 = pd.read_csv(_resolve_data_path(config['data']['plate1_path']))
    plate2 = pd.read_csv(_resolve_data_path(config['data']['plate2_path']))
    # Basic Cleaning
    plate1['time_stamp'] = pd.to_datetime(plate1['leave_time'], errors='coerce')
    plate2['time_stamp'] = pd.to_datetime(plate2['leave_time'], errors='coerce')
    plate1.dropna(subset=['time_stamp', 'vehicle_id'], inplace=True)
    plate2.dropna(subset=['time_stamp', 'vehicle_id'], inplace=True)
    plate2 = plate2[(plate2.leave_time >= '2019-02-19 00:00:00')]

    # Filter by Camera and Turn
    t_up = plate1[plate1.camera_id.isin(config['data']['camera_up']) & (plate1.turn_id == 2)].sort_values('time_stamp')
    t_down = plate2[plate2.camera_id.isin(config['data']['camera_down']) & (plate2.turn_id == 2)].sort_values('time_stamp')

    # Matching
    t_up_prepared = t_up[['time_stamp', 'vehicle_id']].rename(columns={'time_stamp': 'timestamp_up'}).sort_values('timestamp_up')
    t_down_prepared = t_down[['time_stamp', 'vehicle_id']].rename(columns={'time_stamp': 'timestamp_down'}).sort_values('timestamp_down')

    # Merge
    merged_df = pd.merge_asof(
        t_up_prepared, t_down_prepared,
        left_on='timestamp_up', right_on='timestamp_down',
        by='vehicle_id', direction='forward',
        tolerance=pd.Timedelta(minutes=30)
    ).dropna()

    merged_df['travel_time'] = (merged_df['timestamp_down'] - merged_df['timestamp_up']).dt.total_seconds()
    # Filter outliers (Physical implausibility)
    merged_df = (
        merged_df[(merged_df['travel_time'] > 0) & (merged_df['travel_time'] < 1000)]
        .sort_values('timestamp_up')
        .reset_index(drop=True)
    )
    return t_up_prepared, t_down_prepared, merged_df


def load_and_preprocess_data(config, return_sample_metadata=False):
    """Load raw data, compute system state features, and align sequences."""
    try:
        t_up_prepared, _, merged_df = load_matched_data(config)
    except FileNotFoundError as e:
        print(f"Error: {e}, please check file paths.")
        if return_sample_metadata:
            return None, None, None, None, None
        return None, None, None, None

    # Get config params
    delta_obs = config['data'].get('delta_obs', 5)
    window_width = config['data'].get('window_width', 30)

    split_cfg = config.get('data_split', {})
    train_ratio = float(split_cfg.get('train_ratio', 0.7))
    val_ratio = float(split_cfg.get('val_ratio', 0.15))
    test_ratio = float(split_cfg.get('test_ratio', 0.15))
    grid_start = t_up_prepared['timestamp_up'].min().floor(f'{delta_obs}min')
    aligned_df, aligned_times = _aligned_sample_frame(
        merged_df,
        grid_start,
        delta_obs,
        config['model']['sequence_length'],
    )
    period_codes, _ = pd.factorize(pd.to_datetime(aligned_times), sort=False)
    provisional_metadata = pd.DataFrame(
        {
            "case_id": np.arange(len(aligned_df), dtype=np.int64),
            "update_period_id": period_codes.astype(np.int64),
            "prediction_time": pd.to_datetime(aligned_times).to_numpy(),
        }
    )
    boundaries = complete_period_split_boundaries(
        provisional_metadata,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    train_tt = aligned_df.iloc[:boundaries["train_end"]]['travel_time'].values
    if len(train_tt) == 0:
        print("Not enough training samples after filtering.")
        return None, None, None, None
    mu_default = float(np.median(train_tt))
    s2_default = float(np.var(train_tt, ddof=1)) if len(train_tt) > 1 else 0.0

    # 1. Compute Rolling Features (The Information Set x_t)
    print("Computing rolling window features...")
    feature_df = compute_rolling_features(
        t_up_prepared,
        merged_df,
        delta_obs,
        window_width,
        mu_default=mu_default,
        s2_default=s2_default
    )

    # 2. Align Data (Trajectory-indexed samples)
    print("Aligning trajectories with feature history...")
    aligned = create_trajectory_indexed_dataset(
        merged_df,
        feature_df,
        delta_obs,
        config['model']['sequence_length'],
        return_metadata=return_sample_metadata,
    )
    if return_sample_metadata:
        X, y, sample_metadata = aligned
    else:
        X, y = aligned

    if X is None:
        print("Not enough data to create sequences.")
        if return_sample_metadata:
            return None, None, None, None, None
        return None, None, None, None

    result = (X.numpy(), y.numpy(), feature_df.columns.tolist(), merged_df)
    if return_sample_metadata:
        return (*result, sample_metadata)
    return result

def split_time_series_data(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split the already-aligned X, y arrays by strict time order.
    Note: X, y are already sorted by time because merged_df was sorted by timestamp_up.
    """
    n = len(y)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test

def fit_scaler_on_train_only(X_train, X_val, X_test):
    """
    Fit scaler on training data features ONLY.
    Input X is (N, Seq, Feat). We rearrange to (N*Seq, Feat) to fit, then transform back?
    Usually we scale features independently.
    Note: Some features like R_comp [0,1] or sin/cos [-1,1] don't necessarily need scaling or are already bounded.
    But N_up, bar_y, s_y2 need scaling.
    MinMax is fine.
    """
    # X shape: (N, L, D)
    N_train, L, D = X_train.shape

    # Reshape to 2D for scaling
    X_train_reshaped = X_train.reshape(-1, D)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train_reshaped)

    # Transform all
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, D)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, D)).reshape(X_test.shape)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def inverse_scale_y(scaled_y_column, scaler, features_list):
    """Auxiliary function to inverse scale travel_time"""
    # scaled_y_column shape: (N,) or (N,1)

    # We need to reconstruct the full feature vector to use scaler.inverse_transform
    # dimensions: scaler.n_features_in_

    n_features = scaler.n_features_in_
    N = len(scaled_y_column)

    dummy_array = np.zeros((N, n_features))

    # We need to know which column is 'travel_time'.
    # features_list is the list of column names.
    # In load_and_preprocess_data, we constructed X from feature_df (which has N_up, bar_y, etc.)
    # Wait, X does NOT contain 'travel_time' as a feature in the NEW implementation!
    # In strict alignment, current travel time is the LABEL y, not a feature x_t.
    # Lagged travel time is 'bar_y'.

    # BUT, the scaler was fitted on X (features).
    # If y (travel_time) is not in X, we cannot use this scaler to inverse transform y!

    # In the OLD implementation:
    # data_for_scaling = pd.concat([travel_time_df[['travel_time']], time_feature_df], axis=1)
    # So 'travel_time' was column 0.

    # In the NEW implementation:
    # X = feature_values (N_up, N_match, R_comp, bar_y, s_y2, Time...)
    # y = travel_times
    # Scaler is fitted on X_train.

    # Users want to inverse scale 'y' (the predictions).
    # But 'y' is NOT in 'X'.
    # 'bar_y' (lagged average) IS in 'X'.
    # If 'bar_y' is on the same scale as 'y' (seconds), we can use the scaler's 'bar_y' column to inverse scale 'y'?
    # Yes, 'bar_y' is average travel time. 'y' is individual travel time. They are same unit.

    # Let's find index of 'bar_y' in features_list.
    try:
        idx = features_list.index('bar_y')
    except ValueError:
        # Fallback if bar_y not found? Should be there.
        print("Warning: 'bar_y' not found in features. Cannot inverse scale y using X scaler.")
        return scaled_y_column # return raw if fail

    dummy_array[:, idx] = scaled_y_column.flatten()
    original_scale_array = scaler.inverse_transform(dummy_array)
    return original_scale_array[:, idx]
