from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--n-events", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    upstream_time = pd.date_range(
        "2019-02-19 00:00:00", periods=args.n_events, freq="20s"
    )
    travel_seconds = rng.integers(45, 151, size=args.n_events)
    vehicle_id = [f"SYNTHETIC_{index:07d}" for index in range(args.n_events)]

    upstream = pd.DataFrame(
        {
            "leave_time": upstream_time,
            "vehicle_id": vehicle_id,
            "camera_id": "UPSTREAM_CAMERA_ID",
            "turn_id": 2,
        }
    )
    downstream = pd.DataFrame(
        {
            "leave_time": upstream_time
            + pd.to_timedelta(travel_seconds, unit="s"),
            "vehicle_id": vehicle_id,
            "camera_id": "DOWNSTREAM_CAMERA_ID",
            "turn_id": 2,
        }
    )
    upstream.to_csv(output_dir / "upstream_events.csv", index=False)
    downstream.to_csv(output_dir / "downstream_events.csv", index=False)


if __name__ == "__main__":
    main()
