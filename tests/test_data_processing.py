import unittest

import numpy as np
import pandas as pd

from data_processing import (
    complete_period_split_boundaries,
    split_time_series_data_by_period,
)


class CompletePeriodSplitTest(unittest.TestCase):
    def setUp(self):
        counts = [3, 4, 2, 5, 6]
        period_ids = np.repeat(np.arange(len(counts)), counts)
        self.metadata = pd.DataFrame(
            {
                "case_id": np.arange(len(period_ids)),
                "update_period_id": period_ids,
                "prediction_time": pd.Timestamp("2019-01-01")
                + pd.to_timedelta(period_ids * 5, unit="min"),
            }
        )

    def test_boundaries_do_not_split_periods(self):
        result = complete_period_split_boundaries(
            self.metadata,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
        )

        self.assertEqual(result["train_end"], 9)
        self.assertEqual(result["val_end"], 14)
        self.assertEqual(result["case_counts"], {"train": 9, "val": 5, "test": 6})
        self.assertEqual(result["period_counts"], {"train": 3, "val": 1, "test": 1})

    def test_split_arrays_and_periods_stay_aligned(self):
        X = np.arange(len(self.metadata) * 2).reshape(len(self.metadata), 1, 2)
        y = np.arange(len(self.metadata))
        result = split_time_series_data_by_period(
            X,
            y,
            self.metadata,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
        )
        X_train, X_val, X_test, y_train, y_val, y_test, metadata, _ = result

        self.assertEqual((len(X_train), len(X_val), len(X_test)), (9, 5, 6))
        self.assertEqual((len(y_train), len(y_val), len(y_test)), (9, 5, 6))
        self.assertTrue(
            set(metadata["train"]["update_period_id"]).isdisjoint(
                metadata["val"]["update_period_id"]
            )
        )
        self.assertTrue(
            set(metadata["val"]["update_period_id"]).isdisjoint(
                metadata["test"]["update_period_id"]
            )
        )


if __name__ == "__main__":
    unittest.main()
