import unittest

import numpy as np
import pandas as pd

from frozen_evaluation import (
    apply_frozen_rule,
    quantiles_to_gaussian,
    service_rule_metrics,
)


class FrozenEvaluationTest(unittest.TestCase):
    def test_quantile_mapping_reports_crossing_and_positive_scale(self):
        predictions = np.array([[10.0, 20.0, 30.0], [30.0, 20.0, 10.0]])
        mu, sigma, crossing = quantiles_to_gaussian(predictions)
        self.assertEqual(crossing.tolist(), [False, True])
        self.assertTrue((sigma > 0).all())
        np.testing.assert_allclose(mu, [20.0, 20.0])

    def test_service_publication_is_period_weighted(self):
        metadata = pd.DataFrame(
            {
                "case_id": [0, 1, 2],
                "update_period_id": [10, 10, 11],
                "prediction_time": pd.to_datetime(
                    ["2019-01-01 00:00", "2019-01-01 00:00", "2019-01-01 00:05"]
                ),
            }
        )
        test = {"metadata": metadata, "y_raw": np.array([5.0, 6.0, 7.0])}
        modes = pd.Series([True, False], index=[10, 11])
        metrics, broadcast, _, _ = service_rule_metrics(
            test,
            np.array([4.0, 4.0, 4.0]),
            np.array([8.0, 8.0, 8.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([10.0, 10.0, 10.0]),
            modes,
        )
        self.assertEqual(metrics["publication_rate_period"], 0.5)
        self.assertAlmostEqual(metrics["trajectory_exposure_share"], 2.0 / 3.0)
        self.assertEqual(broadcast.tolist(), [True, True, False])

    def test_joint_rule_uses_both_thresholds(self):
        periods = pd.DataFrame(
            {"R_comp": [0.8, 0.4], "rho_t": [0.2, 0.2]}, index=[1, 2]
        )
        rule = {"mode": "joint_gate", "R_low": 0.5, "rho_high": 0.3}
        self.assertEqual(apply_frozen_rule(periods, rule).tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
