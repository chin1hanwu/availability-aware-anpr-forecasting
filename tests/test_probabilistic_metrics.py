import unittest

import numpy as np
import pandas as pd

from probabilistic_metrics import (
    gaussian_crps,
    gaussian_nll,
    mixture_crps,
    mixture_nll,
    negative_mass,
    segmented_acf,
    truncated_mixture_crps,
    truncated_mixture_pit,
)


class ProbabilisticMetricsTest(unittest.TestCase):
    def test_single_component_matches_gaussian_scores(self):
        y = np.array([0.0, 1.0, 2.0])
        mu = np.array([0.5, 1.5, 1.0])
        sigma = np.array([1.0, 2.0, 0.5])
        pis = np.ones((3, 1))
        mus = mu[:, None]
        sigmas = sigma[:, None]

        np.testing.assert_allclose(
            mixture_crps(pis, mus, sigmas, y),
            gaussian_crps(mu, sigma, y),
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            mixture_nll(pis, mus, sigmas, y),
            gaussian_nll(mu, sigma, y),
            rtol=1e-10,
        )

    def test_zero_truncation_rescales_the_cdf(self):
        pis = np.ones((2, 1))
        mus = np.zeros((2, 1))
        sigmas = np.ones((2, 1))
        np.testing.assert_allclose(negative_mass(pis, mus, sigmas), 0.5)
        pit = truncated_mixture_pit(pis, mus, sigmas, np.array([0.0, 1.0]))
        self.assertAlmostEqual(float(pit[0]), 1e-9)
        self.assertGreater(float(pit[1]), 0.0)
        self.assertLess(float(pit[1]), 1.0)

    def test_truncated_crps_grid_converges(self):
        pis = np.ones((3, 1))
        mus = np.array([[20.0], [50.0], [100.0]])
        sigmas = np.array([[10.0], [20.0], [30.0]])
        y = np.array([15.0, 55.0, 130.0])
        coarse = truncated_mixture_crps(pis, mus, sigmas, y, points=2001)
        fine = truncated_mixture_crps(pis, mus, sigmas, y, points=4001)
        self.assertTrue(np.isfinite(coarse).all())
        self.assertLess(float(np.max(np.abs(coarse - fine))), 0.01)

    def test_segmented_acf_excludes_cross_gap_pairs(self):
        values = np.array([1.0, 2.0, 100.0, 101.0])
        times = pd.Series(
            pd.to_datetime(
                [
                    "2019-01-01 00:00",
                    "2019-01-01 00:05",
                    "2019-01-02 00:00",
                    "2019-01-02 00:05",
                ]
            )
        )
        result = segmented_acf(values, times, max_lag=1)
        centered = values - values.mean()
        expected = (
            centered[0] * centered[1] + centered[2] * centered[3]
        ) / np.dot(centered, centered)
        self.assertAlmostEqual(float(result[1]), float(expected))


if __name__ == "__main__":
    unittest.main()
