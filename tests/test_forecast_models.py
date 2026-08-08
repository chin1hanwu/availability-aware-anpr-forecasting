import unittest

import torch

from forecast_models import build_density_model


class ForecastModelShapeTest(unittest.TestCase):
    def test_all_density_families_return_valid_shapes(self):
        specs = [
            {
                "family": "snapshot",
                "hidden_dim": 16,
                "num_layers": 2,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
            },
            {
                "family": "recurrent",
                "hidden_dim": 16,
                "num_layers": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
            },
        ]
        X = torch.randn(5, 30, 14)
        for spec in specs:
            model = build_density_model(spec, input_dim=14, num_mixtures=3)
            pis, mus, sigmas = model(X)
            self.assertEqual(tuple(pis.shape), (5, 3))
            self.assertEqual(tuple(mus.shape), (5, 3))
            self.assertEqual(tuple(sigmas.shape), (5, 3))
            self.assertTrue(torch.allclose(pis.sum(dim=1), torch.ones(5), atol=1e-6))
            self.assertTrue((sigmas > 0).all())


if __name__ == "__main__":
    unittest.main()
