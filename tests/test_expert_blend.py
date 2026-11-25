import unittest

import numpy as np

from benchmarks.airfrans.blend_experts import evaluate_rows, gate_rows, pressure_metrics


class ExpertBlendTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "name": "airFoil2D_SST_10_0_2_4_12",
                "angle": 0.0,
                "surface": np.array([True, False, False]),
                "actual": np.array([1.0, 2.0, 4.0]),
                "baseline": np.array([2.0, 4.0, 8.0]),
                "relative": np.array([3.0, 6.0, 12.0]),
            },
            {
                "name": "airFoil2D_SST_10_10_2_4_12",
                "angle": 10.0,
                "surface": np.array([True, False, False]),
                "actual": np.array([1.0, 2.0, 4.0]),
                "baseline": np.array([3.0, 6.0, 12.0]),
                "relative": np.array([2.0, 4.0, 8.0]),
            },
        ]

    def test_pressure_metrics_separate_surface_and_volume(self):
        metrics = pressure_metrics(
            np.array([1.0, 2.0, 4.0]),
            np.array([2.0, 4.0, 8.0]),
            np.array([True, False, False]),
        )
        self.assertEqual(metrics["surface_pressure_mse"], 1.0)
        self.assertEqual(metrics["volume_pressure_mse"], 10.0)

    def test_alpha_blend_is_linear_in_pressure_and_metrics(self):
        def result(alpha):
            return evaluate_rows(
                self.rows,
                lambda row, alpha=alpha: (
                    alpha * row["baseline"] + (1 - alpha) * row["relative"]
                ),
            )

        baseline = result(1.0)
        relative = result(0.0)
        middle = result(0.5)
        self.assertEqual(baseline["mean_surface_pressure_mse"], 2.5)
        self.assertEqual(relative["mean_surface_pressure_mse"], 2.5)
        self.assertEqual(middle["mean_surface_pressure_mse"], 2.25)

    def test_gate_switches_experts_at_angle_threshold(self):
        result = gate_rows(self.rows, 5.0, "baseline", "relative")
        self.assertEqual(result["mean_surface_pressure_mse"], 1.0)
        reverse = gate_rows(self.rows, 5.0, "relative", "baseline")
        self.assertEqual(reverse["mean_surface_pressure_mse"], 4.0)


if __name__ == "__main__":
    unittest.main()
