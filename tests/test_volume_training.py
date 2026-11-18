import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import train_volume as training
from diagnose_volume import plot_loss_comparison


def make_graph(name="Design_1", speed=2.0):
    return {
        "design_name": name,
        "geom": torch.tensor(
            [[0, 0, 0, 1, 0.1], [1, 0, 0, 3, 0.2]], dtype=torch.float32
        ),
        "params": torch.tensor([1.0, 2.0, speed]),
        "pressure": torch.tensor([[10.0], [20.0]]),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "edge_attr": torch.tensor([[1.0, 0, 1, 1], [-1.0, 0, 1, -1]]),
    }


class VolumeTrainingTests(unittest.TestCase):
    def test_loss_comparison_saves_shared_scale_image_with_panel_metrics(self):
        polygons = [
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
            np.array([[1, 0], [2, 0], [2, 1], [1, 1]]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.png"
            figure = plot_loss_comparison(
                polygons,
                np.array([10.0, 20.0]),
                np.array([0.0, 0.0]),
                np.array([11.0, 19.0]),
                "Design_1",
                2,
                output,
                metrics=[
                    {"rmse_pa": 15.0, "near_wall_rmse_pa": 20.0},
                    {"rmse_pa": 1.0, "near_wall_rmse_pa": 2.0},
                ],
            )
            self.assertGreater(output.stat().st_size, 1000)
            panels = [axis for axis in figure.axes if axis.get_title()]
            self.assertEqual(len(panels), 3)
            self.assertIn("Near-body RMSE: 20.00 Pa", panels[1].get_title())
            self.assertIn("Near-body RMSE: 2.00 Pa", panels[2].get_title())
            self.assertIs(panels[0].collections[0].norm, panels[2].collections[0].norm)

    def test_near_body_loss_preserves_domain_and_near_body_influence(self):
        prediction = torch.tensor([[2.0], [0.0]], requires_grad=True)
        target = torch.zeros(2, 1)
        areas = torch.tensor([1.0, 99.0])
        distance = torch.tensor([0.01, 1.0])
        loss = training.near_body_weighted_mse(
            prediction,
            target,
            areas,
            distance,
            torch.zeros(2, dtype=torch.long),
            1,
        )
        self.assertAlmostEqual(loss.item(), 0.5 * 0.04 + 0.5 * 4.0, places=5)
        loss.backward()
        assert prediction.grad is not None
        self.assertGreater(prediction.grad[0].item(), 0)
        fallback = training.near_body_weighted_mse(
            prediction.detach(),
            target,
            areas,
            torch.ones(2),
            torch.zeros(2, dtype=torch.long),
            1,
        )
        self.assertAlmostEqual(fallback.item(), 0.04, places=5)

    def test_near_body_loss_balances_graphs_not_mesh_sizes(self):
        prediction = torch.tensor([[2.0], [0.0], [3.0], [0.0]])
        loss = training.near_body_weighted_mse(
            prediction,
            torch.zeros(4, 1),
            torch.tensor([1.0, 99.0, 10.0, 990.0]),
            torch.tensor([0.01, 1.0, 0.01, 1.0]),
            torch.tensor([0, 0, 1, 1]),
            2,
        )
        self.assertAlmostEqual(loss.item(), (2.02 + 4.545) / 2, places=5)

    def test_single_design_fit_has_finite_scalers_and_predictions(self):
        data = training.prepare_data([make_graph()], "geometry")
        scalers = training.fit_scalers(data)
        for scaler in scalers:
            self.assertTrue(torch.isfinite(scaler.mean).all())
            self.assertTrue(torch.isfinite(scaler.std).all())
        normalized = training.normalize_data(data, scalers)
        history = []
        model = training.train_model(
            normalized,
            torch.device("cpu"),
            2,
            loss_mode="near-body",
            history=history,
        )
        prediction = training.predict_pressure(
            model, normalized[0], scalers[2], torch.device("cpu")
        )
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertEqual(len(history), 2)

    def test_variants_keep_physical_pressure_and_configure_targets(self):
        raw = make_graph()
        baseline = training.prepare_data([raw], "baseline", 2.0, 2.0)[0]
        coefficient = training.prepare_data([raw], "coefficient", 2.0, 2.0)[0]
        geometry = training.prepare_data([raw], "geometry", 2.0, 2.0)[0]
        torch.testing.assert_close(coefficient["target"], (raw["pressure"] - 2) / 4)
        torch.testing.assert_close(baseline["target"], raw["pressure"])
        self.assertEqual(coefficient["geom"].shape[1], 4)
        self.assertEqual(geometry["geom"].shape[1], 5)
        self.assertNotIn("edge_attr", baseline)
        self.assertIn("edge_attr", geometry)
        self.assertEqual(raw["geom"].shape[1], 5)

    def test_prediction_inverts_coefficient_target(self):
        data = training.prepare_data(
            [make_graph(), make_graph("Design_2", 4)], "geometry", 2.0, 2.0
        )
        scalers = training.fit_scalers(data)
        normalized = training.normalize_data(data, scalers)
        model = training.create_model(torch.device("cpu"), geom_dim=5, edge_dim=4)
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        predicted = training.predict_pressure(
            model, normalized[0], scalers[2], torch.device("cpu")
        )
        expected = scalers[2].mean.expand(2, 1) * 4 + 2
        torch.testing.assert_close(predicted, expected)

    def test_training_is_repeatable_and_checkpoint_round_trips(self):
        data = training.prepare_data(
            [make_graph(), make_graph("Design_2", 4)], "geometry", 2.0, 2.0
        )
        scalers = training.fit_scalers(data)
        normalized = training.normalize_data(data, scalers)
        first = training.train_model(normalized, torch.device("cpu"), 1, seed=7)
        second = training.train_model(normalized, torch.device("cpu"), 1, seed=7)
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            training.save_checkpoint(
                first,
                scalers,
                path,
                {"variant": "geometry", "density": 2.0, "reference_pressure": 2.0},
            )
            restored, restored_scalers, metadata = training.load_checkpoint(
                path, torch.device("cpu")
            )
        torch.testing.assert_close(
            training.predict_pressure(
                first, normalized[0], scalers[2], torch.device("cpu")
            ),
            training.predict_pressure(
                restored, normalized[0], restored_scalers[2], torch.device("cpu")
            ),
        )
        self.assertEqual(metadata["variant"], "geometry")

    def test_metrics_use_physical_units_and_conventional_median(self):
        item = make_graph()
        row = training.measure_design(item["pressure"] + 2, item)
        self.assertAlmostEqual(row["rmse_pa"], 2.0)
        self.assertAlmostEqual(row["mae_pa"], 2.0)
        summary = training.summarize_metrics([row, {**row, "rmse_pa": 4.0}])
        self.assertEqual(summary["median_design_rmse_pa"], 3.0)

    def test_rejects_invalid_scaling_and_small_folds(self):
        with self.assertRaises(ValueError):
            training.prepare_data([make_graph()], "coefficient", 0.0, 0.0)
        with self.assertRaises(ValueError):
            training.cross_validate([make_graph()], torch.device("cpu"), 1)

    def test_cv_saves_only_held_out_predictions_and_training_membership(self):
        data = training.prepare_data(
            [make_graph(f"Design_{index}", float(index + 1)) for index in range(1, 6)],
            "geometry",
            2.0,
            0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "train_volume.train_model",
                return_value=training.create_model(torch.device("cpu"), 5, 4),
            ) as train:
                rows = training.cross_validate(
                    data, torch.device("cpu"), 1, directory, {"seed": 7}
                )
            report = json.loads((Path(directory) / "metrics.json").read_text())
            self.assertEqual(len(rows), 5)
            for fold, call in zip(report["folds"], train.call_args_list):
                actual_training = [item["design_name"] for item in call.args[0]]
                self.assertEqual(actual_training, fold["training_designs"])
                self.assertFalse(set(actual_training) & set(fold["held_out_designs"]))
                self.assertEqual(call.kwargs["seed"], fold["seed"])
            files = list((Path(directory) / "predictions").glob("*.npz"))
            self.assertEqual(len(files), 5)
            with np.load(files[0]) as prediction:
                self.assertEqual(prediction["predicted_pressure_pa"].shape, (2, 1))
                self.assertEqual(prediction["actual_pressure_pa"].shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
