import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from model import Milestone1_GNN
from train import fit_scalers
from validate_side_by_side import (
    load_trained_model,
    plot_comparison,
    predict_fields,
    prepare_comparison,
)


def make_design(name, offset=0.0):
    fields = torch.tensor([[10.0, 2.0], [20.0, 4.0]]) + offset
    return {
        "design_name": name,
        "geom": torch.tensor(
            [[0, 0, 0, 1, 0, 0, 1], [1, 1, 0, -1, 0, 0, 1]],
            dtype=torch.float32,
        ),
        "params": torch.tensor([1.0, 2.0, 3.0]) + offset,
        "fields": fields,
        "coeffs": fields / 5.0,
        "q_dyn": 5.0,
        "true_drag": 10.0,
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "edge_attr": torch.zeros(2, 5),
    }


class ValidationTests(unittest.TestCase):
    def test_loads_two_field_checkpoint_with_its_actual_architecture(self):
        for edge_dim, use_global in [(0, False), (0, True), (5, True)]:
            with self.subTest(edge_dim=edge_dim, use_global=use_global):
                original = Milestone1_GNN(
                    hidden_dim=8,
                    num_layers=2,
                    edge_dim=edge_dim,
                    use_global=use_global,
                )
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory) / "model.pt"
                    torch.save(original.state_dict(), checkpoint)
                    restored = load_trained_model(checkpoint, torch.device("cpu"))
                for key, value in original.state_dict().items():
                    torch.testing.assert_close(restored.state_dict()[key], value)

    def test_prediction_restores_coefficient_scaling_and_dynamic_pressure(self):
        item = make_design("Design_1")
        scalers = fit_scalers([item, make_design("Design_2")])
        model = Milestone1_GNN(hidden_dim=8, num_layers=1)
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)

        prediction = predict_fields(model, item, scalers, torch.device("cpu"))

        expected = item["fields"].mean(dim=0, keepdim=True).expand(2, -1)
        torch.testing.assert_close(prediction, expected)

    def test_holdout_excludes_selected_design_from_training_and_scalers(self):
        data = [
            make_design("Design_1"),
            make_design("Design_2", 100.0),
            make_design("Design_3", 1.0),
        ]
        model = Milestone1_GNN(hidden_dim=8, num_layers=1)
        with (
            patch("validate_side_by_side.train_model", return_value=model) as train,
            patch("validate_side_by_side.fit_scalers", wraps=fit_scalers) as fit,
        ):
            item, prediction, label = prepare_comparison(
                data, "Design_2", Path("unused.pt"), torch.device("cpu"), True
            )

        self.assertIs(item, data[1])
        self.assertEqual(prediction.shape, (2, 2))
        self.assertIn("Held-out", label)
        self.assertEqual(
            [entry["design_name"] for entry in fit.call_args.args[0]],
            ["Design_1", "Design_3"],
        )
        self.assertEqual(
            [entry["design_name"] for entry in train.call_args.args[0]],
            ["Design_1", "Design_3"],
        )

    def test_holdout_rejects_too_few_training_designs(self):
        with self.assertRaisesRegex(ValueError, "at least three designs"):
            prepare_comparison(
                [make_design("Design_1"), make_design("Design_2")],
                "Design_2",
                Path("unused.pt"),
                torch.device("cpu"),
                True,
            )

    def test_plot_has_two_rows_shared_field_limits_and_saves_png(self):
        item = make_design("Design_1")
        figure = plot_comparison(
            item, item["fields"] + 2.0, "Training-data reconstruction"
        )
        try:
            panels = [axis for axis in figure.axes if axis.get_title()]
            self.assertEqual(len(panels), 6)
            for start in (0, 3):
                self.assertEqual(
                    panels[start].collections[0].get_clim(),
                    panels[start + 1].collections[0].get_clim(),
                )
                self.assertEqual(panels[start + 2].collections[0].get_clim()[0], 0)
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "comparison.png"
                figure.savefig(output, dpi=50)
                self.assertGreater(output.stat().st_size, 1000)
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
