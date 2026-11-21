import tempfile
import unittest
from pathlib import Path

import torch

import compare_volume_architectures as experiment
from benchmarks.transolver.model import PhysicsAttention, Transolver
from benchmarks.transolver.train import create_transolver, parse_comparisons


def _architecture():
    return {
        "name": "transolver_test",
        "geom_dim": 5,
        "hidden_dim": 8,
        "num_layers": 2,
        "num_slices": 4,
        "attention_heads": 2,
        "mlp_ratio": 2,
        "activation": "default",
    }


def _item(name, cells=12):
    edge_index = torch.tensor(
        [list(range(cells - 1)) + list(range(1, cells)),
         list(range(1, cells)) + list(range(cells - 1))]
    )  # fmt: skip
    return {
        "design_name": name,
        "geom_norm": torch.randn(cells, 5),
        "geom": torch.rand(cells, 5) + 0.1,
        "params_norm": torch.randn(3),
        "edge_index": edge_index,
        "edge_attr": torch.randn(edge_index.shape[1], 4),
        "cell_areas": torch.rand(cells) + 0.1,
        "wall_distance": torch.rand(cells) * 0.1,
        "pressure": torch.randn(cells, 1),
        "pressure_norm": torch.randn(cells, 1),
    }


class TransolverModelTests(unittest.TestCase):
    def test_physics_attention_preserves_shape(self):
        layer = PhysicsAttention(8, num_slices=4, num_heads=2)
        self.assertEqual(layer(torch.randn(1, 10, 8)).shape, (1, 10, 8))

    def test_rejects_heads_that_do_not_divide_width(self):
        with self.assertRaises(ValueError):
            PhysicsAttention(10, num_slices=4, num_heads=3)

    def test_batched_graphs_are_independent(self):
        torch.manual_seed(0)
        model = Transolver(hidden_dim=8, num_layers=2, num_slices=4, num_heads=2)
        model.eval()
        first, second = torch.randn(6, 5), torch.randn(9, 5)
        params = torch.randn(15, 3)
        batch_idx = torch.tensor([0] * 6 + [1] * 9)
        batched = model(torch.cat([first, second]), params, batch_idx=batch_idx)
        alone = model(first, params[:6])
        self.assertTrue(torch.allclose(batched[:6], alone, atol=1e-6))
        self.assertEqual(batched.shape, (15, 1))

    def test_training_loop_accepts_transolver_factory(self):
        torch.manual_seed(0)
        data = [_item(f"Design_{index}") for index in range(3)]
        with tempfile.TemporaryDirectory() as directory:
            result = experiment.train_with_validation(
                data[:2],
                data[2:],
                torch.device("cpu"),
                _architecture(),
                max_epochs=2,
                validation_interval=1,
                patience=1,
                minimum_epochs=1,
                checkpoint_path=Path(directory) / "best.pt",
                history_path=Path(directory) / "history.json",
                model_factory=create_transolver,
            )
        self.assertIsInstance(result["model"], Transolver)
        self.assertGreaterEqual(result["best_epoch"], 1)

    def test_parse_comparisons_requires_label_and_existing_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            parsed = parse_comparisons([f"GNN={directory}", "Missing=nowhere_xyz"])
            self.assertEqual(parsed, {"GNN": Path(directory)})
            with self.assertRaises(ValueError):
                parse_comparisons([directory])


if __name__ == "__main__":
    unittest.main()
