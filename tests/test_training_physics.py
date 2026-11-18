import unittest
from types import SimpleNamespace

import torch

from dataset import collate_cfd_graphs_with_physics
from physics import compute_batch_drag, compute_design_drag


class TrainingPhysicsTests(unittest.TestCase):
    def test_design_drag_integrates_pressure_and_streamwise_shear(self):
        geom = torch.tensor(
            [[0, 0, 0, 1, 0, 0, 2], [0, 0, 0, -0.6, 0.8, 0, 5]], dtype=torch.float32
        )
        fields = torch.tensor([[10.0, 0.5], [4.0, -0.2]])

        drag = compute_design_drag(fields, geom)

        # (10*1 + 0.5)*2 + (4*-0.6 - 0.2)*5
        self.assertAlmostEqual(drag.item(), 8.0, places=5)

    def test_batch_drag_matches_individual_drag_and_keeps_gradients(self):
        geom = torch.tensor(
            [
                [0, 0, 0, 1, 0, 0, 2],
                [0, 0, 0, 0, 1, 0, 3],
                [0, 0, 0, -1, 0, 0, 4],
            ],
            dtype=torch.float32,
        )
        coeffs = torch.tensor(
            [[5, 2], [7, 3], [11, 4]],
            dtype=torch.float32,
            requires_grad=True,
        )
        batch_idx = torch.tensor([0, 0, 1])
        scaler = SimpleNamespace(mean=torch.zeros(1, 2), std=torch.ones(1, 2))
        q_dyn = torch.tensor([2.0, 10.0])

        batched = compute_batch_drag(coeffs, geom, batch_idx, 2, scaler, q_dyn)
        expected = torch.stack(
            [
                compute_design_drag(coeffs[:2] * 2.0, geom[:2]),
                compute_design_drag(coeffs[2:] * 10.0, geom[2:]),
            ]
        )

        torch.testing.assert_close(batched, expected)
        batched.sum().backward()
        gradients = coeffs.grad
        assert gradients is not None
        self.assertGreater(torch.count_nonzero(gradients).item(), 0)

    def test_physics_collator_offsets_graph_edges_and_preserves_targets(self):
        first = {
            "geom": torch.full((2, 7), 2.0),
            "geom_norm": torch.ones(2, 7),
            "params": torch.tensor([1.0, 2.0, 3.0]),
            "params_norm": torch.tensor([1.0, 2.0, 3.0]),
            "fields": torch.ones(2, 3),
            "fields_norm": torch.ones(2, 3),
            "edge_index": torch.tensor([[0, 1], [1, 0]]),
            "true_drag": 4.0,
            "q_dyn": 50.0,
            "face_weights": torch.ones(2, 2),
            "edge_attr": torch.zeros(2, 5),
        }
        second = {
            "geom": torch.full((2, 7), 3.0),
            "geom_norm": torch.ones(2, 7),
            "params": torch.tensor([4.0, 5.0, 6.0]),
            "params_norm": torch.tensor([4.0, 5.0, 6.0]),
            "fields": torch.ones(2, 3),
            "fields_norm": torch.ones(2, 3),
            "edge_index": torch.tensor([[0], [1]]),
            "true_drag": 7.0,
            "q_dyn": 80.0,
            "face_weights": torch.full((2, 2), 2.0),
            "edge_attr": torch.ones(1, 5),
        }

        batch = collate_cfd_graphs_with_physics([first, second])

        torch.testing.assert_close(
            batch["edge_index"], torch.tensor([[0, 1, 2], [1, 0, 3]])
        )
        torch.testing.assert_close(batch["batch_idx"], torch.tensor([0, 0, 1, 1]))
        torch.testing.assert_close(batch["true_drags"], torch.tensor([4.0, 7.0]))
        torch.testing.assert_close(batch["q_dyn"], torch.tensor([50.0, 80.0]))
        self.assertEqual(tuple(batch["face_weights"].shape), (4, 2))
        torch.testing.assert_close(
            batch["geom_raw"], torch.cat([first["geom"], second["geom"]])
        )


if __name__ == "__main__":
    unittest.main()
