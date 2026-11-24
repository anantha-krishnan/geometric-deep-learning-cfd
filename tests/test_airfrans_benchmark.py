import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from benchmarks.airfrans.data import (
    build_features,
    build_mesh_edge_index,
    fit_normalization,
    induced_subgraph,
    normalize_case,
    vertex_dual_areas,
)
from benchmarks.airfrans.evaluate import summarize
from benchmarks.airfrans.evaluation import predict_full, pressure_scores
from benchmarks.airfrans.slice_dataset import create_reduced_dataset
from benchmarks.airfrans.splits import build_reduced_splits, parse_case
from benchmarks.airfrans.train import sample_case
from benchmarks.airfrans.training import (
    BenchmarkModel,
    physics_loss,
    positional_features,
)


def make_case(index):
    speed = 31 + index * 0.1
    angle = -5 + (index % 20)
    if index % 2:
        return f"airFoil2D_SST_{speed}_{angle}_2.0_4.0_1.0_12.0"
    return f"airFoil2D_SST_{speed}_{angle}_2.0_4.0_12.0"


class AirfransSplitTests(unittest.TestCase):
    def test_parse_airfoil_families(self):
        self.assertEqual(parse_case(make_case(0))["family"], 4)
        self.assertEqual(parse_case(make_case(1))["family"], 5)
        self.assertEqual(parse_case(make_case(1))["speed"], 31.1)

    def test_splits_are_deterministic_and_keep_original_boundaries(self):
        cases = [make_case(index) for index in range(120)]
        manifest = {"full_train": cases[:100], "full_test": cases[100:]}
        first = build_reduced_splits(manifest, 40, 6, 8, seed=2026)
        self.assertEqual(first, build_reduced_splits(manifest, 40, 6, 8, seed=2026))
        self.assertEqual(
            [len(first[key]) for key in ("train", "validation", "test")], [40, 6, 8]
        )
        self.assertTrue(set(first["train"]) <= set(cases[:90]))
        self.assertTrue(set(first["validation"]) <= set(cases[90:100]))
        self.assertTrue(set(first["test"]) <= set(cases[100:]))
        self.assertFalse(set(first["train"]) & set(first["test"]))

    def test_rejects_overlapping_or_oversized_splits(self):
        cases = [make_case(index) for index in range(30)]
        with self.assertRaises(ValueError):
            build_reduced_splits(
                {"full_train": cases[:20], "full_test": cases[19:]}, 4, 1, 4
            )
        with self.assertRaises(ValueError):
            build_reduced_splits(
                {"full_train": cases[:20], "full_test": cases[20:]}, 19, 1, 4
            )


class AirfransSubsetTests(unittest.TestCase):
    def test_creates_self_contained_dataset_with_exact_split_membership(self):
        cases = [make_case(index) for index in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for name in cases:
                folder = source / name
                folder.mkdir()
                (folder / f"{name}_internal.vtu").write_bytes(b"internal")
                (folder / f"{name}_aerofoil.vtp").write_bytes(b"surface")
                (folder / f"{name}_freestream.vtp").write_bytes(b"unused")
            (source / "manifest.json").write_text(
                json.dumps({"full_train": cases[:8], "full_test": cases[8:]})
            )
            splits = {
                "seed": 7,
                "source_manifest": "local path removed",
                "splits": {
                    "train": cases[:4],
                    "validation": cases[4:8],
                    "test": cases[8:11],
                },
            }
            split_file = root / "splits.json"
            split_file.write_text(json.dumps(splits))
            output = root / "reduced"
            create_reduced_dataset(source, split_file, output)
            manifest = json.loads((output / "manifest.json").read_text())
            packaged_splits = json.loads((output / "splits.json").read_text())
            self.assertEqual(len(manifest["full_train"]), 8)
            self.assertEqual(len(manifest["full_test"]), 3)
            self.assertEqual(packaged_splits["splits"], splits["splits"])
            self.assertNotIn(
                "freestream", " ".join(str(path) for path in output.rglob("*"))
            )
            self.assertEqual(len(list(output.rglob("*_internal.vtu"))), 11)
            with self.assertRaises(FileExistsError):
                create_reduced_dataset(source, split_file, output)


class AirfransDataTests(unittest.TestCase):
    def test_mesh_edges_connect_cell_perimeters_in_both_directions(self):
        edge_index = build_mesh_edge_index(np.array([3, 0, 1, 2, 4, 2, 3, 4, 5]))
        actual = set(map(tuple, edge_index.T.tolist()))
        expected = {
            (0, 1),
            (1, 0),
            (1, 2),
            (2, 1),
            (2, 0),
            (0, 2),
            (2, 3),
            (3, 2),
            (3, 4),
            (4, 3),
            (4, 5),
            (5, 4),
            (5, 2),
            (2, 5),
        }
        self.assertEqual(actual, expected)

    def test_vertex_areas_share_each_cell_equally_and_sum_to_mesh_area(self):
        points = np.array([[0, 0], [2, 0], [2, 1], [0, 1], [3, 0]], dtype=float)
        areas = vertex_dual_areas(points, np.array([4, 0, 1, 2, 3, 3, 1, 4, 2]))
        third = 0.5 / 3
        np.testing.assert_allclose(
            areas, [0.5, 0.5 + third, 0.5 + third, 0.5, third], rtol=1e-6
        )
        self.assertAlmostEqual(float(areas.sum()), 2.5, places=5)

    def test_induced_subgraph_remaps_only_edges_between_sampled_nodes(self):
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
        sampled = induced_subgraph(edge_index, np.array([3, 1, 2]), num_nodes=4)
        actual = set(map(tuple, sampled.T.tolist()))
        self.assertEqual(actual, {(1, 2), (2, 1), (2, 0), (0, 2)})

    def test_features_match_surface_coordinates_and_reference_signs(self):
        points = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=float)
        velocity = np.array([[0, 0], [0, 0], [2, 1]], dtype=float)
        data = build_features(
            "airFoil2D_SST_10_0_2_4_12",
            points,
            velocity,
            np.array([3, 4, 5]),
            np.array([0, 0, 0.2]),
            np.array([0, 0, -1]),
            points[[1, 0]],
            np.array([[0, 1], [1, 0]], dtype=float),
        )
        np.testing.assert_array_equal(data["surface"], [True, True, False])
        np.testing.assert_allclose(data["x"][:, 2:4], [[10, 0]] * 3)
        np.testing.assert_allclose(data["x"][:, 5:], [[-1, 0], [0, -1], [0, 0]])
        np.testing.assert_allclose(data["x"][:, 4], [0, 0, 1])
        self.assertEqual(data["y"].shape, (3, 4))

    def test_normalization_uses_node_weighted_training_statistics(self):
        cases = [
            {"x": np.array([[1.0], [3.0]]), "y": np.array([[2.0], [4.0]])},
            {"x": np.array([[5.0]]), "y": np.array([[6.0]])},
        ]
        statistics = fit_normalization(cases)
        np.testing.assert_allclose(statistics["x_mean"], [3.0])
        np.testing.assert_allclose(statistics["y_mean"], [4.0])
        normalized = normalize_case(cases[0], statistics)
        self.assertLess(float(normalized["x"][0, 0]), 0)
        self.assertEqual(float(cases[0]["x"][0, 0]), 1.0)


class AirfransTrainingTests(unittest.TestCase):
    def test_sample_case_returns_induced_mesh_edges_for_gnn(self):
        edge_index = torch.tensor(
            [[0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], [1, 0, 2, 0, 3, 0, 2, 1, 3, 1, 3, 2]]
        )
        case = {
            "name": "case",
            "x": np.zeros((4, 7), dtype=np.float32),
            "pos": np.zeros((4, 2), dtype=np.float32),
            "y": np.zeros((4, 4), dtype=np.float32),
            "surface": np.array([True, True, False, False]),
            "edge_index": edge_index,
            "areas": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        }
        sample = sample_case(
            case,
            2,
            np.random.default_rng(1),
            torch.device("cpu"),
            include_edges=True,
            include_areas=True,
        )
        self.assertEqual(len(sample), 6)
        self.assertEqual(tuple(sample[4].shape), (2, 2))
        self.assertLess(int(sample[4].max()), 2)
        self.assertEqual(tuple(sample[5].shape), (2,))

    def test_area_gnn_kinds_configure_area_options(self):
        for kind, expected in (
            ("gnn", (0, False, False)),
            ("gnn_area_read", (2, False, True)),
            ("gnn_area", (2, True, True)),
        ):
            with self.subTest(kind=kind):
                model = BenchmarkModel(kind)
                self.assertEqual(
                    (
                        model.model.area_heads,
                        model.model.area_aggregation,
                        model.uses_areas,
                    ),
                    expected,
                )
        model = BenchmarkModel("gnn_area")
        edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        predictions = model(torch.randn(3, 7), torch.randn(3, 2), edges, torch.ones(3))
        self.assertEqual(tuple(predictions.shape), (3, 4))
        with self.assertRaises(ValueError):
            model(torch.randn(3, 7), torch.randn(3, 2), edges)

    def test_latent_attention_gnn_maps_airfrans_features_to_four_fields(self):
        model = BenchmarkModel("gnn")
        inputs = torch.randn(6, 7)
        positions = torch.randn(6, 2)
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]]
        )
        predictions = model(inputs, positions, edge_index)
        self.assertEqual(tuple(predictions.shape), (6, 4))
        self.assertEqual(model.model.attention_tokens, 32)
        self.assertTrue(
            any(key.startswith("layers.2.") for key in model.model.state_dict())
        )
        loss = physics_loss(
            predictions,
            torch.zeros_like(predictions),
            torch.tensor([True, False, False, False, False, False]),
        )
        loss.backward()
        self.assertIsNotNone(model.model.encoder[0].weight.grad)

    def test_reference_position_features(self):
        distances = positional_features(torch.tensor([[-2.0, -1.5], [4.0, 1.5]]))
        self.assertEqual(tuple(distances.shape), (2, 64))
        self.assertEqual(float(distances[0, 0]), 0.0)
        self.assertEqual(float(distances[1, -1]), 0.0)

    def test_loss_weights_volume_and_surface_equally(self):
        predictions = torch.tensor([[2.0] * 4, [1.0] * 4, [1.0] * 4])
        targets = torch.zeros_like(predictions)
        surface = torch.tensor([True, False, False])
        self.assertEqual(float(physics_loss(predictions, targets, surface)), 5.0)


class AirfransEvaluationTests(unittest.TestCase):
    def test_pressure_scores_separate_surface_and_volume(self):
        targets = np.zeros((3, 4), dtype=np.float32)
        predictions = targets.copy()
        predictions[:, 2] = [2, 1, 3]
        scores = pressure_scores(predictions, targets, np.array([True, False, False]))
        self.assertEqual(scores["surface_pressure_mse"], 4.0)
        self.assertEqual(scores["volume_pressure_mse"], 5.0)

    def test_inference_covers_every_node_and_enforces_wall_conditions(self):
        class DummyModel(torch.nn.Module):
            def forward(self, inputs, positions):
                return torch.ones((len(inputs), 4), device=inputs.device)

        case = {
            "x": np.zeros((10, 7)),
            "pos": np.zeros((10, 2)),
            "surface": np.array([True] + [False] * 9),
        }
        statistics = {"y_mean": np.zeros(4), "y_std": np.ones(4)}
        predictions, rounds = predict_full(DummyModel(), case, statistics, 4, 42, "cpu")
        np.testing.assert_array_equal(predictions[0], [0, 0, 1, 0])
        np.testing.assert_array_equal(predictions[1:], np.ones((9, 4)))
        self.assertGreaterEqual(rounds, 3)

    def test_gnn_inference_passes_induced_edges_to_the_model(self):
        class DummyGraphModel(torch.nn.Module):
            kind = "gnn"
            uses_graph = True
            uses_areas = True

            def forward(self, inputs, positions, edge_index, areas):
                assert edge_index.shape[0] == 2
                assert edge_index.numel() > 0
                assert areas.shape == (len(inputs),)
                return torch.ones((len(inputs), 4), device=inputs.device)

        case = {
            "x": np.zeros((6, 7)),
            "pos": np.zeros((6, 2)),
            "surface": np.array([True] + [False] * 5),
            "edge_index": torch.tensor(
                [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]]
            ),
            "areas": np.ones(6, dtype=np.float32),
        }
        statistics = {"y_mean": np.zeros(4), "y_std": np.ones(4)}
        predictions, rounds = predict_full(
            DummyGraphModel(), case, statistics, 4, 42, "cpu"
        )
        self.assertEqual(predictions.shape, (6, 4))
        self.assertGreaterEqual(rounds, 2)

    def test_summary_serializes_undefined_rank_correlation_as_null(self):
        summary = summarize(
            {
                "reference": [
                    {
                        "volume_pressure_mse": 1.0,
                        "surface_pressure_mse": 2.0,
                        "lift_relative_error": 0.1,
                        "true_lift": 1.0,
                        "predicted_lift": 0.9,
                    }
                ]
            }
        )
        self.assertIsNone(summary["reference"]["lift_spearman"])


if __name__ == "__main__":
    unittest.main()
