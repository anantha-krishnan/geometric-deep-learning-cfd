import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyvista as pv
import torch

from benchmarks.shapenet_car.data import build_case, load_case
from benchmarks.shapenet_car.splits import (
    build_reduced_splits,
    discover_folds,
    split_fold,
)
from benchmarks.shapenet_car.training import BenchmarkModel, field_loss, field_scores


class ShapeNetCarDataTests(unittest.TestCase):
    def test_build_case_merges_surface_points_and_keeps_field_masks(self):
        volume_points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        surface_points = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        velocity = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        pressure = np.array([10, 20], dtype=float)
        normals = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        case = build_case(volume_points, velocity, surface_points, pressure, normals)

        np.testing.assert_array_equal(case["surface"], [False, False, True, True])
        np.testing.assert_allclose(
            case["pos"], volume_points[[0, 2]].tolist() + surface_points.tolist()
        )
        np.testing.assert_allclose(
            case["y"][:, :3], [[1, 2, 3], [7, 8, 9], [4, 5, 6], [0, 0, 0]]
        )
        np.testing.assert_allclose(case["y"][:, 3], [0, 0, 10, 20])
        np.testing.assert_allclose(case["x"][-2:, 4:], normals)
        np.testing.assert_allclose(case["x"][-2:, 3], 0)

    def test_load_case_reads_vtk_fields_and_graph(self):
        points = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "param0" / "case"
            folder.mkdir(parents=True)
            volume = pv.UnstructuredGrid(
                np.array([8, *range(8)]), np.array([12]), points
            )
            volume.point_data["velocity"] = np.tile([1.0, 2.0, 3.0], (8, 1))
            volume.set_active_vectors("velocity")
            volume.save(folder / "hexvelo_smpl.vtk", binary=False)
            surface = pv.UnstructuredGrid(
                np.array([4, 0, 1, 2, 3]), np.array([9]), points[:4]
            )
            surface.point_data["pressure"] = np.array([4.0, 5.0, 6.0, 7.0])
            surface.set_active_scalars("pressure")
            surface.save(folder / "quadpress_smpl.vtk", binary=False)
            (folder / "cd.txt").write_text("0.42")
            case = load_case(Path(temporary), "param0/case")
            with tarfile.open(Path(temporary) / "param0.tar.gz", "w:gz") as archive:
                archive.add(folder, arcname="param0/case")
            archived = load_case_from_archive(Path(temporary))

        self.assertEqual(case["name"], "param0/case")
        self.assertEqual(case["pos"].shape, (8, 3))
        self.assertEqual(case["x"].shape, (8, 7))
        self.assertEqual(case["surface"].sum(), 4)
        np.testing.assert_allclose(case["y"][case["surface"], 3], [4, 5, 6, 7])
        self.assertAlmostEqual(case["true_cd"], 0.42)
        self.assertEqual(case["edge_index"].shape, (2, 64))
        np.testing.assert_array_equal(archived["x"], case["x"])
        np.testing.assert_array_equal(archived["y"], case["y"])
        torch.testing.assert_close(archived["edge_index"], case["edge_index"])


def load_case_from_archive(root):
    with tempfile.TemporaryDirectory() as temporary:
        archive_root = Path(temporary)
        (archive_root / "param0.tar.gz").write_bytes(
            (root / "param0.tar.gz").read_bytes()
        )
        return load_case(archive_root, "param0/case")


class ShapeNetCarSplitTests(unittest.TestCase):
    def test_materialize_keeps_validation_and_held_out_test_separate(self):
        from benchmarks.shapenet_car.slice_dataset import create_reduced_dataset

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "raw"
            for fold, case in ((0, "test"), (1, "train"), (2, "validation")):
                directory = root / f"param{fold}" / case
                directory.mkdir(parents=True)
                for filename in ("hexvelo_smpl.vtk", "quadpress_smpl.vtk", "cd.txt"):
                    (directory / filename).write_text(f"{case} {filename}")
            splits = {
                "train": ["param1/train"],
                "validation": ["param2/validation"],
                "test": ["param0/test"],
            }
            destination = Path(temporary) / "reduced"
            create_reduced_dataset(root, splits, destination)
            self.assertTrue(
                (destination / "training_data/param1/train/cd.txt").is_file()
            )
            self.assertTrue(
                (destination / "training_data/param0/validation/cd.txt").is_file()
            )
            self.assertTrue((destination / "test_data/param0/test/cd.txt").is_file())
            self.assertFalse((destination / "training_data/param0/test").exists())
            for fold in range(9):
                self.assertTrue(
                    (destination / "training_data" / f"param{fold}").is_dir()
                )

    def test_reduced_split_is_reproducible_disjoint_and_respects_test_fold(self):
        folds = [
            [f"param{fold}/case{index}" for index in range(20)] for fold in range(9)
        ]
        features = {
            name: {
                "extent_x": float(index),
                "extent_y": float(index % 5),
                "extent_z": float(index % 3),
                "surface_points": float(100 + index),
                "volume_points": float(1000 + index),
            }
            for fold in folds
            for index, name in enumerate(fold)
        }
        splits = build_reduced_splits(folds, features, 32, 16, 10, seed=2026)
        self.assertEqual(
            splits, build_reduced_splits(folds, features, 32, 16, 10, seed=2026)
        )
        self.assertEqual(
            [len(splits[key]) for key in ("train", "validation", "test")], [32, 16, 10]
        )
        selected = [case for cases in splits.values() for case in cases]
        self.assertEqual(len(selected), len(set(selected)))
        self.assertTrue(set(splits["test"]).issubset(folds[0]))
        self.assertFalse(set(splits["train"] + splits["validation"]) & set(folds[0]))
        self.assertEqual(
            {name.split("/")[0] for name in splits["train"]},
            {f"param{fold}" for fold in range(1, 9)},
        )

    def test_discovers_archived_and_extracted_folds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for fold in range(9):
                if fold == 0:
                    with tarfile.open(root / "param0.tar.gz", "w:gz") as archive:
                        for member in (
                            "case/hexvelo_smpl.vtk",
                            "case/quadpress_smpl.vtk",
                            "case/cd.txt",
                            "metadata/param1.txt",
                            "metadata/cd.txt",
                        ):
                            info = tarfile.TarInfo(f"param0/{member}")
                            info.size = 1
                            archive.addfile(info, io.BytesIO(b"x"))
                else:
                    folder = root / f"param{fold}" / "case"
                    folder.mkdir(parents=True)
                    for file in ("hexvelo_smpl.vtk", "quadpress_smpl.vtk", "cd.txt"):
                        (folder / file).touch()
            folds = discover_folds(root)
            train, validation = split_fold(folds, 0)
        self.assertEqual(validation, ["param0/case"])
        self.assertEqual(len(train), 8)
        self.assertNotIn(validation[0], train)
        with self.assertRaises(ValueError):
            split_fold(folds, 9)


class ShapeNetCarComparisonTests(unittest.TestCase):
    def test_reference_comparison_accepts_hybrid_prediction(self):
        from benchmarks.shapenet_car.compare_reference import plot_comparison

        case = {
            "positions": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            "surface": np.array([True, True]),
            "actual": np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 2.0]]),
            "prediction": np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 2.0]]),
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "comparison.png"
            plot_comparison(case, case, "case", output, additional=case)
            self.assertTrue(output.is_file())

    def test_reference_comparison_rejects_mismatched_cases(self):
        from benchmarks.shapenet_car.compare_reference import check_matching_cases

        case = {
            "positions": np.array([[0.0, 0.0, 0.0]]),
            "surface": np.array([True]),
            "actual": np.array([[0.0, 0.0, 0.0, 1.0]]),
        }
        check_matching_cases(case, dict(case))
        changed = {**case, "actual": np.array([[0.0, 0.0, 0.0, 2.0]])}
        with self.assertRaises(ValueError):
            check_matching_cases(case, changed)

    def test_upstream_log_normalization_matches_shared_convention(self):
        from benchmarks.shapenet_car.evaluate_reference import normalization_from_log

        normalization = normalization_from_log(
            {"coef_norm": [[1] * 7, [2] * 7, [3] * 4, [4] * 4]}
        )
        np.testing.assert_array_equal(normalization["x_mean"], np.ones(7))
        np.testing.assert_array_equal(normalization["y_std"], np.full(4, 4))
        with self.assertRaises(ValueError):
            normalization_from_log({"coef_norm": [[1], [2], [3], [4]]})

    def test_runner_resolves_validation_mapping_without_test_leakage(self):
        from benchmarks.shapenet_car.run import resolve_split_paths

        splits = {
            "train": ["param1/train"],
            "validation": ["param2/validation"],
            "test": ["param0/test"],
        }
        mapping = {
            "param1/train": "training_data/param1/train",
            "param2/validation": "training_data/param0/validation",
            "param0/test": "test_data/param0/test",
        }
        result = resolve_split_paths(Path("dataset"), splits, mapping)
        self.assertEqual(
            result["validation"], [(Path("dataset/training_data"), "param0/validation")]
        )
        self.assertEqual(result["test"], [(Path("dataset/test_data"), "param0/test")])
        mapping["param0/test"] = "training_data/param0/test"
        with self.assertRaises(ValueError):
            resolve_split_paths(Path("dataset"), splits, mapping)

    def test_loss_and_scores_use_reference_pressure_and_velocity_regions(self):
        targets = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]])
        predictions = torch.tensor([[2.0, 0.0, 0.0, 999.0], [0.0, 0.0, 0.0, 3.0]])
        surface = torch.tensor([False, True])
        self.assertAlmostEqual(
            field_loss(predictions, targets, surface).item(), 1 / 6 + 0.5
        )
        scores = field_scores(predictions.numpy(), targets.numpy(), surface.numpy())
        self.assertAlmostEqual(scores["velocity_relative_l2"], 1.0)
        self.assertAlmostEqual(scores["pressure_relative_l2"], 0.5)

    def test_models_accept_three_dimensional_car_features(self):
        features = torch.randn(8, 7)
        positions = torch.randn(8, 3)
        edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        for kind in ("gnn", "local_transolver"):
            model = BenchmarkModel(kind, hidden_dim=16, num_layers=1)
            self.assertEqual(model(features, positions, edges).shape, (8, 4))

    def test_hybrid_slice_uses_three_dimensional_edges(self):
        from benchmarks.shapenet_car.training import hybrid_edge_features

        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]])
        edges = torch.tensor([[0, 0], [1, 0]])
        attributes = hybrid_edge_features(positions, edges)
        self.assertEqual(attributes.shape, (2, 7))
        torch.testing.assert_close(attributes[0, :3], positions[1] / 3)
        torch.testing.assert_close(attributes[0, 4:], positions[1])
        self.assertTrue(torch.isfinite(attributes).all())
        model = BenchmarkModel("gnn_hybrid_slice", hidden_dim=16, num_layers=1)
        predictions = model(torch.randn(2, 7), positions, edges)
        self.assertEqual(predictions.shape, (2, 4))
        self.assertTrue(torch.isfinite(predictions).all())

    def test_invalid_case_names_and_zero_norm_metrics(self):
        with self.assertRaises(ValueError):
            load_case(".", "param0/..\\other")
        scores = field_scores(np.zeros((2, 4)), np.zeros((2, 4)), [False, True])
        self.assertTrue(np.isnan(scores["velocity_relative_l2"]))
        self.assertTrue(np.isnan(scores["pressure_relative_l2"]))


if __name__ == "__main__":
    unittest.main()
