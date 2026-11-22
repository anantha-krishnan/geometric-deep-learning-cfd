import unittest

import torch

import compare_volume_architectures as experiment
import train_volume as training
from invariance import (
    NEAR_WALL_RELATIVE,
    bounding_box,
    feature_names,
    make_invariant,
)


def _grid_item(cells_per_unit=8, scale=1.0, shift=(0.0, 0.0, 0.0), depth=0):
    """Uniform 2D (or 3D when depth > 0) volume grid around a box body."""
    n = (3 * cells_per_unit, cells_per_unit, max(depth, 1))
    h = 1.0 / cells_per_unit
    index = torch.arange(n[0] * n[1] * n[2]).reshape(n)
    grid = torch.stack(
        torch.meshgrid(*(torch.arange(k) for k in n), indexing="ij"), dim=-1
    ).reshape(-1, 3)
    position = (grid + 0.5) * h
    if not depth:
        position[:, 2] = 0.0
    body = torch.tensor([[0.45, 0.45, 0.0], [0.55, 0.55, depth * h]])
    centre = body.mean(dim=0)
    wall = (position - centre).abs().sub(0.05).clamp_min(0)[:, : 3 if depth else 2]
    edges = []
    for axis in range(3 if depth else 2):
        source = index.narrow(axis, 0, n[axis] - 1).reshape(-1)
        target = index.narrow(axis, 1, n[axis] - 1).reshape(-1)
        edges += [torch.stack([source, target]), torch.stack([target, source])]
    offset = torch.tensor(shift)
    measure = torch.full((len(position), 1), h ** (3 if depth else 2))
    geom = torch.cat(
        [
            position * scale + offset,
            measure * scale ** (3 if depth else 2),
            wall.norm(dim=1, keepdim=True) * scale,
        ],
        dim=1,
    )
    return {
        "geom": geom,
        "params": torch.tensor([0.1, 1.5, 10.0]),
        "pressure": torch.randn(len(geom), 1),
        "edge_index": torch.cat(edges, dim=1),
        "design_name": "Design_test",
        "mesh_kind": "volume",
        "body_bounds": bounding_box(body * scale + offset),
        "domain_bounds": bounding_box(
            torch.tensor([[0.0, 0.0, 0.0], [3.0, 1.0, depth * h]]) * scale + offset
        ),
    }


def _surface_item(segments=16, scale=1.0):
    angle = torch.linspace(0, 2 * torch.pi, segments + 1)[:-1]
    vertices = torch.stack([angle.cos(), angle.sin(), torch.zeros_like(angle)], 1)
    vertices = vertices * 0.05 * scale + torch.tensor([0.5, 0.5, 0.0])
    following = vertices.roll(-1, 0)
    centroid = 0.5 * (vertices + following)
    normal = torch.nn.functional.normalize(centroid - centroid.mean(0), dim=1)
    length = (following - vertices).norm(dim=1, keepdim=True)
    face = torch.arange(segments)
    edge_index = torch.cat(
        [torch.stack([face, face.roll(-1)]), torch.stack([face.roll(-1), face])], 1
    )
    return {
        "geom": torch.cat([centroid, normal, length], dim=1),
        "params": torch.tensor([0.1, 1.5, 10.0]),
        "edge_index": edge_index,
        "design_name": "Design_surface",
        "mesh_kind": "surface",
        "body_bounds": bounding_box(vertices),
        "domain_bounds": bounding_box(
            torch.tensor([[0.0, 0.0, 0.0], [3.0 * scale, 1.0 * scale, 0.0]])
        ),
    }


class InvarianceTests(unittest.TestCase):
    def test_volume_features_are_scale_and_translation_invariant(self):
        base = make_invariant(_grid_item())
        moved = make_invariant(_grid_item(scale=4.0, shift=(2.0, -1.0, 0.0)))
        torch.testing.assert_close(moved["node_features"], base["node_features"])
        torch.testing.assert_close(moved["edge_attr"], base["edge_attr"])
        self.assertAlmostEqual(moved["length_ref"], 4 * base["length_ref"], places=5)
        # Only the Reynolds proxy may change with absolute size.
        delta = moved["design_features"] - base["design_features"]
        torch.testing.assert_close(delta[3], torch.tensor(4.0).log())
        torch.testing.assert_close(delta[[0, 1, 2, 4]], torch.zeros(4))

    def test_edge_features_do_not_depend_on_mesh_resolution(self):
        for cells in (4, 16):
            edges = make_invariant(_grid_item(cells_per_unit=cells))["edge_attr"]
            torch.testing.assert_close(edges[:, 2], torch.ones(len(edges)))
            torch.testing.assert_close(edges[:, 3], torch.zeros(len(edges)))

    def test_nodes_carry_no_cell_size(self):
        coarse = make_invariant(_grid_item(cells_per_unit=4))
        names = feature_names(coarse)
        self.assertEqual(
            names["node"],
            [
                "x_over_L",
                "y_over_L",
                "wall_distance_over_L",
                "log_wall_distance_over_L",
            ],
        )
        self.assertEqual(coarse["node_features"].shape[1], len(names["node"]))
        self.assertEqual(coarse["design_features"].shape[0], len(names["design"]))
        self.assertEqual(coarse["edge_attr"].shape[1], len(names["edge"]))

    def test_planar_mesh_drops_z_and_3d_mesh_keeps_it(self):
        planar = make_invariant(_grid_item(cells_per_unit=4))
        volume = make_invariant(_grid_item(cells_per_unit=4, depth=3))
        self.assertEqual(planar["node_features"].shape[1], 4)
        self.assertEqual(volume["node_features"].shape[1], 5)
        self.assertIn("z_extent_over_L", feature_names(volume)["design"])
        # h = V^(1/3) on a uniform 3D grid gives unit relative spacing.
        torch.testing.assert_close(
            volume["edge_attr"][:, 3], torch.ones(len(volume["edge_attr"]))
        )

    def test_surface_mesh_is_detected_and_invariant(self):
        base = make_invariant(_surface_item())
        large = make_invariant(_surface_item(scale=10.0))
        self.assertNotIn("wall_distance_ref", base)
        self.assertEqual(
            feature_names(base)["node"], ["x_over_L", "y_over_L", "nx", "ny"]
        )
        torch.testing.assert_close(large["node_features"], base["node_features"])
        torch.testing.assert_close(large["edge_attr"], base["edge_attr"])

    def test_missing_metadata_is_rejected(self):
        item = _grid_item(cells_per_unit=4)
        with self.assertRaises(ValueError):
            make_invariant({**item, "mesh_kind": "unknown"})
        with self.assertRaises(ValueError):
            make_invariant(
                {key: value for key, value in item.items() if key != "body_bounds"}
            )

    def test_invariant_variant_uses_body_relative_near_wall_band(self):
        raw = _grid_item(cells_per_unit=4)
        raw["geom"][:3, 4] = 0.0
        item = training.prepare_data([raw], "invariant")[0]
        distance, threshold = training.near_wall_distance(item)
        self.assertEqual(threshold, NEAR_WALL_RELATIVE)
        torch.testing.assert_close(distance, item["wall_distance"] / item["length_ref"])
        row = training.measure_design(item["pressure"] + 1.0, item)
        self.assertAlmostEqual(row["near_wall_rel_rmse_pa"], 1.0, places=5)
        raw = _grid_item(cells_per_unit=4)
        raw["edge_attr"] = torch.zeros(raw["edge_index"].shape[1], 4)
        legacy = training.prepare_data([raw], "geometry")[0]
        self.assertEqual(
            training.near_wall_distance(legacy)[1], training.NEAR_WALL_THRESHOLD
        )

    def test_domain_frame_is_bounded_and_reports_blockage(self):
        item = _grid_item(cells_per_unit=4)
        two_scale = make_invariant(item, "domain")
        position = two_scale["node_features"][:, :2]
        self.assertGreaterEqual(float(position.min()), 0.0)
        self.assertLessEqual(float(position.max()), 1.0)
        names = feature_names(item, "domain")
        self.assertEqual(names["node"][:2], ["x_over_domain", "y_over_domain"])
        self.assertEqual(names["design"][-1], "y_blockage")
        self.assertEqual(two_scale["design_features"].shape[0], len(names["design"]))
        # Body height 0.1 in a domain of height 1.0.
        self.assertAlmostEqual(float(two_scale["design_features"][-1]), 0.1, places=5)
        body = make_invariant(item)
        torch.testing.assert_close(two_scale["edge_attr"], body["edge_attr"])
        torch.testing.assert_close(
            two_scale["node_features"][:, 2:], body["node_features"][:, 2:]
        )
        with self.assertRaises(ValueError):
            make_invariant(
                {k: v for k, v in item.items() if k != "domain_bounds"}, "domain"
            )

    def test_two_scale_variant_prepares_domain_frame(self):
        item = training.prepare_data([_grid_item(cells_per_unit=4)], "two_scale")[0]
        self.assertLessEqual(float(item["node_features"][:, :2].max()), 1.0)
        self.assertIn("wall_distance_ref", item)

    def test_stratified_split_takes_one_validation_design_per_length_stratum(self):
        items = []
        for index, length in enumerate(
            [0.05, 0.06, 0.07, 0.3, 0.31, 0.32, 0.6, 0.61, 0.62, 0.63]
        ):
            bounds = torch.tensor([[0.0, 0.0, 0.0], [length, 0.1, 0.0]])
            items.append({"design_name": f"Design_{index}", "body_bounds": bounds})
        train, validation = experiment.split_inner_validation(
            items, fraction=0.2, seed=3, stratify=True
        )
        lengths = sorted(float(item["body_bounds"][1, 0]) for item in validation)
        self.assertEqual(len(validation), 2)
        self.assertLess(lengths[0], 0.31)
        self.assertGreater(lengths[1], 0.31)
        small_in_train = [i for i in train if float(i["body_bounds"][1, 0]) < 0.1]
        self.assertGreaterEqual(len(small_in_train), 2)
        again = experiment.split_inner_validation(
            items, fraction=0.2, seed=3, stratify=True
        )
        self.assertEqual(
            [i["design_name"] for i in again[1]],
            [i["design_name"] for i in validation],
        )


if __name__ == "__main__":
    unittest.main()
