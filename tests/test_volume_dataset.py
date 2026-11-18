import tempfile
import unittest
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from dataset import collate_cfd_volume_graphs, load_all_starccm_volume_designs
from validate_volume import build_cell_polygons, plot_comparison


class VolumeDatasetTests(unittest.TestCase):
    def test_volume_collator_offsets_cell_graphs(self):
        first = {
            "geom": torch.tensor([[0, 0, 0, 1], [1, 0, 0, 1]], dtype=torch.float32),
            "geom_norm": torch.ones(2, 4),
            "params_norm": torch.tensor([1.0, 2.0, 3.0]),
            "pressure_norm": torch.ones(2, 1),
            "edge_index": torch.tensor([[0, 1], [1, 0]]),
            "edge_attr": torch.tensor([[1.0, 0, 1, 0], [-1.0, 0, 1, 0]]),
        }
        second = {
            "geom": torch.tensor([[0, 1, 0, 2], [1, 1, 0, 2]], dtype=torch.float32),
            "geom_norm": torch.ones(2, 4),
            "params_norm": torch.tensor([4.0, 5.0, 6.0]),
            "pressure_norm": torch.full((2, 1), 2.0),
            "edge_index": torch.tensor([[0], [1]]),
            "edge_attr": torch.tensor([[1.0, 0, 1, 0]]),
        }

        batch = collate_cfd_volume_graphs([first, second])

        torch.testing.assert_close(
            batch["edge_index"], torch.tensor([[0, 1, 2], [1, 0, 3]])
        )
        torch.testing.assert_close(batch["batch_idx"], torch.tensor([0, 0, 1, 1]))
        torch.testing.assert_close(
            batch["pressure_norm"], torch.tensor([[1.0], [1.0], [2.0], [2.0]])
        )
        torch.testing.assert_close(
            batch["cell_areas"], torch.tensor([1, 1, 2, 2], dtype=torch.float32)
        )
        torch.testing.assert_close(
            batch["edge_attr"], torch.cat([first["edge_attr"], second["edge_attr"]])
        )

    def test_load_volume_export_and_derive_cell_areas(self):
        with tempfile.TemporaryDirectory() as root:
            design = Path(root) / "Design_1"
            design.mkdir()
            (design / "Design_1.log").write_text(
                "P_D1_r: 1\nP_D2_r: 2\nP_V: 3\nDrag: 4\nCd: 0.5\nPressureDrop: 6\n"
            )
            path = design / "Volume_Dataset.simh"
            with h5py.File(path, "w") as handle:
                mesh = handle.create_group("Meshes/FvMesh 1/Regions/Domain")
                mesh["Coord"] = [
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [2, 0, 0],
                    [2, 1, 0],
                ]
                mesh["CellGeometryPartIndex"] = [1, 1]
                mesh["FaceCellIndex"] = [[0, 1]]
                mesh.create_group("VertexList")["VertexList"] = [1, 2]
                mesh["VertexList/VertexList_lengths"] = [2]
                boundaries = mesh.create_group("Boundaries/WALL")
                boundaries["FaceCellIndex"] = [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [1, 0],
                    [1, 0],
                    [1, 0],
                ]
                boundaries.create_group("VertexList")["VertexList"] = [
                    0,
                    1,
                    2,
                    3,
                    3,
                    0,
                    1,
                    4,
                    4,
                    5,
                    5,
                    2,
                ]
                boundaries["VertexList/VertexList_lengths"] = [2] * 6
                cells = handle.create_group("States/ROM/Regions/Domain/CellData")
                cells["CentroidFieldFunction"] = [[0.5, 0.5, 0], [1.5, 0.5, 0]]
                cells["Pressure"] = [100, 200]
                cells["AI_ROM_ManualWallDistanceFieldFunction"] = [0.5, 0.75]

            designs = load_all_starccm_volume_designs(root)
            enhanced = load_all_starccm_volume_designs(root, include_physics=True)[0]

        self.assertEqual(len(designs), 1)
        graph = designs[0]
        self.assertEqual(tuple(graph["geom"].shape), (2, 4))
        np.testing.assert_allclose(graph["geom"][:, :3], [[0.5, 0.5, 0], [1.5, 0.5, 0]])
        np.testing.assert_allclose(graph["geom"][:, 3], [1, 1])
        torch.testing.assert_close(graph["pressure"], torch.tensor([[100.0], [200.0]]))
        torch.testing.assert_close(graph["edge_index"], torch.tensor([[0, 1], [1, 0]]))
        self.assertEqual(tuple(enhanced["geom"].shape), (2, 5))
        torch.testing.assert_close(enhanced["geom"][:, 4], torch.tensor([0.5, 0.75]))
        torch.testing.assert_close(
            enhanced["edge_attr"],
            torch.tensor([[1.0, 0, 1, 0], [-1.0, 0, 1, 0]]),
        )


class VolumePlotTests(unittest.TestCase):
    def test_builds_polygons_from_unordered_edges_and_rejects_open_cells(self):
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        edges = [[(2, 3), (0, 1), (3, 0), (1, 2)]]
        polygons = build_cell_polygons(coords, edges)
        self.assertEqual(len(polygons), 1)
        np.testing.assert_array_equal(
            np.sort(polygons[0], axis=0), np.sort(coords, axis=0)
        )
        with self.assertRaises(ValueError):
            build_cell_polygons(coords, [[(0, 1), (1, 2)]])

    def test_volume_plot_uses_cell_polygons_and_shared_pressure_scale(self):
        polygons = [
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
            np.array([[2, 0], [3, 0], [3, 1], [2, 1]]),
        ]
        figure = plot_comparison(
            polygons,
            np.array([10.0, 20.0]),
            np.array([12.0, 24.0]),
            {"design_name": "Design_1", "variant": "geometry", "fold": 1},
        )
        try:
            panels = [axis for axis in figure.axes if axis.get_title(loc="left")]
            self.assertEqual(len(panels), 2)
            self.assertEqual(len(panels[0].collections[0].get_paths()), 2)
            self.assertIs(panels[0].collections[0].norm, panels[1].collections[0].norm)
            np.testing.assert_array_equal(
                panels[0].collections[0].get_array(), [10, 20]
            )
            np.testing.assert_array_equal(
                panels[1].collections[0].get_array(), [12, 24]
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "volume.png"
                figure.savefig(path, dpi=50)
                self.assertGreater(path.stat().st_size, 1000)
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
