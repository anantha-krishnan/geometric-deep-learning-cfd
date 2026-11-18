import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from dataset import collate_cfd_volume_graphs, load_all_starccm_volume_designs


class VolumeDatasetTests(unittest.TestCase):
    def test_volume_collator_offsets_cell_graphs(self):
        first = {
            "geom": torch.tensor([[0, 0, 0, 1], [1, 0, 0, 1]], dtype=torch.float32),
            "geom_norm": torch.ones(2, 4),
            "params_norm": torch.tensor([1.0, 2.0, 3.0]),
            "pressure_norm": torch.ones(2, 1),
            "edge_index": torch.tensor([[0, 1], [1, 0]]),
        }
        second = {
            "geom": torch.tensor([[0, 1, 0, 2], [1, 1, 0, 2]], dtype=torch.float32),
            "geom_norm": torch.ones(2, 4),
            "params_norm": torch.tensor([4.0, 5.0, 6.0]),
            "pressure_norm": torch.full((2, 1), 2.0),
            "edge_index": torch.tensor([[0], [1]]),
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

            designs = load_all_starccm_volume_designs(root)

        self.assertEqual(len(designs), 1)
        graph = designs[0]
        self.assertEqual(tuple(graph["geom"].shape), (2, 4))
        np.testing.assert_allclose(graph["geom"][:, :3], [[0.5, 0.5, 0], [1.5, 0.5, 0]])
        np.testing.assert_allclose(graph["geom"][:, 3], [1, 1])
        torch.testing.assert_close(graph["pressure"], torch.tensor([[100.0], [200.0]]))
        torch.testing.assert_close(graph["edge_index"], torch.tensor([[0, 1], [1, 0]]))


if __name__ == "__main__":
    unittest.main()
