import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from dataset import load_all_starccm_designs


class SurfaceDatasetTests(unittest.TestCase):
    def test_load_nested_starccm_surface_export(self):
        with tempfile.TemporaryDirectory() as root:
            design = Path(root) / "Design_1"
            design.mkdir()
            (design / "Design_1.log").write_text(
                "P_D1_r: 1\nP_D2_r: 2\nP_V: 3\n"
                "Drag: 4\nCd: 0.5\nPressureDrop: 6\n"
            )
            with h5py.File(design / "Surface_Dataset.simh", "w") as handle:
                mesh = handle.create_group(
                    "Meshes/FvMesh 1/Parts/Surfaces/Domain/WALL"
                )
                mesh["Coord"] = [[0, 0, 0], [1, 0, 0], [1, 2, 0]]
                vertices = mesh.create_group("VertexList")
                vertices["VertexList"] = [0, 1, 1, 2]
                vertices["VertexList_lengths"] = [2, 2]
                fields = handle.create_group(
                    "States/ROM/Parts/Surfaces/Domain/WALL/FaceData"
                )
                fields["CentroidFieldFunction"] = [[0.5, 0, 0], [1, 1, 0]]
                fields["NormalFieldFunction"] = [[0, 1, 0], [1, 0, 0]]
                fields["StaticPressureFieldFunction"] = [10, 20]
                fields["WallShearStressFieldFunction"] = [[3, 4, 0], [0, 0, 2]]
                fields["DensityFieldFunction"] = [1.2, 1.2]

            graphs = load_all_starccm_designs(root)

        self.assertEqual(len(graphs), 1)
        graph = graphs[0]
        self.assertEqual(tuple(graph["geom"].shape), (2, 7))
        np.testing.assert_allclose(graph["geom"][:, 6], [1, 2])
        np.testing.assert_allclose(graph["fields"], [[10, 5, 1.2], [20, 2, 1.2]])
        torch.testing.assert_close(
            graph["edge_index"], torch.tensor([[0, 1], [1, 0]])
        )
        self.assertEqual(graph["true_drag"], 4)


if __name__ == "__main__":
    unittest.main()