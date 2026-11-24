import unittest

import numpy as np
import torch

from refine_volume import parent_average, refine_topology, refined_item


def two_quads():
    coords = np.array(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0]], float
    )
    return {
        "coords": coords,
        "loops": [[0, 1, 4, 3], [1, 2, 5, 4]],
        "interior_cells": np.array([[0, 1]]),
        "interior_faces": np.array([[1, 4]]),
        "wall_segments": np.array([[[0.0, 0.0], [2.0, 0.0]]]),
    }


class RefinementTests(unittest.TestCase):
    def test_quads_split_into_four_children_that_tile_and_connect(self):
        refined = refine_topology(two_quads())
        np.testing.assert_allclose(refined["areas"], 0.25)
        np.testing.assert_array_equal(refined["parent"], [0] * 4 + [1] * 4)
        np.testing.assert_allclose(
            np.bincount(refined["parent"], weights=refined["areas"]), [1.0, 1.0]
        )
        np.testing.assert_allclose(refined["wall_distance"], refined["centroids"][:, 1])
        pairs = set(map(tuple, refined["edge_index"].T.tolist()))
        self.assertEqual(len(pairs), 20)
        self.assertIn((1, 4), pairs)
        self.assertIn((2, 7), pairs)
        self.assertNotIn((0, 5), pairs)
        centroid_distance = np.linalg.norm(
            refined["centroids"][1] - refined["centroids"][4]
        )
        self.assertAlmostEqual(centroid_distance, 0.5)

    def test_parent_average_and_refined_item(self):
        refined = refine_topology(two_quads())
        np.testing.assert_allclose(
            parent_average(np.arange(8.0), refined, 2), [1.5, 5.5]
        )
        item = {
            "geom": torch.tensor([[0.5, 0.5, 0, 1, 0.5], [1.5, 0.5, 0, 1, 0.5]]),
            "params": torch.tensor([0.1, 1.0, 10.0]),
            "pressure": torch.tensor([[1.0], [2.0]]),
            "design_name": "Design_test",
            "mesh_kind": "volume",
            "body_bounds": torch.zeros(2, 3),
            "domain_bounds": torch.zeros(2, 3),
        }
        fine = refined_item(item, refined)
        self.assertEqual(tuple(fine["geom"].shape), (8, 5))
        self.assertEqual(fine["edge_attr"].shape[0], fine["edge_index"].shape[1])
        torch.testing.assert_close(
            fine["pressure"][:, 0], torch.tensor([1.0] * 4 + [2.0] * 4)
        )


if __name__ == "__main__":
    unittest.main()
