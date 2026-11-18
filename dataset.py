"""
Handles log parsing, HDF5 extraction from .simh, face-to-graph conversion, normalization, and graph batch collation.
"""

import os
import re

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# ==============================================================================
# 1. NORMALIZATION UTILITY
# ==============================================================================
class RunningStandardizer:
    """Computes mean and std across training designs to normalize inputs/targets."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, tensor_list):
        stacked = torch.cat(tensor_list, dim=0)
        self.mean = stacked.mean(dim=0, keepdim=True)
        std = stacked.std(dim=0, keepdim=True)
        # Constant channels would otherwise amplify float rounding noise.
        self.std = torch.where(std < 1e-6, torch.ones_like(std), std)

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return (x * self.std) + self.mean


# ==============================================================================
# 2. LOG & SIMH PARSERS
# ==============================================================================
def parse_design_log(log_path):
    """
    Extracts design parameters and official STAR-CCM+ report metrics
    from Design_X.log.
    """
    with open(log_path, "r", errors="ignore") as f:
        content = f.read()

    p_d1 = float(re.search(r"P_D1_r\s*:\s*([0-9.eE+-]+)", content).group(1))
    p_d2 = float(re.search(r"P_D2_r\s*:\s*([0-9.eE+-]+)", content).group(1))
    p_v = float(re.search(r"P_V\s*:\s*([0-9.eE+-]+)", content).group(1))

    drag = float(re.search(r"Drag\s*:\s*([0-9.eE+-]+)", content).group(1))
    cd = float(re.search(r"Cd\s*:\s*([0-9.eE+-]+)", content).group(1))
    pdrop = float(re.search(r"PressureDrop\s*:\s*([0-9.eE+-]+)", content).group(1))

    return {
        "params": [p_d1, p_d2, p_v],
        "true_drag": drag,
        "true_cd": cd,
        "true_pdrop": pdrop,
    }


def build_face_graph_from_vertices(vertex_list, num_faces):
    """
    Constructs the dual graph: faces sharing a vertex form graph edges.
    """
    vertex_to_faces = {}
    for face_id in range(num_faces):
        v1 = vertex_list[face_id * 2]
        v2 = vertex_list[face_id * 2 + 1]
        for v in [v1, v2]:
            if v not in vertex_to_faces:
                vertex_to_faces[v] = []
            vertex_to_faces[v].append(face_id)

    edges = []
    for faces in vertex_to_faces.values():
        if len(faces) == 2:  # Shared internal boundary vertex
            f_a, f_b = faces[0], faces[1]
            edges.append([f_a, f_b])
            edges.append([f_b, f_a])

    return torch.tensor(edges, dtype=torch.long).t()


def compute_edge_attr(geom, edge_index):
    """Dimensionless edge features: [unit direction (2), normal change (2), spacing / face length]."""
    src, dst = edge_index[0], edge_index[1]
    delta = geom[dst, 0:2] - geom[src, 0:2]
    dist = delta.norm(dim=1, keepdim=True).clamp_min(1e-12)
    face_len = 0.5 * (geom[src, 6] + geom[dst, 6]).unsqueeze(1)
    normal_change = geom[dst, 3:5] - geom[src, 3:5]
    return torch.cat([delta / dist, normal_change, dist / face_len], dim=1)


def load_all_starccm_designs(root_dir):
    """
    Walks all Design_X folders, extracts mesh and physical fields,
    and returns a structured list of PyTorch graph dictionaries.
    """
    dataset = []
    design_folders = sorted(
        [
            d
            for d in os.listdir(root_dir)
            if d.startswith("Design_") and os.path.isdir(os.path.join(root_dir, d))
        ],
        key=lambda x: int(x.split("_")[1]),
    )

    print(f"Discovered {len(design_folders)} design directories in {root_dir}")

    for folder_name in design_folders:
        folder_path = os.path.join(root_dir, folder_name)
        simh_path = os.path.join(folder_path, "Surface_Dataset.simh")
        log_path = os.path.join(folder_path, f"{folder_name}.log")

        if not os.path.exists(simh_path) or not os.path.exists(log_path):
            print(f"Skipping {folder_name}: Missing .simh or .log file.")
            continue

        log_info = parse_design_log(log_path)
        params_tensor = torch.tensor(log_info["params"], dtype=torch.float32)

        with h5py.File(simh_path, "r") as f:
            wall_mesh = f["Meshes/FvMesh 1/Parts/Surfaces/Domain/WALL"]
            vertex_list = np.array(wall_mesh["VertexList/VertexList"])
            coord = np.array(wall_mesh["Coord"])
            num_faces = len(wall_mesh["VertexList/VertexList_lengths"])

            face_data = f["States/ROM/Parts/Surfaces/Domain/WALL/FaceData"]
            centroids = np.array(face_data["CentroidFieldFunction"])
            normals = np.array(face_data["NormalFieldFunction"])
            pressure = np.array(face_data["StaticPressureFieldFunction"])
            shear_x = np.array(face_data["WallShearStressFieldFunction"])[:, 0]
            density = np.array(face_data["DensityFieldFunction"])

        # Compute 2D face length (area)
        areas = []
        for i in range(num_faces):
            v1_idx = vertex_list[i * 2]
            v2_idx = vertex_list[i * 2 + 1]
            length = np.linalg.norm(coord[v1_idx] - coord[v2_idx])
            areas.append(length)
        areas = np.array(areas, dtype=np.float32).reshape(-1, 1)

        # Pack into tensors
        geom = torch.cat(
            [
                torch.tensor(centroids, dtype=torch.float32),
                torch.tensor(normals, dtype=torch.float32),
                torch.tensor(areas, dtype=torch.float32),
            ],
            dim=1,
        )  # [N_faces, 7]

        fields = torch.stack(
            [
                torch.tensor(pressure, dtype=torch.float32).squeeze(),
                torch.tensor(shear_x, dtype=torch.float32).squeeze(),
            ],
            dim=1,
        )  # [N_faces, 2]

        edge_index = build_face_graph_from_vertices(vertex_list, num_faces)
        rho = float(np.mean(density))
        q_dyn = 0.5 * rho * log_info["params"][2] ** 2
        # Drag sensitivity per face: |n_x|*A for pressure, A for streamwise shear.
        influence = torch.stack([geom[:, 3].abs() * geom[:, 6], geom[:, 6]], dim=1)
        face_weights = influence / influence.mean(dim=0, keepdim=True)

        dataset.append(
            {
                "geom": geom,
                "params": params_tensor,
                "edge_index": edge_index,
                "edge_attr": compute_edge_attr(geom, edge_index),
                "fields": fields,
                "coeffs": fields / q_dyn,
                "face_weights": face_weights,
                "rho": rho,
                "q_dyn": q_dyn,
                "true_drag": log_info["true_drag"],
                "true_cd": log_info["true_cd"],
                "design_name": folder_name,
            }
        )

    print(f"Successfully processed {len(dataset)} designs into PyTorch graphs.")
    return dataset


def _face_edges(face_group):
    vertex_list = np.asarray(face_group["VertexList/VertexList"], dtype=np.int64)
    lengths = np.asarray(face_group["VertexList/VertexList_lengths"], dtype=np.int64)
    if np.any(lengths != 2):
        raise ValueError("Expected a 2D mesh where each face has two vertices.")
    return vertex_list.reshape(-1, 2)


def _cell_areas(coords, interior_cells, interior_edges, boundary_cells, boundary_edges):
    cell_edges = [[] for _ in range(len(coords))]
    num_cells = int(max(interior_cells.max(), boundary_cells.max())) + 1
    cell_edges = [[] for _ in range(num_cells)]

    for (owner, neighbor), edge in zip(interior_cells, interior_edges):
        cell_edges[int(owner)].append(edge)
        cell_edges[int(neighbor)].append(edge)
    for cell, edge in zip(boundary_cells, boundary_edges):
        cell_edges[int(cell)].append(edge)

    areas = np.empty(num_cells, dtype=np.float32)
    for cell, edges in enumerate(cell_edges):
        adjacency = {}
        for start, end in edges:
            adjacency.setdefault(int(start), []).append(int(end))
            adjacency.setdefault(int(end), []).append(int(start))
        if len(adjacency) < 3 or any(
            len(neighbors) != 2 for neighbors in adjacency.values()
        ):
            raise ValueError(f"Cell {cell} does not have a closed 2D face loop.")

        first = next(iter(adjacency))
        ordered = [first]
        previous = None
        current = first
        while True:
            neighbors = adjacency[current]
            following = neighbors[0] if neighbors[0] != previous else neighbors[1]
            if following == first:
                break
            if following in ordered:
                raise ValueError(f"Cell {cell} has a self-intersecting face loop.")
            ordered.append(following)
            previous, current = current, following

        if len(ordered) != len(adjacency):
            raise ValueError(f"Cell {cell} has disconnected polygon boundaries.")
        polygon = coords[ordered, :2]
        area = 0.5 * abs(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
        if not np.isfinite(area) or area <= 0:
            raise ValueError(f"Cell {cell} has invalid area {area}.")
        areas[cell] = area
    return areas


def load_all_starccm_volume_designs(root_dir):
    """Load cell-centred pressure graphs from STAR-CCM+ volume exports."""
    dataset = []
    design_folders = sorted(
        [
            d
            for d in os.listdir(root_dir)
            if d.startswith("Design_") and os.path.isdir(os.path.join(root_dir, d))
        ],
        key=lambda x: int(x.split("_")[1]),
    )
    print(f"Discovered {len(design_folders)} design directories in {root_dir}")

    for folder_name in design_folders:
        folder_path = os.path.join(root_dir, folder_name)
        simh_path = os.path.join(folder_path, "Volume_Dataset.simh")
        log_path = os.path.join(folder_path, f"{folder_name}.log")
        if not os.path.exists(simh_path) or not os.path.exists(log_path):
            print(f"Skipping {folder_name}: Missing volume .simh or log file.")
            continue

        log_info = parse_design_log(log_path)
        with h5py.File(simh_path, "r") as handle:
            mesh = handle["Meshes/FvMesh 1/Regions/Domain"]
            coords = np.asarray(mesh["Coord"], dtype=np.float64)
            if not np.allclose(coords[:, 2], coords[0, 2]):
                raise ValueError(f"{folder_name} is not a planar 2D mesh.")

            cell_indices = np.asarray(mesh["FaceCellIndex"], dtype=np.int64)
            interior_edges = _face_edges(mesh)
            boundary_cells = []
            boundary_edges = []
            for boundary in mesh["Boundaries"].values():
                boundary_cells.extend(
                    np.asarray(boundary["FaceCellIndex"], dtype=np.int64)[:, 0]
                )
                boundary_edges.extend(_face_edges(boundary))
            boundary_cells = np.asarray(boundary_cells, dtype=np.int64)
            boundary_edges = np.asarray(boundary_edges, dtype=np.int64)

            cell_data = handle["States/ROM/Regions/Domain/CellData"]
            centroids = np.asarray(cell_data["CentroidFieldFunction"], dtype=np.float32)
            pressure = np.asarray(cell_data["Pressure"], dtype=np.float32).reshape(
                -1, 1
            )

        num_cells = centroids.shape[0]
        if pressure.shape[0] != num_cells:
            raise ValueError(f"{folder_name} pressure and centroid cell counts differ.")
        if cell_indices.size and cell_indices.max() >= num_cells:
            raise ValueError(f"{folder_name} has a face referencing a missing cell.")
        areas = _cell_areas(
            coords, cell_indices, interior_edges, boundary_cells, boundary_edges
        )
        if len(areas) != num_cells:
            raise ValueError(f"{folder_name} topology and cell data counts differ.")

        edge_index = torch.tensor(
            np.concatenate([cell_indices.T, cell_indices[:, ::-1].T], axis=1),
            dtype=torch.long,
        )
        geom = torch.cat(
            [torch.from_numpy(centroids), torch.from_numpy(areas).unsqueeze(1)], dim=1
        )
        dataset.append(
            {
                "geom": geom,
                "params": torch.tensor(log_info["params"], dtype=torch.float32),
                "pressure": torch.from_numpy(pressure),
                "edge_index": edge_index,
                "design_name": folder_name,
            }
        )

    print(f"Successfully processed {len(dataset)} volume graphs.")
    return dataset


# ==============================================================================
# 3. PYTORCH DATASET & BATCH COLLATOR
# ==============================================================================
class CFDSurfaceDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_cfd_graphs(batch):
    """
    Batches multiple disjoint graphs together by offsetting edge_index.
    """
    geom_list = []
    params_list = []
    fields_list = []
    edge_index_list = []
    batch_idx_list = []

    node_offset = 0
    for b_i, item in enumerate(batch):
        N_f = item["geom"].size(0)

        geom_list.append(item["geom"])
        params_list.append(item["params"].unsqueeze(0).repeat(N_f, 1))
        fields_list.append(item["fields"])

        edge_index_list.append(item["edge_index"] + node_offset)
        batch_idx_list.append(torch.full((N_f,), b_i, dtype=torch.long))

        node_offset += N_f

    return {
        "geom": torch.cat(geom_list, dim=0),
        "params": torch.cat(params_list, dim=0),
        "fields": torch.cat(fields_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "batch_idx": torch.cat(batch_idx_list, dim=0),
        "num_graphs": len(batch),
    }


def collate_cfd_graphs_with_physics(batch):
    """Collate normalized graph inputs alongside raw geometry and drag targets."""
    normalized_batch = [
        {
            "geom": item["geom_norm"],
            "params": item["params_norm"],
            "fields": item["fields_norm"],
            "edge_index": item["edge_index"],
        }
        for item in batch
    ]
    collated = collate_cfd_graphs(normalized_batch)
    return {
        "geom_norm": collated["geom"],
        "geom_raw": torch.cat([item["geom"] for item in batch], dim=0),
        "params_norm": collated["params"],
        "fields_norm": collated["fields"],
        "face_weights": torch.cat([item["face_weights"] for item in batch], dim=0),
        "edge_index": collated["edge_index"],
        "edge_attr": torch.cat([item["edge_attr"] for item in batch], dim=0),
        "batch_idx": collated["batch_idx"],
        "true_drags": torch.tensor(
            [item["true_drag"] for item in batch], dtype=torch.float32
        ),
        "q_dyn": torch.tensor([item["q_dyn"] for item in batch], dtype=torch.float32),
        "num_graphs": collated["num_graphs"],
    }


class CFDVolumeDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


def collate_cfd_volume_graphs(batch):
    """Collate disjoint cell graphs for cell-centred pressure training."""
    geom_list = []
    params_list = []
    pressure_list = []
    area_list = []
    edge_index_list = []
    batch_idx_list = []
    node_offset = 0

    for batch_id, item in enumerate(batch):
        num_cells = item["geom_norm"].size(0)
        geom_list.append(item["geom_norm"])
        params_list.append(item["params_norm"].unsqueeze(0).expand(num_cells, -1))
        pressure_list.append(item["pressure_norm"])
        area_list.append(item["geom"][:, 3])
        edge_index_list.append(item["edge_index"] + node_offset)
        batch_idx_list.append(torch.full((num_cells,), batch_id, dtype=torch.long))
        node_offset += num_cells

    return {
        "geom_norm": torch.cat(geom_list, dim=0),
        "params_norm": torch.cat(params_list, dim=0),
        "pressure_norm": torch.cat(pressure_list, dim=0),
        "cell_areas": torch.cat(area_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "batch_idx": torch.cat(batch_idx_list, dim=0),
        "num_graphs": len(batch),
    }
