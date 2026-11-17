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
        self.std = stacked.std(dim=0, keepdim=True) + 1e-8

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
    with open(log_path, 'r', errors='ignore') as f:
        content = f.read()

    p_d1 = float(re.search(r"P_D1_r\s*:\s*([0-9.eE+-]+)", content).group(1))
    p_d2 = float(re.search(r"P_D2_r\s*:\s*([0-9.eE+-]+)", content).group(1))
    p_v  = float(re.search(r"P_V\s*:\s*([0-9.eE+-]+)", content).group(1))

    drag = float(re.search(r"Drag\s*:\s*([0-9.eE+-]+)", content).group(1))
    cd   = float(re.search(r"Cd\s*:\s*([0-9.eE+-]+)", content).group(1))
    pdrop = float(re.search(r"PressureDrop\s*:\s*([0-9.eE+-]+)", content).group(1))

    return {
        "params": [p_d1, p_d2, p_v],
        "true_drag": drag,
        "true_cd": cd,
        "true_pdrop": pdrop
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


def load_all_starccm_designs(root_dir):
    """
    Walks all Design_X folders, extracts mesh and physical fields,
    and returns a structured list of PyTorch graph dictionaries.
    """
    dataset = []
    design_folders = sorted(
        [d for d in os.listdir(root_dir) if d.startswith("Design_") and os.path.isdir(os.path.join(root_dir, d))],
        key=lambda x: int(x.split("_")[1])
    )

    print(f"Discovered {len(design_folders)} design directories in {root_dir}")

    for folder_name in design_folders:
        folder_path = os.path.join(root_dir, folder_name)
        simh_path = os.path.join(folder_path, "Surface_Dataset.simh")
        log_path  = os.path.join(folder_path, f"{folder_name}.log")

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
            normals   = np.array(face_data["NormalFieldFunction"])
            pressure  = np.array(face_data["StaticPressureFieldFunction"])
            shear     = np.linalg.norm(
                np.array(face_data["WallShearStressFieldFunction"]), axis=1
            )
            density   = np.array(face_data["DensityFieldFunction"])

        # Compute 2D face length (area)
        areas = []
        for i in range(num_faces):
            v1_idx = vertex_list[i * 2]
            v2_idx = vertex_list[i * 2 + 1]
            length = np.linalg.norm(coord[v1_idx] - coord[v2_idx])
            areas.append(length)
        areas = np.array(areas, dtype=np.float32).reshape(-1, 1)

        # Pack into tensors
        geom = torch.cat([
            torch.tensor(centroids, dtype=torch.float32),
            torch.tensor(normals, dtype=torch.float32),
            torch.tensor(areas, dtype=torch.float32)
        ], dim=1)  # [N_faces, 7]

        fields = torch.stack([
            torch.tensor(pressure, dtype=torch.float32).squeeze(),
            torch.tensor(shear, dtype=torch.float32).squeeze(),
            torch.tensor(density, dtype=torch.float32).squeeze()
        ], dim=1)  # [N_faces, 3]

        edge_index = build_face_graph_from_vertices(vertex_list, num_faces)

        dataset.append({
            'geom': geom,
            'params': params_tensor,
            'edge_index': edge_index,
            'fields': fields,
            'true_drag': log_info["true_drag"],
            'true_cd': log_info["true_cd"],
            'design_name': folder_name
        })

    print(f"Successfully processed {len(dataset)} designs into PyTorch graphs.")
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
        N_f = item['geom'].size(0)

        geom_list.append(item['geom'])
        params_list.append(item['params'].unsqueeze(0).repeat(N_f, 1))
        fields_list.append(item['fields'])

        edge_index_list.append(item['edge_index'] + node_offset)
        batch_idx_list.append(torch.full((N_f,), b_i, dtype=torch.long))

        node_offset += N_f

    return {
        'geom': torch.cat(geom_list, dim=0),
        'params': torch.cat(params_list, dim=0),
        'fields': torch.cat(fields_list, dim=0),
        'edge_index': torch.cat(edge_index_list, dim=1),
        'batch_idx': torch.cat(batch_idx_list, dim=0),
        'num_graphs': len(batch)
    }