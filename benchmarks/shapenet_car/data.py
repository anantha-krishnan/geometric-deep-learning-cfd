"""Load ShapeNet-Car CFD data using Transolver's feature conventions."""

import tarfile
from pathlib import Path, PurePosixPath

import numpy as np
import torch
import vtk
from scipy import spatial
from vtkmodules.util.numpy_support import vtk_to_numpy


def build_case(volume_points, velocity, surface_points, pressure, surface_normals):
    volume_points = np.asarray(volume_points)
    velocity = np.asarray(velocity)
    surface_points = np.asarray(surface_points)
    pressure = np.asarray(pressure).reshape(-1)
    surface_normals = np.asarray(surface_normals)
    if (
        volume_points.ndim != 2
        or volume_points.shape[1] != 3
        or velocity.shape != volume_points.shape
        or surface_points.ndim != 2
        or surface_points.shape[1] != 3
        or surface_normals.shape != surface_points.shape
        or len(pressure) != len(surface_points)
        or not len(surface_points)
        or not len(volume_points)
    ):
        raise ValueError("Invalid ShapeNet-Car point or field shapes.")
    if not all(
        np.isfinite(values).all()
        for values in (
            volume_points,
            velocity,
            surface_points,
            pressure,
            surface_normals,
        )
    ):
        raise ValueError("ShapeNet-Car inputs and targets must be finite.")

    surface_lookup = {tuple(point): index for index, point in enumerate(surface_points)}
    exterior = np.array([tuple(point) not in surface_lookup for point in volume_points])
    exterior_points = volume_points[exterior]
    tree_type = getattr(spatial, "cKDTree", None)
    if tree_type is None:
        raise ImportError("SciPy spatial cKDTree is required for ShapeNet-Car.")
    distances, nearest = tree_type(surface_points).query(exterior_points)
    directions = (exterior_points - surface_points[nearest]) / (
        distances[:, None] + 1e-8
    )
    velocity_lookup = {
        tuple(point): value for point, value in zip(volume_points, velocity)
    }
    surface_velocity = np.array(
        [velocity_lookup.get(tuple(point), np.zeros(3)) for point in surface_points]
    )
    positions = np.concatenate((exterior_points, surface_points))
    surface = np.arange(len(positions)) >= len(exterior_points)
    distances = np.concatenate((distances, np.zeros(len(surface_points))))
    directions = np.concatenate((directions, surface_normals))
    targets = np.column_stack(
        (
            np.concatenate((velocity[exterior], surface_velocity)),
            np.concatenate((np.zeros(len(exterior_points)), pressure)),
        )
    )
    return {
        "pos": positions.astype(np.float32),
        "x": np.column_stack((positions, distances, directions)).astype(np.float32),
        "y": targets.astype(np.float32),
        "surface": surface,
    }


def _read_case_files(root, name):
    parts = PurePosixPath(name).parts
    if (
        len(parts) != 2
        or parts[0] not in {f"param{index}" for index in range(9)}
        or not parts[1].replace("_", "").isalnum()
    ):
        raise ValueError(f"Invalid ShapeNet-Car case name: {name}")
    files = ("quadpress_smpl.vtk", "hexvelo_smpl.vtk", "cd.txt")
    root = Path(root)
    if (root / parts[0]).is_dir():
        return {
            file: (root / parts[0] / parts[1] / file).read_bytes() for file in files
        }
    with tarfile.open(root / f"{parts[0]}.tar.gz", "r:gz") as archive:
        result = {}
        for file in files:
            member = archive.getmember(f"{parts[0]}/{parts[1]}/{file}")
            if not member.isfile():
                raise ValueError(f"Expected a regular file: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read archive member: {member.name}")
            result[file] = stream.read()
        return result


def _read_grid(content):
    reader = vtk.vtkUnstructuredGridReader()
    reader.ReadFromInputStringOn()
    reader.SetInputString(content)
    reader.Update()
    grid = reader.GetOutput()
    if not grid.GetNumberOfPoints() or not grid.GetNumberOfCells():
        raise ValueError("Invalid or empty ShapeNet-Car VTK grid.")
    return grid


def _surface_normals(grid):
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(grid)
    surface_filter.Update()
    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.SetInputData(surface_filter.GetOutput())
    normal_filter.SetAutoOrientNormals(1)
    normal_filter.SetConsistency(1)
    normal_filter.SetComputeCellNormals(1)
    normal_filter.SetComputePointNormals(0)
    normal_filter.Update()
    grid.GetCellData().SetNormals(normal_filter.GetOutput().GetCellData().GetNormals())
    point_filter = vtk.vtkCellDataToPointData()
    point_filter.SetInputData(grid)
    point_filter.Update()
    normals = vtk_to_numpy(point_filter.GetOutput().GetPointData().GetNormals()).astype(
        np.float64
    )
    normals /= np.max(np.abs(normals), axis=1, keepdims=True) + 1e-8
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    return normals


def _mesh_edges(grids, positions):
    indices = {tuple(point): index for index, point in enumerate(positions)}
    edge_parts = []
    for grid, cell_size in grids:
        points = vtk_to_numpy(grid.GetPoints().GetData())
        cell_array = grid.GetCells()
        offsets = vtk_to_numpy(cell_array.GetOffsetsArray())
        if np.any(np.diff(offsets) != cell_size):
            raise ValueError("Unexpected ShapeNet-Car mesh cell size.")
        cells = vtk_to_numpy(cell_array.GetConnectivityArray()).reshape(-1, cell_size)
        mapping = np.array([indices[tuple(point)] for point in points], dtype=np.int64)
        nodes = mapping[cells]
        edge_parts.append(
            np.column_stack(
                (
                    np.repeat(nodes, cell_size, axis=1).ravel(),
                    np.tile(nodes, (1, cell_size)).ravel(),
                )
            )
        )
    edges = np.unique(np.concatenate(edge_parts), axis=0)
    return torch.as_tensor(edges.T.copy(), dtype=torch.long)


def load_case(root, name):
    files = _read_case_files(root, name)
    surface_grid = _read_grid(files["quadpress_smpl.vtk"])
    volume_grid = _read_grid(files["hexvelo_smpl.vtk"])
    pressure = surface_grid.GetPointData().GetScalars()
    velocity = volume_grid.GetPointData().GetVectors()
    if pressure is None or velocity is None:
        raise ValueError(f"Missing pressure or velocity fields in {name}.")
    case = build_case(
        vtk_to_numpy(volume_grid.GetPoints().GetData()),
        vtk_to_numpy(velocity),
        vtk_to_numpy(surface_grid.GetPoints().GetData()),
        vtk_to_numpy(pressure),
        _surface_normals(surface_grid),
    )
    case["edge_index"] = _mesh_edges(((surface_grid, 4), (volume_grid, 8)), case["pos"])
    case["name"] = name
    case["true_cd"] = float(files["cd.txt"].decode().strip())
    return case
