"""Discretisation- and scale-invariant model inputs for surface and volume CFD graphs.

Three length scales are separated:
* body (L_ref): streamwise extent of the WALL boundary, never the fluid domain;
  positions and wall distance are expressed in body lengths about the body centre;
* boundary layer: wall distance d / L_ref plus its logarithm to resolve thin layers;
* mesh (h = cell measure ** (1 / manifold_dim)): appears only as neighbour ratios on
  edges, so node inputs carry no absolute cell size.

The same code handles surface meshes (faces on the body, manifold_dim = spatial - 1)
and volume meshes (fluid cells, manifold_dim = spatial), in 2D and 3D. Loaders tag
items with ``mesh_kind``, ``body_bounds`` and ``domain_bounds``; ``make_invariant``
dispatches on them.

Position frames: ``body`` scales positions by L_ref about the body centre (valid for
unbounded flows); ``domain`` maps positions to [0, 1] over the fluid-domain box, for
confined flows whose walls and outlet do not scale with the body. In the domain frame
the body/domain extent ratios (blockage) are added as design inputs.
"""

import math

import torch

FLOW_AXIS = 0
SPEED_INDEX = 2
# Near-body band in body lengths; equals 0.05 m at the dataset's median length (0.33 m).
NEAR_WALL_RELATIVE = 0.15
# Wall-distance floor (body lengths) for the log feature; below the finest cell.
WALL_DISTANCE_FLOOR = 1e-6
FRAMES = ("body", "domain")
_AXIS_NAMES = ("x", "y", "z")


def bounding_box(points):
    """Axis-aligned [min, max] box (2 x 3) of a point set."""
    points = torch.as_tensor(points, dtype=torch.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("Bounding-box points must be a non-empty [N, 3] array.")
    return torch.stack([points.min(dim=0).values, points.max(dim=0).values]).float()


def active_axes(position, tolerance=1e-9):
    """Axes along which the mesh actually extends (drops z for planar 2D meshes)."""
    extent = position.max(dim=0).values - position.min(dim=0).values
    axes = (extent > tolerance * extent.max().clamp_min(1e-30)).nonzero().flatten()
    if FLOW_AXIS not in axes.tolist():
        raise ValueError("The mesh does not extend along the flow axis.")
    return axes


def body_reference(bounds, axes):
    """Reference length, centre and aspect ratios (other extents / length) of the body."""
    extent = (bounds[1] - bounds[0])[axes]
    flow = axes.tolist().index(FLOW_AXIS)
    length = float(extent[flow])
    if not math.isfinite(length) or length <= 0:
        raise ValueError("Body must have a positive streamwise extent.")
    centre = (0.5 * (bounds[0] + bounds[1]))[axes]
    aspect = torch.cat([extent[:flow], extent[flow + 1 :]]) / length
    return length, centre, aspect


def cell_size(measure, manifold_dim):
    """Characteristic cell length h: face length, sqrt(area) or cbrt(volume)."""
    return measure.clamp_min(1e-30).pow(1.0 / manifold_dim)


def invariant_edge_attr(position, measure, edge_index, manifold_dim, normals=None):
    """[unit direction, (normal change), distance / mean h, log measure ratio]."""
    source, destination = edge_index
    displacement = position[destination] - position[source]
    distance = displacement.norm(dim=1, keepdim=True).clamp_min(1e-30)
    size = cell_size(measure, manifold_dim)
    mean_size = (0.5 * (size[source] + size[destination])).unsqueeze(1)
    ratio = measure[destination].clamp_min(1e-30) / measure[source].clamp_min(1e-30)
    parts = [displacement / distance]
    if normals is not None:
        parts.append(normals[destination] - normals[source])
    parts += [distance / mean_size, ratio.log().unsqueeze(1)]
    return torch.cat(parts, dim=1)


def _split_geometry(item):
    geom = item["geom"]
    if item.get("mesh_kind") == "surface":
        return geom[:, :3], geom[:, 6], geom[:, 3:6], None
    if item.get("mesh_kind") == "volume":
        if geom.shape[1] < 5:
            raise ValueError("Volume invariance needs the wall-distance column.")
        return geom[:, :3], geom[:, 3], None, geom[:, 4]
    raise ValueError(f"Unknown mesh kind: {item.get('mesh_kind')!r}")


def _domain_frame(item, axes):
    """Lower corner and extents of the fluid domain along the active axes."""
    if "domain_bounds" not in item:
        raise ValueError(f"{item.get('design_name')} has no domain bounds.")
    bounds = item["domain_bounds"]
    extent = (bounds[1] - bounds[0])[axes]
    if not (extent > 0).all():
        raise ValueError("Domain must have a positive extent along every active axis.")
    return bounds[0][axes], extent


def make_invariant(item, frame="body"):
    """Return a copy of a loaded item with invariant node, design and edge inputs.

    ``geom`` and ``params`` stay physical for loss weights, drag and plotting; models
    read ``node_features``/``design_features`` instead.
    """
    if frame not in FRAMES:
        raise ValueError(f"Unknown position frame: {frame!r}")
    if "body_bounds" not in item:
        raise ValueError(f"{item.get('design_name')} has no body bounds.")
    position, measure, normals, wall_distance = _split_geometry(item)
    axes = active_axes(position)
    length, centre, aspect = body_reference(item["body_bounds"], axes)
    position = position[:, axes]
    surface = normals is not None
    manifold_dim = len(axes) - 1 if surface else len(axes)
    extra_design = [aspect]
    if frame == "domain":
        low, extent = _domain_frame(item, axes)
        nodes = [(position - low) / extent]
        body_extent = (item["body_bounds"][1] - item["body_bounds"][0])[axes]
        extra_design.append(_without_flow_axis(body_extent / extent, axes))
    else:
        nodes = [(position - centre) / length]
    if surface:
        normals = normals[:, axes]
        nodes.append(normals)
    else:
        relative = wall_distance / length
        nodes += [
            relative.unsqueeze(1),
            (relative + WALL_DISTANCE_FLOOR).log().unsqueeze(1),
        ]
    speed = float(item["params"][SPEED_INDEX])
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("Invariant inputs need a positive inlet speed.")
    design = torch.cat(
        [item["params"], torch.tensor([math.log(speed * length)]), *extra_design]
    ).float()
    result = {
        **item,
        "length_ref": length,
        "node_features": torch.cat(nodes, dim=1).float(),
        "design_features": design,
        "edge_attr": invariant_edge_attr(
            position, measure, item["edge_index"], manifold_dim, normals
        ).float(),
    }
    if wall_distance is not None:
        result["wall_distance_ref"] = (wall_distance / length).float()
    return result


def _without_flow_axis(values, axes):
    flow = axes.tolist().index(FLOW_AXIS)
    return torch.cat([values[:flow], values[flow + 1 :]])


def feature_names(item, frame="body"):
    """Names of node, design and edge inputs produced by ``make_invariant``."""
    names = [_AXIS_NAMES[axis] for axis in active_axes(_split_geometry(item)[0])]
    surface = item["mesh_kind"] == "surface"
    suffix = "_over_domain" if frame == "domain" else "_over_L"
    nodes = [f"{name}{suffix}" for name in names]
    nodes += (
        [f"n{name}" for name in names]
        if surface
        else [
            "wall_distance_over_L",
            "log_wall_distance_over_L",
        ]
    )
    design = ["P_D1_r", "P_D2_r", "P_V", "log_speed_times_L"]
    cross_axes = [name for name in names if name != _AXIS_NAMES[FLOW_AXIS]]
    design += [f"{name}_extent_over_L" for name in cross_axes]
    if frame == "domain":
        design += [f"{name}_blockage" for name in cross_axes]
    edges = [f"direction_{name}" for name in names]
    edges += [f"normal_change_{name}" for name in names] if surface else []
    edges += ["distance_over_mean_h", "log_measure_ratio"]
    return {"node": nodes, "design": design, "edge": edges}


def reference_length(item):
    """Body streamwise length from loader bounds, or None when unavailable."""
    if "length_ref" in item:
        return item["length_ref"]
    if "body_bounds" not in item:
        return None
    bounds = item["body_bounds"]
    return float(bounds[1, FLOW_AXIS] - bounds[0, FLOW_AXIS])


def relative_edge_attr(position, edge_index, length):
    """[unit direction, log(distance / L_ref)]: no cell measures on edges."""
    source, destination = edge_index
    displacement = position[destination] - position[source]
    distance = displacement.norm(dim=1, keepdim=True).clamp_min(1e-30)
    return torch.cat([displacement / distance, (distance / length).log()], dim=1)


def continuous_inputs(item):
    """Continuous-coordinate inputs: nodes [position, wall distance], no cell size.

    Positions and wall distance stay in metres (standardised later); edges carry the
    direction and the log of the neighbour distance in body lengths.
    """
    if item.get("mesh_kind") != "volume" or item["geom"].shape[1] < 5:
        raise ValueError("Continuous inputs need a volume item with wall distance.")
    length = reference_length(item)
    if length is None or not math.isfinite(length) or length <= 0:
        raise ValueError(f"{item.get('design_name')} has no positive body length.")
    position = item["geom"][:, :3]
    position = position[:, active_axes(position)]
    return {
        **item,
        "length_ref": length,
        "node_features": torch.cat([position, item["geom"][:, 4:5]], dim=1).float(),
        "design_features": item["params"].float(),
        "edge_attr": relative_edge_attr(position, item["edge_index"], length).float(),
    }
