"""
Aerodynamic post-processing: integrates predicted pressure and shear stress over boundary faces to calculate Drag force.
"""


def _face_drag_contributions(fields_unnorm, geom_raw):
    pressure = fields_unnorm[:, 0]
    shear_x = fields_unnorm[:, 1]
    normal_x = geom_raw[:, 3]
    areas = geom_raw[:, 6]
    return (pressure * normal_x + shear_x) * areas


def compute_batch_drag(
    coeffs_norm, geom_raw, batch_idx, num_graphs, coeffs_scaler, q_dyn
):
    """Integrate normalized predicted coefficients into one differentiable drag per graph."""
    device = coeffs_norm.device
    coeffs = coeffs_norm * coeffs_scaler.std.to(device) + coeffs_scaler.mean.to(
        device
    )
    fields_phys = coeffs * q_dyn[batch_idx].unsqueeze(1)
    face_drag = _face_drag_contributions(fields_phys, geom_raw)

    pred_drags = coeffs_norm.new_zeros(num_graphs)
    pred_drags.index_add_(0, batch_idx, face_drag)
    return pred_drags


def compute_design_drag(fields_unnorm, geom_raw):
    """
    Computes total x-direction Drag force [N] by integrating over the boundary faces:
      Drag = sum((P * n_x + tau_x) * A)

    fields_unnorm: [N_faces, 2] -> (P, tau_x) in physical SI units
    geom_raw: [N_faces, 7] -> (x, y, z, nx, ny, nz, Area)
    """
    return _face_drag_contributions(fields_unnorm, geom_raw).sum()
