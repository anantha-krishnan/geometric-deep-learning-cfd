"""
Aerodynamic post-processing: integrates predicted pressure and shear stress over boundary faces to calculate Drag force.
"""

import torch

def compute_design_drag(fields_unnorm, geom_raw, drag_dir=torch.tensor([1.0, 0.0, 0.0])):
    """
    Computes total Drag force [N] by integrating over the boundary faces:
      Drag = sum(P * A * (n . d) + tau * A * (t . d))

    fields_unnorm: [N_faces, 3] -> (P, tau, rho) in physical SI units
    geom_raw: [N_faces, 7] -> (x, y, z, nx, ny, nz, Area)
    """
    P_f = fields_unnorm[:, 0]
    tau_f = fields_unnorm[:, 1]
    normals = geom_raw[:, 3:6]
    areas = geom_raw[:, 6]

    device = fields_unnorm.device
    drag_dir = drag_dir.to(device)

    # 1. Pressure Drag Force
    n_dot_d = torch.matmul(normals, drag_dir)
    f_pressure = torch.sum(P_f * areas * n_dot_d)

    # 2. Viscous Friction Drag Force
    # Tangent vector orthogonal to normal in xy plane: (-ny, nx, 0)
    tangents = torch.stack([-normals[:, 1], normals[:, 0], torch.zeros_like(normals[:, 0])], dim=1)
    t_dot_d = torch.matmul(tangents, drag_dir)
    f_viscous = torch.sum(tau_f * areas * t_dot_d)

    return f_pressure + f_viscous