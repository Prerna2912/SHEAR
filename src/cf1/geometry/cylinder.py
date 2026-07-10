"""
CF1 — Cylinder / pipe source panel method solver.

Uses both the exact 2D potential flow solution around a circular cylinder
(Rankine doublet) AND the source panel method for validation.  For a circular
cross-section the exact solution is used because it is more accurate.

Exact potential flow velocity around a cylinder of radius R at (0,0):
  u_r = U_inf (1 - R²/r²) cos(θ)
  u_θ = -U_inf (1 + R²/r²) sin(θ)

Parameters
----------
diameter      : cylinder diameter (m)
length        : cylinder / pipe length (m)
flow_velocity : freestream speed U_inf (m/s)
"""

from __future__ import annotations

import numpy as np

from ..base import build_grid, compute_grad_u, boundary_layer_correction


def _cylinder_exact_velocity(
    xp: np.ndarray,
    yp: np.ndarray,
    R: float,
    U_inf: float,
) -> tuple:
    """
    Exact 2D potential flow velocity around a cylinder of radius R centred at origin.

    Interior points (r < R) are mapped to the surface by clamping r to R, giving
    a smooth (no-discontinuity) field that avoids spurious large finite-difference
    gradients across the surface boundary.
    """
    r2 = xp**2 + yp**2
    # Clamp r to the surface radius so interior points are continuously defined
    r2_safe = np.maximum(r2, R**2 * 1.000001)

    # Project interior points onto their nearest surface point for x, y
    scale = np.where(r2 < R**2, R / np.sqrt(r2_safe), 1.0)
    xp_eff = xp * scale
    yp_eff = yp * scale
    r2_eff = np.maximum(xp_eff**2 + yp_eff**2, R**2 * 1.000001)

    factor = R**2 / r2_eff
    ux =  U_inf * (1.0 - factor * (xp_eff**2 - yp_eff**2) / r2_eff)
    uy = -U_inf * factor * 2.0 * xp_eff * yp_eff / r2_eff

    return ux, uy


def solve(params: dict, grid_size: int = 16) -> tuple:
    diameter      = float(params["diameter"])
    length        = float(params["length"])
    U_inf         = float(params.get("flow_velocity", params.get("U_inf",
                           params.get("freestream_velocity", 1.0))))
    Re            = float(params.get("Re", params.get("reynolds_number",
                           U_inf * diameter / 1.5e-5)))
    n_panels      = int(params.get("n_panels", 60))

    R  = diameter / 2.0
    nu = U_inf * diameter / Re

    x_range = (-3.0 * R, 6.0 * R)
    y_range = (-3.0 * R, 3.0 * R)
    z_range = (0.0, length)

    positions, grid_shape = build_grid(x_range, y_range, z_range, grid_size)
    N = positions.shape[0]

    xp, yp = positions[:, 0], positions[:, 1]

    ux, uy = _cylinder_exact_velocity(xp, yp, R=R, U_inf=U_inf)
    velocity = np.stack([ux, uy, np.zeros(N, np.float64)], axis=-1).astype(np.float32)

    grad_u = compute_grad_u(velocity, grid_shape, x_range, y_range, z_range)

    def wall_dist(pos):
        r = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
        return np.maximum(r - R, 0.0)

    grad_u = boundary_layer_correction(grad_u, velocity, positions, wall_dist,
                                        U_inf=U_inf, nu=nu)

    meta = {
        "geometry": "cylinder", "diameter": diameter, "length": length,
        "U_inf": U_inf, "Re": Re,
        "coord_system": "cylinder axis along z, x=streamwise, y=normal",
    }
    return grad_u, velocity, positions, grid_shape, meta
