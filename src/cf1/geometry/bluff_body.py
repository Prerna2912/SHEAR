"""
CF1 — Bluff body / vehicle source panel method solver.

Models the body as an ellipse-like contour using the source panel method.
An elliptic cross-section with aspect ratio (height/length) captures the
main blockage and near-wake pressure gradient.

Parameters
----------
frontal_area : frontal area A (m²)  → width = sqrt(A * height / length)
length       : body length (m)
height       : body height (m)
flow_speed   : freestream speed U_inf (m/s)
"""

from __future__ import annotations

import numpy as np

from ..base import build_grid, compute_grad_u, boundary_layer_correction
from ..panels import solve_source_panels, source_velocity_field


def _ellipse_coords(a: float, b: float, n_panels: int = 60) -> tuple:
    """
    Generate ellipse panel nodes.

    Args:
        a : semi-axis along x (streamwise half-length).
        b : semi-axis along y (half-height).
    """
    theta = np.linspace(0, 2 * np.pi, n_panels + 1)
    # Bunching at the stagnation points
    px = a * np.cos(theta)
    py = b * np.sin(theta)
    return px.astype(np.float64), py.astype(np.float64)


def solve(params: dict, grid_size: int = 16) -> tuple:
    frontal_area = float(params.get("frontal_area", params.get("width", 1.0) * params.get("height", 0.5)))
    length  = float(params["length"])
    height  = float(params["height"])
    U_inf   = float(params.get("flow_speed", params.get("U_inf", params.get("freestream_velocity", 10.0))))
    Re      = float(params.get("Re", params.get("reynolds_number", U_inf * length / 1.5e-5)))
    n_panels= int(params.get("n_panels", 60))

    nu = U_inf * length / Re

    a = length / 2.0          # semi-length
    b = height / 2.0          # semi-height

    px, py = _ellipse_coords(a, b, n_panels=n_panels)
    sigma = solve_source_panels(px, py, alpha=0.0, U_inf=U_inf)

    x_range = (-1.5 * a, 3.0 * a)      # upstream + downstream wake
    y_range = (-2.0 * b, 2.0 * b)
    z_range = (0.0, np.sqrt(frontal_area))   # span proxy from frontal area

    positions, grid_shape = build_grid(x_range, y_range, z_range, grid_size)
    N = positions.shape[0]

    xp, yp = positions[:, 0], positions[:, 1]
    ux, uy = source_velocity_field(xp, yp, px, py, sigma, U_inf=U_inf, alpha=0.0)

    velocity = np.stack([ux, uy, np.zeros(N, np.float64)], axis=-1).astype(np.float32)
    grad_u = compute_grad_u(velocity, grid_shape, x_range, y_range, z_range)

    def wall_dist(pos):
        # Ellipse surface distance
        xp_, yp_ = pos[:, 0], pos[:, 1]
        # Approximate wall distance as distance from nearest surface point
        theta_p = np.arctan2(yp_ / (b + 1e-12), xp_ / (a + 1e-12))
        xs = a * np.cos(theta_p)
        ys = b * np.sin(theta_p)
        return np.sqrt((xp_ - xs)**2 + (yp_ - ys)**2)

    grad_u = boundary_layer_correction(grad_u, velocity, positions, wall_dist,
                                        U_inf=U_inf, nu=nu)

    meta = {
        "geometry": "bluff_body", "frontal_area": frontal_area,
        "length": length, "height": height, "U_inf": U_inf, "Re": Re,
        "coord_system": "centroid at origin, x=streamwise, y=height, z=width",
    }
    return grad_u, velocity, positions, grid_shape, meta
