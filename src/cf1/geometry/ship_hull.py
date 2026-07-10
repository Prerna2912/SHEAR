"""
CF1 — Ship hull source panel method solver.

Models the submerged hull cross-section as a Joukowski-like ellipse.
The hull is slender and non-lifting; source panels handle the blockage.
A free-surface correction (Froude number effect) modifies the y-normal
velocity component above the waterline.

Parameters
----------
length  : hull length between perpendiculars (m)
beam    : beam (maximum width, m)
draft   : design draft (m)
speed   : vessel speed (m/s)
"""

from __future__ import annotations

import numpy as np

from ..base import build_grid, compute_grad_u, boundary_layer_correction
from ..panels import solve_source_panels, source_velocity_field


def _hull_cross_section(a: float, b: float, n_panels: int = 60) -> tuple:
    """
    Semi-elliptic hull cross-section (submerged half only, below waterline y=0).
    """
    theta = np.linspace(np.pi, 2 * np.pi, n_panels + 1)   # lower half
    px = a * np.cos(theta)
    py = b * np.sin(theta)
    # Close the waterline with a flat deck
    px = np.concatenate([px, px[::-1]])
    py = np.concatenate([py, np.zeros_like(py)])
    # Deduplicate endpoints (simple approach: full ellipse for panel method)
    theta_full = np.linspace(0, 2 * np.pi, n_panels + 1)
    px_full = a * np.cos(theta_full)
    py_full = b * np.sin(theta_full)
    return px_full.astype(np.float64), py_full.astype(np.float64)


def solve(params: dict, grid_size: int = 16) -> tuple:
    length = float(params["length"])
    beam   = float(params["beam"])
    draft  = float(params["draft"])
    speed  = float(params.get("speed", params.get("U_inf", 10.0)))
    Re     = float(params.get("Re", params.get("reynolds_number",
                    speed * length / 1.004e-6)))   # water kinematic viscosity
    n_panels = int(params.get("n_panels", 60))

    nu = speed * length / Re

    # Use the LOCAL cross-section (midship) dimensions, not hull length, for the
    # 2D panel solve — avoids extreme aspect ratios that make the influence matrix
    # ill-conditioned.  The full hull length sets the downstream wake grid extent.
    a_panel = beam / 2.0    # midship half-beam
    b_panel = draft         # keel-to-waterline
    Fn = speed / np.sqrt(9.81 * length)     # Froude number

    px, py = _hull_cross_section(a_panel, b_panel, n_panels=n_panels)
    sigma = solve_source_panels(px, py, alpha=0.0, U_inf=speed)

    # Grid centred on midship cross-section
    x_range = (-2.0 * a_panel, 4.0 * a_panel)
    y_range = (-2.0 * b_panel, b_panel)       # below keel to waterline
    z_range = (0.0, length)

    positions, grid_shape = build_grid(x_range, y_range, z_range, grid_size)
    N = positions.shape[0]

    xp, yp = positions[:, 0], positions[:, 1]
    ux, uy = source_velocity_field(xp, yp, px, py, sigma, U_inf=speed, alpha=0.0)

    # Free-surface effect at high Froude number: wave-induced vertical velocity
    y_norm = np.clip(positions[:, 1] / (b_panel + 1e-8), -1.0, 1.0)
    uy_fs = uy * (1.0 + 0.5 * Fn**2 * np.exp(-np.abs(y_norm)))  # shallow correction

    velocity = np.stack([ux, uy_fs, np.zeros(N, np.float64)], axis=-1).astype(np.float32)
    grad_u = compute_grad_u(velocity, grid_shape, x_range, y_range, z_range)

    def wall_dist(pos):
        x_, y_ = pos[:, 0], pos[:, 1]
        theta_p = np.arctan2(y_ / (b_panel + 1e-12), x_ / (a_panel + 1e-12))
        xs = a_panel * np.cos(theta_p)
        ys = b_panel * np.sin(theta_p)
        return np.sqrt((x_ - xs)**2 + (y_ - ys)**2)

    grad_u = boundary_layer_correction(grad_u, velocity, positions, wall_dist,
                                        U_inf=speed, nu=nu)

    meta = {
        "geometry": "ship_hull", "length": length, "beam": beam, "draft": draft,
        "speed": speed, "Fn": float(Fn), "Re": Re,
        "coord_system": "bow at origin, x=forward, y=height (keel=-draft), z=port",
    }
    return grad_u, velocity, positions, grid_shape, meta
