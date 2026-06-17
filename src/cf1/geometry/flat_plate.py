"""
CF1 — Flat plate panel method solver.

Thin flat plate using the vortex panel method.  Because the plate has zero
thickness the Keldysh–Sedov exact solution γ(x) = 2 U_inf sin(α) sqrt((1-x)/x)
is used to set the initial guess; the panel solve then corrects for finite
plate geometry and grid effects.

Parameters
----------
length    : plate length in streamwise direction (m)
width     : plate width (span, m)
alpha_deg : angle of attack (degrees)
U_inf     : freestream speed (m/s)
Re        : Reynolds number
"""

from __future__ import annotations

import numpy as np

from ..base import build_grid, compute_grad_u, boundary_layer_correction
from ..panels import solve_vortex_panels, vortex_velocity_field


def _flat_plate_coords(length: float, n_panels: int = 60) -> tuple:
    """Cosine-spaced flat plate nodes from TE (x=L) → LE (x=0) → TE."""
    beta = np.linspace(0, np.pi, n_panels // 2 + 1)
    x_half = 0.5 * length * (1.0 - np.cos(beta))   # 0 at LE, L at TE
    y_upper = np.zeros_like(x_half)
    y_lower = np.zeros_like(x_half)

    # Upper surface: TE → LE, very slight camber bump to avoid singular geometry
    eps = length * 1e-4
    px = np.concatenate([x_half[::-1], x_half[1:]])
    py = np.concatenate([y_upper[::-1] + eps, y_lower[1:] - eps])
    return px.astype(np.float64), py.astype(np.float64)


def solve(params: dict, grid_size: int = 16) -> tuple:
    length    = float(params["length"])
    width     = float(params.get("width", params.get("span", length)))
    alpha_deg = float(params.get("alpha_deg", params.get("angle_of_attack", 5.0)))
    U_inf     = float(params.get("U_inf", params.get("freestream_velocity", 1.0)))
    Re        = float(params.get("Re", params.get("reynolds_number", 5e5)))
    n_panels  = int(params.get("n_panels", 60))

    alpha = np.deg2rad(alpha_deg)
    nu = U_inf * length / Re

    px, py = _flat_plate_coords(length, n_panels=n_panels)
    gamma = solve_vortex_panels(px, py, alpha, U_inf=U_inf)

    x_range = (-0.25 * length, 1.5 * length)
    y_range = (-0.5 * length, 0.5 * length)
    z_range = (0.0, width)

    positions, grid_shape = build_grid(x_range, y_range, z_range, grid_size)
    N = positions.shape[0]

    xp, yp = positions[:, 0], positions[:, 1]
    ux, uy = vortex_velocity_field(xp, yp, px, py, gamma, U_inf=U_inf, alpha=alpha)

    velocity = np.stack([ux, uy, np.zeros(N, np.float64)], axis=-1).astype(np.float32)
    grad_u = compute_grad_u(velocity, grid_shape, x_range, y_range, z_range)

    def wall_dist(pos):
        # Flat plate occupies y≈0, 0≤x≤L
        on_plate = (pos[:, 0] >= 0) & (pos[:, 0] <= length)
        d = np.abs(pos[:, 1])
        d[~on_plate] = np.sqrt(
            np.minimum(pos[~on_plate, 0]**2,
                       (pos[~on_plate, 0] - length)**2)
            + pos[~on_plate, 1]**2
        )
        return d

    grad_u = boundary_layer_correction(grad_u, velocity, positions, wall_dist,
                                        U_inf=U_inf, nu=nu)

    meta = {
        "geometry": "flat_plate", "length": length, "width": width,
        "alpha_deg": alpha_deg, "U_inf": U_inf, "Re": Re,
        "coord_system": "LE at origin, x=streamwise, y=normal, z=span",
    }
    return grad_u, velocity, positions, grid_shape, meta
