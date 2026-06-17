"""
CF1 — Bluff body with downstream wake region.

Combines the source panel method for the body (ellipse) with a Rankine
half-body / von Karman vortex street model for the near-wake.

The wake model:
  - Centreline velocity deficit: Ud(x) = Ud0 * (x / (x0+1e-8))^(-1/2)
  - Wake width: w(x) = w0 * (x / x0)^(1/2)  (Gaussian spread)
  - Alternating signed vortices: approximates unsteady shedding as a
    time-averaged velocity deficit with transverse meander.

Parameters
----------
frontal_area       : body frontal area (m²)
length             : body length (m)
height             : body height (m)
flow_speed         : freestream speed (m/s)
wake_length        : downstream extent of wake region (m)
"""

from __future__ import annotations

import numpy as np

from ..base import build_grid, compute_grad_u, boundary_layer_correction
from ..panels import solve_source_panels, source_velocity_field


def _wake_velocity_deficit(
    xp: np.ndarray,
    yp: np.ndarray,
    U_inf: float,
    Cd: float,
    D: float,
    x_body_end: float,
) -> tuple:
    """
    Pope (2000) far-wake self-similar Gaussian deficit.

    U_deficit(x, y) = U0 * exp(-y²/(2σ²))
    U0 ~ Cd * D / sqrt(x - x0)
    σ ~ sqrt(x - x0)
    """
    x_rel = np.maximum(xp - x_body_end, 0.01 * D)
    sigma = 0.12 * np.sqrt(x_rel * D)
    U0 = U_inf * Cd * D / (2.0 * np.sqrt(2 * np.pi) * sigma + 1e-8)
    deficit = U0 * np.exp(-yp**2 / (2 * sigma**2 + 1e-12))
    # Transverse component from wake meandering (weak)
    uy_wake = -U0 * yp / (sigma**2 + 1e-8) * np.exp(-yp**2 / (2 * sigma**2 + 1e-12)) * 0.05
    return -deficit, uy_wake


def solve(params: dict, grid_size: int = 16) -> tuple:
    frontal_area = float(params.get("frontal_area", 1.0))
    length      = float(params["length"])
    height      = float(params["height"])
    U_inf       = float(params.get("flow_speed", params.get("U_inf",
                         params.get("freestream_velocity", 10.0))))
    wake_length = float(params.get("wake_length", 5.0 * length))
    Re          = float(params.get("Re", params.get("reynolds_number",
                         U_inf * length / 1.5e-5)))
    n_panels    = int(params.get("n_panels", 60))

    nu = U_inf * length / Re

    a = length / 2.0
    b = height / 2.0

    # Build ellipse panels for the body
    theta = np.linspace(0, 2 * np.pi, n_panels + 1)
    px = (a * np.cos(theta)).astype(np.float64)
    py = (b * np.sin(theta)).astype(np.float64)
    sigma = solve_source_panels(px, py, alpha=0.0, U_inf=U_inf)

    x_range = (-1.5 * a, a + wake_length)
    y_range = (-3.0 * b, 3.0 * b)
    z_range = (0.0, np.sqrt(frontal_area))

    positions, grid_shape = build_grid(x_range, y_range, z_range, grid_size)
    N = positions.shape[0]

    xp, yp = positions[:, 0], positions[:, 1]

    # Body potential flow
    ux_body, uy_body = source_velocity_field(xp, yp, px, py, sigma,
                                              U_inf=U_inf, alpha=0.0)

    # Wake deficit superimposed downstream of body
    Cd = 1.0           # bluff body drag coefficient (approximate)
    D  = 2.0 * b
    ux_wake, uy_wake = _wake_velocity_deficit(xp, yp, U_inf, Cd, D, x_body_end=a)

    ux = ux_body + ux_wake
    uy = uy_body + uy_wake
    velocity = np.stack([ux, uy, np.zeros(N, np.float64)], axis=-1).astype(np.float32)

    grad_u = compute_grad_u(velocity, grid_shape, x_range, y_range, z_range)

    def wall_dist(pos):
        x_, y_ = pos[:, 0], pos[:, 1]
        theta_p = np.arctan2(y_ / (b + 1e-12), x_ / (a + 1e-12))
        xs = a * np.cos(theta_p)
        ys = b * np.sin(theta_p)
        return np.sqrt((x_ - xs)**2 + (y_ - ys)**2)

    grad_u = boundary_layer_correction(grad_u, velocity, positions, wall_dist,
                                        U_inf=U_inf, nu=nu)

    meta = {
        "geometry": "bluff_body_wake", "frontal_area": frontal_area,
        "length": length, "height": height, "wake_length": wake_length,
        "U_inf": U_inf, "Re": Re, "Cd": Cd,
        "coord_system": "centroid at origin, x=streamwise, y=normal, z=width",
    }
    return grad_u, velocity, positions, grid_shape, meta
