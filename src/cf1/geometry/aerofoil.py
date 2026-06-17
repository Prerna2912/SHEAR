"""
CF1 — Aerofoil panel method solver.

Geometry: NACA 4-digit series airfoil (or symmetric NACA 00xx by default).
Method  : 2D vortex panel method → 3D via lifting-line spanwise correction.

Parameters
----------
chord     : chord length (m)
span      : wing span (m)
alpha_deg : angle of attack (degrees)
U_inf     : freestream speed (m/s)
Re        : Reynolds number
naca_code : 4-digit NACA code string (default '2412')
n_panels  : number of 2D panels (default 80)
grid_size : grid resolution per side (default 16)
"""

from __future__ import annotations

import numpy as np

from ..base import build_grid, compute_grad_u, boundary_layer_correction
from ..panels import naca4_coords, solve_vortex_panels, vortex_velocity_field


def solve(params: dict, grid_size: int = 16) -> tuple:
    """
    Generate ∇u field around a NACA aerofoil.

    Returns:
        grad_u   : [N, 3, 3] velocity gradient.
        velocity : [N, 3]    velocity field.
        positions: [N, 3]    grid coordinates.
        grid_shape: (Nx, Ny, Nz).
        meta     : dict of diagnostics.
    """
    chord     = float(params["chord"])
    span      = float(params["span"])
    alpha_deg = float(params.get("alpha_deg", params.get("angle_of_attack", 5.0)))
    U_inf     = float(params.get("U_inf", params.get("freestream_velocity", 1.0)))
    Re        = float(params.get("Re", params.get("reynolds_number", 1e6)))
    naca_code = str(params.get("naca_code", "2412"))
    n_panels  = int(params.get("n_panels", 80))

    alpha = np.deg2rad(alpha_deg)
    nu = U_inf * chord / Re

    # --- NACA profile --------------------------------------------------------
    m = int(naca_code[0]) / 100.0
    p = int(naca_code[1]) / 10.0
    t = int(naca_code[2:]) / 100.0
    px, py = naca4_coords(m, p, t, n_panels=n_panels, chord=chord)

    # --- Solve 2D vortex panel -----------------------------------------------
    gamma = solve_vortex_panels(px, py, alpha, U_inf=U_inf)

    # --- Build 3D grid (body-centred, LE at origin) --------------------------
    # x: streamwise [-0.5c, 1.5c], y: normal [-0.5c, 0.5c], z: spanwise [0, span]
    x_range = (-0.5 * chord, 1.5 * chord)
    y_range = (-0.5 * chord, 0.5 * chord)
    z_range = (0.0, span)

    positions, grid_shape = build_grid(x_range, y_range, z_range, grid_size)
    Nx, Ny, Nz = grid_shape
    N = Nx * Ny * Nz

    # --- Evaluate 2D velocity on xy slice (z-independent for infinite span) --
    xp = positions[:, 0]   # [N]
    yp = positions[:, 1]   # [N]

    ux, uy = vortex_velocity_field(xp, yp, px, py, gamma, U_inf=U_inf, alpha=alpha)

    # Spanwise: apply Prandtl lifting-line downwash correction
    # Simple finite-span correction: CL = CL_2d * AR / (AR + 2)
    AR = span / chord
    cl_2d = 2.0 * np.pi * alpha                     # thin-airfoil theory
    cl_3d = cl_2d * AR / (AR + 2.0)
    span_correction = cl_3d / (cl_2d + 1e-12)        # multiplicative scale

    # Apply correction to normal component (y) only, tapered toward wingtips
    z = positions[:, 2]
    z_norm = 2.0 * z / (span + 1e-8) - 1.0           # -1..1
    taper = np.sqrt(np.clip(1.0 - z_norm**2, 0.0, 1.0))   # elliptic taper

    uy_corrected = uy * (span_correction * taper + (1.0 - taper))
    uz = np.zeros(N, dtype=np.float32)

    velocity = np.stack([ux, uy_corrected, uz], axis=-1).astype(np.float32)

    # --- Compute ∇u via finite differences -----------------------------------
    grad_u = compute_grad_u(velocity, grid_shape, x_range, y_range, z_range)

    # --- Boundary layer correction -------------------------------------------
    x_panel = np.array(px)
    y_panel = np.array(py)

    def wall_dist(pos: np.ndarray) -> np.ndarray:
        xp_, yp_ = pos[:, 0], pos[:, 1]
        # Min distance to any panel midpoint (approximate)
        xmid = 0.5 * (x_panel[:-1] + x_panel[1:])
        ymid = 0.5 * (y_panel[:-1] + y_panel[1:])
        d2 = (xp_[:, None] - xmid[None, :])**2 + (yp_[:, None] - ymid[None, :])**2
        return np.sqrt(d2.min(axis=1))

    grad_u = boundary_layer_correction(
        grad_u, velocity, positions, wall_dist,
        U_inf=U_inf, nu=nu,
    )

    meta = {
        "geometry": "aerofoil",
        "naca_code": naca_code,
        "chord": chord,
        "span": span,
        "alpha_deg": alpha_deg,
        "U_inf": U_inf,
        "Re": Re,
        "AR": AR,
        "n_panels": n_panels,
        "coord_system": "LE at origin, x=streamwise, y=normal, z=spanwise",
    }

    return grad_u, velocity, positions, grid_shape, meta
