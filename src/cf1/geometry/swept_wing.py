"""
CF1 — Swept wing panel method solver.

Applies a sweep-corrected 2D vortex panel method:
  V_eff = U_inf * cos(sweep_angle)   (component normal to leading edge)
Dihedral is handled as a y-axis rotation of the outboard sections.

Parameters
----------
chord        : chord length (m)
span         : wing span (m)
sweep_angle  : sweep angle (degrees, leading-edge sweep)
dihedral     : dihedral angle (degrees)
alpha_deg    : angle of attack (degrees)
U_inf        : freestream speed (m/s)
Re           : Reynolds number
naca_code    : 4-digit NACA code (default '2412')
"""

from __future__ import annotations

import numpy as np

from ..base import build_grid, compute_grad_u, boundary_layer_correction
from ..panels import naca4_coords, solve_vortex_panels, vortex_velocity_field


def solve(params: dict, grid_size: int = 16) -> tuple:
    chord       = float(params["chord"])
    span        = float(params["span"])
    sweep_deg   = float(params.get("sweep_angle", 25.0))
    dihedral_deg= float(params.get("dihedral", 5.0))
    alpha_deg   = float(params.get("alpha_deg", params.get("angle_of_attack", 4.0)))
    U_inf       = float(params.get("U_inf", params.get("freestream_velocity", 1.0)))
    Re          = float(params.get("Re", params.get("reynolds_number", 5e6)))
    naca_code   = str(params.get("naca_code", "2412"))
    n_panels    = int(params.get("n_panels", 80))

    alpha       = np.deg2rad(alpha_deg)
    sweep       = np.deg2rad(sweep_deg)
    dihedral    = np.deg2rad(dihedral_deg)
    nu          = U_inf * chord / Re

    # Effective freestream normal to the leading edge (sweep correction)
    V_eff = U_inf * np.cos(sweep)

    m = int(naca_code[0]) / 100.0
    p = int(naca_code[1]) / 10.0
    t = int(naca_code[2:]) / 100.0
    px, py = naca4_coords(m, p, t, n_panels=n_panels, chord=chord)

    gamma = solve_vortex_panels(px, py, alpha, U_inf=V_eff)

    # Grid: slightly larger than aerofoil case to capture swept wake
    x_range = (-0.5 * chord, 2.0 * chord)
    y_range = (-0.5 * chord, 0.5 * chord)
    z_range = (0.0, span)

    positions, grid_shape = build_grid(x_range, y_range, z_range, grid_size)
    Nx, Ny, Nz = grid_shape
    N = Nx * Ny * Nz

    xp = positions[:, 0]
    yp = positions[:, 1]
    z  = positions[:, 2]

    ux_2d, uy_2d = vortex_velocity_field(xp, yp, px, py, gamma, U_inf=V_eff, alpha=alpha)

    # The spanwise (z) component induced by sweep: U_inf * sin(sweep)
    uz_sweep = U_inf * np.sin(sweep) * np.ones(N, dtype=np.float64)

    # Dihedral: rotate uy → uy * cos(dihedral), uz += uy * sin(dihedral)
    uy_dih = uy_2d * np.cos(dihedral)
    uz_dih = uy_2d * np.sin(dihedral)

    # Elliptic span loading taper
    AR = span / chord
    cl_2d = 2.0 * np.pi * alpha
    cl_3d = cl_2d * AR / (AR + 2.0)
    taper = np.sqrt(np.clip(1.0 - (2.0 * z / (span + 1e-8) - 1.0)**2, 0.0, 1.0))
    uy_final = uy_dih * (cl_3d / (cl_2d + 1e-12)) * taper + uy_dih * (1.0 - taper)

    velocity = np.stack([
        ux_2d,
        uy_final,
        uz_sweep + uz_dih,
    ], axis=-1).astype(np.float32)

    grad_u = compute_grad_u(velocity, grid_shape, x_range, y_range, z_range)

    x_panel = np.array(px)
    y_panel = np.array(py)
    def wall_dist(pos):
        xmid = 0.5 * (x_panel[:-1] + x_panel[1:])
        ymid = 0.5 * (y_panel[:-1] + y_panel[1:])
        d2 = (pos[:, 0, None] - xmid[None, :])**2 + (pos[:, 1, None] - ymid[None, :])**2
        return np.sqrt(d2.min(axis=1))

    grad_u = boundary_layer_correction(grad_u, velocity, positions, wall_dist,
                                        U_inf=V_eff, nu=nu)

    meta = {
        "geometry": "swept_wing",
        "chord": chord, "span": span,
        "sweep_deg": sweep_deg, "dihedral_deg": dihedral_deg,
        "alpha_deg": alpha_deg, "U_inf": U_inf, "Re": Re,
        "V_eff": V_eff, "AR": AR,
        "coord_system": "LE at origin, x=streamwise, y=normal, z=spanwise",
    }
    return grad_u, velocity, positions, grid_shape, meta
