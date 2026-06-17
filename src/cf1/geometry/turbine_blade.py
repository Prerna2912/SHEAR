"""
CF1 — Wind turbine blade panel method solver.

Uses a NACA 4-digit vortex panel method for the blade cross-section.
Rotation effects are captured via:
  - Modified angle of attack at each radial station: α_eff = atan(V_axial / V_tang)
  - Centrifugal pumping term: adds Coriolis-induced radial component to ∇u

Parameters
----------
radius        : rotor radius (m)
pitch_angle   : blade pitch angle (degrees)
tip_speed_ratio: tip speed ratio λ = ω*R / U_wind
rotation_speed: rotor angular velocity ω (rad/s)
U_inf         : freestream wind speed (m/s)
"""

from __future__ import annotations

import numpy as np

from ..base import build_grid, compute_grad_u, boundary_layer_correction
from ..panels import naca4_coords, solve_vortex_panels, vortex_velocity_field


def solve(params: dict, grid_size: int = 16) -> tuple:
    R       = float(params["radius"])
    pitch   = float(params.get("pitch_angle", 5.0))
    tsr     = float(params.get("tip_speed_ratio", 7.0))
    omega   = float(params.get("rotation_speed", tsr * params.get("U_inf", 10.0) / R))
    U_inf   = float(params.get("U_inf", params.get("freestream_velocity", 10.0)))
    Re      = float(params.get("Re", params.get("reynolds_number", U_inf * R / 1.5e-5)))
    n_panels= int(params.get("n_panels", 80))
    naca_code = str(params.get("naca_code", "2412"))

    nu = U_inf * R / Re

    # Blade chord is typically c = 0.1*R (simplification)
    chord = params.get("chord", 0.1 * R)

    # Blade element: solve at the 75% radial station (representative)
    r_rep = 0.75 * R
    V_tang = omega * r_rep                      # rotational speed at 75% span
    V_axial = U_inf

    # Effective inflow angle
    phi = np.arctan2(V_axial, V_tang)
    alpha = phi - np.deg2rad(pitch)             # effective angle of attack

    m = int(naca_code[0]) / 100.0
    p = int(naca_code[1]) / 10.0
    t = int(naca_code[2:]) / 100.0
    px, py = naca4_coords(m, p, t, n_panels=n_panels, chord=chord)

    V_rel = np.sqrt(V_tang**2 + V_axial**2)    # relative inflow speed
    gamma = solve_vortex_panels(px, py, alpha, U_inf=V_rel)

    x_range = (-0.5 * chord, 1.5 * chord)
    y_range = (-0.5 * chord, 0.5 * chord)
    z_range = (0.0, R)

    positions, grid_shape = build_grid(x_range, y_range, z_range, grid_size)
    Nx, Ny, Nz = grid_shape
    N = positions.shape[0]

    xp, yp = positions[:, 0], positions[:, 1]
    r_pos = positions[:, 2]                     # radial position along blade

    ux, uy = vortex_velocity_field(xp, yp, px, py, gamma, U_inf=V_rel, alpha=alpha)

    # Coriolis radial component: u_r = -2 * Ω × u_tang  (simplified)
    v_tang_local = omega * np.maximum(r_pos, 0.01 * R)
    uz_rotation = -2.0 * omega * ux * (r_pos / (R + 1e-8))   # simplified Coriolis

    velocity = np.stack([ux, uy, uz_rotation], axis=-1).astype(np.float32)
    grad_u = compute_grad_u(velocity, grid_shape, x_range, y_range, z_range)

    x_panel = np.array(px)
    y_panel = np.array(py)
    def wall_dist(pos):
        xmid = 0.5 * (x_panel[:-1] + x_panel[1:])
        ymid = 0.5 * (y_panel[:-1] + y_panel[1:])
        d2 = (pos[:, 0, None] - xmid[None, :])**2 + (pos[:, 1, None] - ymid[None, :])**2
        return np.sqrt(d2.min(axis=1))

    grad_u = boundary_layer_correction(grad_u, velocity, positions, wall_dist,
                                        U_inf=V_rel, nu=nu)

    meta = {
        "geometry": "turbine_blade", "radius": R, "pitch_angle": pitch,
        "tsr": tsr, "omega": omega, "U_inf": U_inf, "Re": Re,
        "chord": chord, "alpha_eff_deg": float(np.rad2deg(alpha)), "V_rel": float(V_rel),
        "coord_system": "LE at origin, x=chord-normal, y=profile-normal, z=radial",
    }
    return grad_u, velocity, positions, grid_shape, meta
