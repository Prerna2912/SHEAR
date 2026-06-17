"""
CF1 — Panel Method ∇u Synthesis.

Coordinate system (agreed with team, Day 1):
  x : streamwise direction (freestream U_inf along +x)
  y : normal / wall-normal direction
  z : spanwise direction

Velocity gradient convention matching JHTDB training data:
  ∇u[i,j] = ∂u_i/∂x_j   (row = velocity component, col = spatial direction)

All physical lengths are in SI units (metres).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Any

import numpy as np


# ---------------------------------------------------------------------------
# Coordinate system constant (published to teammates on Day 1)
# ---------------------------------------------------------------------------

COORDINATE_SYSTEM = {
    "x": "streamwise — freestream U_inf along +x",
    "y": "normal / wall-normal",
    "z": "spanwise",
    "grad_u_convention": "∂u_i/∂x_j  (row=velocity, col=spatial) — matches JHTDB",
}


# ---------------------------------------------------------------------------
# Geometry type registry
# ---------------------------------------------------------------------------

class GeometryType(str, Enum):
    AEROFOIL          = "aerofoil"
    SWEPT_WING        = "swept_wing"
    BLUFF_BODY        = "bluff_body"
    CYLINDER          = "cylinder"
    TURBINE_BLADE     = "turbine_blade"
    FLAT_PLATE        = "flat_plate"
    SHIP_HULL         = "ship_hull"
    BLUFF_BODY_WAKE   = "bluff_body_wake"


# ---------------------------------------------------------------------------
# CF1 output contract  (the integration surface for CF2 / V1)
# ---------------------------------------------------------------------------

@dataclass
class CF1Output:
    """
    Velocity gradient field and derived quantities from the panel method solver.

    The `.graph` attribute is a torch_geometric Data object ready for V1 inference.
    Its node features `.x` are the 9-component e3nn irreps of ∇u.

    Attributes:
        grad_u      : [N, 3, 3] velocity gradient ∂u_i/∂x_j at each grid node.
        velocity    : [N, 3]    mean-flow velocity (u_x, u_y, u_z) at each node.
        positions   : [N, 3]    physical coordinates (x, y, z) of each node.
        graph               : torch_geometric.data.Data with .x, .pos, .edge_index
                              (None until build_graph() is called).
        grid_shape  : (Nx, Ny, Nz) dimensions of the structured grid.
        geometry_type       : geometry type string.
        params      : raw input parameters dict.
        meta        : diagnostic metadata (Re, U_inf, char_length, BL correction flag).
    """
    grad_u:        np.ndarray
    velocity:      np.ndarray
    positions:     np.ndarray
    grid_shape:    tuple
    geometry_type: str
    params:        Dict[str, Any]
    graph:         Any = field(default=None)          # torch_geometric Data
    meta:          Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Structured 3D grid builder
# ---------------------------------------------------------------------------

def build_grid(
    x_range: tuple,
    y_range: tuple,
    z_range: tuple,
    grid_size: int = 16,
) -> tuple:
    """
    Build a uniform Cartesian grid for flow field evaluation.

    Args:
        x_range  : (x_min, x_max) in metres.
        y_range  : (y_min, y_max) in metres.
        z_range  : (z_min, z_max) in metres.
        grid_size: number of points per side (default 16 → 4096 nodes).

    Returns:
        positions : [N, 3] grid point coordinates (flattened, row-major).
        grid_shape: (grid_size, grid_size, grid_size).
    """
    xs = np.linspace(x_range[0], x_range[1], grid_size, dtype=np.float32)
    ys = np.linspace(y_range[0], y_range[1], grid_size, dtype=np.float32)
    zs = np.linspace(z_range[0], z_range[1], grid_size, dtype=np.float32)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')           # each [Nx, Ny, Nz]
    positions = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1).astype(np.float32)
    return positions, (grid_size, grid_size, grid_size)


# ---------------------------------------------------------------------------
# Finite-difference ∇u from velocity field on structured grid
# ---------------------------------------------------------------------------

def compute_grad_u(
    velocity: np.ndarray,
    grid_shape: tuple,
    x_range: tuple,
    y_range: tuple,
    z_range: tuple,
) -> np.ndarray:
    """
    Compute velocity gradient ∂u_i/∂x_j on a structured grid via 2nd-order
    central differences (1st-order at boundaries).

    Args:
        velocity  : [N, 3] velocity field (flattened).
        grid_shape: (Nx, Ny, Nz).
        x_range, y_range, z_range: physical extents.

    Returns:
        grad_u: [N, 3, 3]   ∂u_i/∂x_j  (row=velocity, col=spatial).
    """
    Nx, Ny, Nz = grid_shape
    N = Nx * Ny * Nz

    U = velocity.reshape(Nx, Ny, Nz, 3).astype(np.float64)     # [Nx, Ny, Nz, 3]

    dx = (x_range[1] - x_range[0]) / max(Nx - 1, 1)
    dy = (y_range[1] - y_range[0]) / max(Ny - 1, 1)
    dz = (z_range[1] - z_range[0]) / max(Nz - 1, 1)

    G = np.zeros((Nx, Ny, Nz, 3, 3), dtype=np.float64)          # ∂u_i/∂x_j

    # Central differences along x (j=0)
    G[1:-1, :, :, :, 0] = (U[2:, :, :] - U[:-2, :, :]) / (2 * dx)
    G[0,    :, :, :, 0] = (U[1,  :, :] - U[0,   :, :]) / dx
    G[-1,   :, :, :, 0] = (U[-1, :, :] - U[-2,  :, :]) / dx

    # Central differences along y (j=1)
    G[:, 1:-1, :, :, 1] = (U[:, 2:, :] - U[:, :-2, :]) / (2 * dy)
    G[:, 0,    :, :, 1] = (U[:, 1,  :] - U[:, 0,   :]) / dy
    G[:, -1,   :, :, 1] = (U[:, -1, :] - U[:, -2,  :]) / dy

    # Central differences along z (j=2)
    G[:, :, 1:-1, :, 2] = (U[:, :, 2:] - U[:, :, :-2]) / (2 * dz)
    G[:, :, 0,    :, 2] = (U[:, :, 1]  - U[:, :, 0]  ) / dz
    G[:, :, -1,   :, 2] = (U[:, :, -1] - U[:, :, -2])  / dz

    return G.reshape(N, 3, 3).astype(np.float32)


# ---------------------------------------------------------------------------
# Boundary layer correction
# ---------------------------------------------------------------------------

def boundary_layer_correction(
    grad_u:    np.ndarray,
    velocity:  np.ndarray,
    positions: np.ndarray,
    wall_distance_fn,
    U_inf:   float = 1.0,
    nu:      float = 1.5e-5,
    blend_factor: float = 3.0,
) -> np.ndarray:
    """
    Blend panel-method ∇u with a Blasius boundary-layer correction near walls.

    The Blasius solution gives ∂u_x/∂y ≈ 0.332 * U_e / δ within the boundary
    layer (δ = 5 * x / sqrt(Re_x)), where U_e is the local edge velocity.
    Beyond 3δ from the wall, the panel solution is used unmodified.

    Args:
        grad_u    : [N, 3, 3] panel-method velocity gradient (modified in-place).
        velocity  : [N, 3]    panel-method velocity.
        positions : [N, 3]    grid coordinates.
        wall_distance_fn: callable(positions) → [N] signed distance from nearest wall.
        U_inf     : freestream speed (m/s).
        nu        : kinematic viscosity (m²/s).
        blend_factor: number of δ thicknesses for blending window.

    Returns:
        grad_u_corrected: [N, 3, 3] (copy of input with near-wall values updated).
    """
    grad_u = grad_u.copy()
    d = wall_distance_fn(positions)                         # [N] wall distance ≥ 0

    x = np.clip(positions[:, 0], 1e-6, None)               # streamwise coord ≥ ε
    Re_x = np.clip(U_inf * x / nu, 1.0, None)
    delta = 5.0 * x / np.sqrt(Re_x)                        # Blasius BL thickness

    U_e = np.linalg.norm(velocity[:, :2], axis=1)          # edge speed (in xy-plane)
    U_e = np.where(U_e < 1e-6, U_inf, U_e)

    # BL velocity gradient ∂u_x/∂y from Blasius (dominant near-wall shear component)
    du_dy_bl = 0.332 * U_e / np.maximum(delta, 1e-8)       # [N]

    # Blending weight: 1 inside BL, 0 beyond blend_factor * delta
    weight = np.clip(1.0 - d / (blend_factor * np.maximum(delta, 1e-8)), 0.0, 1.0)

    # Apply correction only to ∂u_x/∂y (index [0, 1])
    grad_u[:, 0, 1] = (
        weight * du_dy_bl
        + (1.0 - weight) * grad_u[:, 0, 1]
    )

    return grad_u
