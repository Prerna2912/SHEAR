"""
CF1 — Panel Method ∇u Synthesis.

Provides the physics engine that generates the velocity gradient field ∇u
for each geometry type.  V1 (the SE(3)-equivariant CFM model) uses this
as its input.

Quick start
-----------
    from src.cf1 import solve

    output = solve("aerofoil", {
        "chord": 1.0, "span": 5.0,
        "alpha_deg": 5.0, "U_inf": 50.0, "Re": 3e6,
    })
    # output.graph is a torch_geometric Data object ready for V1

Coordinate system  (documented here for teammates on Day 1)
-------------------
    x : streamwise  — freestream U_inf along +x
    y : normal / wall-normal
    z : spanwise
    ∇u[i,j] = ∂u_i/∂x_j   (row=velocity component, col=spatial direction)
    This convention matches the JHTDB training data.

Supported geometry types
------------------------
    aerofoil | swept_wing | bluff_body | cylinder |
    turbine_blade | flat_plate | ship_hull | bluff_body_wake
"""

from .solver import solve
from .base import CF1Output, GeometryType, COORDINATE_SYSTEM
from .validation import ValidationError

__all__ = ["solve", "CF1Output", "GeometryType", "ValidationError", "COORDINATE_SYSTEM"]
