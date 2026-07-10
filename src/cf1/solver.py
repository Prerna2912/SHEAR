"""
CF1 — Main entry point.

Usage
-----
    from src.cf1 import solve

    output = solve("aerofoil", {
        "chord": 1.0,
        "span":  5.0,
        "alpha_deg": 5.0,
        "U_inf": 50.0,
        "Re":    3e6,
    })

    # output.graph  →  torch_geometric Data ready for V1 inference
    # output.grad_u →  [N, 3, 3] velocity gradient field
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

from .base import CF1Output, GeometryType
from .validation import validate, ValidationError
from .graph import build_cf1_graph


def solve(
    geometry_type: str,
    params: Dict[str, Any],
    grid_size: int = 16,
    k_neighbours: int = 26,
    build_graph: bool = True,
) -> CF1Output:
    """
    Run the CF1 panel method solver for a given geometry type.

    Args:
        geometry_type: one of the 8 supported geometry strings (see GeometryType).
        params       : geometry-specific parameter dict (keys vary by geometry).
        grid_size    : grid resolution per side (default 16 → 4096 nodes).
        k_neighbours : kNN connectivity for the V1 graph (must match training k=26).
        build_graph  : if True (default), assemble the torch_geometric Data object.

    Returns:
        CF1Output with .graph ready for V1, .grad_u, .velocity, .positions.

    Raises:
        ValidationError : if params fail physical plausibility checks.
        ValueError      : if geometry_type is unknown.
    """
    # Normalise geometry type string
    gtype = str(geometry_type).lower().replace("-", "_")

    # Validate parameters
    validate(gtype, params)

    # Dispatch to geometry-specific solver
    grad_u, velocity, positions, grid_shape, meta = _dispatch(gtype, params, grid_size)

    # Clip near-surface numerical spikes: keep 99.5th-percentile range
    # (Spurious large values appear when grid nodes land inside or very near body surfaces)
    p995 = np.percentile(np.abs(grad_u), 99.5)
    grad_u = np.clip(grad_u, -p995, p995)

    # Build torch_geometric graph
    graph = build_cf1_graph(grad_u, positions, k_neighbours=k_neighbours) if build_graph else None

    return CF1Output(
        grad_u=grad_u,
        velocity=velocity,
        positions=positions,
        grid_shape=grid_shape,
        geometry_type=gtype,
        params=params,
        graph=graph,
        meta=meta,
    )


def _dispatch(gtype: str, params: Dict[str, Any], grid_size: int) -> tuple:
    """Import and call the appropriate geometry solver module."""
    if gtype == "aerofoil":
        from .geometry.aerofoil import solve as _solve
    elif gtype == "swept_wing":
        from .geometry.swept_wing import solve as _solve
    elif gtype == "flat_plate":
        from .geometry.flat_plate import solve as _solve
    elif gtype == "bluff_body":
        from .geometry.bluff_body import solve as _solve
    elif gtype == "cylinder":
        from .geometry.cylinder import solve as _solve
    elif gtype == "ship_hull":
        from .geometry.ship_hull import solve as _solve
    elif gtype == "turbine_blade":
        from .geometry.turbine_blade import solve as _solve
    elif gtype == "bluff_body_wake":
        from .geometry.bluff_body_wake import solve as _solve
    else:
        raise ValueError(f"Unsupported geometry type: '{gtype}'")

    return _solve(params, grid_size=grid_size)
