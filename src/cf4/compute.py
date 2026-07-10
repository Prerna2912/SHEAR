"""
CF4 — Uncertainty & Regime Map computations.

All functions take raw numpy arrays directly from CF1/CF2 outputs,
so CF4 has no import dependency on those modules — safe to call with
the mock fixture too.

Coordinate convention: x=streamwise, y=normal, z=spanwise.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class CF4Output:
    # Regime
    q_values:       np.ndarray   # [N] Q-criterion
    regime_labels:  np.ndarray   # [N] int 0=strain-dom 1=mixed 2=vortex-dom
    regime_thresholds: Tuple[float, float]   # (p33, p66)

    # Uncertainty
    uncertainty:    np.ndarray   # [N] scalar per-node (L2 norm of variance_tau)
    high_unc_mask:  np.ndarray   # [N] bool — top-25% uncertainty nodes

    # SGS dissipation
    dissipation:    np.ndarray   # [N] Π = -τ:S (negative = backscatter)

    # Backscatter
    backscatter_mask: np.ndarray   # [N] bool — Π < 0
    backscatter_frac: float        # fraction of nodes with Π < 0


# ---------------------------------------------------------------------------
# Q-criterion & regime
# ---------------------------------------------------------------------------

def compute_q_criterion(grad_u_flat: np.ndarray) -> np.ndarray:
    """
    Q = 0.5 * (||Ω||² - ||S||²)

    Args:
        grad_u_flat: [N, 9] flat ∇u (row-major: ∂u_i/∂x_j)

    Returns:
        Q: [N]
    """
    G = grad_u_flat.reshape(-1, 3, 3)
    S = 0.5 * (G + G.transpose(0, 2, 1))      # symmetric part
    Om = 0.5 * (G - G.transpose(0, 2, 1))     # antisymmetric part
    return 0.5 * ((Om * Om).sum(axis=(-2, -1)) - (S * S).sum(axis=(-2, -1)))


def compute_regime(q_values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Partition into 3 regimes by Q-criterion percentile.

    Returns:
        labels [N] — 0=strain-dominated, 1=mixed, 2=vortex-dominated
        p33, p66   — threshold values
    """
    p33 = float(np.percentile(q_values, 33.3))
    p66 = float(np.percentile(q_values, 66.7))
    labels = np.zeros(len(q_values), dtype=np.int32)
    labels[q_values >= p33] = 1
    labels[q_values >= p66] = 2
    return labels, p33, p66


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------

def compute_uncertainty(variance_tau: np.ndarray) -> np.ndarray:
    """
    Scalar uncertainty per node: L2 norm of per-component variance.

    Args:
        variance_tau: [N, 6]

    Returns:
        uncertainty: [N]
    """
    return np.sqrt(np.sum(np.array(variance_tau), axis=-1))


# ---------------------------------------------------------------------------
# SGS dissipation & backscatter
# ---------------------------------------------------------------------------

def compute_dissipation(
    mean_tau_flat: np.ndarray,   # [N, 6] stress irreps
    grad_u_flat: np.ndarray,     # [N, 9]
) -> np.ndarray:
    """
    SGS dissipation: Π = -τ:S = -sum_{ij} τ_{ij} * S_{ij}

    Converts mean_tau from e3nn irreps [N, 6] → symmetric 3×3 [N,3,3],
    then contracts with strain rate S.

    Sign convention: Π > 0 = forward scatter (energy removed from resolved),
                     Π < 0 = backscatter (energy returned to resolved).

    Returns:
        Pi: [N]
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[2]))

    from src.data.dataset import irreps_to_stress
    import torch

    tau_irr = torch.from_numpy(np.array(mean_tau_flat, dtype=np.float32))
    tau_3x3 = irreps_to_stress(tau_irr).numpy()   # [N, 3, 3]

    G = np.array(grad_u_flat, dtype=np.float32).reshape(-1, 3, 3)
    S = 0.5 * (G + G.transpose(0, 2, 1))          # [N, 3, 3]

    Pi = -(tau_3x3 * S).sum(axis=(-2, -1))        # [N]
    return Pi


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyse(
    mean_tau:     list | np.ndarray,   # [N, 6]
    variance_tau: list | np.ndarray,   # [N, 6]
    grad_u:       list | np.ndarray,   # [N, 9]
) -> CF4Output:
    """
    Run all CF4 diagnostics from raw CF2 / CF1 arrays.

    Accepts lists (from JSON fixture) or numpy arrays.
    """
    mean_tau     = np.array(mean_tau,     dtype=np.float32)
    variance_tau = np.array(variance_tau, dtype=np.float32)
    grad_u       = np.array(grad_u,       dtype=np.float32)

    q_values = compute_q_criterion(grad_u)
    regime_labels, p33, p66 = compute_regime(q_values)

    uncertainty = compute_uncertainty(variance_tau)
    unc_thresh  = np.percentile(uncertainty, 75)
    high_unc    = uncertainty >= unc_thresh

    Pi = compute_dissipation(mean_tau, grad_u)
    backscatter_mask = Pi < 0
    backscatter_frac = float(backscatter_mask.mean())

    return CF4Output(
        q_values=q_values,
        regime_labels=regime_labels,
        regime_thresholds=(p33, p66),
        uncertainty=uncertainty,
        high_unc_mask=high_unc,
        dissipation=Pi,
        backscatter_mask=backscatter_mask,
        backscatter_frac=backscatter_frac,
    )
