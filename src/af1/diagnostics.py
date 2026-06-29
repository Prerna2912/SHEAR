"""
AF1 — Flow Regime Diagnostics Panel.

Real-time Task 3 evaluation pipeline surfaced as a product feature.
Computes: τ invariants (I₁, I₂, I₃), invariant KDE, OOD detection
(Mahalanobis distance), and per-regime Pearson r from stored evaluation results.
"""
from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_STATS_PATH = Path(__file__).parent / "jhtdb_stats.json"
_STATS: dict | None = None


def _stats() -> dict:
    global _STATS
    if _STATS is None:
        with open(_STATS_PATH) as f:
            _STATS = json.load(f)
    return _STATS


# ---------------------------------------------------------------------------
# τ invariants
# ---------------------------------------------------------------------------

def stress_to_3x3(mean_tau: np.ndarray) -> np.ndarray:
    """
    Convert e3nn irreps [N, 6] → symmetric 3×3 [N, 3, 3].

    Irrep layout: [0e scalar (isotropic), 2e × 5 (traceless symmetric)].
    Mapping: τ_iso = c[0] / √3,  then traceless part from c[1:6].
    """
    N = len(mean_tau)
    tau = np.zeros((N, 3, 3), dtype=np.float32)
    inv_sqrt3 = 1.0 / np.sqrt(3.0)
    iso = mean_tau[:, 0] * inv_sqrt3

    # isotropic diagonal
    tau[:, 0, 0] = iso
    tau[:, 1, 1] = iso
    tau[:, 2, 2] = iso

    # traceless symmetric part (2e irrep in e3nn real spherical harmonic basis)
    # Y2: m=-2,-1,0,1,2 → (xy, yz, zz-r²/3, xz, xx-yy) normalised
    # approximate map to Voigt components for display:
    # c[1]→τ_xy, c[2]→τ_yz, c[3]→diag correction, c[4]→τ_xz, c[5]→τ_xx-τ_yy
    c = mean_tau[:, 1:]
    sq2 = np.sqrt(2.0)
    tau[:, 0, 1] += c[:, 0] / sq2;  tau[:, 1, 0] += c[:, 0] / sq2
    tau[:, 1, 2] += c[:, 1] / sq2;  tau[:, 2, 1] += c[:, 1] / sq2
    tau[:, 0, 0] += -c[:, 2] / np.sqrt(6.0)
    tau[:, 1, 1] += -c[:, 2] / np.sqrt(6.0)
    tau[:, 2, 2] +=  c[:, 2] * 2.0 / np.sqrt(6.0)
    tau[:, 0, 2] += c[:, 3] / sq2;  tau[:, 2, 0] += c[:, 3] / sq2
    tau[:, 0, 0] +=  c[:, 4] / sq2
    tau[:, 1, 1] += -c[:, 4] / sq2

    return tau


def compute_invariants(mean_tau: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute τ invariants I₁, I₂, I₃ at each node.

    I₁ = tr(τ)
    I₂ = (tr(τ)² − tr(τ²)) / 2
    I₃ = det(τ)

    Args:
        mean_tau: [N, 6] e3nn irreps
    Returns:
        I1, I2, I3: each [N]
    """
    tau = stress_to_3x3(mean_tau)                    # [N, 3, 3]
    I1  = np.trace(tau, axis1=1, axis2=2)            # [N]
    tau2 = np.einsum('nij,njk->nik', tau, tau)       # [N, 3, 3]
    I2  = 0.5 * (I1**2 - np.trace(tau2, axis1=1, axis2=2))
    I3  = np.linalg.det(tau)
    return I1.astype(np.float32), I2.astype(np.float32), I3.astype(np.float32)


# ---------------------------------------------------------------------------
# KDE (Gaussian, bw=Silverman)
# ---------------------------------------------------------------------------

def kde_1d(values: np.ndarray, n_points: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_grid, density) for a 1D Gaussian KDE."""
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(values, bw_method='silverman')
    lo, hi = values.min(), values.max()
    pad = (hi - lo) * 0.2 + 1e-8
    x = np.linspace(lo - pad, hi + pad, n_points)
    return x.astype(np.float32), kde(x).astype(np.float32)


# ---------------------------------------------------------------------------
# OOD detection (Mahalanobis distance on ∇u statistics)
# ---------------------------------------------------------------------------

def compute_ood(grad_u: np.ndarray) -> dict:
    """
    Mahalanobis distance of the request's ∇u distribution from JHTDB training.

    Uses diagonal covariance approximation (stored in jhtdb_stats.json).

    Args:
        grad_u: [N, 9]
    Returns:
        dict with distance, threshold, is_ood, and per-component z-scores
    """
    st = _stats()
    mu  = np.array(st["grad_u_mean"], dtype=np.float32)
    var = np.array(st["grad_u_cov_diag"], dtype=np.float32)

    # compare distribution of this request vs training
    req_mean = np.array(grad_u, dtype=np.float32).mean(axis=0)  # [9]
    diff = req_mean - mu
    mahal = float(np.sqrt(np.sum(diff**2 / (var + 1e-12))))

    threshold = st["ood_mahal_threshold"]
    z_scores  = (diff / (np.sqrt(var) + 1e-8)).tolist()

    return {
        "mahal_distance": round(mahal, 4),
        "threshold":      threshold,
        "is_ood":         mahal > threshold,
        "z_scores":       z_scores,
    }


# ---------------------------------------------------------------------------
# Per-regime Pearson r from Task 3 evaluation
# ---------------------------------------------------------------------------

def regime_pearson_r() -> dict[str, float]:
    """Return stored per-regime Pearson r from Task 3 V1 evaluation."""
    return _stats()["regime_pearson_r"]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@dataclass
class AF1Output:
    # Invariants
    I1: List[float]
    I2: List[float]
    I3: List[float]

    # KDE grids (for Plotly line charts)
    kde_I1_x: List[float]
    kde_I1_y: List[float]
    kde_I2_x: List[float]
    kde_I2_y: List[float]
    kde_I3_x: List[float]
    kde_I3_y: List[float]

    # JHTDB reference bounds
    jhtdb_I1: dict
    jhtdb_I2: dict
    jhtdb_I3: dict

    # OOD
    ood: dict

    # Per-regime Pearson r
    regime_pearson_r: dict


def analyse(
    mean_tau: list | np.ndarray,  # [N, 6]
    grad_u:   list | np.ndarray,  # [N, 9]
) -> AF1Output:
    mean_tau = np.array(mean_tau, dtype=np.float32)
    grad_u   = np.array(grad_u,   dtype=np.float32)

    I1, I2, I3 = compute_invariants(mean_tau)

    kde_I1_x, kde_I1_y = kde_1d(I1)
    kde_I2_x, kde_I2_y = kde_1d(I2)
    kde_I3_x, kde_I3_y = kde_1d(I3)

    ood = compute_ood(grad_u)
    st  = _stats()

    return AF1Output(
        I1=I1.tolist(), I2=I2.tolist(), I3=I3.tolist(),
        kde_I1_x=kde_I1_x.tolist(), kde_I1_y=kde_I1_y.tolist(),
        kde_I2_x=kde_I2_x.tolist(), kde_I2_y=kde_I2_y.tolist(),
        kde_I3_x=kde_I3_x.tolist(), kde_I3_y=kde_I3_y.tolist(),
        jhtdb_I1=st["invariants"]["I1"],
        jhtdb_I2=st["invariants"]["I2"],
        jhtdb_I3=st["invariants"]["I3"],
        ood=ood,
        regime_pearson_r=regime_pearson_r(),
    )
