"""
Task 3 V3: Q-criterion computation and regime-partitioned evaluation.

Q = 0.5 * (||Ω||² - ||S||²)
  where S = (∇u + ∇uᵀ)/2  (strain rate, symmetric)
        Ω = (∇u - ∇uᵀ)/2  (rotation rate, antisymmetric)

Regime labels (by Q-criterion percentile):
  0 = low turbulence intensity    (Q < p33)
  1 = medium turbulence intensity (p33 ≤ Q < p66)
  2 = high turbulence intensity   (Q ≥ p66)
"""

import numpy as np
import torch
from typing import Optional


# ---------------------------------------------------------------------------
# Q-criterion
# ---------------------------------------------------------------------------

def q_criterion(grad_u: torch.Tensor) -> torch.Tensor:
    """
    Compute Q-criterion at each point.

    Args:
        grad_u: [N, 3, 3] velocity gradient tensor (or [N, 9] flat).

    Returns:
        Q: [N] Q-criterion values.
    """
    if grad_u.shape[-1] != 3:
        grad_u = grad_u.reshape(-1, 3, 3)

    S = 0.5 * (grad_u + grad_u.transpose(-1, -2))        # strain rate
    Omega = 0.5 * (grad_u - grad_u.transpose(-1, -2))    # rotation rate

    # ||A||² = tr(A Aᵀ) = sum of squared entries (Frobenius²)
    S_norm2 = (S * S).sum(dim=(-2, -1))
    Omega_norm2 = (Omega * Omega).sum(dim=(-2, -1))

    return 0.5 * (Omega_norm2 - S_norm2)


def q_criterion_np(grad_u: np.ndarray) -> np.ndarray:
    """NumPy version of q_criterion.  grad_u: [N, 3, 3] or [N, 9]."""
    return q_criterion(torch.from_numpy(np.asarray(grad_u, dtype=np.float32))).numpy()


# ---------------------------------------------------------------------------
# Regime partitioning
# ---------------------------------------------------------------------------

def regime_partition(q_values: torch.Tensor,
                     n_regimes: int = 3) -> torch.Tensor:
    """
    Partition samples by Q-criterion into low / medium / high regimes.

    Args:
        q_values: [N] Q-criterion values.
        n_regimes: number of regimes (default 3).

    Returns:
        labels: [N] integer labels 0, 1, …, n_regimes-1.
    """
    q_np = q_values.cpu().numpy() if isinstance(q_values, torch.Tensor) else q_values
    thresholds = [np.percentile(q_np, 100 * k / n_regimes)
                  for k in range(1, n_regimes)]

    labels = np.zeros(len(q_np), dtype=np.int64)
    for i, thr in enumerate(thresholds):
        labels[q_np >= thr] = i + 1
    return torch.from_numpy(labels)


REGIME_NAMES = {0: 'low', 1: 'medium', 2: 'high'}


# ---------------------------------------------------------------------------
# Regime-partitioned metrics
# ---------------------------------------------------------------------------

def metrics_by_regime(
    pred_3x3: torch.Tensor,
    true_3x3: torch.Tensor,
    grad_3x3: torch.Tensor,
    compute_fn,
    n_regimes: int = 3,
) -> dict:
    """
    Compute metrics separately for each turbulence-intensity regime.

    Args:
        pred_3x3:   [N, 3, 3] predicted SGS stress.
        true_3x3:   [N, 3, 3] true SGS stress.
        grad_3x3:   [N, 3, 3] velocity gradient (used for Q and dissipation).
        compute_fn: callable(pred_3x3, true_3x3, grad_3x3) → metrics dict.
                    Typically evaluation.metrics.compute_all_metrics_np.
        n_regimes:  number of regimes.

    Returns:
        dict with keys 'low', 'medium', 'high' (and 'all') each containing
        the metrics dict for that subset.
    """
    Q = q_criterion(grad_3x3)                          # [N]
    labels = regime_partition(Q, n_regimes)             # [N]

    results = {'all': compute_fn(pred_3x3, true_3x3, grad_3x3)}

    for label in range(n_regimes):
        mask = labels == label
        name = REGIME_NAMES.get(label, str(label))
        n_pts = mask.sum().item()
        if n_pts == 0:
            results[name] = {}
            continue
        results[name] = compute_fn(
            pred_3x3[mask],
            true_3x3[mask],
            grad_3x3[mask],
        )
        results[name]['n_points'] = n_pts

    return results


# ---------------------------------------------------------------------------
# Per-sample Q for sorting / worst-case analysis
# ---------------------------------------------------------------------------

def per_sample_q(grad_3x3: torch.Tensor,
                 batch_indices: Optional[torch.Tensor] = None,
                 n_nodes_per_sample: int = 512) -> torch.Tensor:
    """
    Mean Q-criterion per sub-cube (sample-level, not node-level).

    Args:
        grad_3x3:          [N_total_nodes, 3, 3] velocity gradients.
        batch_indices:     [N_total_nodes] PyG batch vector (which sub-cube each node belongs to).
                           If None, assumes all nodes belong to one sub-cube or uses n_nodes_per_sample.
        n_nodes_per_sample: nodes per sub-cube when batch_indices is None.

    Returns:
        q_per_sample: [n_samples] mean Q per sub-cube.
    """
    Q = q_criterion(grad_3x3)   # [N_total_nodes]

    if batch_indices is not None:
        n_samples = int(batch_indices.max().item()) + 1
        q_per = torch.zeros(n_samples, device=Q.device)
        counts = torch.zeros(n_samples, device=Q.device)
        q_per.scatter_add_(0, batch_indices, Q)
        counts.scatter_add_(0, batch_indices, torch.ones_like(Q))
        return q_per / counts.clamp(min=1)

    N = Q.shape[0]
    n_samples = N // n_nodes_per_sample
    return Q[:n_samples * n_nodes_per_sample].reshape(n_samples, n_nodes_per_sample).mean(dim=1)
