"""
CF2 — Core V1 inference function.

Stateless callable: takes a CF1 graph, model, and stats; returns samples.
The FastAPI layer and the queue both call run_inference() directly.

Normalisation
-------------
V1 was trained on normalised features:
    x_norm = (x - x_mean) / x_std   ← applied to graph.x before forward pass
    y_norm = (y - y_mean) / y_std   ← model output is in this space
    tau    = y_norm * y_std + y_mean ← denormalise to physical units

Progress callback
-----------------
progress_fn(completed: int, total: int) is called after each ODE solve.
CF2 app wires this to the WebSocket broadcaster.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Callable, Optional

import numpy as np
import torch
from torch_geometric.data import Data, Batch


# ---------------------------------------------------------------------------
# Geometry hash (cache key)
# ---------------------------------------------------------------------------

def geometry_hash(geometry_type: str, params: dict) -> str:
    """SHA-256 of the canonical geometry params dict (stable across sessions)."""
    canonical = json.dumps(
        {"geometry_type": geometry_type, **params},
        sort_keys=True,
        default=str,      # handle non-JSON types (e.g. numpy scalars)
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    graph: Data,
    model,
    stats: dict,
    device: torch.device,
    n_samples: int = 20,
    ode_steps: int = 100,
    ode_method: str = "dopri5",
    progress_fn: Optional[Callable[[int, int], None]] = None,
    include_samples: bool = False,
) -> dict:
    """
    Run V1 CFM inference on a CF1 graph.

    Each of the n_samples calls integrates the learned ODE independently
    from a fresh Gaussian noise draw, giving one sample of τ per call.
    The mean and variance over all samples form the core output.

    Args:
        graph        : torch_geometric Data from CF1 (unnormalised .x).
        model        : EGNNFlowMatching in eval mode.
        stats        : normalisation dict with x_mean, x_std, y_mean, y_std.
        device       : target device.
        n_samples    : number of ODE samples (default 20).
        ode_steps    : solver step budget (passed to torchdiffeq).
        ode_method   : ODE solver name ('dopri5', 'euler', 'rk4').
        progress_fn  : optional callback(completed, total) after each sample.
        include_samples: if True, return full [n_samples, N, 6] array.

    Returns:
        dict with keys:
            samples      : np.ndarray [n_samples, N, 6]  (or None)
            mean_tau     : np.ndarray [N, 6]
            variance_tau : np.ndarray [N, 6]
            n_nodes      : int
    """
    # --- Normalise graph features and move to device -------------------------
    g = graph.clone().to(device)
    g.x = (g.x - stats['x_mean']) / stats['x_std']

    # Set ode_steps on the model ODE solver
    # EGNNFlowMatching.sample() accepts n_steps only for fixed solvers;
    # for dopri5 we use rtol/atol (already set in model).
    # We monkey-patch the model's rtol/atol for tighter/looser solves.
    original_rtol = model.ode_rtol
    original_atol = model.ode_atol

    samples_list = []

    try:
        for i in range(n_samples):
            try:
                tau_norm = model.sample(g, n_steps=ode_steps, method=ode_method)  # [N, 6]
            except Exception as ode_err:
                # ODE solver diverged — return a structured error payload
                raise RuntimeError(
                    f"ODE solve diverged at sample {i+1}/{n_samples}: {ode_err}"
                ) from ode_err

            # Denormalise
            tau = tau_norm * stats['y_std'] + stats['y_mean']             # [N, 6]
            samples_list.append(tau.cpu().float().numpy())

            if progress_fn is not None:
                progress_fn(i + 1, n_samples)

    finally:
        model.ode_rtol = original_rtol
        model.ode_atol = original_atol

    samples = np.stack(samples_list, axis=0)   # [n_samples, N, 6]

    return {
        "samples":      samples if include_samples else None,
        "mean_tau":     samples.mean(axis=0),   # [N, 6]
        "variance_tau": samples.var(axis=0),    # [N, 6]
        "n_nodes":      graph.pos.shape[0],
    }


# ---------------------------------------------------------------------------
# Backscatter computation (used by CF4)
# ---------------------------------------------------------------------------

def compute_backscatter(
    samples: np.ndarray,           # [n_samples, N, 6] stress irreps
    strain_rate: np.ndarray,       # [N, 9] flat strain rate (S_ij)
) -> np.ndarray:
    """
    Compute per-node backscatter fraction across samples.

    SGS dissipation: Π = -τ:S = -sum_{ij} τ_{ij} S_{ij}
    Backscatter at node k: fraction of samples where Π_k < 0.

    Args:
        samples    : [n_samples, N, 6] stress irreps from run_inference.
        strain_rate: [N, 9] flat 3x3 strain rate tensor.

    Returns:
        backscatter_fraction: [N] fraction of samples with Π < 0.
    """
    from src.data.dataset import irreps_to_stress

    import torch as th
    tau_3x3 = irreps_to_stress(th.from_numpy(samples.reshape(-1, 6))).numpy()
    tau_3x3 = tau_3x3.reshape(samples.shape[0], -1, 3, 3)   # [S, N, 3, 3]

    S_3x3 = strain_rate.reshape(-1, 3, 3)                    # [N, 3, 3]

    # Pi = -tau:S = -sum_{ij} tau_{ij} * S_{ij}
    Pi = -(tau_3x3 * S_3x3[None, :, :, :]).sum(axis=(-2, -1))   # [S, N]
    return (Pi < 0).mean(axis=0).astype(np.float32)              # [N]
