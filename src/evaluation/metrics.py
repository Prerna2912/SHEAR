"""
Evaluation metrics for SGS closure:

1. Pearson correlation per tensor component.
2. Jensen-Shannon divergence on stress tensor invariants (I1, I2, I3).
3. Sub-grid dissipation error.
4. Principal-axis alignment angle between predicted and true tau eigenvectors.
5. Inference time (ms) and peak GPU memory (MB) — secondary efficiency metrics.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr
from typing import Optional


# ------------------------------------------------------------------
# Collect predictions
# ------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader,
    variant: str,
    device: torch.device,
    n_ode_steps: int = 100,
    stats: Optional[dict] = None,
) -> tuple:
    """
    Run inference on the full test set and return stacked tensors.

    Returns:
        pred_irr: [N_total, 6]  predicted stress in irreps basis.
        true_irr: [N_total, 6]  true stress in irreps basis.
        pred_3x3: [N_total, 3, 3] predicted stress as 3x3 tensors.
        true_3x3: [N_total, 3, 3] true stress as 3x3 tensors.
    """
    from data.dataset import irreps_to_stress

    model.eval()
    preds_irr, trues_irr = [], []

    for batch in loader:
        if variant in ('v1', 'v2'):
            batch = batch.to(device)

        if variant == 'v1':
            pred = model.sample(batch, n_steps=n_ode_steps)   # [N, 6]
        elif variant == 'v2':
            pred = model(x=batch.x, pos=batch.pos, edge_index=batch.edge_index)
        elif variant == 'v3':
            from data.dataset import grad_to_irreps
            batch = batch.to(device)
            grad3x3 = batch.grad_full.reshape(-1, 3, 3)
            grad_irr = grad_to_irreps(grad3x3)
            # Normalise gradient to match training distribution
            if stats is not None:
                x_mean = stats['x_mean'].to(device)
                x_std  = stats['x_std'].to(device)
                grad_irr = (grad_irr - x_mean) / x_std
            pred = model.sample(grad_irr)   # MLPFlowMatching.sample takes no n_steps arg

        preds_irr.append(pred.cpu())
        trues_irr.append(batch.y.cpu())

    pred_irr = torch.cat(preds_irr, dim=0)
    true_irr = torch.cat(trues_irr, dim=0)
    pred_3x3 = irreps_to_stress(pred_irr)
    true_3x3 = irreps_to_stress(true_irr)

    return pred_irr, true_irr, pred_3x3, true_3x3


# ------------------------------------------------------------------
# 1. Pearson correlation per component
# ------------------------------------------------------------------

def pearson_per_component(pred_3x3: torch.Tensor, true_3x3: torch.Tensor) -> dict:
    """
    Compute Pearson r for each of the 6 independent components of tau.

    Returns dict with keys like 'tau_xx', 'tau_xy', etc.
    """
    components = {
        'tau_xx': (0, 0), 'tau_yy': (1, 1), 'tau_zz': (2, 2),
        'tau_xy': (0, 1), 'tau_xz': (0, 2), 'tau_yz': (1, 2),
    }
    result = {}
    for name, (i, j) in components.items():
        p = pred_3x3[:, i, j].numpy()
        t = true_3x3[:, i, j].numpy()
        r, _ = pearsonr(p, t)
        result[name] = float(r)
    result['mean'] = float(np.mean(list(result.values())))
    return result


# ------------------------------------------------------------------
# 2. JSD on stress tensor invariants
# ------------------------------------------------------------------

def stress_invariants(tau: torch.Tensor) -> torch.Tensor:
    """
    Compute the three invariants of the stress tensor per sample.

    I1 = tr(tau)
    I2 = 0.5 * (tr(tau)^2 - tr(tau^2))
    I3 = det(tau)

    Args:
        tau: [N, 3, 3]

    Returns:
        [N, 3] invariants.
    """
    I1 = tau[:, 0, 0] + tau[:, 1, 1] + tau[:, 2, 2]
    tau2 = torch.bmm(tau, tau)
    I2 = 0.5 * (I1**2 - (tau2[:, 0, 0] + tau2[:, 1, 1] + tau2[:, 2, 2]))
    I3 = torch.linalg.det(tau)
    return torch.stack([I1, I2, I3], dim=-1)


def jsd_invariants(
    pred_3x3: torch.Tensor,
    true_3x3: torch.Tensor,
    n_bins: int = 100,
) -> dict:
    """
    Jensen-Shannon divergence on each of the three stress tensor invariants.

    Returns dict with keys 'I1', 'I2', 'I3' and 'mean'.
    """
    pred_inv = stress_invariants(pred_3x3).numpy()   # [N, 3]
    true_inv = stress_invariants(true_3x3).numpy()   # [N, 3]

    result = {}
    for k, name in enumerate(['I1', 'I2', 'I3']):
        lo = min(true_inv[:, k].min(), pred_inv[:, k].min()) - 1e-10
        hi = max(true_inv[:, k].max(), pred_inv[:, k].max()) + 1e-10
        bins = np.linspace(lo, hi, n_bins + 1)

        p_hist, _ = np.histogram(pred_inv[:, k], bins=bins, density=True)
        t_hist, _ = np.histogram(true_inv[:, k], bins=bins, density=True)

        # Add small constant to avoid zero-mass bins
        p_hist = p_hist + 1e-10
        t_hist = t_hist + 1e-10
        p_hist /= p_hist.sum()
        t_hist /= t_hist.sum()

        result[name] = float(jensenshannon(p_hist, t_hist, base=2))

    result['mean'] = float(np.mean([result['I1'], result['I2'], result['I3']]))
    return result


# ------------------------------------------------------------------
# 3. Sub-grid dissipation error
# ------------------------------------------------------------------

def sgs_dissipation(tau: torch.Tensor, grad_u: torch.Tensor) -> torch.Tensor:
    """
    Sub-grid dissipation: Pi = -tau_{ij} * S_{ij}
    where S_{ij} = (du_i/dx_j + du_j/dx_i) / 2

    Args:
        tau:    [N, 3, 3] SGS stress tensor.
        grad_u: [N, 3, 3] filtered velocity gradient.

    Returns:
        [N] sub-grid dissipation per point.
    """
    S = 0.5 * (grad_u + grad_u.transpose(-1, -2))       # [N, 3, 3]
    Pi = -(tau * S).sum(dim=(-1, -2))                    # [N]
    return Pi


def dissipation_error(
    pred_3x3: torch.Tensor,
    true_3x3: torch.Tensor,
    grad_u_3x3: torch.Tensor,
) -> dict:
    """
    Returns dict with 'rmse', 'correlation', 'mean_pred', 'mean_true'.
    """
    pred_pi = sgs_dissipation(pred_3x3, grad_u_3x3).numpy()
    true_pi = sgs_dissipation(true_3x3, grad_u_3x3).numpy()

    rmse = float(np.sqrt(np.mean((pred_pi - true_pi)**2)))
    r, _ = pearsonr(pred_pi, true_pi)
    return {
        'rmse': rmse,
        'correlation': float(r),
        'mean_pred': float(pred_pi.mean()),
        'mean_true': float(true_pi.mean()),
    }


# ------------------------------------------------------------------
# 4. Principal-axis alignment angle
# ------------------------------------------------------------------

def principal_axis_alignment(
    pred_3x3: torch.Tensor,
    true_3x3: torch.Tensor,
) -> dict:
    """
    Angle (degrees) between the leading eigenvectors of predicted and true tau.

    Returns dict with 'mean_deg', 'median_deg', 'std_deg'.
    """
    # Eigendecomposition
    _, pred_vecs = torch.linalg.eigh(pred_3x3)   # eigh for symmetric: [N, 3, 3]
    _, true_vecs = torch.linalg.eigh(true_3x3)   # eigenvectors in columns

    # Leading eigenvector (last column, largest eigenvalue)
    pred_v = pred_vecs[:, :, -1]   # [N, 3]
    true_v = true_vecs[:, :, -1]   # [N, 3]

    # cos(angle), clamped to [-1, 1] to handle numerical issues
    cos_angle = (pred_v * true_v).sum(dim=-1).abs().clamp(0, 1)  # [N]
    angles_deg = torch.acos(cos_angle) * (180.0 / np.pi)

    return {
        'mean_deg': float(angles_deg.mean()),
        'median_deg': float(angles_deg.median()),
        'std_deg': float(angles_deg.std()),
    }


# ------------------------------------------------------------------
# 5. Inference timing and memory
# ------------------------------------------------------------------

def benchmark_inference(
    model: nn.Module,
    loader,
    variant: str,
    device: torch.device,
    n_batches: int = 10,
) -> dict:
    """
    Measure wall-clock inference time per sub-cube and peak GPU memory.

    Returns dict with 'ms_per_subcube', 'peak_gpu_mb'.
    """
    model.eval()
    times = []

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break

            if variant in ('v1', 'v2'):
                batch = batch.to(device)

            t0 = time.perf_counter()
            if variant == 'v1':
                model.sample(batch)
            elif variant == 'v2':
                model(x=batch.x, pos=batch.pos, edge_index=batch.edge_index)
            elif variant == 'v3':
                from data.dataset import grad_to_irreps
                batch = batch.to(device)
                grad3x3 = batch.grad_full.reshape(-1, 3, 3)
                grad_irr = grad_to_irreps(grad3x3)
                model.sample(grad_irr)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            n_subcubes = batch.num_graphs
            times.append((t1 - t0) * 1000.0 / n_subcubes)   # ms per sub-cube

    peak_mb = 0.0
    if device.type == 'cuda':
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6

    return {
        'ms_per_subcube': float(np.mean(times)),
        'peak_gpu_mb': peak_mb,
    }


# ------------------------------------------------------------------
# Master evaluation function
# ------------------------------------------------------------------

def compute_all_metrics(
    model: nn.Module,
    test_loader,
    variant: str,
    device: torch.device,
    stats: Optional[dict] = None,
) -> dict:
    """
    Run the full evaluation suite and return a nested dict of all metrics.
    """
    print(f"  Collecting predictions ({variant.upper()})...")
    pred_irr, true_irr, pred_3x3, true_3x3 = collect_predictions(
        model, test_loader, variant, device, stats=stats
    )

    # Denormalize if stats provided
    if stats is not None:
        y_mean = stats['y_mean']
        y_std  = stats['y_std']
        pred_irr = pred_irr * y_std + y_mean
        true_irr = true_irr * y_std + y_mean
        from data.dataset import irreps_to_stress
        pred_3x3 = irreps_to_stress(pred_irr)
        true_3x3 = irreps_to_stress(true_irr)

    # Need velocity gradient for dissipation (from the test loader)
    # Collect grad_u from all test batches
    grad_list = []
    for batch in test_loader:
        grad_list.append(batch.grad_full.reshape(-1, 3, 3))
    grad_3x3 = torch.cat(grad_list, dim=0)

    print("  Pearson correlation...")
    pearson = pearson_per_component(pred_3x3, true_3x3)

    print("  JSD on invariants...")
    jsd = jsd_invariants(pred_3x3, true_3x3)

    print("  Dissipation error...")
    diss = dissipation_error(pred_3x3, true_3x3, grad_3x3)

    print("  Principal-axis alignment...")
    align = principal_axis_alignment(pred_3x3, true_3x3)

    print("  Inference benchmark...")
    timing = benchmark_inference(model, test_loader, variant, device)

    return {
        'pearson': pearson,
        'jsd_invariants': jsd,
        'dissipation': diss,
        'alignment': align,
        'efficiency': timing,
    }


# ------------------------------------------------------------------
# Numpy-only evaluation (for classical baselines — no model needed)
# ------------------------------------------------------------------

def compute_all_metrics_np(
    pred_3x3: torch.Tensor,
    true_3x3: torch.Tensor,
    grad_3x3: torch.Tensor,
) -> dict:
    """
    Compute all a-priori metrics from 3×3 tensor arrays.
    Accepts torch.Tensor; returns a plain dict of floats.

    This is the entry point used by Task 3 for classical baselines and
    regime-partitioned evaluation.
    """
    pearson = pearson_per_component(pred_3x3, true_3x3)
    jsd     = jsd_invariants(pred_3x3, true_3x3)
    diss    = dissipation_error(pred_3x3, true_3x3, grad_3x3)
    align   = principal_axis_alignment(pred_3x3, true_3x3)
    bs      = backscatter_fraction(pred_3x3, grad_3x3)
    return {
        'pearson': pearson,
        'jsd_invariants': jsd,
        'dissipation': diss,
        'alignment': align,
        'backscatter_fraction': bs,
    }


# ------------------------------------------------------------------
# 6. Backscatter fraction
# ------------------------------------------------------------------

def backscatter_fraction(
    pred_3x3: torch.Tensor,
    grad_3x3: torch.Tensor,
) -> float:
    """
    Fraction of grid points where the predicted SGS dissipation is negative
    (i.e. model predicts energy transfer from sub-grid to resolved scales).

    Π = τ_ij S_ij  (< 0 means forward scatter, > 0 means backscatter).

    Args:
        pred_3x3: [N, 3, 3] predicted stress.
        grad_3x3: [N, 3, 3] velocity gradient.

    Returns:
        Scalar fraction in [0, 1].
    """
    Pi = sgs_dissipation(pred_3x3, grad_3x3)   # [N]; positive = forward scatter
    return float((Pi < 0).float().mean().item())


# ------------------------------------------------------------------
# 7. Results CSV helpers  (Task 2/3 shared output)
# ------------------------------------------------------------------

def save_results_csv(
    results_dict: dict,
    model_name: str,
    csv_path,
) -> None:
    """
    Append one row (model_name + all metric values) to a shared CSV.

    Args:
        results_dict: output of compute_all_metrics or compute_all_metrics_np.
        model_name:   string identifier, e.g. 'smagorinsky', 'se3_cfm'.
        csv_path:     pathlib.Path or str; file is created if absent.
    """
    import csv
    from pathlib import Path

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    flat = _flatten_metrics(results_dict)
    flat['model'] = model_name

    write_header = not csv_path.exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(flat.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(flat)


def load_results_csv(csv_path) -> dict:
    """
    Load the results CSV and return a dict keyed by model_name.

    Returns:
        { model_name: { metric: value } }
    """
    import csv
    from pathlib import Path

    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {}

    results = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.pop('model')
            results[name] = {k: _try_float(v) for k, v in row.items()}
    return results


def _try_float(v: str):
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def _flatten_metrics(d: dict, prefix: str = '') -> dict:
    """Recursively flatten a nested metrics dict into dot-separated keys."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_metrics(v, key))
        else:
            try:
                out[key] = float(v)
            except (TypeError, ValueError):
                out[key] = v
    return out
