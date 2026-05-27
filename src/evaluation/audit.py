"""
Rotational equivariance audit (Section 5 of the task spec).

Protocol (following McConkey et al. #8):
  - Take 50 held-out test sub-cubes.
  - Apply 100 random SO(3) rotations to each.
  - Measure equivariance error: ||f(Rx) - R f(x)|| / ||R f(x)||

Expected:
  - V1 (SE(3)-equivariant EGNN): machine-precision error, independent of rotation angle.
  - V3 (augmentation only): error grows as rotation deviates from the 24 training orientations.

Also reproduces the equivariance error vs. generalisation error scatter plot.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.data import Data, Batch

from data.augmentation import random_so3_rotation
from data.dataset import grad_to_irreps, stress_to_irreps, irreps_to_stress


# ------------------------------------------------------------------
# Rotate a graph
# ------------------------------------------------------------------

def rotate_graph(data: Data, R: torch.Tensor) -> Data:
    """
    Apply rotation R to a torch_geometric Data object.

    Rotates:
      - .pos        (node positions, polar vectors)
      - .x          (velocity gradient irreps: need to reconstruct 3x3, rotate, re-encode)
      - .y          (stress irreps: same)
      - .grad_full  (flat 3x3 gradient)
      - .tau_full   (flat 3x3 stress)

    Args:
        data: single graph Data object.
        R:    [3, 3] rotation matrix.

    Returns:
        New Data object with all tensors rotated.
    """
    # Rotate positions
    new_pos = data.pos @ R.T    # [N, 3]

    # Rotate velocity gradient: G' = R G R^T
    grad3x3 = data.grad_full.reshape(-1, 3, 3)
    new_grad3x3 = R @ grad3x3 @ R.T
    new_grad_full = new_grad3x3.reshape(-1, 9)
    new_x = grad_to_irreps(new_grad3x3)

    # Rotate stress: tau' = R tau R^T
    tau3x3 = data.tau_full.reshape(-1, 3, 3)
    new_tau3x3 = R @ tau3x3 @ R.T
    new_tau_full = new_tau3x3.reshape(-1, 9)
    new_y = stress_to_irreps(new_tau3x3)

    return Data(
        x=new_x,
        y=new_y,
        pos=new_pos,
        edge_index=data.edge_index.clone(),
        tau_full=new_tau_full,
        grad_full=new_grad_full,
    )


# ------------------------------------------------------------------
# Equivariance error per sample
# ------------------------------------------------------------------

@torch.no_grad()
def equivariance_error_v1(
    model: nn.Module,
    data: Data,
    R: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Compute equivariance error for V1 (EGNN-based CFM).

    Error = ||f(Rx) - R·f(x)|| / ||R·f(x)||

    For the EGNN deterministic component (regression mode):
    we use the V2-style forward (no ODE) for speed during the audit.

    Args:
        model: SE3EquivariantEGNN (called with use_flow_matching=False) or
               EGNNFlowMatching (we use its .vf directly with fixed t=0.5).
        data:  single graph (already on CPU).
        R:     [3, 3] rotation matrix on device.
    """
    from models.egnn import SE3EquivariantEGNN
    from models.flow_matching import EGNNFlowMatching

    model.eval()

    def _forward(d: Data) -> torch.Tensor:
        d = d.to(device)
        batch = Batch.from_data_list([d])
        # Always compute x from grad_full so original and rotated graphs
        # both use unnormalized irreps — avoids normalization inconsistency.
        x_raw = grad_to_irreps(batch.grad_full.reshape(-1, 3, 3))
        if isinstance(model, EGNNFlowMatching):
            n = x_raw.shape[0]
            x_t = torch.zeros(n, 6, device=device)
            t = torch.full((n, 1), 0.5, device=device)
            return model.vf(x=x_raw, pos=batch.pos,
                             edge_index=batch.edge_index, x_t=x_t, t=t)
        elif isinstance(model, SE3EquivariantEGNN):
            return model(x=x_raw, pos=batch.pos, edge_index=batch.edge_index)
        else:
            raise TypeError

    # f(x) — forward on original
    fx = _forward(data)   # [N, 6]

    # f(Rx) — forward on rotated input
    R_cpu = R.cpu()
    data_rot = rotate_graph(data, R_cpu)
    fRx = _forward(data_rot)   # [N, 6]

    # R·f(x): rotate the stress output
    # irreps -> 3x3 -> rotate -> irreps
    fx_3x3 = irreps_to_stress(fx.cpu())              # [N, 3, 3]
    Rfx_3x3 = R_cpu @ fx_3x3 @ R_cpu.T              # [N, 3, 3]
    Rfx = stress_to_irreps(Rfx_3x3).to(device)      # [N, 6]

    num = (fRx - Rfx).norm(dim=-1)                  # [N]
    den = Rfx.norm(dim=-1).clamp(min=1e-8)          # [N]
    return float((num / den).mean().item())


@torch.no_grad()
def equivariance_error_v3(
    model: nn.Module,
    data: Data,
    R: torch.Tensor,
    device: torch.device,
) -> float:
    """
    Compute equivariance error for V3 (MLP-based CFM).

    Uses fixed t=0.5, x_t=0 for deterministic comparison.
    """
    from models.flow_matching import MLPFlowMatching

    model.eval()

    def _mlp_forward(d: Data) -> torch.Tensor:
        d = d.to(device)
        grad3x3 = d.grad_full.reshape(-1, 3, 3)
        grad_irr = grad_to_irreps(grad3x3)
        n = grad_irr.shape[0]
        x_t = torch.zeros(n, 6, device=device)
        t = torch.full((n,), 0.5, device=device)
        return model.vf(grad_irr, x_t, t)

    fx = _mlp_forward(data)

    R_cpu = R.cpu()
    data_rot = rotate_graph(data, R_cpu)
    fRx = _mlp_forward(data_rot)

    fx_3x3 = irreps_to_stress(fx.cpu())
    Rfx_3x3 = R_cpu @ fx_3x3 @ R_cpu.T
    Rfx = stress_to_irreps(Rfx_3x3).to(device)

    num = (fRx - Rfx).norm(dim=-1)
    den = Rfx.norm(dim=-1).clamp(min=1e-8)
    return float((num / den).mean().item())


# ------------------------------------------------------------------
# Full audit
# ------------------------------------------------------------------

def run_equivariance_audit(
    models: dict,
    test_loader,
    device: torch.device,
    n_subcubes: int = 50,
    n_rotations: int = 100,
    out_dir: str = './results',
) -> dict:
    """
    Run the equivariance audit for all variants and produce the scatter plot.

    Args:
        models: dict mapping variant name ('v1', 'v2', 'v3') to nn.Module.
        test_loader: test DataLoader.
        n_subcubes:  number of held-out test sub-cubes to use.
        n_rotations: number of random SO(3) rotations per sub-cube.
        out_dir:     directory for saving plots.

    Returns:
        results dict with equivariance errors per variant.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect held-out sub-cubes (one graph = one sub-cube from a batch)
    test_graphs = []
    for batch in test_loader:
        for g in batch.to_data_list():
            test_graphs.append(g)
            if len(test_graphs) >= n_subcubes:
                break
        if len(test_graphs) >= n_subcubes:
            break
    test_graphs = test_graphs[:n_subcubes]
    print(f"  Audit: {len(test_graphs)} sub-cubes × {n_rotations} rotations")

    # Sample random SO(3) rotations
    rotations = random_so3_rotation(n_rotations, device=device)  # [100, 3, 3]

    results = {}
    all_errors = {}

    for variant, model in models.items():
        if model is None:
            continue
        errors = np.zeros((n_subcubes, n_rotations))
        err_fn = equivariance_error_v1 if variant in ('v1', 'v2') else equivariance_error_v3

        for i, graph in enumerate(test_graphs):
            for j, R in enumerate(rotations):
                errors[i, j] = err_fn(model, graph, R, device)

        mean_err = errors.mean()
        std_err  = errors.std()
        all_errors[variant] = errors
        results[variant] = {
            'mean': float(mean_err),
            'std': float(std_err),
            'median': float(np.median(errors)),
        }
        print(f"  {variant.upper()}: equivariance error = {mean_err:.4e} ± {std_err:.4e}")

    # Scatter plot: equivariance error vs. rotation angle
    _plot_equivariance_scatter(all_errors, rotations.cpu().numpy(), out_dir)

    return results


def _rotation_angle_deg(R: np.ndarray) -> float:
    """Compute the rotation angle (degrees) from a 3x3 rotation matrix."""
    cos_angle = (np.trace(R) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _plot_equivariance_scatter(
    all_errors: dict,
    rotations: np.ndarray,
    out_dir: Path,
):
    """Reproduce the McConkey et al. scatter plot: equivariance error vs. rotation angle."""
    angles = np.array([_rotation_angle_deg(R) for R in rotations])  # [n_rot]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {'v1': '#1f77b4', 'v2': '#2ca02c', 'v3': '#d62728'}
    labels = {'v1': 'V1 (SE(3)-equivariant CFM)', 'v2': 'V2 (Equivariant regression)', 'v3': 'V3 (Aug. CFM)'}

    for variant, errors in all_errors.items():
        # Mean over sub-cubes for each rotation
        mean_per_rot = errors.mean(axis=0)   # [n_rotations]
        ax.scatter(angles, mean_per_rot, s=8, alpha=0.5,
                   color=colors.get(variant, 'gray'),
                   label=labels.get(variant, variant))

    ax.set_xlabel('Rotation angle (deg)', fontsize=12)
    ax.set_ylabel('Equivariance error  ||f(Rx)-Rf(x)|| / ||Rf(x)||', fontsize=10)
    ax.set_title('Equivariance error vs. rotation angle (McConkey et al. protocol)', fontsize=11)
    ax.legend(fontsize=10)
    ax.set_yscale('log')
    plt.tight_layout()
    fig.savefig(out_dir / 'equivariance_audit.png', dpi=150)
    plt.close(fig)
    print(f"  Saved equivariance plot to {out_dir / 'equivariance_audit.png'}")
