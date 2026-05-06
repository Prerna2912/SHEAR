"""
Task 3: Failure-case visualisation and diagnostic plots.

Functions:
  worst_predictions         — indices of N worst-predicted sub-cubes by correlation.
  plot_tau_slices           — side-by-side predicted vs true τ_xy 2D mid-plane slices.
  plot_q_overlay            — τ_xy slices with Q-criterion contour overlay.
  plot_correlation_scatter  — per-component scatter predicted vs true.
  plot_invariant_distributions — I1/I2/I3 overlaid histograms.
  plot_equivariance_scatter — equivariance error vs rotation angle (McConkey protocol).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMP_MAP = {
    'tau_xx': (0, 0), 'tau_yy': (1, 1), 'tau_zz': (2, 2),
    'tau_xy': (0, 1), 'tau_xz': (0, 2), 'tau_yz': (1, 2),
}


def _get_component(tau_3x3: np.ndarray, component: str) -> np.ndarray:
    """Extract one 3×3 component from [N, 3, 3] array."""
    i, j = _COMP_MAP[component]
    return tau_3x3[:, i, j]


def _subcube_midslice(tau_vals: np.ndarray, patch_size: int = 8) -> np.ndarray:
    """
    Reshape [patch_size³] → [patch_size, patch_size, patch_size] and
    return the mid-plane (xy slice at z = patch_size//2).
    """
    cube = tau_vals.reshape(patch_size, patch_size, patch_size)
    return cube[:, :, patch_size // 2]


# ---------------------------------------------------------------------------
# Worst-prediction ranking
# ---------------------------------------------------------------------------

def worst_predictions(
    pred_3x3: np.ndarray,
    true_3x3: np.ndarray,
    n: int = 10,
    n_nodes_per_sample: int = 512,
) -> np.ndarray:
    """
    Find the n worst-predicted sub-cubes by mean Pearson r across τ components.

    Args:
        pred_3x3: [N_total, 3, 3] predictions.
        true_3x3: [N_total, 3, 3] ground truth.
        n:        number of worst cases to return.
        n_nodes_per_sample: nodes per sub-cube.

    Returns:
        indices: [n] indices of worst sub-cubes (sorted, worst first).
    """
    from scipy.stats import pearsonr

    N = pred_3x3.shape[0]
    n_samples = N // n_nodes_per_sample
    components = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]

    per_sample_r = np.zeros(n_samples)
    for s in range(n_samples):
        sl = slice(s * n_nodes_per_sample, (s + 1) * n_nodes_per_sample)
        rs = []
        for i, j in components:
            p = pred_3x3[sl, i, j]
            t = true_3x3[sl, i, j]
            if p.std() < 1e-10 or t.std() < 1e-10:
                rs.append(0.0)
            else:
                r, _ = pearsonr(p, t)
                rs.append(float(r))
        per_sample_r[s] = float(np.mean(rs))

    worst_idx = np.argsort(per_sample_r)[:n]
    return worst_idx


# ---------------------------------------------------------------------------
# 2D slice plots
# ---------------------------------------------------------------------------

def plot_tau_slices(
    pred_3x3: np.ndarray,
    true_3x3: np.ndarray,
    indices: List[int],
    component: str = 'tau_xy',
    patch_size: int = 8,
    save_path: Optional[str] = None,
):
    """
    Side-by-side predicted vs true τ mid-plane slices for a list of sub-cubes.

    Args:
        pred_3x3: [N_total, 3, 3] predictions.
        true_3x3: [N_total, 3, 3] ground truth.
        indices:  sub-cube indices to plot.
        component: τ component name ('tau_xy', 'tau_xx', etc.).
        patch_size: spatial side of one sub-cube.
        save_path: if given, save figure here.
    """
    n = len(indices)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    nodes = patch_size ** 3
    i_idx, j_idx = _COMP_MAP[component]

    for row, s in enumerate(indices):
        sl = slice(s * nodes, (s + 1) * nodes)
        p_vals = pred_3x3[sl, i_idx, j_idx]
        t_vals = true_3x3[sl, i_idx, j_idx]

        p_slice = _subcube_midslice(p_vals, patch_size)
        t_slice = _subcube_midslice(t_vals, patch_size)

        vmin = min(p_slice.min(), t_slice.min())
        vmax = max(p_slice.max(), t_slice.max())

        im0 = axes[row, 0].imshow(p_slice, vmin=vmin, vmax=vmax, cmap='RdBu_r')
        axes[row, 0].set_title(f'Pred  (sub-cube {s})', fontsize=9)
        axes[row, 0].axis('off')

        im1 = axes[row, 1].imshow(t_slice, vmin=vmin, vmax=vmax, cmap='RdBu_r')
        axes[row, 1].set_title(f'True  (sub-cube {s})', fontsize=9)
        axes[row, 1].axis('off')

        fig.colorbar(im1, ax=axes[row, :].tolist(), shrink=0.8)

    fig.suptitle(f'{component}  mid-plane slices  (worst predictions)', fontsize=11)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_q_overlay(
    pred_3x3: np.ndarray,
    true_3x3: np.ndarray,
    grad_3x3: np.ndarray,
    indices: List[int],
    component: str = 'tau_xy',
    patch_size: int = 8,
    save_path: Optional[str] = None,
):
    """
    τ mid-plane slices with Q-criterion contour overlay.

    Args:
        grad_3x3: [N_total, 3, 3] velocity gradients (used for Q).
        other args: same as plot_tau_slices.
    """
    from evaluation.regime import q_criterion_np

    n = len(indices)
    fig, axes = plt.subplots(n, 2, figsize=(7, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    nodes = patch_size ** 3
    i_idx, j_idx = _COMP_MAP[component]

    for row, s in enumerate(indices):
        sl = slice(s * nodes, (s + 1) * nodes)
        p_slice = _subcube_midslice(pred_3x3[sl, i_idx, j_idx], patch_size)
        t_slice = _subcube_midslice(true_3x3[sl, i_idx, j_idx], patch_size)
        q_slice = _subcube_midslice(
            q_criterion_np(grad_3x3[sl]), patch_size)

        vmin = min(p_slice.min(), t_slice.min())
        vmax = max(p_slice.max(), t_slice.max())

        for col, (data, label) in enumerate([(p_slice, 'Pred'), (t_slice, 'True')]):
            ax = axes[row, col]
            im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap='RdBu_r', origin='lower')
            ax.contour(q_slice, levels=5, colors='k', linewidths=0.5, alpha=0.6)
            ax.set_title(f'{label}  +Q (sub-cube {s})', fontsize=9)
            ax.axis('off')
            fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f'{component}  with Q-criterion contours', fontsize=11)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Diagnostic scatter and distribution plots
# ---------------------------------------------------------------------------

def plot_correlation_scatter(
    pred_3x3: np.ndarray,
    true_3x3: np.ndarray,
    component: str = 'tau_xy',
    save_path: Optional[str] = None,
    max_points: int = 5000,
):
    """Predicted vs true scatter for one τ component."""
    from scipy.stats import pearsonr

    i_idx, j_idx = _COMP_MAP[component]
    p = pred_3x3[:, i_idx, j_idx]
    t = true_3x3[:, i_idx, j_idx]

    if len(p) > max_points:
        idx = np.random.default_rng(0).choice(len(p), max_points, replace=False)
        p, t = p[idx], t[idx]

    r, _ = pearsonr(p, t)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(t, p, s=4, alpha=0.3, color='#1f77b4')
    lims = [min(t.min(), p.min()), max(t.max(), p.max())]
    ax.plot(lims, lims, 'k--', lw=1, label='perfect prediction')
    ax.set_xlabel(f'True {component}', fontsize=11)
    ax.set_ylabel(f'Predicted {component}', fontsize=11)
    ax.set_title(f'{component}  Pearson r = {r:.4f}', fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_invariant_distributions(
    pred_3x3: np.ndarray,
    true_3x3: np.ndarray,
    save_path: Optional[str] = None,
    n_bins: int = 60,
):
    """
    Overlaid histograms for I1, I2, I3 with JS divergence annotation.

    Args:
        pred_3x3: [N, 3, 3] predictions.
        true_3x3: [N, 3, 3] ground truth.
    """
    from scipy.spatial.distance import jensenshannon
    import torch

    def _invariants(tau: np.ndarray) -> np.ndarray:
        I1 = tau[:, 0, 0] + tau[:, 1, 1] + tau[:, 2, 2]
        t = torch.from_numpy(tau)
        t2 = torch.bmm(t, t)
        I2 = 0.5 * (I1 ** 2 - (t2[:, 0, 0] + t2[:, 1, 1] + t2[:, 2, 2]).numpy())
        I3 = np.linalg.det(tau)
        return np.stack([I1, I2, I3], axis=1)

    pred_inv = _invariants(pred_3x3)
    true_inv = _invariants(true_3x3)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = ['$I_1$', '$I_2$', '$I_3$']

    for k, (ax, lbl) in enumerate(zip(axes, labels)):
        lo = min(true_inv[:, k].min(), pred_inv[:, k].min()) - 1e-10
        hi = max(true_inv[:, k].max(), pred_inv[:, k].max()) + 1e-10
        bins = np.linspace(lo, hi, n_bins + 1)

        ph, _ = np.histogram(pred_inv[:, k], bins=bins, density=True)
        th, _ = np.histogram(true_inv[:, k], bins=bins, density=True)
        ph = ph + 1e-10; ph /= ph.sum()
        th = th + 1e-10; th /= th.sum()
        jsd = float(jensenshannon(ph, th, base=2))

        centers = 0.5 * (bins[:-1] + bins[1:])
        ax.plot(centers, th, label='True', color='#1f77b4', lw=1.5)
        ax.plot(centers, ph, label='Pred', color='#d62728', lw=1.5, ls='--')
        ax.set_title(f'{lbl}  JSD={jsd:.4f}', fontsize=11)
        ax.set_xlabel(lbl, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle('Stress tensor invariant distributions', fontsize=12)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_equivariance_scatter(
    results: dict,
    save_path: Optional[str] = None,
):
    """
    Equivariance error vs rotation angle scatter (McConkey et al. protocol).

    Args:
        results: dict from evaluation.audit.run_equivariance_audit.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {'v1': '#1f77b4', 'v2': '#2ca02c', 'v3': '#d62728'}
    labels = {
        'v1': 'V1 SE(3)-CFM (arch. equivariant)',
        'v2': 'V2 Equivariant regression',
        'v3': 'V3 MLP-CFM (aug. only)',
    }
    for variant, stats in results.items():
        if variant == 'equivariance_audit':
            continue
        mean = stats.get('mean', 0)
        std = stats.get('std', 0)
        ax.errorbar(
            [mean], [mean], xerr=[std], yerr=[std],
            fmt='o', markersize=10,
            color=colors.get(variant, 'gray'),
            label=f"{labels.get(variant, variant)}  {mean:.2e}±{std:.2e}",
        )
    ax.set_xlabel('Equivariance error  ||f(Rx)−Rf(x)|| / ||Rf(x)||', fontsize=11)
    ax.set_ylabel('Equivariance error (same axis)', fontsize=11)
    ax.set_title('Equivariance audit — mean ± std across 50 sub-cubes × 100 rotations',
                 fontsize=10)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig
