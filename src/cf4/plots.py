"""
CF4 — 4-panel diagnostic figure.

Panel layout:
  [0] Flow Regime Map      — Q-criterion coloured mid-plane slice
  [1] Prediction Uncertainty — per-node uncertainty with high-unc hatching
  [2] SGS Dissipation      — Π = -τ:S, diverging colormap (forward/backscatter)
  [3] Backscatter Fraction — per-node backscatter indicator + summary bar

All panels are 2D mid-plane slices (z = Nz//2 for a 3D grid).
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from typing import Tuple

from .compute import CF4Output


REGIME_COLOURS = ['#4575b4', '#74add1', '#d73027']   # blue=strain, mid=mixed, red=vortex
REGIME_LABELS  = ['Strain-dominated', 'Mixed', 'Vortex-dominated']


def _mid_slice(values: np.ndarray, grid_shape: list | tuple) -> np.ndarray:
    """Reshape [N] → [Nx, Ny, Nz] and return the z=Nz//2 mid-plane [Nx, Ny]."""
    Nx, Ny, Nz = grid_shape
    vol = values.reshape(Nx, Ny, Nz)
    return vol[:, :, Nz // 2].T   # transpose so x=columns, y=rows


def make_figure(
    cf4: CF4Output,
    grid_shape: list | tuple,
    geometry_type: str = "",
    params: dict | None = None,
    figsize: Tuple[int, int] = (16, 14),
) -> plt.Figure:
    """
    Build the 4-panel CF4 diagnostic figure.

    Args:
        cf4          : CF4Output from compute.analyse()
        grid_shape   : [Nx, Ny, Nz] from CF1/CF2 result
        geometry_type: label string for the figure title
        params       : geometry params dict (shown in subtitle)
        figsize      : matplotlib figure size

    Returns:
        fig: matplotlib Figure (caller can .savefig() or pass to Gradio)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.patch.set_facecolor('#0f1117')
    for ax in axes.flat:
        ax.set_facecolor('#1a1d27')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    _panel_regime(axes[0, 0], cf4, grid_shape)
    _panel_uncertainty(axes[0, 1], cf4, grid_shape)
    _panel_dissipation(axes[1, 0], cf4, grid_shape)
    _panel_backscatter(axes[1, 1], cf4, grid_shape)

    # Title
    subtitle = ""
    if params:
        kv = ", ".join(f"{k}={v}" for k, v in list(params.items())[:4])
        subtitle = f"{kv}"
    fig.suptitle(
        f"SHEAR CF4 — {geometry_type.replace('_', ' ').title()} Regime Diagnostics\n"
        f"{subtitle}",
        fontsize=13, color='white', y=0.98, fontweight='bold'
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# Individual panels
# ---------------------------------------------------------------------------

def _panel_regime(ax, cf4: CF4Output, grid_shape):
    """Panel 0: Flow regime map (Q-criterion coloured)."""
    q_slice = _mid_slice(cf4.q_values, grid_shape)
    labels_slice = _mid_slice(cf4.regime_labels.astype(np.float32), grid_shape)

    cmap = mcolors.ListedColormap(REGIME_COLOURS)
    im = ax.imshow(
        labels_slice, origin='lower', cmap=cmap,
        vmin=-0.5, vmax=2.5, aspect='auto', interpolation='nearest'
    )

    # Q-criterion contour overlay
    try:
        cs = ax.contour(
            q_slice, levels=5,
            colors='white', linewidths=0.5, alpha=0.4
        )
    except Exception:
        pass

    legend_patches = [
        Patch(color=REGIME_COLOURS[i], label=REGIME_LABELS[i])
        for i in range(3)
    ]
    ax.legend(handles=legend_patches, loc='upper right',
              fontsize=7, framealpha=0.6, labelcolor='white',
              facecolor='#222')

    ax.set_title('Flow Regime Map (Q-criterion)', color='white', fontsize=10, pad=8)
    ax.set_xlabel('x (streamwise)', color='#aaa', fontsize=8)
    ax.set_ylabel('y (normal)', color='#aaa', fontsize=8)
    ax.tick_params(colors='#888', labelsize=7)

    p33, p66 = cf4.regime_thresholds
    ax.text(
        0.02, 0.04,
        f"Q thresholds: {p33:.2e} | {p66:.2e}",
        transform=ax.transAxes, color='#aaa', fontsize=7
    )


def _panel_uncertainty(ax, cf4: CF4Output, grid_shape):
    """Panel 1: Prediction uncertainty heatmap."""
    unc_slice  = _mid_slice(cf4.uncertainty, grid_shape)
    mask_slice = _mid_slice(cf4.high_unc_mask.astype(np.float32), grid_shape)

    im = ax.imshow(
        unc_slice, origin='lower', cmap='plasma',
        aspect='auto', interpolation='bilinear'
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label='Uncertainty (L₂ of variance)')

    # Hatch high-uncertainty region
    ax.contourf(
        mask_slice, levels=[0.5, 1.5],
        hatches=['///'], colors='none', alpha=0.0
    )
    ax.contour(
        mask_slice, levels=[0.5],
        colors=['#ff6b6b'], linewidths=1.2, linestyles='--'
    )

    ax.set_title('Prediction Uncertainty', color='white', fontsize=10, pad=8)
    ax.set_xlabel('x (streamwise)', color='#aaa', fontsize=8)
    ax.set_ylabel('y (normal)', color='#aaa', fontsize=8)
    ax.tick_params(colors='#888', labelsize=7)

    ax.text(
        0.02, 0.04,
        f"Top-25% high-unc zone (dashed red)",
        transform=ax.transAxes, color='#ff6b6b', fontsize=7
    )


def _panel_dissipation(ax, cf4: CF4Output, grid_shape):
    """Panel 2: SGS dissipation Π = -τ:S (diverging, signed)."""
    pi_slice = _mid_slice(cf4.dissipation, grid_shape)

    abs_max = max(np.abs(pi_slice).max(), 1e-12)
    im = ax.imshow(
        pi_slice, origin='lower', cmap='RdBu_r',
        vmin=-abs_max, vmax=abs_max,
        aspect='auto', interpolation='bilinear'
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Π = −τ:S  (forward ← 0 → backscatter)', color='#aaa', fontsize=7)
    cbar.ax.tick_params(colors='#888', labelsize=6)

    # Zero contour (transition between forward scatter and backscatter)
    try:
        ax.contour(pi_slice, levels=[0], colors=['white'], linewidths=0.8, linestyles=['-'])
    except Exception:
        pass

    ax.set_title('SGS Dissipation  Π = −τ:S', color='white', fontsize=10, pad=8)
    ax.set_xlabel('x (streamwise)', color='#aaa', fontsize=8)
    ax.set_ylabel('y (normal)', color='#aaa', fontsize=8)
    ax.tick_params(colors='#888', labelsize=7)

    fwd  = (cf4.dissipation > 0).mean() * 100
    back = cf4.backscatter_frac * 100
    ax.text(
        0.02, 0.04,
        f"Forward: {fwd:.1f}%   Backscatter: {back:.1f}%",
        transform=ax.transAxes, color='white', fontsize=7
    )


def _panel_backscatter(ax, cf4: CF4Output, grid_shape):
    """Panel 3: Backscatter map + per-regime bar chart."""
    bs_slice = _mid_slice(cf4.backscatter_mask.astype(np.float32), grid_shape)
    regime_slice = _mid_slice(cf4.regime_labels.astype(np.float32), grid_shape)

    # Left portion: spatial backscatter map
    # Split axis: map on left, bar chart on right
    divider_x = int(grid_shape[0] * 0.65)

    cmap_bs = mcolors.ListedColormap(['#1a1d27', '#ff6b35'])
    im = ax.imshow(
        bs_slice, origin='lower', cmap=cmap_bs,
        vmin=0, vmax=1, aspect='auto', interpolation='nearest'
    )

    ax.set_title(
        f'Backscatter Map  (fraction={cf4.backscatter_frac:.1%})',
        color='white', fontsize=10, pad=8
    )
    ax.set_xlabel('x (streamwise)', color='#aaa', fontsize=8)
    ax.set_ylabel('y (normal)', color='#aaa', fontsize=8)
    ax.tick_params(colors='#888', labelsize=7)

    legend_patches = [
        Patch(color='#1a1d27', label='Forward scatter'),
        Patch(color='#ff6b35', label='Backscatter (Π < 0)'),
    ]
    ax.legend(handles=legend_patches, loc='upper right',
              fontsize=7, framealpha=0.6, labelcolor='white',
              facecolor='#222')

    # Per-regime backscatter inset text
    for r, name in enumerate(REGIME_LABELS):
        mask_r = cf4.regime_labels == r
        frac_r = cf4.backscatter_mask[mask_r].mean() if mask_r.any() else 0.0
        ax.text(
            0.02, 0.22 - r * 0.07,
            f"{name[:7]}: {frac_r:.1%} backscatter",
            transform=ax.transAxes, color=REGIME_COLOURS[r], fontsize=6.5
        )

    # Smagorinsky comparison note
    ax.text(
        0.02, 0.04,
        "Smagorinsky always Π ≥ 0 (no backscatter)",
        transform=ax.transAxes, color='#888', fontsize=6.5, fontstyle='italic'
    )
