"""
Quick dataset visualisation for demonstration purposes.

Usage:
    python scripts/show_sample.py                  # synthetic data
    python scripts/show_sample.py --real           # 16^3 real JHTDB sample
    python scripts/show_sample.py --real --save    # save figure to figures/

Produces a 2x3 figure:
  Row 1 — z-midplane slices: u velocity | SGS stress tau_11 | |S| strain rate
  Row 2 — energy spectrum | SGS stress PDF | tau vs grad-u scatter
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ------------------------------------------------------------------
def load_synthetic(n: int = 64, seed: int = 42):
    from data.jhtdb import generate_synthetic_les_data
    tau, grad_u, vels = generate_synthetic_les_data(n_les=n, seed=seed, return_vels=True)
    return vels, tau, grad_u


def load_real_sample():
    import os
    token = os.environ.get("JHTDB_TOKEN", "edu.jhu.pha.turbulence.testing-201406")
    from givernylocal.turbulence_dataset import turb_dataset
    from givernylocal.turbulence_toolkit import getCutout

    cube = turb_dataset(
        dataset_title="isotropic1024coarse",
        output_path="/tmp/jhtdb_demo",
        auth_token=token,
    )
    axes    = np.array([[1, 16], [1, 16], [1, 16], [1, 1]], dtype=np.int32)
    strides = np.array([1, 1, 1, 1], dtype=np.int32)
    ds      = getCutout(cube, "velocity", axes, strides, verbose=False)
    vel     = ds["velocity_0001"].values.astype(np.float32)  # [16,16,16,3]
    vels    = vel

    # Compute velocity gradient and Smagorinsky SGS stress on the 16^3 patch
    from data.jhtdb import compute_velocity_gradient
    dx = 2 * np.pi / 1024   # DNS grid spacing
    u, v, w = vel[..., 0], vel[..., 1], vel[..., 2]
    grad_u  = compute_velocity_gradient(u, v, w, dx)
    S       = 0.5 * (grad_u + grad_u.transpose(0, 1, 2, 4, 3))
    Smag    = np.sqrt(2.0 * np.sum(S ** 2, axis=(-2, -1)))
    Cs, Delta = 0.17, dx
    tau     = (-2.0 * (Cs * Delta) ** 2
               * Smag[..., np.newaxis, np.newaxis] * S).astype(np.float32)

    return vels, tau, grad_u


# ------------------------------------------------------------------
def energy_spectrum(u, v, w):
    N = u.shape[0]
    u_hat = np.fft.fftn(u) / N**3
    v_hat = np.fft.fftn(v) / N**3
    w_hat = np.fft.fftn(w) / N**3

    k1d = np.fft.fftfreq(N, d=1.0 / N).astype(int)
    KX, KY, KZ = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    K = np.sqrt(KX**2 + KY**2 + KZ**2).astype(int)

    E_hat = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2)
    k_max = N // 2
    E_k   = np.zeros(k_max + 1)
    for k in range(1, k_max + 1):
        mask   = K == k
        E_k[k] = E_hat[mask].sum()

    return np.arange(k_max + 1), E_k


# ------------------------------------------------------------------
def make_figure(vels, tau, grad_u, title: str, save_path=None):
    N  = vels.shape[0]
    z0 = N // 2   # midplane slice

    u  = vels[..., 0] if vels.ndim == 4 else vels[0]
    v  = vels[..., 1] if vels.ndim == 4 else vels[1]
    w  = vels[..., 2] if vels.ndim == 4 else vels[2]

    S    = 0.5 * (grad_u + grad_u.transpose(0, 1, 2, 4, 3))
    Smag = np.sqrt(2.0 * np.sum(S**2, axis=(-2, -1)))

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # (0,0) u-velocity slice
    im = axes[0, 0].imshow(u[:, :, z0], origin="lower", cmap="RdBu_r")
    fig.colorbar(im, ax=axes[0, 0])
    axes[0, 0].set_title(f"u velocity  (z={z0})")
    axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("y")

    # (0,1) tau_11 slice
    im = axes[0, 1].imshow(tau[:, :, z0, 0, 0], origin="lower", cmap="seismic")
    fig.colorbar(im, ax=axes[0, 1])
    axes[0, 1].set_title(r"SGS stress $\tau_{11}$  (z=" + str(z0) + ")")
    axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("y")

    # (0,2) strain-rate magnitude slice
    im = axes[0, 2].imshow(Smag[:, :, z0], origin="lower", cmap="hot")
    fig.colorbar(im, ax=axes[0, 2])
    axes[0, 2].set_title(r"Strain-rate magnitude $|S|$  (z=" + str(z0) + ")")
    axes[0, 2].set_xlabel("x"); axes[0, 2].set_ylabel("y")

    # (1,0) energy spectrum
    ks, Ek = energy_spectrum(u, v, w)
    k_plot = ks[1:]
    axes[1, 0].loglog(k_plot, Ek[1:], lw=1.8, color="steelblue", label="E(k)")
    k_ref  = k_plot[(k_plot >= 2) & (k_plot <= N // 4)]
    axes[1, 0].loglog(k_ref, 0.3 * Ek[2] * (k_ref / 2)**(-5/3), "k--",
                      lw=1, label=r"$k^{-5/3}$")
    axes[1, 0].set_xlabel("wavenumber k"); axes[1, 0].set_ylabel("E(k)")
    axes[1, 0].set_title("Energy spectrum"); axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, which="both")

    # (1,1) tau_11 PDF
    tau11_flat = tau[..., 0, 0].ravel()
    axes[1, 1].hist(tau11_flat, bins=60, density=True, color="steelblue",
                    alpha=0.8, edgecolor="white", linewidth=0.3)
    axes[1, 1].set_xlabel(r"$\tau_{11}$"); axes[1, 1].set_ylabel("density")
    axes[1, 1].set_title(r"PDF of $\tau_{11}$")
    axes[1, 1].grid(True, alpha=0.3)

    # (1,2) tau_11 vs grad_u_11 scatter (1000 random points)
    rng  = np.random.default_rng(0)
    idx  = rng.integers(0, tau[..., 0, 0].size, size=min(1000, tau[..., 0, 0].size))
    g11  = grad_u[..., 0, 0].ravel()[idx]
    t11  = tau[..., 0, 0].ravel()[idx]
    axes[1, 2].scatter(g11, t11, s=6, alpha=0.5, color="steelblue")
    axes[1, 2].set_xlabel(r"$\partial u / \partial x$")
    axes[1, 2].set_ylabel(r"$\tau_{11}$")
    axes[1, 2].set_title(r"$\tau_{11}$ vs $\partial u/\partial x$ (1000 pts)")
    axes[1, 2].grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real",  action="store_true",
                        help="Use real 16³ JHTDB DNS sample (testing token)")
    parser.add_argument("--save",  action="store_true",
                        help="Save figure to figures/dataset_sample.png")
    args = parser.parse_args()

    if args.real:
        print("Fetching 16³ real JHTDB DNS sample …")
        vels, tau, grad_u = load_real_sample()
        title = "JHTDB isotropic1024coarse — real DNS sample (16³ patch, t=1)"
    else:
        print("Generating synthetic 64³ LES sample …")
        vels, tau, grad_u = load_synthetic()
        title = "Synthetic LES data — 64³ grid, Kolmogorov spectrum + Smagorinsky τ"

    save_path = None
    if args.save:
        out = Path(__file__).resolve().parents[1] / "figures" / "dataset_sample.png"
        out.parent.mkdir(exist_ok=True)
        save_path = str(out)

    make_figure(vels, tau, grad_u, title, save_path)


if __name__ == "__main__":
    main()
