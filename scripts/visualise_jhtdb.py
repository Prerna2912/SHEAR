"""
JHTDB DNS dashboard — rich 4×3 figure showing velocity, vorticity, SGS stress, and spectra.

Usage:
    uv run scripts/visualise_jhtdb.py
    uv run scripts/visualise_jhtdb.py --save
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ── data ──────────────────────────────────────────────────────────────────────

def fetch_jhtdb_patch(n: int = 16) -> np.ndarray:
    """Return [n,n,n,3] float32 velocity from JHTDB (testing token)."""
    token = os.environ.get("JHTDB_TOKEN", "edu.jhu.pha.turbulence.testing-201406")
    from givernylocal.turbulence_dataset import turb_dataset
    from givernylocal.turbulence_toolkit import getCutout

    cube = turb_dataset(
        dataset_title="isotropic1024coarse",
        output_path="/tmp/jhtdb_viz",
        auth_token=token,
    )
    axes    = np.array([[1, n], [1, n], [1, n], [1, 1]], dtype=np.int32)
    strides = np.array([1, 1, 1, 1], dtype=np.int32)
    ds      = getCutout(cube, "velocity", axes, strides, verbose=False)
    return ds["velocity_0001"].values.astype(np.float32)  # [n,n,n,3]


def derived_fields(vel: np.ndarray, dx: float):
    """Compute grad_u, vorticity, strain-rate, SGS stress, energy from velocity."""
    from data.jhtdb import compute_velocity_gradient

    u, v, w = vel[..., 0], vel[..., 1], vel[..., 2]
    grad_u  = compute_velocity_gradient(u, v, w, dx)

    # vorticity  ω = curl(u)
    omega_x = grad_u[..., 2, 1] - grad_u[..., 1, 2]   # dw/dy - dv/dz
    omega_y = grad_u[..., 0, 2] - grad_u[..., 2, 0]   # du/dz - dw/dx
    omega_z = grad_u[..., 1, 0] - grad_u[..., 0, 1]   # dv/dx - du/dy

    S     = 0.5 * (grad_u + grad_u.transpose(0, 1, 2, 4, 3))
    Smag  = np.sqrt(2.0 * np.sum(S ** 2, axis=(-2, -1)))
    R     = 0.5 * (grad_u - grad_u.transpose(0, 1, 2, 4, 3))
    Omega = np.sqrt(2.0 * np.sum(R ** 2, axis=(-2, -1)))

    Q = 0.5 * (Omega**2 - Smag**2)          # Q-criterion

    Cs, Delta = 0.17, dx
    tau = (-2.0 * (Cs * Delta) ** 2 * Smag[..., np.newaxis, np.newaxis] * S).astype(np.float32)

    # energy spectrum
    N = u.shape[0]
    u_hat = np.fft.fftn(u) / N**3
    v_hat = np.fft.fftn(v) / N**3
    w_hat = np.fft.fftn(w) / N**3
    k1d   = np.fft.fftfreq(N, d=1.0 / N).astype(int)
    KX, KY, KZ = np.meshgrid(k1d, k1d, k1d, indexing="ij")
    K  = np.sqrt(KX**2 + KY**2 + KZ**2).astype(int)
    Ek = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2)
    k_max = N // 2
    E_k   = np.zeros(k_max + 1)
    for k in range(1, k_max + 1):
        E_k[k] = Ek[K == k].sum()
    ks = np.arange(k_max + 1)

    return dict(
        u=u, v=v, w=w,
        omega_x=omega_x, omega_y=omega_y, omega_z=omega_z,
        Smag=Smag, Q=Q,
        tau=tau, grad_u=grad_u,
        ks=ks, E_k=E_k,
    )


# ── plotting helpers ──────────────────────────────────────────────────────────

def _div_im(ax, data, title, fig, cmap="RdBu_r", label=""):
    vmax = np.abs(data).max() * 0.95
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im   = ax.imshow(data, origin="lower", cmap=cmap, norm=norm)
    cb   = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.2g")
    cb.set_label(label, fontsize=7)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    _stats(ax, data)


def _pos_im(ax, data, title, fig, cmap="inferno", label=""):
    im = ax.imshow(data, origin="lower", cmap=cmap)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, format="%.2g")
    cb.set_label(label, fontsize=7)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    _stats(ax, data)


def _stats(ax, data):
    ax.text(0.02, 0.97, f"μ={data.mean():.2g}  σ={data.std():.2g}",
            transform=ax.transAxes, fontsize=6.5, va="top",
            color="white", bbox=dict(fc="black", alpha=0.5, pad=1.5))


# ── main figure ───────────────────────────────────────────────────────────────

def make_dashboard(f: dict, save_path=None):
    N  = f["u"].shape[0]
    z0 = N // 2

    style = {
        "axes.facecolor":   "#0d1117",
        "figure.facecolor": "#0d1117",
        "text.color":       "#e6edf3",
        "axes.labelcolor":  "#e6edf3",
        "xtick.color":      "#e6edf3",
        "ytick.color":      "#e6edf3",
        "axes.edgecolor":   "#30363d",
        "grid.color":       "#21262d",
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(17, 20), dpi=120)
        fig.patch.set_facecolor("#0d1117")

        gs = fig.add_gridspec(4, 3, hspace=0.38, wspace=0.28,
                              left=0.05, right=0.97, top=0.93, bottom=0.05)

        # ── Row 0: velocity components ─────────────────────────────────
        ax = fig.add_subplot(gs[0, 0])
        _div_im(ax, f["u"][:, :, z0], f"u  velocity  (z={z0})", fig, label="m/s")

        ax = fig.add_subplot(gs[0, 1])
        _div_im(ax, f["v"][:, :, z0], f"v  velocity  (z={z0})", fig, label="m/s")

        ax = fig.add_subplot(gs[0, 2])
        _div_im(ax, f["w"][:, :, z0], f"w  velocity  (z={z0})", fig, label="m/s")

        # ── Row 1: vorticity ───────────────────────────────────────────
        ax = fig.add_subplot(gs[1, 0])
        _div_im(ax, f["omega_x"][:, :, z0], r"Vorticity  $\omega_x$  (z="+str(z0)+")",
                fig, cmap="PuOr", label="1/s")

        ax = fig.add_subplot(gs[1, 1])
        _div_im(ax, f["omega_y"][:, :, z0], r"Vorticity  $\omega_y$  (z="+str(z0)+")",
                fig, cmap="PuOr", label="1/s")

        ax = fig.add_subplot(gs[1, 2])
        _div_im(ax, f["omega_z"][:, :, z0], r"Vorticity  $\omega_z$  (z="+str(z0)+")",
                fig, cmap="PuOr", label="1/s")

        # ── Row 2: SGS stress + strain rate ────────────────────────────
        ax = fig.add_subplot(gs[2, 0])
        _div_im(ax, f["tau"][:, :, z0, 0, 0], r"SGS stress  $\tau_{11}$  (z="+str(z0)+")",
                fig, cmap="seismic", label="Pa")

        ax = fig.add_subplot(gs[2, 1])
        _div_im(ax, f["tau"][:, :, z0, 0, 1], r"SGS stress  $\tau_{12}$  (z="+str(z0)+")",
                fig, cmap="seismic", label="Pa")

        ax = fig.add_subplot(gs[2, 2])
        _pos_im(ax, f["Smag"][:, :, z0], r"Strain-rate magnitude  $|S|$  (z="+str(z0)+")",
                fig, cmap="hot", label="1/s")

        # ── Row 3: spectra, PDF, scatter ───────────────────────────────
        ax = fig.add_subplot(gs[3, 0])
        ks, Ek = f["ks"], f["E_k"]
        k_plot = ks[1:]
        ax.loglog(k_plot, Ek[1:], lw=2.0, color="#58a6ff", label="E(k)")
        k_ref = k_plot[(k_plot >= 2) & (k_plot <= N // 4)]
        if len(k_ref):
            ax.loglog(k_ref, 0.3 * Ek[max(2, k_ref[0])] * (k_ref / k_ref[0])**(-5/3),
                      "--", color="#f0883e", lw=1.5, label=r"$k^{-5/3}$")
        ax.set_xlabel("wavenumber  k", fontsize=8)
        ax.set_ylabel("E(k)", fontsize=8)
        ax.set_title("Energy spectrum", fontsize=9, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.3)
        ax.grid(True, alpha=0.25, which="both")
        ax.tick_params(labelsize=7)

        ax = fig.add_subplot(gs[3, 1])
        tau11 = f["tau"][..., 0, 0].ravel()
        ax.hist(tau11, bins=60, density=True, color="#58a6ff",
                alpha=0.85, edgecolor="#0d1117", linewidth=0.4)
        mu, sig = tau11.mean(), tau11.std()
        xg = np.linspace(tau11.min(), tau11.max(), 200)
        ax.plot(xg, np.exp(-0.5 * ((xg - mu) / sig)**2) / (sig * np.sqrt(2 * np.pi)),
                color="#f0883e", lw=1.8, label="Gaussian")
        ax.set_xlabel(r"$\tau_{11}$", fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.set_title(r"PDF of  $\tau_{11}$", fontsize=9, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.3)
        ax.grid(True, alpha=0.25)
        ax.text(0.97, 0.97, f"kurt={float(np.mean(((tau11-mu)/sig)**4)):.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                color="#e6edf3", bbox=dict(fc="#161b22", alpha=0.7, pad=2))
        ax.tick_params(labelsize=7)

        ax = fig.add_subplot(gs[3, 2])
        rng = np.random.default_rng(42)
        n_pts = min(800, tau11.size)
        idx   = rng.integers(0, tau11.size, n_pts)
        g11   = f["grad_u"][..., 0, 0].ravel()[idx]
        t11   = tau11[idx]
        sc = ax.scatter(g11, t11, s=7, alpha=0.55, c=np.abs(t11),
                        cmap="plasma", linewidths=0)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=r"$|\tau_{11}|$",
                     format="%.2g")
        ax.set_xlabel(r"$\partial u/\partial x$", fontsize=8)
        ax.set_ylabel(r"$\tau_{11}$", fontsize=8)
        ax.set_title(r"$\tau_{11}$ vs  $\partial u/\partial x$", fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)

        # ── title ──────────────────────────────────────────────────────
        fig.suptitle(
            f"JHTDB  isotropic1024coarse — real DNS  ({N}³ patch, t=1)\n"
            "Forced isotropic turbulence at  Reλ ≈ 433,  1024³ DNS",
            fontsize=12, fontweight="bold", color="#e6edf3", y=0.97,
        )

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"Saved → {save_path}")
        else:
            plt.show()

        plt.close(fig)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="Save to figures/jhtdb_dashboard.png")
    args = ap.parse_args()

    print("Fetching 16³ real JHTDB DNS patch …")
    vel = fetch_jhtdb_patch(n=16)
    print(f"  vel shape: {vel.shape}  |u|_max={np.abs(vel).max():.4f}")

    dx = 2 * np.pi / 1024   # DNS grid spacing
    print("Computing derived fields (vorticity, SGS stress, spectrum) …")
    fields = derived_fields(vel, dx)

    out = str(Path(__file__).resolve().parents[1] / "figures" / "jhtdb_dashboard.png")
    make_dashboard(fields, save_path=out if args.save else None)

    if not args.save:
        print("Re-saving anyway so you can open the file …")
        make_dashboard(fields, save_path=out)
        print(f"  → {out}")


if __name__ == "__main__":
    main()
