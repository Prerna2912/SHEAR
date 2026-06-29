import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def compute_differences(tau_eng, tau_ref):
    diff = tau_eng - tau_ref
    diff_mag = torch.norm(diff, dim=(-2,-1))
    eng_mag = torch.norm(tau_eng, dim=(-2,-1))
    ref_mag = torch.norm(tau_ref, dim=(-2,-1))
    t1 = tau_eng.reshape(-1).float()
    t2 = tau_ref.reshape(-1).float()
    vx = t1 - t1.mean(); vy = t2 - t2.mean()
    denom = vx.norm() * vy.norm()
    corr = (vx*vy).sum() / denom if denom > 0 else torch.tensor(0.0)
    metrics = {
        "mean_diff": diff_mag.mean().item(),
        "max_diff": diff_mag.max().item(),
        "peak_tau_eng": eng_mag.max().item(),
        "peak_tau_ref": ref_mag.max().item(),
        "mean_tau_eng": eng_mag.mean().item(),
        "mean_tau_ref": ref_mag.mean().item(),
        "pearson_r": corr.item(),
    }
    return diff_mag, metrics

def plot_comparison(tau_eng, tau_ref, grad_eng, grad_ref, grid_size=(32, 32)):
    diff_mag, metrics = compute_differences(tau_eng, tau_ref)
    H, W = grid_size
    n = min(H*W, tau_eng.shape[0])
    eng_mag = torch.norm(tau_eng[:n], dim=(-2,-1)).numpy().reshape(H, W)
    ref_mag = torch.norm(tau_ref[:n], dim=(-2,-1)).numpy().reshape(H, W)
    dif_map = diff_mag[:n].detach().numpy().reshape(H, W)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    vmax = max(eng_mag.max(), ref_mag.max(), 1e-10)
    axes[0].imshow(eng_mag, cmap="viridis", vmin=0, vmax=vmax, origin="lower", aspect="auto")
    axes[0].set_title("V1 prediction  |tau|", color="white", fontsize=10)
    axes[1].imshow(ref_mag, cmap="viridis", vmin=0, vmax=vmax, origin="lower", aspect="auto")
    axes[1].set_title("Reference  |tau|", color="white", fontsize=10)
    im = axes[2].imshow(dif_map, cmap="magma", origin="lower", aspect="auto")
    axes[2].set_title(f"Difference  max={metrics['max_diff']:.3f}", color="white", fontsize=10)
    plt.colorbar(im, ax=axes[2]).ax.tick_params(colors="#888")

    for ax in axes:
        ax.tick_params(colors="#888", labelsize=7)
        ax.set_xlabel("x", color="#aaa", fontsize=8)
        ax.set_ylabel("y", color="#aaa", fontsize=8)

    r = metrics["pearson_r"]
    md = metrics["mean_diff"]
    fig.suptitle(f"AF3 Comparative Geometry Benchmarking  |  Pearson r={r:.3f}  mean diff={md:.4f}",
                 color="white", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig, metrics
