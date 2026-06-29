import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REFERENCE_CASES = {
    "naca0012": {"name": "NACA 0012", "chord": 1.0, "alpha_deg": 5.0, "Re": 3000000.0},
    "flat_plate": {"name": "Flat Plate", "chord": 1.0, "alpha_deg": 0.0, "Re": 1000000.0},
    "cylinder": {"name": "Cylinder", "diameter": 1.0, "Re": 100.0},
}

def get_reference_case(name):
    if name not in REFERENCE_CASES:
        raise ValueError(f"unknown case: {name}")
    return REFERENCE_CASES[name]

def compute_differences(tau_eng, tau_ref):
    diff = tau_eng - tau_ref
    t1 = tau_eng.reshape(-1).float()
    t2 = tau_ref.reshape(-1).float()
    vx = t1 - t1.mean()
    vy = t2 - t2.mean()
    denom = vx.norm() * vy.norm()
    corr = (vx * vy).sum() / denom if denom > 0 else torch.tensor(0.0)
    metrics = {
        "mean_diff": diff.abs().mean().item(),
        "max_diff": diff.abs().max().item(),
        "pearson_r": corr.item(),
    }
    return diff, metrics

def plot_comparison(tau_eng, tau_ref, grad_eng, grad_ref, grid_size=(32, 32)):
    diff, metrics = compute_differences(tau_eng, tau_ref)
    nx, ny = grid_size
    n = min(nx * ny, tau_eng.shape[0])
    t_eng = tau_eng[:n, 0, 0].numpy().reshape(nx, ny)
    t_ref = tau_ref[:n, 0, 0].numpy().reshape(nx, ny)
    t_dif = diff[:n, 0, 0].detach().numpy().reshape(nx, ny)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    vmax = max(abs(t_eng).max(), abs(t_ref).max(), 1e-10)
    axes[0].imshow(t_eng, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower", aspect="auto")
    axes[0].set_title("V1 prediction  tau_xx", color="white", fontsize=10)
    axes[1].imshow(t_ref, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower", aspect="auto")
    axes[1].set_title("Reference  tau_xx", color="white", fontsize=10)
    axes[2].imshow(t_dif, cmap="plasma", origin="lower", aspect="auto")
    axes[2].set_title("Difference", color="white", fontsize=10)

    for ax in axes:
        ax.tick_params(colors="#888", labelsize=7)
        ax.set_xlabel("x", color="#aaa", fontsize=8)
        ax.set_ylabel("y", color="#aaa", fontsize=8)

    r = metrics["pearson_r"]
    md = metrics["mean_diff"]
    fig.suptitle(f"AF3  |  Pearson r={r:.3f}  mean diff={md:.4f}",
                 color="white", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig, metrics
