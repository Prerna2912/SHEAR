import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pathlib import Path

# load the mock response that prerona added
fixture_path = Path(__file__).parent.parent / "tests" / "fixtures" / "mock_cf2_response.json"
with open(fixture_path) as f:
    result = json.load(f)

# pull out what we need
grad_u       = np.array(result["grad_u"],       dtype=np.float32)
mean_tau     = np.array(result["mean_tau"],     dtype=np.float32)
variance_tau = np.array(result["variance_tau"], dtype=np.float32)
grid_shape   = result["grid_shape"]
params       = result["params"]
n_nodes      = result["n_nodes"]
n_samples    = result["n_samples"]

print("grad_u shape :", grad_u.shape)
print("mean_tau shape:", mean_tau.shape)
print("grid_shape    :", grid_shape)
print("n_nodes       :", n_nodes)
print("geometry      :", result["geometry_type"])

# -----------------------------------------------------------------
# Q-criterion
# Q = 0.5 * (||Omega||^2 - ||S||^2)
# positive Q means vortex dominated, negative means strain dominated
# -----------------------------------------------------------------

G  = grad_u.reshape(-1, 3, 3)
S  = 0.5 * (G + G.transpose(0, 2, 1))
Om = 0.5 * (G - G.transpose(0, 2, 1))
Q  = 0.5 * ((Om * Om).sum(axis=(-2, -1)) - (S * S).sum(axis=(-2, -1)))

# split into 3 regimes using 33rd and 66th percentile same as task 3
p33 = float(np.percentile(Q, 33.3))
p66 = float(np.percentile(Q, 66.7))

regime = np.zeros(len(Q), dtype=np.int32)
regime[Q >= p33] = 1
regime[Q >= p66] = 2

print(f"\nQ range: {Q.min():.3f} to {Q.max():.3f}")
print(f"p33={p33:.4f}  p66={p66:.4f}")
print(f"strain: {(regime==0).sum()}  mixed: {(regime==1).sum()}  vortex: {(regime==2).sum()}")

# -----------------------------------------------------------------
# uncertainty - just take L2 norm of variance across 6 components
# -----------------------------------------------------------------

unc = np.sqrt((variance_tau ** 2).sum(axis=-1))
high_unc = unc >= np.percentile(unc, 75)

print(f"\nuncertainty mean={unc.mean():.4f}  max={unc.max():.4f}")
print(f"high uncertainty nodes: {high_unc.sum()} out of {n_nodes}")

# -----------------------------------------------------------------
# SGS dissipation  Pi = - tau : S
# need to expand mean_tau from voigt [N,6] to full [N,3,3]
# voigt order assumed: xx yy zz xy xz yz
# -----------------------------------------------------------------

tau = np.zeros((n_nodes, 3, 3), dtype=np.float32)
tau[:, 0, 0] = mean_tau[:, 0]
tau[:, 1, 1] = mean_tau[:, 1]
tau[:, 2, 2] = mean_tau[:, 2]
tau[:, 0, 1] = mean_tau[:, 3]
tau[:, 1, 0] = mean_tau[:, 3]
tau[:, 0, 2] = mean_tau[:, 4]
tau[:, 2, 0] = mean_tau[:, 4]
tau[:, 1, 2] = mean_tau[:, 5]
tau[:, 2, 1] = mean_tau[:, 5]

Pi = -(tau * S).sum(axis=(-2, -1))

backscatter = Pi < 0
bs_frac     = float(backscatter.mean())

print(f"\ndissipation range: {Pi.min():.4f} to {Pi.max():.4f}")
print(f"backscatter fraction: {bs_frac:.1%}")
print("note: smagorinsky always 0% backscatter by construction")

# -----------------------------------------------------------------
# helper to get the mid z-slice for plotting
# -----------------------------------------------------------------

def get_slice(arr, gs):
    nx, ny, nz = gs
    return arr.reshape(nx, ny, nz)[:, :, nz // 2].T


# -----------------------------------------------------------------
# 4 panel plot
# -----------------------------------------------------------------

RCOLS  = ["#4575b4", "#74add1", "#d73027"]
RNAMES = ["Strain-dominated", "Mixed", "Vortex-dominated"]

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.patch.set_facecolor("#0f1117")

for ax in axes.flat:
    ax.set_facecolor("#1a1d27")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

# panel 0 - regime map
ax = axes[0, 0]
reg_sl = get_slice(regime.astype(np.float32), grid_shape)
q_sl   = get_slice(Q, grid_shape)

cmap_reg = mcolors.ListedColormap(RCOLS)
ax.imshow(reg_sl, origin="lower", cmap=cmap_reg,
          vmin=-0.5, vmax=2.5, aspect="auto", interpolation="nearest")

try:
    ax.contour(q_sl, levels=5, colors="white", linewidths=0.5, alpha=0.35)
except Exception:
    pass

legend_p = [Patch(color=RCOLS[i], label=RNAMES[i]) for i in range(3)]
ax.legend(handles=legend_p, loc="upper right", fontsize=7,
          framealpha=0.6, labelcolor="white", facecolor="#222")

ax.set_title("Flow Regime Map  (Q-criterion)", color="white", fontsize=10, fontweight="bold")
ax.set_xlabel("x streamwise", color="#aaa", fontsize=8)
ax.set_ylabel("y normal", color="#aaa", fontsize=8)
ax.tick_params(colors="#888", labelsize=7)
ax.text(0.02, 0.04, f"thresholds  p33={p33:.2e}  p66={p66:.2e}",
        transform=ax.transAxes, color="#aaa", fontsize=6.5)

for r, nm in enumerate(RNAMES):
    cnt = (regime == r).sum()
    ax.text(0.02, 0.17 - r * 0.08,
            f"{nm}: {cnt} nodes ({cnt/n_nodes:.0%})",
            transform=ax.transAxes, color=RCOLS[r], fontsize=6.5)

# panel 1 - uncertainty
ax = axes[0, 1]
unc_sl  = get_slice(unc, grid_shape)
mask_sl = get_slice(high_unc.astype(np.float32), grid_shape)

im1 = ax.imshow(unc_sl, origin="lower", cmap="plasma",
                aspect="auto", interpolation="bilinear")
cb1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
cb1.set_label("L2 norm of variance_tau", color="#aaa", fontsize=7)
cb1.ax.tick_params(colors="#888", labelsize=6)

try:
    ax.contour(mask_sl, levels=[0.5], colors=["#ff6b6b"], linewidths=1.4, linestyles="--")
except Exception:
    pass

ax.set_title("Prediction Uncertainty  (V1 ensemble spread)", color="white",
             fontsize=10, fontweight="bold")
ax.set_xlabel("x streamwise", color="#aaa", fontsize=8)
ax.set_ylabel("y normal", color="#aaa", fontsize=8)
ax.tick_params(colors="#888", labelsize=7)
ax.text(0.02, 0.08, "dashed red = top 25% uncertainty",
        transform=ax.transAxes, color="#ff6b6b", fontsize=7)
ax.text(0.02, 0.03, f"mean={unc.mean():.3f}  max={unc.max():.3f}",
        transform=ax.transAxes, color="#aaa", fontsize=7)

# panel 2 - sgs dissipation
ax = axes[1, 0]
pi_sl   = get_slice(Pi, grid_shape)
abs_max = max(np.abs(pi_sl).max(), 1e-10)

im2 = ax.imshow(pi_sl, origin="lower", cmap="RdBu_r",
                vmin=-abs_max, vmax=abs_max,
                aspect="auto", interpolation="bilinear")
cb2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
cb2.set_label("Pi = -tau:S   (forward < 0 < backscatter)", color="#aaa", fontsize=6.5)
cb2.ax.tick_params(colors="#888", labelsize=6)

try:
    ax.contour(pi_sl, levels=[0], colors=["white"], linewidths=0.8)
except Exception:
    pass

ax.set_title("SGS Dissipation  Pi = -tau:S", color="white", fontsize=10, fontweight="bold")
ax.set_xlabel("x streamwise", color="#aaa", fontsize=8)
ax.set_ylabel("y normal", color="#aaa", fontsize=8)
ax.tick_params(colors="#888", labelsize=7)
ax.text(0.02, 0.09, f"forward scatter: {(1-bs_frac):.1%}",
        transform=ax.transAxes, color="#e8747c", fontsize=8, fontweight="bold")
ax.text(0.02, 0.03, f"backscatter: {bs_frac:.1%}  (smagorinsky = 0% always)",
        transform=ax.transAxes, color="#6baed6", fontsize=7)

# panel 3 - backscatter map
ax = axes[1, 1]
bs_sl = get_slice(backscatter.astype(np.float32), grid_shape)

cmap_bs = mcolors.ListedColormap(["#1a1d27", "#ff6b35"])
ax.imshow(bs_sl, origin="lower", cmap=cmap_bs,
          vmin=0, vmax=1, aspect="auto", interpolation="nearest")

ax.set_title(f"Backscatter Map  (fraction = {bs_frac:.1%})",
             color="white", fontsize=10, fontweight="bold")
ax.set_xlabel("x streamwise", color="#aaa", fontsize=8)
ax.set_ylabel("y normal", color="#aaa", fontsize=8)
ax.tick_params(colors="#888", labelsize=7)

leg_bs = [
    Patch(color="#1a1d27", label="Forward  (Pi >= 0)"),
    Patch(color="#ff6b35", label="Backscatter  (Pi < 0)"),
]
ax.legend(handles=leg_bs, loc="upper right", fontsize=7,
          framealpha=0.7, labelcolor="white", facecolor="#222")

for r, nm in enumerate(RNAMES):
    mask_r = regime == r
    fr = backscatter[mask_r].mean() if mask_r.any() else 0.0
    ax.text(0.02, 0.22 - r * 0.07,
            f"{nm[:9]}: {fr:.1%}",
            transform=ax.transAxes, color=RCOLS[r], fontsize=7)

ax.text(0.02, 0.03, "smagorinsky = no backscatter by construction",
        transform=ax.transAxes, color="#888", fontsize=6.5, fontstyle="italic")

# title
param_str = "  ".join(f"{k}={v}" for k, v in list(params.items())[:3])
fig.suptitle(
    f"SHEAR  CF4 Diagnostics  —  {result['geometry_type']}  |  {param_str}\n"
    f"{n_samples} V1 samples  |  {result['inference_time_ms']}ms",
    fontsize=12, color="white", y=0.997, fontweight="bold"
)

fig.tight_layout(rect=[0, 0, 1, 0.96])

out = Path(__file__).parent.parent / "figures" / "cf4_aerofoil_demo.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1117")
print(f"\nsaved to {out}")