"""
Generate plots from the quick training run results.
Run after run_quick_train.py completes.

Usage:
    uv run python scripts/plot_quick_results.py
"""

import sys, json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS  = REPO / 'results' / 'quick_results.json'
AUDIT    = REPO / 'results' / 'audit_results.json'
FIG_DIR  = REPO / 'figures'
FIG_DIR.mkdir(exist_ok=True)

with open(RESULTS) as f:
    res = json.load(f)
with open(AUDIT) as f:
    audit = json.load(f)

COLORS  = {'v1': '#2196F3', 'v2': '#FF5722', 'v3': '#4CAF50'}
LABELS  = {'v1': 'V1: SE(3)-CFM', 'v2': 'V2: SE(3)-MSE', 'v3': 'V3: MLP-CFM'}

# ── Figure 1: Training loss curves ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

for ax, v in zip(axes, ('v1', 'v2', 'v3')):
    h = res[v]['history']
    train_loss = h['train_loss']
    steps = list(range(1, len(train_loss) + 1))
    ax.plot(steps, train_loss, color=COLORS[v], alpha=0.5, linewidth=1, label='Train')

    if h['val_loss']:
        val_steps, val_vals = zip(*h['val_loss'])
        ax.plot(val_steps, val_vals, 'o-', color=COLORS[v], linewidth=2,
                markersize=5, label='Val')
        best = res[v]['best_val_loss']
        ax.axhline(best, color=COLORS[v], linestyle='--', alpha=0.4,
                   label=f'Best val: {best:.4f}')

    ax.set_title(LABELS[v], fontsize=12, fontweight='bold')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('Training curves — synthetic data, 500 steps', fontsize=13)
plt.tight_layout()
out1 = FIG_DIR / 'training_curves.png'
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {out1}')

# ── Figure 2: Equivariance audit bar chart ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

variants = ['v1', 'v2', 'v3']
means = [audit[v]['mean'] for v in variants]
stds  = [audit[v]['std']  for v in variants]
xlabels = [LABELS[v] for v in variants]

bars = ax.bar(xlabels, means, yerr=stds, capsize=6,
              color=[COLORS[v] for v in variants],
              alpha=0.8, edgecolor='black', linewidth=0.8)

ax.set_yscale('log')
ax.set_ylabel('Equivariance error  ‖f(Rx)−Rf(x)‖ / ‖Rf(x)‖', fontsize=10)
ax.set_title('SE(3) Equivariance Audit\n(50 sub-cubes × 100 SO(3) rotations, untrained models)',
             fontsize=11)
ax.grid(True, axis='y', alpha=0.3)

for bar, v in zip(bars, variants):
    val = audit[v]['mean']
    ax.text(bar.get_x() + bar.get_width() / 2, val * 1.8,
            f'{val:.2e}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.annotate('Machine\nprecision', xy=(0, means[0]), xytext=(0.4, means[0] * 50),
            fontsize=8, color='gray',
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

plt.tight_layout()
out2 = FIG_DIR / 'equivariance_audit_bar.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {out2}')

# ── Figure 3: Combined summary (for slides) ──────────────────────────────────
fig = plt.figure(figsize=(14, 5))
gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.35)

# Left: all three loss curves on one plot
ax_left = fig.add_subplot(gs[0])
for v in ('v1', 'v2', 'v3'):
    h = res[v]['history']
    smooth = np.convolve(h['train_loss'], np.ones(10)/10, mode='valid')
    ax_left.plot(range(len(smooth)), smooth, color=COLORS[v],
                 linewidth=2, label=LABELS[v])
    if h['val_loss']:
        val_steps, val_vals = zip(*h['val_loss'])
        ax_left.plot(val_steps, val_vals, 'o', color=COLORS[v], markersize=6)

ax_left.set_xlabel('Step'); ax_left.set_ylabel('Loss (log scale)')
ax_left.set_yscale('log')
ax_left.set_title('Training Loss (10-step moving average)', fontsize=11)
ax_left.legend(fontsize=9); ax_left.grid(True, alpha=0.3)

# Right: equivariance bars
ax_right = fig.add_subplot(gs[1])
bars = ax_right.bar(range(3), means, yerr=stds, capsize=5,
                    color=[COLORS[v] for v in variants],
                    alpha=0.85, edgecolor='black', linewidth=0.8)
ax_right.set_yscale('log')
ax_right.set_xticks(range(3)); ax_right.set_xticklabels(['V1', 'V2', 'V3'])
ax_right.set_ylabel('Equivariance error')
ax_right.set_title('SE(3) Equivariance\nAudit', fontsize=11)
ax_right.grid(True, axis='y', alpha=0.3)
for bar, v in zip(bars, variants):
    ax_right.text(bar.get_x() + bar.get_width()/2, audit[v]['mean'] * 2,
                  f'{audit[v]["mean"]:.1e}', ha='center', va='bottom', fontsize=9)

fig.suptitle('SE(3)-Equivariant Flow Matching for SGS Closure — Preliminary Results',
             fontsize=12, fontweight='bold')

out3 = FIG_DIR / 'summary_slide.png'
plt.savefig(out3, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {out3}')

print('\n=== Training Summary ===')
for v in ('v1', 'v2', 'v3'):
    print(f'  {v.upper()}: best_val={res[v]["best_val_loss"]:.6f}  '
          f'({res[v]["elapsed_s"]/60:.1f} min)')
print('\n=== Equivariance Audit ===')
for v in ('v1', 'v2', 'v3'):
    print(f'  {v.upper()}: {audit[v]["mean"]:.2e} ± {audit[v]["std"]:.2e}')
