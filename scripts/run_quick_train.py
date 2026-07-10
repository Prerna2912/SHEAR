"""
Quick local training run: V1, V2, V3 on synthetic data, 500 steps each.
Results saved to runs/quick/{v1,v2,v3}/ and results/quick_results.json.

Usage:
    uv run python scripts/run_quick_train.py
"""

import sys, json, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

import torch
import yaml
import numpy as np
from data.jhtdb import generate_synthetic_les_data
from data.dataset import build_dataloaders
from models.flow_matching import build_model
from training.trainer import Trainer

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {DEVICE}\n')

STEPS      = 500
VAL_EVERY  = 100
LOG_EVERY  = 50
OUT_ROOT   = REPO / 'runs' / 'quick'
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# --- Synthetic data ---
print('Generating synthetic LES data (64³)...')
tau_field, grad_field = generate_synthetic_les_data(n_les=64, seed=0)

train_loader, val_loader, test_loader, stats = build_dataloaders(
    tau_field, grad_field,
    n_train=2000, n_val=300, n_test=300,
    patch_size=8, k_neighbours=26,
    batch_size=1, num_workers=0, seed=42,
)
print(f'Train: {len(train_loader)}  Val: {len(val_loader)}  Test: {len(test_loader)}\n')
torch.save(stats, OUT_ROOT / 'stats.pt')

# --- Models ---
cfgs = {}
for v in ('v1', 'v2', 'v3'):
    with open(REPO / f'configs/{v}.yaml') as f:
        cfgs[v] = yaml.safe_load(f)

models = {v: build_model(v, cfgs[v]['model']).to(DEVICE) for v in ('v1', 'v2', 'v3')}
for v, m in models.items():
    print(f'{v.upper()}: {sum(p.numel() for p in m.parameters()):,} params')

# --- Training ---
summary = {}
for v in ('v1', 'v2', 'v3'):
    print(f'\n{"="*50}')
    out_dir = OUT_ROOT / v
    trainer = Trainer(
        model=models[v], variant=v,
        train_loader=train_loader, val_loader=val_loader,
        cfg={**cfgs[v]['training'],
             'total_steps': STEPS,
             'val_every':   VAL_EVERY,
             'log_every':   LOG_EVERY},
        device=DEVICE,
        out_dir=str(out_dir),
    )
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    summary[v] = {
        'best_val_loss': trainer.best_val_loss,
        'elapsed_s': round(elapsed, 1),
        'history': trainer.history,
    }
    print(f'  {v.upper()} done in {elapsed/60:.1f} min  |  best_val={trainer.best_val_loss:.6f}')

# --- Save summary ---
results_dir = REPO / 'results'
results_dir.mkdir(exist_ok=True)
out = results_dir / 'quick_results.json'
with open(out, 'w') as f:
    # history has tensors → convert
    def _clean(d):
        if isinstance(d, dict):
            return {k: _clean(v) for k, v in d.items()}
        if isinstance(d, list):
            return [_clean(x) for x in d]
        if isinstance(d, (np.floating, np.integer)):
            return float(d)
        return d
    json.dump(_clean(summary), f, indent=2)

print(f'\nSaved → {out}')
print('\n=== SUMMARY ===')
for v, r in summary.items():
    print(f'  {v.upper()}: best_val={r["best_val_loss"]:.6f}  ({r["elapsed_s"]/60:.1f} min)')
