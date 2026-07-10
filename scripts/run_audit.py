"""
Standalone equivariance audit on untrained models.

Runs the architectural equivariance check for V1, V2, V3 without needing
trained checkpoints. Results saved to results/audit_results.json and
figures/equivariance_audit.png.

Usage:
    uv run python scripts/run_audit.py
"""

import sys
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

import torch
import yaml
from data.jhtdb import generate_synthetic_les_data
from data.dataset import build_dataloaders
from models.flow_matching import build_model
from evaluation.audit import run_equivariance_audit

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# --- Data (synthetic, small) ---
print('Generating synthetic LES data...')
tau_field, grad_field = generate_synthetic_les_data(n_les=32, seed=0)

_, _, test_loader, _ = build_dataloaders(
    tau_field, grad_field,
    n_train=100, n_val=20, n_test=60,
    patch_size=8, k_neighbours=26,
    batch_size=1, num_workers=0, seed=42,
)

# --- Untrained models ---
print('Building untrained models...')
with open(REPO / 'configs/v1.yaml') as f: cfg_v1 = yaml.safe_load(f)
with open(REPO / 'configs/v2.yaml') as f: cfg_v2 = yaml.safe_load(f)
with open(REPO / 'configs/v3.yaml') as f: cfg_v3 = yaml.safe_load(f)

model_v1 = build_model('v1', cfg_v1['model']).to(DEVICE).eval()
model_v2 = build_model('v2', cfg_v2['model']).to(DEVICE).eval()
model_v3 = build_model('v3', cfg_v3['model']).to(DEVICE).eval()

print(f'V1 params: {sum(p.numel() for p in model_v1.parameters()):,}')
print(f'V2 params: {sum(p.numel() for p in model_v2.parameters()):,}')
print(f'V3 params: {sum(p.numel() for p in model_v3.parameters()):,}')

# --- Audit ---
out_dir = REPO / 'figures'
out_dir.mkdir(exist_ok=True)

print('\nRunning equivariance audit (50 sub-cubes × 100 rotations)...')
results = run_equivariance_audit(
    models={'v1': model_v1, 'v2': model_v2, 'v3': model_v3},
    test_loader=test_loader,
    device=DEVICE,
    n_subcubes=50,
    n_rotations=100,
    out_dir=str(out_dir),
)

# --- Save numbers ---
results_dir = REPO / 'results'
results_dir.mkdir(exist_ok=True)
out_json = results_dir / 'audit_results.json'
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nSaved numbers to {out_json}')
print(f'Saved plot to {out_dir}/equivariance_audit.png')
print('\n=== EQUIVARIANCE AUDIT RESULTS ===')
for variant, r in results.items():
    print(f'  {variant.upper()}: mean={r["mean"]:.4e}  std={r["std"]:.4e}')
print('\nNote: these are ARCHITECTURAL results — valid on untrained models.')
print('V1/V2 (SE3-equivariant) should be ~1e-6 to 1e-7.')
print('V3 (MLP, no equivariance) should be ~1.')
