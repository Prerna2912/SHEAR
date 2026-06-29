"""
Print a single raw data sample from real JHTDB DNS data.

Usage:
    python scripts/raw_sample.py
    JHTDB_TOKEN=<your-token> python scripts/raw_sample.py
"""

import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.dataset import SGSDataset

print("=" * 60)
print("SGS FLOW MATCHING — RAW JHTDB DATASET SAMPLE")
print("=" * 60)

print("\n[1] Fetching 16³ real JHTDB DNS sample …")
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
ds_raw  = getCutout(cube, "velocity", axes, strides, verbose=False)
vel     = ds_raw["velocity_0001"].values.astype(np.float32)  # [16,16,16,3]

from data.jhtdb import compute_velocity_gradient
dx    = 2 * np.pi / 1024
u, v, w = vel[..., 0], vel[..., 1], vel[..., 2]
grad_u  = compute_velocity_gradient(u, v, w, dx)
S       = 0.5 * (grad_u + grad_u.transpose(0, 1, 2, 4, 3))
Smag    = np.sqrt(2.0 * np.sum(S ** 2, axis=(-2, -1)))
Cs, Delta = 0.17, dx
tau     = (-2.0 * (Cs * Delta) ** 2
           * Smag[..., np.newaxis, np.newaxis] * S).astype(np.float32)

print(f"\n[2] Field shapes (16³ DNS patch)")
print(f"    vel     [velocity field]      : {vel.shape}  — raw DNS velocities")
print(f"    grad_u  [velocity gradient]   : {grad_u.shape}  — input to model")
print(f"    tau     [SGS stress tensor]   : {tau.shape}  — target for model")

np.set_printoptions(precision=6, suppress=True)

cx, cy, cz = 8, 8, 8
print(f"\n[3] Raw tensors at grid point [{cx}, {cy}, {cz}]")

print(f"\n    Velocity (u, v, w):")
print(vel[cx, cy, cz])

print(f"\n    Velocity gradient ∂uᵢ/∂xⱼ  (input):")
print(grad_u[cx, cy, cz])

print(f"\n    SGS stress τᵢⱼ  (target):")
print(tau[cx, cy, cz])
print(f"    (τ is symmetric: τᵢⱼ = τⱼᵢ — only 6 independent components)")

print(f"\n[4] Graph sample (one 8³ sub-cube patch)")
ds     = SGSDataset(tau, grad_u, n_samples=1, patch_size=8, normalize=True)
sample = ds[0]

print(f"    Nodes        : {sample.x.shape[0]}  (8×8×8 local neighbourhood)")
print(f"    Node features: {list(sample.x.shape)}  — grad_u decomposed into SE(3) irreps (ℓ=0,1,2)")
print(f"    Node targets : {list(sample.y.shape)}  — τ decomposed into SE(3) irreps (ℓ=0,2)")
print(f"    Edges        : {sample.edge_index.shape[1]}  (each node connected to 26 neighbours)")

print(f"\n    Feature vector at node 0  (9-dim SE(3) irreps of grad_u):")
print(np.array2string(sample.x[0].numpy(), precision=6, separator=', '))

print(f"\n    Target vector at node 0  (6-dim SE(3) irreps of τ):")
print(np.array2string(sample.y[0].numpy(), precision=6, separator=', '))

print("\n" + "=" * 60)
