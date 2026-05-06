"""
SGS closure dataset.

Extracts 5,000 point-wise samples from the 64^3 LES field.
Each sample is a local neighborhood graph centred on one LES grid point.

Graph construction:
  - Nodes: all points in a patch_size^3 sub-cube.
  - Edges: k-nearest neighbours (k=6 face-adjacent + 12 edge-adjacent = 26).
  - Node features: velocity gradient (9 components) at each node.
  - Node targets: SGS stress tau_{ij} (6 independent components) at each node.

The centre node's index within the sub-cube is stored for output extraction.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data
from pathlib import Path
from typing import Optional


# ------------------------------------------------------------------
# Irreps helpers: convert 3x3 tensors to irreps vectors and back
# ------------------------------------------------------------------

def grad_to_irreps(grad_u: torch.Tensor) -> torch.Tensor:
    """
    Decompose velocity gradient [..., 3, 3] into SE(3) irreducible representations.

    Decomposition:
      l=0 (0e): trace / sqrt(3)                                     → 1 component
      l=1 (1o): antisymmetric part (vorticity dual) / sqrt(2)       → 3 components
      l=2 (2e): symmetric traceless part (spherical-harmonic basis)  → 5 components
    Total: 9 components, matching the 9 entries of a 3×3 matrix.

    Returns [..., 9] irreps vector in order [l0, l1x, l1y, l1z, l2_0..4].
    """
    # Symmetric and antisymmetric parts
    S = 0.5 * (grad_u + grad_u.transpose(-1, -2))   # symmetric
    A = 0.5 * (grad_u - grad_u.transpose(-1, -2))   # antisymmetric

    # l=0: trace
    tr = (S[..., 0, 0] + S[..., 1, 1] + S[..., 2, 2]) / np.sqrt(3)  # [..., 1]

    # Symmetric traceless part
    trace_val = S[..., 0, 0] + S[..., 1, 1] + S[..., 2, 2]
    I = torch.eye(3, device=grad_u.device, dtype=grad_u.dtype)
    S_tl = S - (trace_val / 3.0).unsqueeze(-1).unsqueeze(-1) * I

    # l=2 basis (real solid harmonics, rank-2):
    # Y_{2,-2} ~ xy, Y_{2,-1} ~ yz, Y_{2,0} ~ (2zz-xx-yy)/2, Y_{2,1} ~ xz, Y_{2,2} ~ (xx-yy)/2
    l2 = torch.stack([
        S_tl[..., 0, 1],                                             # xy
        S_tl[..., 1, 2],                                             # yz
        (2*S_tl[..., 2, 2] - S_tl[..., 0, 0] - S_tl[..., 1, 1]) / np.sqrt(12),
        S_tl[..., 0, 2],                                             # xz
        (S_tl[..., 0, 0] - S_tl[..., 1, 1]) / 2.0,                 # (xx-yy)
    ], dim=-1)  # [..., 5]

    # l=1: dual of antisymmetric part (vorticity / 2)
    l1 = torch.stack([
        A[..., 2, 1],   # omega_x / 2
        A[..., 0, 2],   # omega_y / 2
        A[..., 1, 0],   # omega_z / 2
    ], dim=-1)  # [..., 3]

    return torch.cat([tr.unsqueeze(-1), l1, l2], dim=-1)  # [..., 9]


def stress_to_irreps(tau: torch.Tensor) -> torch.Tensor:
    """
    Decompose symmetric SGS stress [..., 3, 3] into irreps.

    l=0 (0e): trace / sqrt(3)          → 1 component
    l=2 (2e): symmetric traceless       → 5 components
    Total: 6 components.

    Returns [..., 6] irreps vector.
    """
    tr = (tau[..., 0, 0] + tau[..., 1, 1] + tau[..., 2, 2]) / np.sqrt(3)

    trace_val = tau[..., 0, 0] + tau[..., 1, 1] + tau[..., 2, 2]
    I = torch.eye(3, device=tau.device, dtype=tau.dtype)
    tau_tl = tau - (trace_val / 3.0).unsqueeze(-1).unsqueeze(-1) * I

    l2 = torch.stack([
        tau_tl[..., 0, 1],
        tau_tl[..., 1, 2],
        (2*tau_tl[..., 2, 2] - tau_tl[..., 0, 0] - tau_tl[..., 1, 1]) / np.sqrt(12),
        tau_tl[..., 0, 2],
        (tau_tl[..., 0, 0] - tau_tl[..., 1, 1]) / 2.0,
    ], dim=-1)

    return torch.cat([tr.unsqueeze(-1), l2], dim=-1)  # [..., 6]


def irreps_to_stress(x: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct symmetric 3x3 stress tensor from 6-component irreps vector.

    Args:
        x: [..., 6]  [l0, l2_xy, l2_yz, l2_m0, l2_xz, l2_xx_yy]

    Returns:
        tau: [..., 3, 3] symmetric tensor.
    """
    tr_scaled = x[..., 0]
    tr = tr_scaled * np.sqrt(3)

    l2_xy  = x[..., 1]
    l2_yz  = x[..., 2]
    l2_m0  = x[..., 3]   # (2zz - xx - yy) / sqrt(12)
    l2_xz  = x[..., 4]
    l2_diag = x[..., 5]  # (xx - yy) / 2

    # Recover diagonal elements
    # m0 = (2*zz - xx - yy) / sqrt(12), diag = (xx-yy)/2
    # tr = xx + yy + zz
    # => zz = sqrt(12)*m0/3 + tr/3
    # => xx = tr/3 + diag - sqrt(12)*m0/6
    # => yy = tr/3 - diag - sqrt(12)*m0/6
    sqrt12 = np.sqrt(12)
    zz = sqrt12 * l2_m0 / 3 + tr / 3
    xx = tr / 3 + l2_diag - sqrt12 * l2_m0 / 6
    yy = tr / 3 - l2_diag - sqrt12 * l2_m0 / 6

    tau = torch.zeros(*x.shape[:-1], 3, 3, device=x.device, dtype=x.dtype)
    tau[..., 0, 0] = xx
    tau[..., 1, 1] = yy
    tau[..., 2, 2] = zz
    tau[..., 0, 1] = tau[..., 1, 0] = l2_xy
    tau[..., 1, 2] = tau[..., 2, 1] = l2_yz
    tau[..., 0, 2] = tau[..., 2, 0] = l2_xz
    return tau


# ------------------------------------------------------------------
# Graph building from sub-cube
# ------------------------------------------------------------------

def _knn_graph_torch(pos: torch.Tensor, k: int) -> torch.Tensor:
    """
    Pure-PyTorch kNN graph. No torch-cluster needed.
    Returns edge_index [2, N*k] in (destination, source) order.
    """
    dist = torch.cdist(pos, pos)                       # [N, N]
    dist.fill_diagonal_(float('inf'))                  # exclude self-loops
    k_eff = min(k, pos.shape[0] - 1)
    _, col = dist.topk(k_eff, largest=False, dim=-1)   # [N, k]
    row = torch.arange(pos.shape[0]).unsqueeze(1).expand_as(col)
    return torch.stack([row.reshape(-1), col.reshape(-1)], dim=0)  # [2, N*k]


def build_subcube_graph(
    grad_u_patch: np.ndarray,
    tau_patch: np.ndarray,
    positions: np.ndarray,
    k_neighbours: int = 26,
) -> Data:
    """
    Build a torch_geometric Data object for one sub-cube patch.

    Args:
        grad_u_patch: [P^3, 3, 3] velocity gradient at each node.
        tau_patch:    [P^3, 3, 3] SGS stress at each node.
        positions:    [P^3, 3] 3-D grid coordinates.
        k_neighbours: number of nearest neighbours for edges.

    Returns:
        torch_geometric Data with:
          .x      [P^3, 9]  velocity gradient as irreps (node features)
          .y      [P^3, 6]  SGS stress as irreps (targets)
          .pos    [P^3, 3]  node positions
          .edge_index [2, E] COO edge index
          .tau_full [P^3, 9] full 3x3 stress as flat vector (for metric computation)
    """
    n = positions.shape[0]
    pos_t = torch.from_numpy(positions.astype(np.float32))
    grad_t = torch.from_numpy(grad_u_patch.astype(np.float32))  # [n, 3, 3]
    tau_t = torch.from_numpy(tau_patch.astype(np.float32))      # [n, 3, 3]

    x = grad_to_irreps(grad_t)    # [n, 9]
    y = stress_to_irreps(tau_t)   # [n, 6]

    # Build kNN graph via brute-force pairwise distances (no extra deps)
    edge_index = _knn_graph_torch(pos_t, k=k_neighbours)

    return Data(
        x=x,
        y=y,
        pos=pos_t,
        edge_index=edge_index,
        tau_full=tau_t.reshape(n, 9),     # [n, 9] for metrics
        grad_full=grad_t.reshape(n, 9),   # [n, 9]
    )


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class SGSDataset(Dataset):
    """
    Dataset of local sub-cube graphs sampled from the LES field.

    Args:
        tau_field:    [64, 64, 64, 3, 3] SGS stress.
        grad_field:   [64, 64, 64, 3, 3] filtered velocity gradient.
        n_samples:    total number of sub-cubes to draw.
        patch_size:   side length of each sub-cube patch (e.g. 8 -> 8^3 nodes).
        k_neighbours: kNN connectivity within each sub-cube.
        seed:         random seed for sampling.
        normalize:    if True, standardize features using per-channel statistics.
    """

    def __init__(
        self,
        tau_field: np.ndarray,
        grad_field: np.ndarray,
        n_samples: int = 5000,
        patch_size: int = 8,
        k_neighbours: int = 26,
        seed: int = 42,
        normalize: bool = True,
        stats: Optional[dict] = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.k_neighbours = k_neighbours
        self.n_samples = n_samples

        N = tau_field.shape[0]
        P = patch_size
        rng = np.random.default_rng(seed)

        # Sample random top-left corners for non-overlapping patches
        origins = rng.integers(0, N - P + 1, size=(n_samples, 3))

        # Build coordinate grid for one patch [P^3, 3]
        coords_1d = np.arange(P, dtype=np.float32)
        gx, gy, gz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
        local_coords = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)  # [P^3, 3]

        self.graphs = []
        for i in range(n_samples):
            ox, oy, oz = origins[i]
            # Extract patch
            g_patch = grad_field[ox:ox+P, oy:oy+P, oz:oz+P].reshape(-1, 3, 3)
            t_patch = tau_field[ox:ox+P, oy:oy+P, oz:oz+P].reshape(-1, 3, 3)
            # Global coordinates (add origin)
            pos = local_coords + origins[i].astype(np.float32)
            self.graphs.append(build_subcube_graph(
                g_patch, t_patch, pos, k_neighbours=k_neighbours
            ))

        # Compute normalization statistics from training data
        if normalize and stats is None:
            all_x = torch.cat([g.x for g in self.graphs], dim=0)
            all_y = torch.cat([g.y for g in self.graphs], dim=0)
            self.stats = {
                'x_mean': all_x.mean(0), 'x_std': all_x.std(0).clamp(min=1e-8),
                'y_mean': all_y.mean(0), 'y_std': all_y.std(0).clamp(min=1e-8),
            }
        elif stats is not None:
            self.stats = stats
        else:
            d_x = self.graphs[0].x.shape[-1]
            d_y = self.graphs[0].y.shape[-1]
            self.stats = {
                'x_mean': torch.zeros(d_x), 'x_std': torch.ones(d_x),
                'y_mean': torch.zeros(d_y), 'y_std': torch.ones(d_y),
            }

        if normalize:
            for g in self.graphs:
                g.x = (g.x - self.stats['x_mean']) / self.stats['x_std']
                g.y = (g.y - self.stats['y_mean']) / self.stats['y_std']

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def build_dataloaders(
    tau_field: np.ndarray,
    grad_field: np.ndarray,
    n_total: int = 5000,
    n_train: int = 4000,
    n_val: int = 500,
    n_test: int = 500,
    patch_size: int = 8,
    k_neighbours: int = 26,
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple:
    """
    Build train / val / test DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, stats
    """
    from torch_geometric.loader import DataLoader as GeoDataLoader

    # Draw non-overlapping samples with different seeds per split to avoid leakage
    train_ds = SGSDataset(
        tau_field, grad_field, n_samples=n_train,
        patch_size=patch_size, k_neighbours=k_neighbours,
        seed=seed, normalize=True,
    )
    stats = train_ds.stats

    val_ds = SGSDataset(
        tau_field, grad_field, n_samples=n_val,
        patch_size=patch_size, k_neighbours=k_neighbours,
        seed=seed + 1, normalize=True, stats=stats,
    )
    test_ds = SGSDataset(
        tau_field, grad_field, n_samples=n_test,
        patch_size=patch_size, k_neighbours=k_neighbours,
        seed=seed + 2, normalize=True, stats=stats,
    )

    train_loader = GeoDataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = GeoDataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = GeoDataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, stats
