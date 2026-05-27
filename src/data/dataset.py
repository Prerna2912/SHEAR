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
# Precomputed change-of-basis matrices (e3nn convention)
# These map Cartesian 3x3 tensors to e3nn irreducible representations,
# guaranteed to transform under the Wigner D-matrices used internally
# by the equivariant EGNN.
#
# Velocity gradient (general rank-2): 1x0e + 1x1e + 1x2e  (9 components)
# SGS stress (symmetric rank-2):      1x0e + 1x2e          (6 components)
# ------------------------------------------------------------------

def _build_change_of_basis():
    from e3nn.o3 import ReducedTensorProducts
    rtp_grad   = ReducedTensorProducts('ij',    i='1o', j='1o')
    rtp_stress = ReducedTensorProducts('ij=ji', i='1o', j='1o')
    # Q_GRAD:   [9, 3, 3]  — x[k] = einsum('kij,ij', Q_GRAD, G)
    # Q_STRESS: [6, 3, 3]  — x[k] = einsum('kij,ij', Q_STRESS, tau)
    return (rtp_grad.change_of_basis.detach().float(),
            rtp_stress.change_of_basis.detach().float())

_Q_GRAD, _Q_STRESS = _build_change_of_basis()


# ------------------------------------------------------------------
# Irreps helpers: convert 3x3 tensors to irreps vectors and back
# ------------------------------------------------------------------

def grad_to_irreps(grad_u: torch.Tensor) -> torch.Tensor:
    """
    Decompose velocity gradient [..., 3, 3] into e3nn irreducible representations.

    Uses the e3nn-canonical basis (1x0e + 1x1e + 1x2e) so that the resulting
    9-vector transforms correctly under Wigner D-matrices for any SO(3) rotation.

    Returns [..., 9] irreps vector.
    """
    Q = _Q_GRAD.to(grad_u.device)
    return torch.einsum('kij,...ij->...k', Q, grad_u)


def stress_to_irreps(tau: torch.Tensor) -> torch.Tensor:
    """
    Decompose symmetric SGS stress [..., 3, 3] into e3nn irreps (1x0e + 1x2e).

    Returns [..., 6] irreps vector.
    """
    Q = _Q_STRESS.to(tau.device)
    return torch.einsum('kij,...ij->...k', Q, tau)


def irreps_to_stress(x: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct symmetric 3x3 stress tensor from 6-component e3nn irreps vector.

    Args:
        x: [..., 6]

    Returns:
        tau: [..., 3, 3] symmetric tensor.
    """
    Q = _Q_STRESS.to(x.device)
    return torch.einsum('kij,...k->...ij', Q, x)


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
