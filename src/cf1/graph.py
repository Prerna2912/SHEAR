"""
CF1 — kNN graph builder for V1 inference.

Converts a ∇u field (positions + grad_u) into a torch_geometric Data object
using the same irreps encoding and kNN connectivity as the training data.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data


def build_cf1_graph(
    grad_u: np.ndarray,
    positions: np.ndarray,
    k_neighbours: int = 26,
) -> Data:
    """
    Build a torch_geometric Data object from a CF1 ∇u field.

    Reuses the irreps encoding and kNN graph construction from the training
    pipeline (src.data.dataset) so V1 sees an identical input format.

    Args:
        grad_u    : [N, 3, 3] velocity gradient ∂u_i/∂x_j.
        positions : [N, 3]    grid coordinates.
        k_neighbours: kNN connectivity (must match training k, default 26).

    Returns:
        torch_geometric Data with:
          .x          [N, 9]  velocity gradient encoded as e3nn irreps
          .pos        [N, 3]  node positions
          .edge_index [2, E]  kNN graph edges
          .grad_full  [N, 9]  flat velocity gradient (for metric computation)
    """
    from src.data.dataset import grad_to_irreps, _knn_graph_torch

    pos_t  = torch.from_numpy(positions.astype(np.float32))
    grad_t = torch.from_numpy(grad_u.astype(np.float32))            # [N, 3, 3]

    x = grad_to_irreps(grad_t)                                       # [N, 9]
    edge_index = _knn_graph_torch(pos_t, k=k_neighbours)

    return Data(
        x=x,
        pos=pos_t,
        edge_index=edge_index,
        grad_full=grad_t.reshape(grad_u.shape[0], 9),
    )
