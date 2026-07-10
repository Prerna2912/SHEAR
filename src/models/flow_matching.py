"""
Conditional Flow Matching (CFM) framework following Lipman et al. (ICLR 2023).

Straight-path interpolant:
    x_t = (1 - t) * x_0  +  t * x_1
Target vector field:
    u(x_t | x_1) = x_1 - x_0
Loss:
    L(theta) = E_{t, x_0, x_1} || v_theta(x_t, t, c) - (x_1 - x_0) ||^2

At inference, integrate the learned ODE from t=0 to t=1 using torchdiffeq.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint
from torch_geometric.data import Data, Batch
from typing import Union


class ConditionalFlowMatcher:
    """
    Stateless helper for the CFM training objective (not a nn.Module).
    Works for both V1 (EGNN) and V3 (MLP).
    """

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample t ~ Uniform(0, 1)."""
        return torch.rand(batch_size, device=device)

    def interpolate(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple:
        """
        Compute x_t and the target vector field.

        Args:
            x_0: [B, D] source (Gaussian noise).
            x_1: [B, D] target (true SGS stress).
            t:   [B]    times in [0, 1].

        Returns:
            x_t:    [B, D] interpolated point.
            target: [B, D] target vector field = x_1 - x_0.
        """
        t_bc = t.view(-1, *([1] * (x_1.dim() - 1)))  # [B, 1, ...]
        x_t = (1.0 - t_bc) * x_0 + t_bc * x_1
        target = x_1 - x_0
        return x_t, target


# ------------------------------------------------------------------
# Graph-aware CFM wrapper for EGNN (V1)
# ------------------------------------------------------------------

class EGNNFlowMatching(nn.Module):
    """
    Wraps the SE3EquivariantEGNN as a CFM model.

    Training:
        loss = cfm_loss(batch)
    Inference:
        tau_pred = sample(batch)   # ODE integration via torchdiffeq
    """

    def __init__(self, vector_field: nn.Module, ode_rtol: float = 1e-4, ode_atol: float = 1e-5):
        super().__init__()
        self.vf = vector_field
        self.cfm = ConditionalFlowMatcher()
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _vf_forward(
        self,
        x_t: torch.Tensor,
        t_scalar: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        """Evaluate vector field for all nodes in a batched graph."""
        n_nodes = batch.x.shape[0]
        t_node = t_scalar.expand(n_nodes, 1)   # [N, 1]
        return self.vf(
            x=batch.x,
            pos=batch.pos,
            edge_index=batch.edge_index,
            x_t=x_t,
            t=t_node,
        )                                        # [N, 6]

    def loss(self, batch: Batch) -> torch.Tensor:
        """
        Compute CFM training loss for a batched graph.

        Args:
            batch: torch_geometric Batch with .x (condition) and .y (targets).

        Returns:
            Scalar MSE loss between predicted and true vector fields.
        """
        y = batch.y                  # [N, 6] true stress irreps (normalized)
        n_nodes = y.shape[0]

        # Sample t per graph, then broadcast to nodes
        n_graphs = batch.num_graphs
        t_graph = self.cfm.sample_t(n_graphs, y.device)     # [G]
        t_node = t_graph[batch.batch]                        # [N]

        # Sample source Gaussian noise in stress irreps space
        x_0 = torch.randn_like(y)                           # [N, 6]

        # Interpolate
        x_t, target = self.cfm.interpolate(x_0, y, t_node)  # [N, 6]

        # Predict vector field
        pred = self._vf_forward(x_t, t_node.unsqueeze(-1), batch)  # [N, 6]

        return torch.mean((pred - target) ** 2)

    @torch.no_grad()
    def sample(
        self,
        batch: Batch,
        n_steps: int = 100,
        method: str = 'dopri5',
    ) -> torch.Tensor:
        """
        Generate SGS stress prediction by integrating the ODE.

        Args:
            batch:   Batched graph with conditioning features.
            n_steps: number of ODE time steps (for fixed solvers).
            method:  ODE solver ('dopri5', 'euler', 'rk4').

        Returns:
            [N, 6] predicted stress irreps.
        """
        n_nodes = batch.x.shape[0]
        x_0 = torch.randn(n_nodes, 6, device=batch.x.device, dtype=batch.x.dtype)

        # Store batch for use inside the ODE function
        _batch = batch

        def ode_fn(t, x):
            t_node = t.expand(n_nodes, 1)
            return self.vf(
                x=_batch.x,
                pos=_batch.pos,
                edge_index=_batch.edge_index,
                x_t=x,
                t=t_node,
            )

        t_span = torch.tensor([0.0, 1.0], device=x_0.device)
        traj = odeint(
            ode_fn, x_0, t_span,
            method=method,
            rtol=self.ode_rtol,
            atol=self.ode_atol,
        )
        return traj[-1]   # [N, 6] at t=1


# ------------------------------------------------------------------
# MLP-based CFM wrapper for V3
# ------------------------------------------------------------------

class MLPFlowMatching(nn.Module):
    """
    Wraps the VectorFieldMLP as a CFM model (V3, non-equivariant).

    Node-wise (no graph structure): for each LES grid point, uses its
    flattened 9-component velocity gradient as the condition.
    """

    def __init__(self, vector_field: nn.Module, ode_rtol: float = 1e-4, ode_atol: float = 1e-5):
        super().__init__()
        self.vf = vector_field
        self.cfm = ConditionalFlowMatcher()
        self.ode_rtol = ode_rtol
        self.ode_atol = ode_atol

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def loss(self, grad_flat: torch.Tensor, tau_flat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grad_flat: [B, 9] velocity gradient (flat 3x3).
            tau_flat:  [B, 6] stress irreps (target).

        Returns:
            Scalar CFM loss.
        """
        B = tau_flat.shape[0]
        t = self.cfm.sample_t(B, tau_flat.device)   # [B]
        x_0 = torch.randn_like(tau_flat)             # [B, 6]
        x_t, target = self.cfm.interpolate(x_0, tau_flat, t)
        pred = self.vf(grad_flat, x_t, t)
        return torch.mean((pred - target) ** 2)

    @torch.no_grad()
    def sample(
        self,
        grad_flat: torch.Tensor,
        method: str = 'dopri5',
    ) -> torch.Tensor:
        """
        Args:
            grad_flat: [B, 9] velocity gradient condition.

        Returns:
            [B, 6] predicted stress irreps.
        """
        B = grad_flat.shape[0]
        x_0 = torch.randn(B, 6, device=grad_flat.device, dtype=grad_flat.dtype)
        _cond = grad_flat

        def ode_fn(t, x):
            t_batch = t.expand(B)
            return self.vf(_cond, x, t_batch)

        t_span = torch.tensor([0.0, 1.0], device=x_0.device)
        traj = odeint(ode_fn, x_0, t_span, method=method,
                      rtol=self.ode_rtol, atol=self.ode_atol)
        return traj[-1]


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def build_model(variant: str, cfg: dict) -> nn.Module:
    """
    Build V1, V2, or V3 model from config dict.

    Args:
        variant: 'v1', 'v2', or 'v3'.
        cfg:     model config dict (from YAML).

    Returns:
        nn.Module (EGNNFlowMatching, SE3EquivariantEGNN, or MLPFlowMatching).
    """
    from .egnn import SE3EquivariantEGNN
    from .mlp import VectorFieldMLP
    import e3nn.o3 as o3

    irreps_hidden = o3.Irreps(cfg.get('irreps_hidden', "32x0e + 16x1o + 8x2e"))

    if variant == 'v1':
        egnn = SE3EquivariantEGNN(
            use_flow_matching=True,
            n_layers=cfg.get('n_layers', 5),
            irreps_hidden=irreps_hidden,
            lmax_sh=cfg.get('lmax_sh', 2),
            n_radial_hidden=cfg.get('n_radial_hidden', 64),
        )
        return EGNNFlowMatching(
            egnn,
            ode_rtol=cfg.get('ode_rtol', 1e-4),
            ode_atol=cfg.get('ode_atol', 1e-5),
        )

    elif variant == 'v2':
        return SE3EquivariantEGNN(
            use_flow_matching=False,
            n_layers=cfg.get('n_layers', 5),
            irreps_hidden=irreps_hidden,
            lmax_sh=cfg.get('lmax_sh', 2),
            n_radial_hidden=cfg.get('n_radial_hidden', 64),
        )

    elif variant == 'v3':
        mlp = VectorFieldMLP(
            dim_grad=9,
            dim_stress=6,
            dim_hidden=cfg.get('dim_hidden', 256),
            n_layers=cfg.get('n_layers', 6),
            time_embed_dim=cfg.get('time_embed_dim', 64),
        )
        return MLPFlowMatching(
            mlp,
            ode_rtol=cfg.get('ode_rtol', 1e-4),
            ode_atol=cfg.get('ode_atol', 1e-5),
        )

    else:
        raise ValueError(f"Unknown variant: {variant}. Choose 'v1', 'v2', or 'v3'.")
