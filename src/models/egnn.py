"""
SE(3)-Equivariant Graph Neural Network using e3nn.

Used as:
  - V1: vector field for conditional flow matching (takes x_t + condition + time).
  - V2: deterministic regression (takes condition only, outputs tau directly).

Architecture:
  - Irreps embedding layer
  - N equivariant message-passing layers with tensor product + radial MLP
  - Irreps output projection

Input node features:
  - Velocity gradient: "1x0e + 1x1e + 1x2e"  (9 components)
    (0e=trace, 1e=antisym/vorticity axial vector, 2e=sym traceless)
  - [V1 only] Noisy stress x_t: "1x0e + 1x2e" (6 components)
  - [V1 only] Time t:  "1x0e"                  (1 component)

Output:
  - "1x0e + 1x2e"  (6 components, representing the symmetric SGS stress)
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_ckpt
from e3nn import o3
from e3nn.nn import BatchNorm as E3BatchNorm
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


# Irreps used throughout.
# Gradient: 1o⊗1o = 0e + 1e + 2e (antisymmetric part is axial vector → 1e)
IRREPS_GRAD  = o3.Irreps("1x0e + 1x1e + 1x2e")   # velocity gradient (9)
IRREPS_STRESS = o3.Irreps("1x0e + 1x2e")           # SGS stress (6)
IRREPS_TIME  = o3.Irreps("1x0e")                   # time scalar (1)


def _build_irreps_in(use_flow_matching: bool) -> o3.Irreps:
    """Node feature irreps for the EGNN."""
    irreps = IRREPS_GRAD
    if use_flow_matching:
        irreps = irreps + IRREPS_STRESS + IRREPS_TIME
    return irreps


# ------------------------------------------------------------------
# Radial MLP (scalar, equivariance-safe)
# ------------------------------------------------------------------

class RadialMLP(nn.Module):
    """MLP mapping radial distance -> TP weights (fully scalar, no equivariance needed)."""

    def __init__(self, n_out: int, n_hidden: int = 64, n_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(1, n_hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.SiLU()]
        layers.append(nn.Linear(n_hidden, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, r_norm: torch.Tensor) -> torch.Tensor:
        return self.net(r_norm.unsqueeze(-1))


# ------------------------------------------------------------------
# Equivariant message-passing layer
# ------------------------------------------------------------------

class EquivariantConv(MessagePassing):
    """
    One equivariant message-passing step.

    Message: m_{ij} = TP(h_j, Y(r̂_{ij})) * f_radial(|r_{ij}|)
    Update:  h_i <- LayerNorm(h_i + sum_j m_{ij})

    The tensor product (TP) maps:
        irreps_node x irreps_sh -> irreps_node
    and its scalar weights come from a radial MLP conditioned on edge length.
    """

    def __init__(
        self,
        irreps_node: o3.Irreps,
        irreps_sh: o3.Irreps,
        n_radial_hidden: int = 64,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.irreps_node = irreps_node
        self.irreps_sh = irreps_sh

        self.tp = o3.FullyConnectedTensorProduct(
            irreps_node, irreps_sh, irreps_node, shared_weights=False
        )
        self.radial = RadialMLP(self.tp.weight_numel, n_radial_hidden)

        self.norm = E3BatchNorm(irreps_node)

        # Self-interaction linear
        self.self_lin = o3.Linear(irreps_node, irreps_node)

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        row, col = edge_index  # row=destination, col=source

        r_vec = pos[col] - pos[row]                           # [E, 3]
        r_norm = r_vec.norm(dim=-1, keepdim=True).clamp(1e-8) # [E, 1]
        r_hat = r_vec / r_norm                                 # [E, 3]

        sh = o3.spherical_harmonics(
            self.irreps_sh, r_hat, normalize=True, normalization='component'
        )                                                      # [E, dim_sh]
        weights = self.radial(r_norm.squeeze(-1))              # [E, n_weights]

        # Propagate
        agg = self.propagate(
            edge_index, h=h, sh=sh, weights=weights,
            size=(h.shape[0], h.shape[0])
        )                                                      # [N, dim_node]

        out = h + agg + self.self_lin(h)
        out = self.norm(out)
        return out

    def message(self, h_j, sh, weights):
        return self.tp(h_j, sh, weights)


# ------------------------------------------------------------------
# Full EGNN
# ------------------------------------------------------------------

class SE3EquivariantEGNN(nn.Module):
    """
    SE(3)-equivariant EGNN for SGS stress prediction / flow matching.

    Args:
        use_flow_matching: if True, also accepts noisy stress x_t and time t
                           as node features (for V1). If False, acts as a
                           deterministic regression model (V2).
        n_layers:          number of equivariant message-passing layers.
        irreps_hidden:     hidden irreps for intermediate representations.
        lmax_sh:           maximum L for spherical harmonics on edges.
        n_radial_hidden:   hidden units in radial MLP.
    """

    LMAX = 2

    def __init__(
        self,
        use_flow_matching: bool = True,
        n_layers: int = 5,
        irreps_hidden: o3.Irreps = o3.Irreps("32x0e + 16x1o + 8x2e"),
        lmax_sh: int = 2,
        n_radial_hidden: int = 64,
    ):
        super().__init__()
        self.use_flow_matching = use_flow_matching

        irreps_in = _build_irreps_in(use_flow_matching)
        irreps_sh = o3.Irreps.spherical_harmonics(lmax_sh)

        # Input embedding
        self.embed = o3.Linear(irreps_in, irreps_hidden)

        # Message-passing layers
        self.layers = nn.ModuleList([
            EquivariantConv(irreps_hidden, irreps_sh, n_radial_hidden)
            for _ in range(n_layers)
        ])

        # Output projection -> stress irreps
        self.output = o3.Linear(irreps_hidden, IRREPS_STRESS)

    def forward(
        self,
        x: torch.Tensor,          # [N, 9] velocity gradient irreps
        pos: torch.Tensor,         # [N, 3] node positions
        edge_index: torch.Tensor,  # [2, E]
        x_t: torch.Tensor = None,  # [N, 6] noisy stress (V1 only)
        t: torch.Tensor = None,    # [N, 1] time (V1 only)
    ) -> torch.Tensor:
        """
        Returns:
            [N, 6] predicted stress irreps (V2) or vector field (V1).
        """
        if self.use_flow_matching:
            assert x_t is not None and t is not None, \
                "V1 requires x_t and t inputs"
            node_feat = torch.cat([x, x_t, t], dim=-1)   # [N, 9+6+1=16]
        else:
            node_feat = x                                  # [N, 9]

        h = self.embed(node_feat)

        for layer in self.layers:
            if self.training:
                h = grad_ckpt(layer, h, pos, edge_index, use_reentrant=False)
            else:
                h = layer(h, pos, edge_index)

        return self.output(h)   # [N, 6]

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
