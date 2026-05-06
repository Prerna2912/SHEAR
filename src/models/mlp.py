"""
Standard MLP vector field for V3 (non-equivariant conditional flow matching).

Input: [B, 9 (grad_u) + 6 (x_t) + 1 (t)] = [B, 16]
Output: [B, 6] (vector field in stress component space)

Equivariance is enforced through data augmentation (24 cubic rotations) during
training, not through architectural constraints.
"""

import torch
import torch.nn as nn


class VectorFieldMLP(nn.Module):
    """
    MLP-based vector field for non-equivariant CFM (V3).

    Time is embedded with a sinusoidal encoding and concatenated with the
    velocity gradient and noisy stress inputs.

    Args:
        dim_grad:   velocity gradient dimension (9).
        dim_stress: SGS stress dimension (6).
        dim_hidden: width of hidden layers.
        n_layers:   depth of the MLP (number of hidden layers).
        time_embed_dim: sinusoidal time embedding dimension.
    """

    def __init__(
        self,
        dim_grad: int = 9,
        dim_stress: int = 6,
        dim_hidden: int = 256,
        n_layers: int = 6,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.dim_grad = dim_grad
        self.dim_stress = dim_stress
        self.time_embed_dim = time_embed_dim

        # Sinusoidal time embedding
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )

        dim_in = dim_grad + dim_stress + time_embed_dim
        layers = [nn.Linear(dim_in, dim_hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(dim_hidden, dim_hidden),
                nn.LayerNorm(dim_hidden),
                nn.SiLU(),
            ]
        layers.append(nn.Linear(dim_hidden, dim_stress))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,    # [B, 9]  velocity gradient (flat, not irreps)
        x_t: torch.Tensor,  # [B, 6]  noisy stress at time t
        t: torch.Tensor,    # [B] or [B, 1]  flow matching time in [0, 1]
    ) -> torch.Tensor:
        """
        Returns:
            [B, 6] vector field prediction.
        """
        t = t.view(-1, 1).float() if t.dim() == 1 else t.float()
        t_emb = self.time_embed(t.squeeze(-1))          # [B, time_embed_dim]
        inp = torch.cat([x, x_t, t_emb], dim=-1)        # [B, 9+6+dim_t]
        return self.net(inp)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SinusoidalEmbedding(nn.Module):
    """Fixed sinusoidal position embedding for the time scalar t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2
        freqs = torch.exp(-torch.arange(half, dtype=torch.float32) * (8.0 / half))
        self.register_buffer('freqs', freqs)   # [half]

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] scalar in [0,1]. Returns [B, dim]."""
        t = t.float().unsqueeze(-1)            # [B, 1]
        args = t * self.freqs.unsqueeze(0)     # [B, half]
        return torch.cat([args.sin(), args.cos()], dim=-1)  # [B, dim]
