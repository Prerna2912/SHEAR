"""
Task 2 neural baselines.

V3 — BeckMLP:              3-layer MLP, MSE loss (Beck et al. J. Comp. Phys. 2019, bib #10).
V4 — EnergyConservingMLP:  same backbone, positive-SGS-dissipation output constraint
                            via Cholesky parameterisation (van Gastelen et al. 2025, bib #9).

Tensor conventions (both classes):
  Input:  velocity gradient flat [B, 9] (row-major 3×3).
  Output: SGS stress flat [B, 9] (3×3, symmetric → only upper triangle used in loss).

Helper utilities:
  tau_3x3_to_voigt  — [N, 3, 3] → [N, 6] Voigt [τ11, τ22, τ33, τ12, τ13, τ23]
  voigt_to_tau_3x3  — [N, 6] → [N, 3, 3]
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# ---------------------------------------------------------------------------
# Voigt ↔ 3×3 helpers
# ---------------------------------------------------------------------------

def tau_3x3_to_voigt(tau: torch.Tensor) -> torch.Tensor:
    """[N, 3, 3] → [N, 6]  Voigt: [τ11, τ22, τ33, τ12, τ13, τ23]"""
    return torch.stack([
        tau[:, 0, 0], tau[:, 1, 1], tau[:, 2, 2],
        tau[:, 0, 1], tau[:, 0, 2], tau[:, 1, 2],
    ], dim=-1)


def voigt_to_tau_3x3(v: torch.Tensor) -> torch.Tensor:
    """[N, 6] → [N, 3, 3]  from Voigt [τ11, τ22, τ33, τ12, τ13, τ23]"""
    B = v.shape[0]
    tau = torch.zeros(B, 3, 3, dtype=v.dtype, device=v.device)
    tau[:, 0, 0] = v[:, 0]; tau[:, 1, 1] = v[:, 1]; tau[:, 2, 2] = v[:, 2]
    tau[:, 0, 1] = tau[:, 1, 0] = v[:, 3]
    tau[:, 0, 2] = tau[:, 2, 0] = v[:, 4]
    tau[:, 1, 2] = tau[:, 2, 1] = v[:, 5]
    return tau


# ---------------------------------------------------------------------------
# Beck MLP (V3)
# ---------------------------------------------------------------------------

class BeckMLP(nn.Module):
    """
    3-layer MLP SGS closure (Beck et al. 2019).

    Input:  velocity gradient [B, 9] (flat 3×3).
    Output: SGS stress [B, 6] (Voigt).

    No equivariance constraint — replicates the original architecture.
    """

    def __init__(self, input_dim: int = 9, hidden_dim: int = 256,
                 output_dim: int = 6, n_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 9] → [B, 6]"""
        return self.net(x)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Energy-Conserving MLP (V4) — van Gastelen et al. 2025
# ---------------------------------------------------------------------------

class EnergyConservingMLP(nn.Module):
    """
    MLP with a Cholesky-parameterised output layer that guarantees
    non-positive SGS dissipation τ:S ≤ 0 everywhere (no spurious backscatter).

    Architecture:
      - Same 3-layer backbone as BeckMLP.
      - Final head outputs 6 Cholesky entries → positive semi-definite W.
      - SGS stress: τ = -(W S + S W) where S is the strain rate.
        This gives τ:S = -2 tr(W S²) ≤ 0 since W and S² are both PSD.

    Input:  velocity gradient [B, 9] (flat 3×3).
    Output: SGS stress [B, 6] (Voigt).
    """

    def __init__(self, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(9, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        self.backbone = nn.Sequential(*layers)
        # 6 Cholesky entries: L_11, L_21, L_22, L_31, L_32, L_33
        self.cholesky_head = nn.Linear(hidden_dim, 6)

    def _cholesky_to_w(self, c: torch.Tensor) -> torch.Tensor:
        """Convert 6 Cholesky components → PSD matrix W = L Lᵀ.  [B,6] → [B,3,3]"""
        B = c.shape[0]
        L = torch.zeros(B, 3, 3, dtype=c.dtype, device=c.device)
        L[:, 0, 0] = F.softplus(c[:, 0])      # positive diagonal
        L[:, 1, 0] = c[:, 1]
        L[:, 1, 1] = F.softplus(c[:, 2])
        L[:, 2, 0] = c[:, 3]
        L[:, 2, 1] = c[:, 4]
        L[:, 2, 2] = F.softplus(c[:, 5])
        return L @ L.transpose(-1, -2)         # W = L Lᵀ ≥ 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 9] velocity gradient (flat 3×3).

        Returns:
            tau_voigt: [B, 6] SGS stress in Voigt notation.
        """
        h = self.backbone(x)                              # [B, hidden]
        c = self.cholesky_head(h)                         # [B, 6]
        W = self._cholesky_to_w(c)                        # [B, 3, 3], PSD

        grad3x3 = x.reshape(-1, 3, 3)                     # [B, 3, 3]
        S = 0.5 * (grad3x3 + grad3x3.transpose(-1, -2))  # strain rate [B,3,3]

        # τ = -(W S + S W)  →  τ:S = -2 tr(W S²) ≤ 0
        tau_3x3 = -(W @ S + S @ W)
        return tau_3x3_to_voigt(tau_3x3)                  # [B, 6]

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Shared training loop
# ---------------------------------------------------------------------------

def train_mlp(
    model: nn.Module,
    train_loader,
    val_loader,
    n_steps: int = 50_000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    warmup_fraction: float = 0.05,
    grad_clip: float = 1.0,
    log_every: int = 500,
    val_every: int = 500,
    out_dir: str = './runs/task2',
    device: torch.device = None,
) -> dict:
    """
    Train BeckMLP or EnergyConservingMLP.

    DataLoader must yield batches with attributes:
        .grad_full  [N_nodes, 9]  flat velocity gradient
        .tau_full   [N_nodes, 9]  flat 3×3 SGS stress

    Returns history dict with 'train_loss', 'val_loss'.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    model = model.to(device)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-6)
    warmup_steps = int(n_steps * warmup_fraction)

    history = {'train_loss': [], 'val_loss': []}
    best_val = float('inf')

    def _extract(batch):
        """Extract flat gradient + Voigt stress from a PyG batch."""
        grad9 = batch.grad_full.reshape(-1, 9).to(device)     # [B,9]
        tau9  = batch.tau_full.reshape(-1, 9).to(device)      # [B,9]
        tau3x3 = tau9.reshape(-1, 3, 3)
        return grad9, tau_3x3_to_voigt(tau3x3)                # [B,6]

    loader_iter = iter(train_loader)
    t0 = time.time()

    for step in range(1, n_steps + 1):
        # Warmup
        if step <= warmup_steps:
            scale = step / warmup_steps
            for pg in optimizer.param_groups:
                pg['lr'] = scale * lr

        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        grad9, tau_voigt = _extract(batch)
        pred = model(grad9)
        loss = F.mse_loss(pred, tau_voigt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        history['train_loss'].append(loss.item())

        if step % log_every == 0:
            print(f"  step {step:>6d}/{n_steps} | loss={loss.item():.6f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                  f"t={time.time()-t0:.0f}s")

        if step % val_every == 0:
            model.eval()
            total, n = 0.0, 0
            with torch.no_grad():
                for vbatch in val_loader:
                    g, t = _extract(vbatch)
                    vl = F.mse_loss(model(g), t).item()
                    total += vl * g.shape[0]; n += g.shape[0]
            val_loss = total / max(n, 1)
            history['val_loss'].append((step, val_loss))
            print(f"  >>> val={val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({'step': step,
                            'model_state': model.state_dict(),
                            'val_loss': best_val},
                           out_path / 'best.pt')

    torch.save({'step': n_steps,
                'model_state': model.state_dict()},
               out_path / 'final.pt')
    print(f"Training done. Best val loss: {best_val:.6f}")
    return history


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model: nn.Module,
                  grad_u_np: np.ndarray,
                  batch_size: int = 4096,
                  device: torch.device = None) -> np.ndarray:
    """
    Run inference on a numpy array of velocity gradients.

    Args:
        grad_u_np: [N, 9] or [N, 3, 3] velocity gradient (numpy).
        batch_size: internal mini-batch size for large arrays.

    Returns:
        tau_pred: [N, 6] predicted stress in Voigt notation (numpy).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    if grad_u_np.ndim == 3:
        grad_u_np = grad_u_np.reshape(-1, 9)

    N = grad_u_np.shape[0]
    preds = []
    for start in range(0, N, batch_size):
        chunk = torch.from_numpy(
            grad_u_np[start:start + batch_size].astype(np.float32)
        ).to(device)
        preds.append(model(chunk).cpu().numpy())
    return np.concatenate(preds, axis=0)
