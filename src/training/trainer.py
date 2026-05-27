"""
Unified trainer for all three variants (V1, V2, V3).

Hyperparameters fixed across variants per the paper spec:
  - AdamW: lr=1e-3, weight_decay=1e-4
  - 50,000 training steps
  - Cosine LR schedule with linear warmup (5% of total steps)

V3 additionally applies the 24 cubic rotation augmentations before each forward pass.
"""

import time
import yaml
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch

from .losses import flow_matching_loss, mse_loss
from data.augmentation import apply_cubic_rotations, rotate_velocity_gradient, rotate_sgs_stress
import numpy as np


class Trainer:
    """
    Trains a single variant (V1, V2, or V3) for a fixed number of steps.

    Args:
        model:       nn.Module (EGNNFlowMatching, SE3EquivariantEGNN, or MLPFlowMatching).
        variant:     'v1', 'v2', or 'v3'.
        train_loader: DataLoader for training graphs.
        val_loader:   DataLoader for validation.
        cfg:          dict with training hyperparameters.
        device:       torch.device.
        out_dir:      directory for checkpoints and logs.
    """

    DEFAULTS = dict(
        lr=1e-3,
        weight_decay=1e-4,
        total_steps=50_000,
        warmup_fraction=0.05,
        val_every=500,
        log_every=100,
        grad_clip=1.0,
        accumulation_steps=1,
    )

    def __init__(
        self,
        model: nn.Module,
        variant: str,
        train_loader,
        val_loader,
        cfg: Optional[dict] = None,
        device: torch.device = torch.device('cpu'),
        out_dir: str = './runs',
    ):
        self.model = model.to(device)
        self.variant = variant.lower()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        hparams = {**self.DEFAULTS, **(cfg or {})}
        self.total_steps = hparams['total_steps']
        self.val_every = hparams['val_every']
        self.log_every = hparams['log_every']
        self.grad_clip = hparams['grad_clip']
        self.accumulation_steps = int(hparams.get('accumulation_steps', 1))

        self.optimizer = AdamW(
            model.parameters(),
            lr=hparams['lr'],
            weight_decay=hparams['weight_decay'],
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.total_steps, eta_min=1e-6
        )

        self.warmup_steps = int(self.total_steps * hparams['warmup_fraction'])
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}

        # V3: precompute rotation matrices on device
        if self.variant == 'v3':
            from data.augmentation import CUBIC_ROTATIONS_SO3
            self.cube_rots = torch.from_numpy(CUBIC_ROTATIONS_SO3).to(device)  # [24, 3, 3]

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _apply_warmup(self):
        if self.global_step < self.warmup_steps:
            scale = (self.global_step + 1) / self.warmup_steps
            for pg in self.optimizer.param_groups:
                pg['lr'] = scale * self.DEFAULTS['lr']

    # ------------------------------------------------------------------
    # Batch preparation
    # ------------------------------------------------------------------

    def _prepare_batch_v1(self, batch: Batch):
        return batch.to(self.device)

    def _prepare_batch_v2(self, batch: Batch):
        return batch.to(self.device)

    def _prepare_batch_v3(self, batch: Batch):
        """
        Flatten graph into per-node tensors and apply a random cubic rotation.
        """
        batch = batch.to(self.device)

        # Reconstruct 3x3 tensors from irreps storage
        # .grad_full [N, 9], .tau_full [N, 9]
        grad_flat = batch.grad_full   # [N, 9]  (flat 3x3, not irreps)
        tau_full  = batch.tau_full    # [N, 9]

        grad3x3 = grad_flat.reshape(-1, 3, 3)
        tau3x3  = tau_full.reshape(-1, 3, 3)

        # Random rotation from the 24 cubic group
        idx = torch.randint(0, 24, (1,)).item()
        R = self.cube_rots[idx]   # [3, 3]

        grad3x3 = R @ grad3x3 @ R.T     # [N, 3, 3]
        tau3x3  = R @ tau3x3  @ R.T     # [N, 3, 3]

        # Convert to irreps for the MLP
        from data.dataset import grad_to_irreps, stress_to_irreps
        grad_irr = grad_to_irreps(grad3x3)    # [N, 9]
        tau_irr  = stress_to_irreps(tau3x3)   # [N, 6]

        return grad_irr, tau_irr

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def _forward_loss(self, batch) -> torch.Tensor:
        """Single forward pass; returns unscaled loss."""
        if self.variant == 'v1':
            batch = self._prepare_batch_v1(batch)
            return self.model.loss(batch)
        elif self.variant == 'v2':
            batch = self._prepare_batch_v2(batch)
            pred = self.model(x=batch.x, pos=batch.pos, edge_index=batch.edge_index)
            return mse_loss(pred, batch.y)
        elif self.variant == 'v3':
            grad_irr, tau_irr = self._prepare_batch_v3(batch)
            return self.model.loss(grad_irr, tau_irr)

    def _train_step(self, batches: list) -> float:
        """One optimizer update over `accumulation_steps` mini-batches."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for batch in batches:
            loss = self._forward_loss(batch) / self.accumulation_steps
            loss.backward()
            total_loss += loss.item()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self._apply_warmup()
        self.optimizer.step()
        self.scheduler.step()
        return total_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total_loss, total_nodes = 0.0, 0

        for batch in self.val_loader:
            if self.variant in ('v1', 'v2'):
                batch = batch.to(self.device)

            if self.variant == 'v1':
                loss = self.model.loss(batch)
                n = batch.num_nodes
            elif self.variant == 'v2':
                pred = self.model(
                    x=batch.x, pos=batch.pos, edge_index=batch.edge_index
                )
                loss = mse_loss(pred, batch.y)
                n = batch.num_nodes
            elif self.variant == 'v3':
                grad_irr, tau_irr = self._prepare_batch_v3(batch)
                loss = self.model.loss(grad_irr, tau_irr)
                n = tau_irr.shape[0]

            total_loss += loss.item() * n
            total_nodes += n

        return total_loss / max(total_nodes, 1)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        print(f"\n=== Training {self.variant.upper()} | "
              f"{self.model.n_params if hasattr(self.model, 'n_params') else '?'} params | "
              f"{self.total_steps} steps ===\n")

        loader_iter = iter(self.train_loader)
        t_start = time.time()

        def _next_batch():
            nonlocal loader_iter
            try:
                return next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.train_loader)
                return next(loader_iter)

        while self.global_step < self.total_steps:
            batches = [_next_batch() for _ in range(self.accumulation_steps)]
            loss = self._train_step(batches)
            self.global_step += 1
            self.history['train_loss'].append(loss)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            if self.global_step % self.log_every == 0:
                elapsed = time.time() - t_start
                print(f"  step {self.global_step:>6d}/{self.total_steps} | "
                      f"loss={loss:.6f} | "
                      f"lr={self.optimizer.param_groups[0]['lr']:.2e} | "
                      f"elapsed={elapsed:.0f}s")
                self._save_checkpoint('last.pt')

            if self.global_step % self.val_every == 0:
                val_loss = self._validate()
                self.history['val_loss'].append((self.global_step, val_loss))
                print(f"  >>> val loss={val_loss:.6f}  (step {self.global_step})")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint('best.pt')

        self._save_checkpoint('final.pt')
        self._save_history()
        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.6f}")

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _save_checkpoint(self, name: str):
        ckpt = {
            'step': self.global_step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'val_loss': self.best_val_loss,
            'variant': self.variant,
        }
        torch.save(ckpt, self.out_dir / name)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.scheduler.load_state_dict(ckpt['scheduler_state'])
        self.global_step = ckpt['step']
        self.best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f"Loaded checkpoint from step {self.global_step}")

    def _save_history(self):
        import json
        with open(self.out_dir / 'history.json', 'w') as f:
            json.dump(self.history, f)
