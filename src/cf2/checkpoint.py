"""
CF2 — V1 model and normalisation statistics loader.

Checkpoint format (written by src/training/trainer.py _save_checkpoint):
    {
        'step':            int,
        'model_state':     OrderedDict,
        'optimizer_state': ...,
        'scheduler_state': ...,
        'val_loss':        float,
        'variant':         str,   # 'v1'
    }

Stats format (written by build_dataloaders, saved separately):
    {
        'x_mean': Tensor[9],   # velocity gradient feature mean
        'x_std':  Tensor[9],   # velocity gradient feature std
        'y_mean': Tensor[6],   # stress target mean
        'y_std':  Tensor[6],   # stress target std
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import yaml


def load_v1(
    checkpoint_path: Union[str, Path],
    stats_path: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> tuple:
    """
    Load V1 EGNNFlowMatching model from checkpoint.

    Args:
        checkpoint_path: path to .pt checkpoint file.
        stats_path      : path to stats.pt (normalisation statistics).
                          If None, infers from checkpoint parent directory.
        config_path     : path to v1.yaml config.
                          If None, searches ./configs/v1.yaml.
        device          : target device. Auto-selects CUDA if available.

    Returns:
        model  : EGNNFlowMatching in eval() mode on device.
        stats  : dict with x_mean, x_std, y_mean, y_std tensors on device.
        device : the resolved device.
    """
    import sys, os
    sys.path.insert(0, str(Path(__file__).parents[2]))

    checkpoint_path = Path(checkpoint_path)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- locate stats ---------------------------------------------------------
    if stats_path is None:
        # look in parent dir, then grandparent (runs/quick/stats.pt vs runs/quick/v1/stats.pt)
        for candidate in [
            checkpoint_path.parent / 'stats.pt',
            checkpoint_path.parent.parent / 'stats.pt',
        ]:
            if candidate.exists():
                stats_path = candidate
                break

    if stats_path is None or not Path(stats_path).exists():
        raise FileNotFoundError(
            f"Could not find stats.pt near {checkpoint_path}. "
            "Pass stats_path= explicitly."
        )

    # --- locate config --------------------------------------------------------
    if config_path is None:
        for candidate in [
            Path('configs/v1.yaml'),
            Path(__file__).parents[2] / 'configs' / 'v1.yaml',
        ]:
            if candidate.exists():
                config_path = candidate
                break

    if config_path is None or not Path(config_path).exists():
        # Fall back to minimal defaults matching the quick-train config
        model_cfg = {
            'irreps_hidden': '16x0e + 8x1e + 4x2e',
            'n_layers': 4,
            'lmax_sh': 2,
            'n_radial_hidden': 64,
            'ode_rtol': 1e-4,
            'ode_atol': 1e-5,
        }
    else:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg['model']

    # --- build model ----------------------------------------------------------
    from src.models.flow_matching import build_model
    model = build_model('v1', model_cfg)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()

    # --- load stats -----------------------------------------------------------
    raw_stats = torch.load(stats_path, map_location=device, weights_only=False)
    stats = {
        'x_mean': raw_stats['x_mean'].to(device),
        'x_std':  raw_stats['x_std'].to(device).clamp(min=1e-8),
        'y_mean': raw_stats['y_mean'].to(device),
        'y_std':  raw_stats['y_std'].to(device).clamp(min=1e-8),
    }

    print(
        f"[CF2] Loaded V1 checkpoint (step={ckpt.get('step', '?')}, "
        f"val_loss={ckpt.get('val_loss', '?'):.4g}) on {device}"
    )
    return model, stats, device


# Singleton model holder — loaded once per process at FastAPI startup
_MODEL   = None
_STATS   = None
_DEVICE  = None


def get_model():
    """Return the globally loaded (model, stats, device) tuple."""
    if _MODEL is None:
        raise RuntimeError(
            "V1 model not loaded. Call init_model() first "
            "(this is done automatically at FastAPI startup)."
        )
    return _MODEL, _STATS, _DEVICE


def init_model(
    checkpoint_path: Union[str, Path],
    stats_path: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> None:
    """Load model into global singleton (called once at app startup)."""
    global _MODEL, _STATS, _DEVICE
    _MODEL, _STATS, _DEVICE = load_v1(
        checkpoint_path, stats_path, config_path, device
    )
