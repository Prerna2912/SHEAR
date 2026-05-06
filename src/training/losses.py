"""
Loss functions for all three variants.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch


def flow_matching_loss(model, batch_or_inputs) -> torch.Tensor:
    """
    Dispatch to the appropriate CFM loss based on model type.

    For V1 (EGNNFlowMatching): batch_or_inputs is a torch_geometric Batch.
    For V3 (MLPFlowMatching):  batch_or_inputs is a tuple (grad_flat, tau_flat).
    """
    from models.flow_matching import EGNNFlowMatching, MLPFlowMatching
    if isinstance(model, EGNNFlowMatching):
        return model.loss(batch_or_inputs)
    elif isinstance(model, MLPFlowMatching):
        grad_flat, tau_flat = batch_or_inputs
        return model.loss(grad_flat, tau_flat)
    else:
        raise TypeError(f"Unexpected model type: {type(model)}")


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Plain MSE loss for V2 (deterministic regression).

    Args:
        pred:   [N, 6] predicted stress irreps.
        target: [N, 6] true stress irreps.
    """
    return F.mse_loss(pred, target)
