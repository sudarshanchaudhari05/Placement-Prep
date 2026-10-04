"""
Validation Step Logic
=====================
Computes validation losses and accumulates path detection metrics.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn

from src.losses.structural_loss import TotalRowAnchorLoss
from src.evaluation.metrics import RowAnchorEvaluator


@torch.no_grad()
def validate_one_step(
    model: nn.Module,
    criterion: TotalRowAnchorLoss,
    evaluator: RowAnchorEvaluator,
    images: torch.Tensor,
    targets: torch.Tensor,
    u_coords: torch.Tensor,
    presence: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """
    Execute a single validation evaluation step.
    
    Args:
        model: RowAnchorNet.
        criterion: TotalRowAnchorLoss.
        evaluator: RowAnchorEvaluator metric accumulator.
        images: Input batch tensor (B, 3, 288, 384).
        targets: Target class indices (B, 18).
        u_coords: Ground truth u-coordinates (B, 18).
        presence: Ground truth presence mask (B, 18).
        device: Torch device.
        
    Returns:
        Dict of step loss values.
    """
    model.eval()
    images = images.to(device)
    targets = targets.to(device)

    logits = model(images)  # (B, 18, 101)
    loss_dict = criterion(logits, targets)

    evaluator.update(logits, u_coords, presence)

    return {k: float(v.item()) for k, v in loss_dict.items()}
