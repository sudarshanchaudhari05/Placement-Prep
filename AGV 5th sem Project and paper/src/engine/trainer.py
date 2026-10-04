"""
Training Step and Optimizer Setup
=================================
Implements the exact training specifications:
- Optimizer: AdamW (beta1=0.9, beta2=0.999, weight_decay=1e-4)
- Learning Rate: Initial 4e-3, Cosine Annealing to 1e-5
- Total Loss: L_focal + 0.5 * L_smooth + 0.75 * L_curv
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs.config import ProjectConfig, TrainingConfig
from src.losses.structural_loss import TotalRowAnchorLoss


def build_optimizer(model: nn.Module, config: TrainingConfig) -> AdamW:
    """Build AdamW optimizer strictly matching research specifications."""
    return AdamW(
        model.parameters(),
        lr=config.initial_lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )


def build_scheduler(
    optimizer: AdamW,
    config: TrainingConfig,
    total_epochs: int = 50,
) -> CosineAnnealingLR:
    """Build Cosine Annealing learning rate scheduler."""
    return CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=config.final_lr,
    )


def train_one_step(
    model: nn.Module,
    criterion: TotalRowAnchorLoss,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """
    Execute a single training optimization step.
    
    Args:
        model: RowAnchorNet.
        criterion: TotalRowAnchorLoss.
        optimizer: AdamW optimizer.
        images: Input batch tensor (B, 3, 288, 384).
        targets: Target class indices (B, 18).
        device: Torch device.
        
    Returns:
        Dict of scalar loss values.
    """
    model.train()
    images = images.to(device)
    targets = targets.to(device)

    optimizer.zero_grad()
    logits = model(images)  # (B, 18, 101)
    loss_dict = criterion(logits, targets)

    loss_dict["loss_total"].backward()
    optimizer.step()

    return {k: float(v.item()) for k, v in loss_dict.items()}
