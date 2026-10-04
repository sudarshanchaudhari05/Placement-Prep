"""
Structural Loss Module for Row-Anchor Network
=============================================
Computes:
1. Focal Loss (L_focal) with gamma = 2.0 for handling class imbalance (spatial bins vs absence).
2. First-order Smoothness Loss (L_smooth) enforcing spatial continuity along path.
3. Second-order Curvature Loss (L_curv) penalizing sharp steering transitions.

Formulation (Section 2.2):
    L_total = L_focal + lambda_smooth * L_smooth + lambda_curv * L_curv
    lambda_smooth = 0.5
    lambda_curv = 0.75
    focal_gamma = 2.0
"""

from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import LossConfig, ModelConfig


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.
    FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits of shape (N, C) or (B, M, C).
            targets: Ground-truth class indices of shape (N,) or (B, M).
        Returns:
            Scalar focal loss.
        """
        # Flatten to (N, C) and (N,)
        if logits.ndim > 2:
            num_classes = logits.shape[-1]
            logits = logits.reshape(-1, num_classes)
            targets = targets.reshape(-1)

        # Log-softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)
        # Gather log prob of target class
        log_pt = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        pt = torch.exp(log_pt)

        focal_weight = torch.pow(1.0 - pt, self.gamma)
        loss = -focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SmoothnessLoss(nn.Module):
    """
    First-order spatial continuity loss:
    L_smooth = sum_{i=1}^{M-1} |w_hat_{i+1} - w_hat_i|_1
    """

    def __init__(self):
        super().__init__()

    def forward(self, w_hat: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            w_hat: Soft-argmax coordinates of shape (B, M).
            mask: Optional boolean presence mask of shape (B, M).
        Returns:
            Scalar smoothness loss.
        """
        diff = torch.abs(w_hat[:, 1:] - w_hat[:, :-1])  # (B, M-1)

        if mask is not None:
            # Pairwise mask: both rows must be present
            pair_mask = mask[:, 1:] & mask[:, :-1]
            if pair_mask.sum() > 0:
                diff = diff * pair_mask.float()
                return diff.sum() / (pair_mask.sum().float() + 1e-6)
            else:
                return (diff * 0.0).sum()

        return diff.mean()


class CurvatureLoss(nn.Module):
    """
    Second-order curvature continuity loss:
    L_curv = sum_{i=1}^{M-2} |(w_hat_{i+2} - w_hat_{i+1}) - (w_hat_{i+1} - w_hat_i)|_1
    """

    def __init__(self):
        super().__init__()

    def forward(self, w_hat: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            w_hat: Soft-argmax coordinates of shape (B, M).
            mask: Optional boolean presence mask of shape (B, M).
        Returns:
            Scalar curvature loss.
        """
        # Second derivative along row anchor sequence
        diff1 = w_hat[:, 1:] - w_hat[:, :-1]  # (B, M-1)
        diff2 = torch.abs(diff1[:, 1:] - diff1[:, :-1])  # (B, M-2)

        if mask is not None:
            # Triplet mask: all three rows must be present
            triplet_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
            if triplet_mask.sum() > 0:
                diff2 = diff2 * triplet_mask.float()
                return diff2.sum() / (triplet_mask.sum().float() + 1e-6)
            else:
                return (diff2 * 0.0).sum()

        return diff2.mean()


class TotalRowAnchorLoss(nn.Module):
    """
    Total structural row-anchor training loss:
        L_total = L_focal + lambda_smooth * L_smooth + lambda_curv * L_curv
    """

    def __init__(
        self,
        loss_config: Optional[LossConfig] = None,
        model_config: Optional[ModelConfig] = None,
    ):
        super().__init__()
        self.loss_cfg = loss_config or LossConfig()
        self.model_cfg = model_config or ModelConfig()

        self.focal_loss = FocalLoss(gamma=self.loss_cfg.focal_gamma)
        self.smoothness_loss = SmoothnessLoss()
        self.curvature_loss = CurvatureLoss()

        self.num_spatial_bins = self.model_cfg.num_spatial_bins

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss and component losses.
        
        Args:
            logits: Raw network predictions (B, M, K + 1), e.g. (B, 18, 101).
            targets: Ground-truth class labels (B, M) with values in [0, 100].
            
        Returns:
            Dictionary containing:
                'loss_total': Weighted sum of all losses
                'loss_focal': Classification focal loss
                'loss_smooth': First-order smoothness loss
                'loss_curv': Second-order curvature loss
        """
        b, m, total_c = logits.shape
        assert total_c == self.model_cfg.total_classes, (
            f"Expected {self.model_cfg.total_classes} classes, got {total_c}"
        )
        assert targets.shape == (b, m), (
            f"Target shape mismatch. Expected ({b}, {m}), got {targets.shape}"
        )

        # 1. Focal loss over all row classifications
        l_focal = self.focal_loss(logits, targets)

        # 2. Derive soft-argmax coordinates for structural regularization
        # Softmax over all K+1 classes
        probs = torch.softmax(logits, dim=-1)
        spatial_probs = probs[:, :, :self.num_spatial_bins]  # (B, M, K)
        bin_indices = torch.arange(
            self.num_spatial_bins, dtype=torch.float32, device=logits.device
        ).view(1, 1, self.num_spatial_bins)
        w_hat = torch.sum(spatial_probs * bin_indices, dim=-1)  # (B, M)

        # Presence mask based on ground truth (exclude absence class 100)
        presence_mask = (targets != self.num_spatial_bins)  # (B, M)

        # 3. Structural continuity losses
        l_smooth = self.smoothness_loss(w_hat, mask=presence_mask)
        l_curv = self.curvature_loss(w_hat, mask=presence_mask)

        # 4. Total weighted loss
        l_total = (
            l_focal
            + self.loss_cfg.lambda_smooth * l_smooth
            + self.loss_cfg.lambda_curv * l_curv
        )

        return {
            "loss_total": l_total,
            "loss_focal": l_focal,
            "loss_smooth": l_smooth,
            "loss_curv": l_curv,
        }
