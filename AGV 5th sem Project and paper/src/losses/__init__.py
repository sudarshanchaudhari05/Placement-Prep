"""Losses module for row-anchor training."""

from src.losses.structural_loss import (
    FocalLoss,
    SmoothnessLoss,
    CurvatureLoss,
    TotalRowAnchorLoss,
)

__all__ = [
    "FocalLoss",
    "SmoothnessLoss",
    "CurvatureLoss",
    "TotalRowAnchorLoss",
]
