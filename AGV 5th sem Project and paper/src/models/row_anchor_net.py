"""
Row-Anchor Classification Network
==================================
Ultra-lightweight structural line detection paradigm discretized over M horizontal anchor rows.
Backbone: ShuffleNetV2-0.5x (ImageNet pretrained, original 1000-class head removed).
Input: (batch, 3, 288, 384)
Output: (batch, 18, 101)
"""

from typing import Optional
import torch
import torch.nn as nn

from configs.config import ProjectConfig, ModelConfig, SICTConfig
from src.models.backbone import ShuffleNetV2Backbone
from src.sict.sict_filter import SICTModule


class RowAnchorNet(nn.Module):
    """
    Row-Anchor Network for AGV path tracking.
    
    Given an input image of shape (B, 3, 288, 384), predicts classification logits
    over M horizontal rows across (K + 1) classes:
      - K spatial bins (horizontal grid cells)
      - 1 absence / no-line class
      
    Output shape: [batch_size, num_row_anchors, num_spatial_bins + 1] -> [B, 18, 101].
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        sict_config: Optional[SICTConfig] = None,
        apply_sict_in_forward: bool = False,
    ):
        super().__init__()
        self.config = model_config or ModelConfig()
        self.apply_sict_in_forward = apply_sict_in_forward

        if self.apply_sict_in_forward:
            self.sict = SICTModule(config=sict_config or SICTConfig(), out_channels=3)
        else:
            self.sict = None

        self.backbone = ShuffleNetV2Backbone(pretrained=self.config.pretrained)

        # Classification head: 1024 -> M * (K + 1)
        in_dim = self.backbone.out_features
        out_dim = self.config.num_row_anchors * self.config.total_classes  # 18 * 101 = 1818

        self.head = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(in_dim, out_dim),
        )

        self._init_head()

    def _init_head(self) -> None:
        """Initialize classification head weights."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, 288, 384).
               Values should be normalized or in range expected by transform pipeline.
               
        Returns:
            Logits tensor of shape (B, 18, 101).
        """
        # Strict dimension and shape checks
        assert x.ndim == 4, (
            f"Expected 4D input tensor (B, C, H, W), got {x.ndim}D tensor with shape {x.shape}"
        )
        b, c, h, w = x.shape
        assert c == self.config.input_channels, (
            f"Expected {self.config.input_channels} input channels, got {c}"
        )
        assert h == self.config.input_height and w == self.config.input_width, (
            f"Expected spatial resolution ({self.config.input_height}, {self.config.input_width}), "
            f"got ({h}, {w})"
        )

        # Optional in-network SICT filtering
        if self.sict is not None:
            x = self.sict(x)

        # Extract backbone features: (B, 1024)
        features = self.backbone(x)

        # Predict logits: (B, M * (K + 1))
        logits_flat = self.head(features)

        # Reshape to (B, M, K + 1) -> (B, 18, 101)
        logits = logits_flat.view(
            b, self.config.num_row_anchors, self.config.total_classes
        )

        # Output shape assertion
        expected_shape = (b, self.config.num_row_anchors, self.config.total_classes)
        assert logits.shape == expected_shape, (
            f"Output shape mismatch! Expected {expected_shape}, got {logits.shape}"
        )

        return logits

    def count_parameters(self) -> dict:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "backbone": backbone_params,
            "head": head_params,
        }


def build_model(
    config: ProjectConfig,
    apply_sict_in_forward: bool = False,
) -> RowAnchorNet:
    """Build RowAnchorNet from project configuration."""
    model = RowAnchorNet(
        model_config=config.model,
        sict_config=config.sict,
        apply_sict_in_forward=apply_sict_in_forward,
    )
    return model
