"""
ShuffleNetV2-0.5x Backbone Feature Extractor
============================================
Extracts compact visual features using torchvision's pretrained ShuffleNetV2-0.5x.
The original 1000-class ImageNet classification head is removed.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ShuffleNet_V2_X0_5_Weights


class ShuffleNetV2Backbone(nn.Module):
    """
    ShuffleNetV2-0.5x feature extraction backbone.
    Outputs a 1024-dimensional feature vector per sample.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ShuffleNet_V2_X0_5_Weights.DEFAULT if pretrained else None
        base_model = models.shufflenet_v2_x0_5(weights=weights)

        # Retain feature extraction stages
        self.conv1 = base_model.conv1
        self.maxpool = base_model.maxpool
        self.stage2 = base_model.stage2
        self.stage3 = base_model.stage3
        self.stage4 = base_model.stage4
        self.conv5 = base_model.conv5

        # Feature dimension after conv5 is 1024
        self.out_features = 1024

        # Adaptive global average pool
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input tensor.
        Args:
            x: Tensor of shape (B, 3, H, W)
        Returns:
            Tensor of shape (B, 1024)
        """
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x
