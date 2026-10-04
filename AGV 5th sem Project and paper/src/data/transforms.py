"""
Image Preprocessing and Transformation Pipeline
===============================================
Operations:
1. ROI Cropping: Extracts y in [200, 470], x in [0, 640] from 640x480 input frame.
2. Resizing: Scales cropped ROI to (height=288, width=384).
3. SICT Glare Filtering (configurable):
   - If enabled: applies Specular-Invariant Chromaticity Transform.
   - If disabled: passes raw RGB (enabling baseline ablation studies).
4. Tensor Conversion & Normalization: Produces (3, 288, 384) float32 tensor.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
import torch

from configs.config import ProjectConfig, ROIConfig, ModelConfig, SICTConfig
from src.sict.sict_filter import sict_transform_np


class PreprocessingPipeline:
    """
    Standard preprocessing pipeline conforming to frozen research specification.
    """

    def __init__(
        self,
        roi_config: Optional[ROIConfig] = None,
        model_config: Optional[ModelConfig] = None,
        sict_config: Optional[SICTConfig] = None,
        apply_sict: bool = True,
        normalize: bool = True,
    ):
        self.roi = roi_config or ROIConfig()
        self.model = model_config or ModelConfig()
        self.sict = sict_config or SICTConfig()
        self.apply_sict = apply_sict and self.sict.enabled
        self.normalize = normalize

        # Standard ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def crop_roi(self, img: np.ndarray) -> np.ndarray:
        """
        Crop original 640x480 frame to ROI [y_min:y_max, x_min:x_max].
        """
        assert img.ndim == 3 and img.shape[2] == 3, f"Expected (H, W, 3) image, got {img.shape}"
        h, w, _ = img.shape
        assert h >= self.roi.y_max and w >= self.roi.x_max, (
            f"Image dimensions ({h}, {w}) smaller than ROI ({self.roi.y_max}, {self.roi.x_max})"
        )
        return img[self.roi.y_min : self.roi.y_max, self.roi.x_min : self.roi.x_max]

    def resize(self, img_roi: np.ndarray) -> np.ndarray:
        """
        Resize cropped ROI to neural network input resolution (W=384, H=288).
        """
        target_w = self.model.input_width   # 384
        target_h = self.model.input_height  # 288
        return cv2.resize(img_roi, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def __call__(self, img_bgr_or_rgb: np.ndarray, is_bgr: bool = True) -> torch.Tensor:
        """
        Run full preprocessing pipeline.
        
        Args:
            img_bgr_or_rgb: Original input frame (H=480, W=640, C=3).
            is_bgr: True if image is in OpenCV BGR format.
            
        Returns:
            torch.Tensor of shape (3, 288, 384), float32.
        """
        if is_bgr:
            img_rgb = cv2.cvtColor(img_bgr_or_rgb, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_bgr_or_rgb

        # FROZEN PIPELINE ORDER (Option A):
        # 640x480 RGB -> SICT Pre-filter -> ROI Crop [200:470, 0:640] -> Resize [288, 384] -> Tensor
        
        # 1. SICT Pre-filtering on full 640x480 frame (or raw RGB for ablation baseline)
        if self.apply_sict:
            # Operates on full incoming optical stream
            filtered_full = sict_transform_np(img_rgb, config=self.sict, out_channels=3)
        else:
            filtered_full = img_rgb.astype(np.float32) / 255.0

        # 2. ROI Crop: (480, 640, 3) -> (270, 640, 3)
        cropped = self.crop_roi(filtered_full)

        # 3. Resize: (270, 640, 3) -> (288, 384, 3)
        processed = self.resize(cropped)

        # 4. Optional normalization
        if self.normalize:
            processed = (processed - self.mean) / self.std

        # 5. HWC -> CHW tensor conversion: (288, 384, 3) -> (3, 288, 384)
        tensor = torch.from_numpy(processed.transpose(2, 0, 1)).float()

        assert tensor.shape == (3, self.model.input_height, self.model.input_width), (
            f"Preprocessed tensor shape mismatch: expected (3, {self.model.input_height}, "
            f"{self.model.input_width}), got {tensor.shape}"
        )
        return tensor
