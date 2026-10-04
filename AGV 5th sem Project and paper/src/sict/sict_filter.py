"""
Specular-Invariant Chromaticity Transform (SICT) Module
======================================================

Mathematical Formulation:
Given an RGB pixel vector I(u,v) = [R, G, B]^T with total intensity:
    S(u,v) = R + G + B

Normalized chromaticity coordinates:
    r = R / (S + epsilon)
    g = G / (S + epsilon)
    b = B / (S + epsilon)
    (epsilon = 1e-5)

Specular chromatic distance to equal-energy illuminant (1/3, 1/3):
    Phi(u,v) = sqrt((r(u,v) - 1/3)^2 + (g(u,v) - 1/3)^2)

Conditional filtering:
    I_filtered(u,v) =
        I_ambient                           if S(u,v) >= Gamma_lum and Phi(u,v) <= delta_spec
        alpha * r(u,v) + beta * g(u,v)     otherwise

Parameters:
    Gamma_lum = 220.0
    delta_spec = 0.05
    alpha = 0.6
    beta = 0.4
    I_ambient = 0.2 * frame-average intensity
"""

from typing import Union, Optional
import numpy as np
import torch
import torch.nn as nn

from configs.config import SICTConfig


def sict_transform_np(
    img_rgb: np.ndarray,
    config: Optional[SICTConfig] = None,
    out_channels: int = 3,
) -> np.ndarray:
    """
    Apply SICT to an RGB NumPy image (H, W, 3).
    
    Args:
        img_rgb: Input image of shape (H, W, 3) in RGB order.
                 Accepts uint8 [0, 255] or float32 [0.0, 255.0] or [0.0, 1.0].
        config: SICTConfig dataclass containing research parameters.
        out_channels: Number of output channels (1 or 3).
                      If 3, the single-channel map is replicated across 3 channels.
                      
    Returns:
        np.ndarray of shape (H, W, out_channels), float32 in [0, 1].
    """
    if config is None:
        config = SICTConfig()

    assert img_rgb.ndim == 3 and img_rgb.shape[2] == 3, (
        f"Expected image with shape (H, W, 3), got {img_rgb.shape}"
    )

    img = img_rgb.astype(np.float32)
    # If image is normalized to [0, 1], scale to [0, 255] for intensity threshold check
    if img.max() <= 1.0 and img.max() > 0.0:
        img_255 = img * 255.0
    else:
        img_255 = img

    R = img_255[:, :, 0]
    G = img_255[:, :, 1]
    B = img_255[:, :, 2]

    # Total intensity: S = R + G + B
    S = R + G + B

    # Normalized chromaticities
    denom = S + config.epsilon
    r = R / denom
    g = G / denom

    # Specular chromatic distance Phi = sqrt((r - 1/3)^2 + (g - 1/3)^2)
    diff_r = r - (1.0 / 3.0)
    diff_g = g - (1.0 / 3.0)
    Phi = np.sqrt(diff_r * diff_r + diff_g * diff_g)

    # Base chromaticity response: alpha * r + beta * g
    base_response = config.alpha * r + config.beta * g

    # Frame-average raw RGB intensity:
    # Strictly conforming to frozen specification: I_ambient = 0.2 * frame_average_intensity
    # where total intensity is S(u,v) = R + G + B, and normalized raw pixel intensity is S / 765.0
    if config.frame_avg_mode in ("raw_intensity", "intensity_norm"):
        frame_avg = float(np.mean(S / 765.0))
    elif config.frame_avg_mode == "chromatic":
        # Legacy/ablation mode only
        frame_avg = float(np.mean(base_response))
    else:
        # Default to frozen specification: raw RGB intensity
        frame_avg = float(np.mean(S / 765.0))

    i_ambient = config.ambient_ratio * frame_avg

    # Specular condition: S >= Gamma_lum and Phi <= delta_spec
    specular_mask = (S >= config.gamma_lum) & (Phi <= config.delta_spec)

    # Conditional filtering
    filtered = np.where(specular_mask, i_ambient, base_response).astype(np.float32)

    # Clip to valid [0, 1] range
    filtered = np.clip(filtered, 0.0, 1.0)

    if out_channels == 3:
        return np.repeat(filtered[:, :, np.newaxis], 3, axis=2)
    elif out_channels == 1:
        return filtered[:, :, np.newaxis]
    else:
        raise ValueError(f"Unsupported out_channels: {out_channels}. Must be 1 or 3.")


def sict_transform_torch(
    tensor_rgb: torch.Tensor,
    config: Optional[SICTConfig] = None,
    out_channels: int = 3,
) -> torch.Tensor:
    """
    Differentiable / GPU-compatible PyTorch implementation of SICT.
    
    Args:
        tensor_rgb: Tensor of shape (B, 3, H, W) or (3, H, W) in RGB channel order.
                    Values in [0, 255] or [0, 1].
        config: SICTConfig dataclass.
        out_channels: Output channels (1 or 3).
        
    Returns:
        torch.Tensor of shape (B, out_channels, H, W) or (out_channels, H, W).
    """
    if config is None:
        config = SICTConfig()

    has_batch = tensor_rgb.ndim == 4
    if not has_batch:
        assert tensor_rgb.ndim == 3 and tensor_rgb.shape[0] == 3, (
            f"Expected tensor shape (3, H, W), got {tensor_rgb.shape}"
        )
        tensor_rgb = tensor_rgb.unsqueeze(0)
    else:
        assert tensor_rgb.shape[1] == 3, (
            f"Expected tensor shape (B, 3, H, W), got {tensor_rgb.shape}"
        )

    t = tensor_rgb.float()
    if t.max() <= 1.0 and t.max() > 0.0:
        t_255 = t * 255.0
    else:
        t_255 = t

    R = t_255[:, 0:1, :, :]
    G = t_255[:, 1:2, :, :]
    B = t_255[:, 2:3, :, :]

    S = R + G + B
    denom = S + config.epsilon
    r = R / denom
    g = G / denom

    diff_r = r - (1.0 / 3.0)
    diff_g = g - (1.0 / 3.0)
    Phi = torch.sqrt(diff_r * diff_r + diff_g * diff_g)

    base_response = config.alpha * r + config.beta * g

    # Frame-average raw RGB intensity:
    if config.frame_avg_mode in ("raw_intensity", "intensity_norm"):
        frame_avg = (S / 765.0).mean(dim=[-2, -1], keepdim=True)
    elif config.frame_avg_mode == "chromatic":
        frame_avg = base_response.mean(dim=[-2, -1], keepdim=True)
    else:
        frame_avg = (S / 765.0).mean(dim=[-2, -1], keepdim=True)

    i_ambient = config.ambient_ratio * frame_avg

    specular_mask = (S >= config.gamma_lum) & (Phi <= config.delta_spec)
    filtered = torch.where(specular_mask, i_ambient, base_response)
    filtered = torch.clamp(filtered, 0.0, 1.0)

    if out_channels == 3:
        out = filtered.repeat(1, 3, 1, 1)
    elif out_channels == 1:
        out = filtered
    else:
        raise ValueError(f"Unsupported out_channels: {out_channels}")

    if not has_batch:
        out = out.squeeze(0)
    return out


class SICTModule(nn.Module):
    """PyTorch Module wrapper for SICT transform."""

    def __init__(self, config: Optional[SICTConfig] = None, out_channels: int = 3):
        super().__init__()
        self.config = config or SICTConfig()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sict_transform_torch(x, config=self.config, out_channels=self.out_channels)
