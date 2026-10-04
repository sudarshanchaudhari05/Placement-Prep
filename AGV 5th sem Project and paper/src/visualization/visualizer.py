"""
Visualization Utilities for Path Tracking Perception
=====================================================
Renders:
1. Row-anchor ground truth and prediction overlays on original 640x480 frame.
2. Comparative plots of raw RGB vs SICT pre-filtered representations.
"""

from typing import Optional, List, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt

from configs.config import ProjectConfig
from src.labels.coder import get_anchor_row_y_coords


def draw_row_anchors(
    img_bgr: np.ndarray,
    pred_coords: Optional[np.ndarray] = None,
    pred_present: Optional[np.ndarray] = None,
    gt_coords: Optional[np.ndarray] = None,
    gt_present: Optional[np.ndarray] = None,
    config: Optional[ProjectConfig] = None,
) -> np.ndarray:
    """
    Overlay row-anchor predictions and ground-truth onto frame.
    
    Args:
        img_bgr: Base image (640x480) in BGR.
        pred_coords: Predicted u-coords (18,).
        pred_present: Predicted presence mask (18,).
        gt_coords: Ground truth u-coords (18,).
        gt_present: Ground truth presence mask (18,).
        config: Project configuration.
        
    Returns:
        Annotated BGR image.
    """
    if config is None:
        config = ProjectConfig()

    out = img_bgr.copy()
    row_y = get_anchor_row_y_coords(
        config.model.num_row_anchors,
        config.roi.y_min,
        config.roi.y_max,
    )

    # Draw ROI boundaries
    cv2.line(out, (0, config.roi.y_min), (config.camera.input_width, config.roi.y_min), (255, 0, 0), 1)
    cv2.line(out, (0, config.roi.y_max), (config.camera.input_width, config.roi.y_max), (255, 0, 0), 1)

    # Draw horizontal anchor guideline dashes
    for y in row_y:
        cv2.line(out, (0, int(y)), (config.camera.input_width, int(y)), (50, 50, 50), 1)

    # Draw Ground Truth in GREEN
    if gt_coords is not None and gt_present is not None:
        gt_pts = []
        for i, y in enumerate(row_y):
            if gt_present[i] and not np.isnan(gt_coords[i]) and gt_coords[i] >= 0:
                pt = (int(gt_coords[i]), int(y))
                gt_pts.append(pt)
                cv2.circle(out, pt, 5, (0, 255, 0), -1)
        if len(gt_pts) > 1:
            for k in range(len(gt_pts) - 1):
                cv2.line(out, gt_pts[k], gt_pts[k + 1], (0, 255, 0), 2)

    # Draw Prediction in RED
    if pred_coords is not None and pred_present is not None:
        pred_pts = []
        for i, y in enumerate(row_y):
            if pred_present[i] and not np.isnan(pred_coords[i]) and pred_coords[i] >= 0:
                pt = (int(pred_coords[i]), int(y))
                pred_pts.append(pt)
                cv2.circle(out, pt, 4, (0, 0, 255), -1)
        if len(pred_pts) > 1:
            for k in range(len(pred_pts) - 1):
                cv2.line(out, pred_pts[k], pred_pts[k + 1], (0, 0, 255), 2)

    return out


def plot_sict_comparison(
    raw_rgb: np.ndarray,
    sict_img: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot side-by-side comparison of raw RGB image and SICT pre-filtered image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(raw_rgb)
    axes[0].set_title("Input Raw RGB (with Glare)")
    axes[0].axis("off")

    if sict_img.ndim == 3 and sict_img.shape[2] == 3:
        axes[1].imshow(sict_img)
    else:
        axes[1].imshow(sict_img.squeeze(), cmap="viridis")
    axes[1].set_title("SICT Filtered (Specular-Suppressed)")
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
