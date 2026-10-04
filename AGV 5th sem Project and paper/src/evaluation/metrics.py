"""
Evaluation Metrics for Row-Anchor AGV Path Tracking
===================================================
Evaluates path detection performance:
- F1 Score with 8-pixel horizontal tolerance
- Precision & Recall
- Coordinate Root-Mean-Square Error (RMSE) on True Positives
- Anchor Row Presence Accuracy
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch

from configs.config import ProjectConfig
from src.labels.coder import decode_row_anchor_predictions


def compute_row_anchor_metrics(
    pred_coords: np.ndarray,
    pred_present: np.ndarray,
    gt_coords: np.ndarray,
    gt_present: np.ndarray,
    coord_tolerance_px: float = 8.0,
) -> Dict[str, float]:
    """
    Compute rigorous path tracking metrics.
    
    Args:
        pred_coords: Predicted u-coordinates (N, M), float32.
        pred_present: Predicted presence boolean mask (N, M).
        gt_coords: Ground-truth u-coordinates (N, M), float32 (or -1/NaN for absent).
        gt_present: Ground-truth presence boolean mask (N, M).
        coord_tolerance_px: Coordinate error tolerance for True Positive (default 8.0 px).
        
    Returns:
        Dict containing:
            'precision', 'recall', 'f1_score', 'rmse_tp_px', 'presence_accuracy'
    """
    assert pred_coords.shape == gt_coords.shape, (
        f"Shape mismatch: preds {pred_coords.shape} vs gt {gt_coords.shape}"
    )

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    sq_errors: List[float] = []

    n_samples, n_rows = pred_coords.shape

    for i in range(n_samples):
        for r in range(n_rows):
            p_pres = bool(pred_present[i, r])
            g_pres = bool(gt_present[i, r])

            if p_pres and g_pres:
                err = abs(float(pred_coords[i, r]) - float(gt_coords[i, r]))
                if err <= coord_tolerance_px:
                    total_tp += 1
                    sq_errors.append(err ** 2)
                else:
                    # Predicted present but exceeds coordinate tolerance
                    total_fp += 1
                    total_fn += 1
            elif p_pres and not g_pres:
                total_fp += 1
            elif not p_pres and g_pres:
                total_fn += 1
            else:
                total_tn += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2.0 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    rmse = float(np.sqrt(np.mean(sq_errors))) if len(sq_errors) > 0 else 0.0
    total_slots = n_samples * n_rows
    presence_acc = (
        (total_tp + total_tn) / total_slots if total_slots > 0 else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "rmse_tp_px": float(rmse),
        "presence_accuracy": float(presence_acc),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
    }


class RowAnchorEvaluator:
    """Accumulator for batch-by-batch evaluation during validation."""

    def __init__(self, config: Optional[ProjectConfig] = None):
        self.config = config or ProjectConfig()
        self.tolerance = self.config.dataset.eval_tolerance_px
        self.reset()

    def reset(self) -> None:
        self.all_pred_coords: List[np.ndarray] = []
        self.all_pred_present: List[np.ndarray] = []
        self.all_gt_coords: List[np.ndarray] = []
        self.all_gt_present: List[np.ndarray] = []

    def update(
        self,
        logits: torch.Tensor,
        gt_coords: torch.Tensor,
        gt_present: torch.Tensor,
    ) -> None:
        """Update metrics with a batch."""
        pred_coords, is_present, _ = decode_row_anchor_predictions(
            logits,
            image_width=self.config.camera.input_width,
            num_spatial_bins=self.config.model.num_spatial_bins,
        )

        self.all_pred_coords.append(pred_coords.detach().cpu().numpy())
        self.all_pred_present.append(is_present.detach().cpu().numpy())
        self.all_gt_coords.append(gt_coords.detach().cpu().numpy())
        self.all_gt_present.append(gt_present.detach().cpu().numpy())

    def compute(self) -> Dict[str, float]:
        if not self.all_pred_coords:
            return {"f1_score": 0.0, "precision": 0.0, "recall": 0.0, "rmse_tp_px": 0.0}

        pred_coords = np.concatenate(self.all_pred_coords, axis=0)
        pred_present = np.concatenate(self.all_pred_present, axis=0)
        gt_coords = np.concatenate(self.all_gt_coords, axis=0)
        gt_present = np.concatenate(self.all_gt_present, axis=0)

        return compute_row_anchor_metrics(
            pred_coords,
            pred_present,
            gt_coords,
            gt_present,
            coord_tolerance_px=self.tolerance,
        )
