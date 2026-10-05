"""
Phase 4B.2 Decoder Ablation Script (Validation Set Only)
======================================================
Evaluates candidate row-anchor decoding formulations on the 900-image VALIDATION SET
using the untouched checkpoint:
experiments/sict_shufflenet_baseline_50ep/checkpoints/best_model.pth

Ablation Methods:
A. Current Global Soft-Argmax (README Section 4.3 formulation)
B. Pure Argmax over Spatial Bins (0..99)
C. Local/Windowed Soft-Argmax (radius R in {1, 2, 3})
D. Temperature-Scaled Soft-Argmax (T in {0.25, 0.5, 1.0, 2.0, 4.0}):
   - D1: Renormalized over spatial bins 0..99
   - D2: Unnormalized slice of 101-class softmax
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import load_config
from src.data.dataset import RowAnchorDataset, build_dataloader
from src.models.row_anchor_net import build_model
from src.evaluation.metrics import compute_row_anchor_metrics


def extract_validation_logits(
    ckpt_path: Path,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run one forward pass over the 900 validation images and return raw tensors.
    """
    cfg = load_config()
    val_ds = RowAnchorDataset("data/synthetic/val/annotations.json", config=cfg)
    val_loader = build_dataloader(val_ds, batch_size=32, shuffle=False)

    model = build_model(cfg, apply_sict_in_forward=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    all_logits = []
    all_gt_coords = []
    all_gt_present = []
    all_gt_targets = []

    print(f"Extracting validation logits using {device}...")
    t0 = time.time()
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            logits = model(images)  # (B, 18, 101)
            all_logits.append(logits.cpu())
            all_gt_coords.append(batch["u_coords"])
            all_gt_present.append(batch["presence"])
            all_gt_targets.append(batch["targets"])
    print(f"Logits extracted in {time.time() - t0:.2f}s.")

    logits_tensor = torch.cat(all_logits, dim=0)       # (900, 18, 101)
    gt_coords_tensor = torch.cat(all_gt_coords, dim=0)   # (900, 18)
    gt_present_tensor = torch.cat(all_gt_present, dim=0) # (900, 18)
    gt_targets_tensor = torch.cat(all_gt_targets, dim=0) # (900, 18)

    return logits_tensor, gt_coords_tensor, gt_present_tensor, gt_targets_tensor


# =========================================================================
# DECODER IMPLEMENTATIONS
# =========================================================================

def decode_method_A_current_global_soft_argmax(
    logits: torch.Tensor,
    image_width: float = 640.0,
    num_bins: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method A: Current global soft-argmax per README Section 4.3.
    w_hat = sum_{j=0}^{K-1} j * softmax_101(logits)_j
    u_hat = ((w_hat + 0.5) / K) * W
    """
    probs = F.softmax(logits, dim=-1)  # (N, M, 101)
    argmax_classes = torch.argmax(probs, dim=-1)  # (N, M)
    is_present = (argmax_classes != num_bins)  # class 100 is absence

    spatial_probs = probs[:, :, :num_bins]  # (N, M, 100)
    bin_indices = torch.arange(num_bins, dtype=torch.float32).view(1, 1, num_bins)
    w_hat = torch.sum(spatial_probs * bin_indices, dim=-1)  # (N, M)
    pred_coords = ((w_hat + 0.5) / float(num_bins)) * image_width

    return pred_coords.numpy(), is_present.numpy()


def decode_method_B_pure_argmax(
    logits: torch.Tensor,
    image_width: float = 640.0,
    num_bins: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method B: Pure argmax over spatial bins 0..99, absence = class 100 is argmax.
    """
    argmax_all = torch.argmax(logits, dim=-1)  # (N, M)
    is_present = (argmax_all != num_bins)

    spatial_logits = logits[:, :, :num_bins]
    best_bin = torch.argmax(spatial_logits, dim=-1).float()  # (N, M)
    pred_coords = ((best_bin + 0.5) / float(num_bins)) * image_width

    return pred_coords.numpy(), is_present.numpy()


def decode_method_C_local_windowed_soft_argmax(
    logits: torch.Tensor,
    radius: int = 2,
    image_width: float = 640.0,
    num_bins: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method C: Local/windowed soft-argmax around peak spatial bin.
    1. Find peak spatial bin j* = argmax_{0..K-1}(logits).
    2. Local window: [max(0, j* - radius), min(K-1, j* + radius)].
    3. Renormalize probabilities strictly over the local window.
    4. Compute expectation within that window.
    """
    argmax_all = torch.argmax(logits, dim=-1)
    is_present = (argmax_all != num_bins)

    spatial_logits = logits[:, :, :num_bins]  # (N, M, 100)
    best_bin = torch.argmax(spatial_logits, dim=-1)  # (N, M)

    n_samples, n_rows, _ = logits.shape
    w_hat = torch.zeros((n_samples, n_rows), dtype=torch.float32)

    for i in range(n_samples):
        for r in range(n_rows):
            j_star = best_bin[i, r].item()
            j_min = max(0, j_star - radius)
            j_max = min(num_bins - 1, j_star + radius)

            win_logits = spatial_logits[i, r, j_min : j_max + 1]
            win_probs = F.softmax(win_logits, dim=-1)
            win_indices = torch.arange(j_min, j_max + 1, dtype=torch.float32)
            w_hat[i, r] = torch.sum(win_probs * win_indices)

    pred_coords = ((w_hat + 0.5) / float(num_bins)) * image_width
    return pred_coords.numpy(), is_present.numpy()


def decode_method_D1_temperature_spatial_soft_argmax(
    logits: torch.Tensor,
    temperature: float = 1.0,
    image_width: float = 640.0,
    num_bins: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method D1: Temperature-scaled soft-argmax normalized over the 100 spatial bins.
    probs_spatial = softmax(logits_{0..99} / T)
    w_hat = sum_{j=0}^{99} j * probs_spatial_j
    Presence rule: argmax over all 101 classes != 100.
    """
    argmax_all = torch.argmax(logits, dim=-1)
    is_present = (argmax_all != num_bins)

    spatial_logits = logits[:, :, :num_bins] / temperature
    spatial_probs = F.softmax(spatial_logits, dim=-1)
    bin_indices = torch.arange(num_bins, dtype=torch.float32).view(1, 1, num_bins)
    w_hat = torch.sum(spatial_probs * bin_indices, dim=-1)
    pred_coords = ((w_hat + 0.5) / float(num_bins)) * image_width

    return pred_coords.numpy(), is_present.numpy()


def decode_method_D2_temperature_101_soft_argmax(
    logits: torch.Tensor,
    temperature: float = 1.0,
    image_width: float = 640.0,
    num_bins: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method D2: Temperature-scaled slice of 101-class softmax (README formula with temperature).
    probs_101 = softmax(logits / T)
    w_hat = sum_{j=0}^{99} j * probs_101_j
    Presence rule: argmax over all 101 classes != 100.
    """
    probs_101 = F.softmax(logits / temperature, dim=-1)
    argmax_all = torch.argmax(probs_101, dim=-1)
    is_present = (argmax_all != num_bins)

    spatial_probs = probs_101[:, :, :num_bins]
    bin_indices = torch.arange(num_bins, dtype=torch.float32).view(1, 1, num_bins)
    w_hat = torch.sum(spatial_probs * bin_indices, dim=-1)
    pred_coords = ((w_hat + 0.5) / float(num_bins)) * image_width

    return pred_coords.numpy(), is_present.numpy()


# =========================================================================
# EVALUATION & METRIC CALCULATION
# =========================================================================

def evaluate_method(
    name: str,
    pred_coords: np.ndarray,
    pred_present: np.ndarray,
    gt_coords: np.ndarray,
    gt_present: np.ndarray,
    tolerance: float = 8.0,
) -> Dict[str, Any]:
    metrics = compute_row_anchor_metrics(
        pred_coords=pred_coords,
        pred_present=pred_present,
        gt_coords=gt_coords,
        gt_present=gt_present,
        coord_tolerance_px=tolerance,
    )

    # Compute coordinate statistics on valid ground truth vs predicted
    gt_valid_coords = gt_coords[gt_present]
    pred_valid_coords = pred_coords[pred_present]

    # Coordinate stats on rows where BOTH are present
    both_present = pred_present & gt_present
    if both_present.sum() > 0:
        coord_diff = np.abs(pred_coords[both_present] - gt_coords[both_present])
        mean_abs_err = float(np.mean(coord_diff))
        median_abs_err = float(np.median(coord_diff))
        within_8px_pct = float((coord_diff <= 8.0).mean() * 100)
    else:
        mean_abs_err = 0.0
        median_abs_err = 0.0
        within_8px_pct = 0.0

    return {
        "method_name": name,
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "rmse_tp_px": metrics["rmse_tp_px"],
        "pred_coord_mean": float(np.mean(pred_valid_coords)) if len(pred_valid_coords) > 0 else 0.0,
        "pred_coord_std": float(np.std(pred_valid_coords)) if len(pred_valid_coords) > 0 else 0.0,
        "gt_coord_mean": float(np.mean(gt_valid_coords)),
        "gt_coord_std": float(np.std(gt_valid_coords)),
        "num_pred_present": int(np.sum(pred_present)),
        "num_gt_present": int(np.sum(gt_present)),
        "both_present_count": int(np.sum(both_present)),
        "mean_abs_err": mean_abs_err,
        "median_abs_err": median_abs_err,
        "within_8px_pct": within_8px_pct,
    }


def main():
    print("=" * 80)
    print("PHASE 4B.2: VALIDATION-ONLY ROW-ANCHOR DECODER ABLATION AUDIT")
    print("=" * 80)

    ckpt_path = PROJECT_ROOT / "experiments" / "sict_shufflenet_baseline_50ep" / "checkpoints" / "best_model.pth"
    assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits, gt_coords, gt_present, gt_targets = extract_validation_logits(ckpt_path, device)

    gt_c_np = gt_coords.numpy()
    gt_p_np = gt_present.numpy()

    results = []

    # 1. Method A: Current global soft-argmax
    p_coords_a, p_pres_a = decode_method_A_current_global_soft_argmax(logits)
    res_a = evaluate_method("A: Global Soft-Argmax (Current Baseline)", p_coords_a, p_pres_a, gt_c_np, gt_p_np)
    results.append(res_a)

    # 2. Method B: Pure argmax
    p_coords_b, p_pres_b = decode_method_B_pure_argmax(logits)
    res_b = evaluate_method("B: Pure Argmax (Spatial 0..99)", p_coords_b, p_pres_b, gt_c_np, gt_p_np)
    results.append(res_b)

    # 3. Method C: Local windowed soft-argmax (radius 1, 2, 3)
    for r in [1, 2, 3]:
        p_coords_c, p_pres_c = decode_method_C_local_windowed_soft_argmax(logits, radius=r)
        res_c = evaluate_method(f"C: Local Windowed Soft-Argmax (radius={r}, win={2*r+1})", p_coords_c, p_pres_c, gt_c_np, gt_p_np)
        results.append(res_c)

    # 4. Method D1: Temperature-scaled soft-argmax (Spatial 100 bins renormalized)
    for T in [0.25, 0.5, 1.0, 2.0, 4.0]:
        p_coords_d1, p_pres_d1 = decode_method_D1_temperature_spatial_soft_argmax(logits, temperature=T)
        res_d1 = evaluate_method(f"D1: Temp-Scaled Softmax (Spatial 100, T={T})", p_coords_d1, p_pres_d1, gt_c_np, gt_p_np)
        results.append(res_d1)

    # 5. Method D2: Temperature-scaled slice of 101-class softmax
    for T in [0.25, 0.5, 1.0, 2.0, 4.0]:
        p_coords_d2, p_pres_d2 = decode_method_D2_temperature_101_soft_argmax(logits, temperature=T)
        res_d2 = evaluate_method(f"D2: Temp-Scaled Softmax (101-Class Slice, T={T})", p_coords_d2, p_pres_d2, gt_c_np, gt_p_np)
        results.append(res_d2)

    # Output detailed comparative table
    print("\n" + "=" * 140)
    print(f"{'Method':<48} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'Prec':<7} | {'Recall':<7} | {'F1':<7} | {'RMSE':<7} | {'Pred Std':<9} | {'<=8px %':<8}")
    print("-" * 140)
    for r in results:
        print(
            f"{r['method_name']:<48} | "
            f"{r['tp']:<5} | {r['fp']:<5} | {r['fn']:<5} | "
            f"{r['precision']*100:<6.2f}% | {r['recall']*100:<6.2f}% | "
            f"{r['f1_score']:.4f} | {r['rmse_tp_px']:<6.2f}px | "
            f"{r['pred_coord_std']:<8.2f}px | {r['within_8px_pct']:<6.2f}%"
        )
    print("=" * 140)

    # Save structured audit results JSON
    out_json = PROJECT_ROOT / "experiments" / "decoder_ablation_validation_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved structured audit report to: {out_json}")


if __name__ == "__main__":
    main()
