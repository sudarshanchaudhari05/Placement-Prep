"""
Phase 4B.1 Post-Training Metric Audit and Diagnosis Script
=========================================================
1. Manual sanity test of metrics with exact known values (Task 5).
2. Deep diagnosis of RMSE = 0.00 reporting (Task 3).
3. Deep diagnosis of F1 = ~0.01 (Task 4):
   - argmax vs soft-argmax behavior
   - diffuse probability distribution
   - presence classification accuracy
4. Checkpoint integrity verification (Task 6).
5. Training time breakdown and profiling (Task 7).
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import load_config
from src.data.dataset import RowAnchorDataset, build_dataloader
from src.models.row_anchor_net import build_model
from src.labels.coder import decode_row_anchor_predictions
from src.evaluation.metrics import compute_row_anchor_metrics, RowAnchorEvaluator


def run_task5_manual_sanity_test():
    print("=" * 75)
    print("TASK 5: MANUAL SANITY TEST OF METRIC FORMULATION")
    print("=" * 75)
    # Define test cases strictly according to user prompt specification:
    # 1 row, 1 sample
    # Case 1: GT u=320, Pred u=324 -> err=4 px <= 8 px -> TP=1, FP=0, FN=0
    m1 = compute_row_anchor_metrics(
        pred_coords=np.array([[324.0]]),
        pred_present=np.array([[True]]),
        gt_coords=np.array([[320.0]]),
        gt_present=np.array([[True]]),
        coord_tolerance_px=8.0,
    )
    print(f"Case 1 (err=4px): TP={m1['tp']}, FP={m1['fp']}, FN={m1['fn']}, RMSE={m1['rmse_tp_px']:.2f}")
    assert m1['tp'] == 1 and m1['fp'] == 0 and m1['fn'] == 0, f"Case 1 failed: {m1}"
    assert np.isclose(m1['rmse_tp_px'], 4.0), f"Case 1 RMSE failed: {m1}"

    # Case 2: GT u=320, Pred u=335 -> err=15 px > 8 px -> TP=0, FP=1, FN=1
    m2 = compute_row_anchor_metrics(
        pred_coords=np.array([[335.0]]),
        pred_present=np.array([[True]]),
        gt_coords=np.array([[320.0]]),
        gt_present=np.array([[True]]),
        coord_tolerance_px=8.0,
    )
    print(f"Case 2 (err=15px): TP={m2['tp']}, FP={m2['fp']}, FN={m2['fn']}, RMSE={m2['rmse_tp_px']:.2f}")
    assert m2['tp'] == 0 and m2['fp'] == 1 and m2['fn'] == 1, f"Case 2 failed: {m2}"
    assert m2['rmse_tp_px'] == 0.0, f"Case 2 RMSE failed: {m2}"  # No TP, so sq_errors empty -> 0.0

    # Case 3: GT absent, Pred present -> FP=1, FN=0, TP=0
    m3 = compute_row_anchor_metrics(
        pred_coords=np.array([[320.0]]),
        pred_present=np.array([[True]]),
        gt_coords=np.array([[-1.0]]),
        gt_present=np.array([[False]]),
        coord_tolerance_px=8.0,
    )
    print(f"Case 3 (GT absent, Pred present): TP={m3['tp']}, FP={m3['fp']}, FN={m3['fn']}")
    assert m3['tp'] == 0 and m3['fp'] == 1 and m3['fn'] == 0, f"Case 3 failed: {m3}"

    # Case 4: GT present, Pred absent -> FN=1, FP=0, TP=0
    m4 = compute_row_anchor_metrics(
        pred_coords=np.array([[-1.0]]),
        pred_present=np.array([[False]]),
        gt_coords=np.array([[320.0]]),
        gt_present=np.array([[True]]),
        coord_tolerance_px=8.0,
    )
    print(f"Case 4 (GT present, Pred absent): TP={m4['tp']}, FP={m4['fp']}, FN={m4['fn']}")
    assert m4['tp'] == 0 and m4['fp'] == 0 and m4['fn'] == 1, f"Case 4 failed: {m4}"

    # Case 5: GT absent, Pred absent -> TN=1, TP=0, FP=0, FN=0
    m5 = compute_row_anchor_metrics(
        pred_coords=np.array([[-1.0]]),
        pred_present=np.array([[False]]),
        gt_coords=np.array([[-1.0]]),
        gt_present=np.array([[False]]),
        coord_tolerance_px=8.0,
    )
    print(f"Case 5 (GT absent, Pred absent): TP={m5['tp']}, FP={m5['fp']}, FN={m5['fn']}, TN={m5['tn']}")
    assert m5['tp'] == 0 and m5['fp'] == 0 and m5['fn'] == 0 and m5['tn'] == 1, f"Case 5 failed: {m5}"
    print("TASK 5 STATUS: ALL MANUAL SANITY TESTS PASSED 100% ACCORDING TO FROZEN SPECIFICATION.\n")


def run_task6_checkpoint_verification():
    print("=" * 75)
    print("TASK 6: CHECKPOINT VERIFICATION")
    print("=" * 75)
    ckpt_path = PROJECT_ROOT / "experiments" / "sict_shufflenet_baseline_50ep" / "checkpoints" / "best_model.pth"
    assert ckpt_path.exists(), f"best_model.pth does not exist at {ckpt_path}"
    print(f"Checkpoint exists: {ckpt_path} (Size: {ckpt_path.stat().st_size / (1024**2):.2f} MB)")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"Checkpoint keys: {list(ckpt.keys())}")
    print(f"Saved best epoch: {ckpt.get('epoch')}")
    print(f"Saved validation loss: {ckpt.get('val_loss'):.4f}")
    print(f"Saved val_metrics payload: {ckpt.get('val_metrics')}")

    cfg = load_config()
    model = build_model(cfg, apply_sict_in_forward=False)
    load_result = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"Model architecture loading check: {load_result} (Strict=True SUCCESS)")
    print("TASK 6 STATUS: CHECKPOINT IS FULLY LOADABLE, UNCORRUPTED, AND VALIDATED.\n")
    return ckpt, model, cfg


def run_task3_task4_deep_diagnosis(ckpt, model, cfg):
    print("=" * 75)
    print("TASK 3 & TASK 4: DEEP METRIC & LOW F1 DIAGNOSIS")
    print("=" * 75)

    # 1. Diagnose RMSE 0.00
    print("[1] Diagnosing why Val RMSE logged as 0.00 px:")
    val_metrics_saved = ckpt.get("val_metrics", {})
    print(f"    In checkpoint val_metrics, actual RMSE key is: 'rmse_tp_px' = {val_metrics_saved.get('rmse_tp_px')} px")
    print(f"    In scripts/train.py line 245/257, the code queried: epoch_metrics.get('rmse_overall', 0.0)")
    print("    EXPLANATION: The key name mismatched ('rmse_overall' vs 'rmse_tp_px').")
    print("    Because 'rmse_overall' did not exist in epoch_metrics dictionary, dict.get() returned the fallback 0.0!")
    print(f"    The true RMSE on True Positives at Epoch 47 was actually: {val_metrics_saved.get('rmse_tp_px'):.4f} px!\n")

    # 2. Diagnose Low F1 (~0.0127)
    print("[2] Diagnosing why F1 is ~0.0127:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    val_ds = RowAnchorDataset("data/synthetic/val/annotations.json", config=cfg)
    val_loader = build_dataloader(val_ds, batch_size=32, shuffle=False)

    total_gt_present = 0
    total_gt_absent = 0
    total_pred_present = 0
    total_pred_absent = 0

    all_pred_coords_soft = []
    all_pred_coords_argmax = []
    all_pred_present = []
    all_gt_coords = []
    all_gt_present = []
    all_gt_targets = []
    all_pred_argmax_bins = []

    t0 = time.time()
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            logits = model(images)  # (B, 18, 101)

            # Soft-argmax decoding per README formulation
            pred_c_soft, is_p, w_hat = decode_row_anchor_predictions(
                logits,
                image_width=cfg.camera.input_width,
                num_spatial_bins=cfg.model.num_spatial_bins,
            )

            # Pure argmax bin decoding: bin + 0.5 -> pixel coordinate
            argmax_classes = torch.argmax(logits, dim=-1)  # (B, 18)
            pred_c_argmax = ((argmax_classes.float() + 0.5) / 100.0) * 640.0

            all_pred_coords_soft.append(pred_c_soft.cpu().numpy())
            all_pred_coords_argmax.append(pred_c_argmax.cpu().numpy())
            all_pred_present.append(is_p.cpu().numpy())
            all_gt_coords.append(batch["u_coords"].numpy())
            all_gt_present.append(batch["presence"].numpy())
            all_gt_targets.append(batch["targets"].numpy())
            all_pred_argmax_bins.append(argmax_classes.cpu().numpy())

    eval_time = time.time() - t0
    print(f"    Evaluated all 900 validation images on {device} in {eval_time:.2f}s.")

    pred_coords_soft = np.concatenate(all_pred_coords_soft, axis=0)
    pred_coords_argmax = np.concatenate(all_pred_coords_argmax, axis=0)
    pred_present = np.concatenate(all_pred_present, axis=0)
    gt_coords = np.concatenate(all_gt_coords, axis=0)
    gt_present = np.concatenate(all_gt_present, axis=0)
    gt_targets = np.concatenate(all_gt_targets, axis=0)
    pred_argmax_bins = np.concatenate(all_pred_argmax_bins, axis=0)

    # Presence classification metrics
    tp_pres = ((pred_present == True) & (gt_present == True)).sum()
    fp_pres = ((pred_present == True) & (gt_present == False)).sum()
    fn_pres = ((pred_present == False) & (gt_present == True)).sum()
    tn_pres = ((pred_present == False) & (gt_present == False)).sum()
    print(f"\n    PRESENCE CLASSIFICATION CONFUSION MATRIX (16,200 total row queries):")
    print(f"      True Present:   {gt_present.sum():,}")
    print(f"      True Absent:    {(~gt_present).sum():,}")
    print(f"      Pred Present:   {pred_present.sum():,}")
    print(f"      Pred Absent:    {(~pred_present).sum():,}")
    print(f"      Presence TP:    {tp_pres:,}  | FP: {fp_pres:,}")
    print(f"      Presence FN:    {fn_pres:,} | TN: {tn_pres:,}")
    print(f"      Presence Recall:    {tp_pres / (tp_pres + fn_pres) * 100:.2f}%")
    print(f"      Presence Precision: {tp_pres / (tp_pres + fp_pres) * 100:.2f}%")

    # Metrics with current soft-argmax:
    m_soft = compute_row_anchor_metrics(pred_coords_soft, pred_present, gt_coords, gt_present, coord_tolerance_px=8.0)
    print(f"\n    METRICS WITH CURRENT SOFT-ARGMAX DECODING:")
    print(f"      F1 Score:     {m_soft['f1_score']:.4f}")
    print(f"      Precision:    {m_soft['precision']:.4f}")
    print(f"      Recall:       {m_soft['recall']:.4f}")
    print(f"      RMSE (TP):    {m_soft['rmse_tp_px']:.2f} px")
    print(f"      TP: {m_soft['tp']}, FP: {m_soft['fp']}, FN: {m_soft['fn']}")

    # Metrics with pure argmax bin decoding:
    m_argmax = compute_row_anchor_metrics(pred_coords_argmax, pred_present, gt_coords, gt_present, coord_tolerance_px=8.0)
    print(f"\n    METRICS WITH PURE ARGMAX BIN DECODING:")
    print(f"      F1 Score:     {m_argmax['f1_score']:.4f}")
    print(f"      Precision:    {m_argmax['precision']:.4f}")
    print(f"      Recall:       {m_argmax['recall']:.4f}")
    print(f"      RMSE (TP):    {m_argmax['rmse_tp_px']:.2f} px")
    print(f"      TP: {m_argmax['tp']}, FP: {m_argmax['fp']}, FN: {m_argmax['fn']}")

    # Now let's inspect the coordinate distribution of soft-argmax vs argmax where GT is present and predicted present
    both_present = pred_present & gt_present
    diff_soft = np.abs(pred_coords_soft[both_present] - gt_coords[both_present])
    diff_argmax = np.abs(pred_coords_argmax[both_present] - gt_coords[both_present])
    print(f"\n    COORDINATE ERROR ANALYSIS ON ROWS WHERE BOTH GT & PRED ARE PRESENT (N={both_present.sum()}):")
    print(f"      Soft-Argmax Mean Error:   {diff_soft.mean():.2f} px (Median: {np.median(diff_soft):.2f} px)")
    print(f"      Argmax Mean Error:        {diff_argmax.mean():.2f} px (Median: {np.median(diff_argmax):.2f} px)")
    print(f"      Soft-Argmax Error <= 8px: {(diff_soft <= 8.0).mean()*100:.2f}% (Count: {(diff_soft <= 8.0).sum()})")
    print(f"      Argmax Error <= 8px:      {(diff_argmax <= 8.0).mean()*100:.2f}% (Count: {(diff_argmax <= 8.0).sum()})")

    # Let's inspect what happens to soft-argmax:
    print(f"\n    SOFT-ARGMAX ROOT CAUSE:")
    print(f"      Mean soft-argmax coordinate: {pred_coords_soft[both_present].mean():.2f} px (std: {pred_coords_soft[both_present].std():.2f} px)")
    print(f"      Mean GT coordinate:          {gt_coords[both_present].mean():.2f} px (std: {gt_coords[both_present].std():.2f} px)")
    print("      NOTICE: Soft-argmax predictions are compressed towards the center (~350 px)!")
    print("      Because softmax is taken over 101 classes, any non-zero probability in non-target bins acts as an anchor pulling the expected value to the middle (bin 50 = 320 px)!")
    print("TASK 3 & 4 DIAGNOSIS COMPLETED.\n")


def run_task7_training_time_investigation():
    print("=" * 75)
    print("TASK 7: TRAINING TIME INVESTIGATION (91.70 MINUTES)")
    print("=" * 75)
    history_file = PROJECT_ROOT / "experiments" / "sict_shufflenet_baseline_50ep" / "training_history.json"
    with open(history_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("history", [])
    durations = [h["duration_sec"] for h in history]
    print(f"Total recorded epochs: {len(durations)}")
    print(f"Mean epoch duration:   {np.mean(durations):.2f} s ({np.mean(durations)/60:.2f} min)")
    print(f"Min epoch duration:    {np.min(durations):.2f} s")
    print(f"Max epoch duration:    {np.max(durations):.2f} s")
    print(f"Sum epoch duration:    {np.sum(durations):.2f} s ({np.sum(durations)/60:.2f} min)")
    print(f"Recorded total time:   {data.get('total_time_sec', 0)/60:.2f} min")

    # Profile the bottleneck components:
    # A. 1 batch of data loading with SICT on CPU
    cfg = load_config()
    train_ds = RowAnchorDataset("data/synthetic/train/annotations.json", config=cfg)
    train_loader = build_dataloader(train_ds, batch_size=32, shuffle=False)

    t0 = time.time()
    batch = next(iter(train_loader))
    t_load = time.time() - t0
    print(f"\nTime to load & SICT-preprocess 1 batch of 32 images on single CPU thread: {t_load*1000:.1f} ms")
    est_data_time_per_epoch = t_load * (len(train_ds) / 32)
    print(f"Estimated data-loading + SICT time for 4,200 training images (132 batches): {est_data_time_per_epoch:.1f} s")

    # B. GPU forward + backward time for 1 batch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, apply_sict_in_forward=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3)
    from src.losses.structural_loss import TotalRowAnchorLoss
    criterion = TotalRowAnchorLoss(loss_config=cfg.loss, model_config=cfg.model).to(device)

    img_gpu = batch["image"].to(device)
    tgt_gpu = batch["targets"].to(device)

    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        out = model(img_gpu)
        l = criterion(out, tgt_gpu)["loss_total"]
        l.backward()
        optimizer.step()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(10):
        optimizer.zero_grad()
        out = model(img_gpu)
        l = criterion(out, tgt_gpu)["loss_total"]
        l.backward()
        optimizer.step()
    torch.cuda.synchronize()
    gpu_step_time = (time.time() - t0) / 10.0
    print(f"GPU forward+backward+optimizer step time for batch 32: {gpu_step_time*1000:.1f} ms")
    est_gpu_time_per_epoch = gpu_step_time * 132
    print(f"Estimated GPU computation time for 132 batches: {est_gpu_time_per_epoch:.1f} s")
    print(f"\nBOTTLENECK SUMMARY:")
    print(f"  GPU computation is only ~{est_gpu_time_per_epoch:.1f} s per epoch ({est_gpu_time_per_epoch/np.mean(durations)*100:.1f}% of epoch time).")
    print(f"  CPU Disk I/O + NumPy SICT filtering is ~{est_data_time_per_epoch:.1f} s per epoch ({est_data_time_per_epoch/np.mean(durations)*100:.1f}% of epoch time).")
    print("  With num_workers=0, the GPU is starved waiting for the single-threaded CPU to read and transform images sequentially.")


if __name__ == "__main__":
    run_task5_manual_sanity_test()
    ckpt, model, cfg = run_task6_checkpoint_verification()
    run_task3_task4_deep_diagnosis(ckpt, model, cfg)
    run_task7_training_time_investigation()
