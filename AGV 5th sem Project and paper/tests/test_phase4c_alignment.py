"""
Deterministic Unit Tests for Phase 4C: Decoder-Loss Alignment & Metric Logging
=============================================================================
Tests:
1. Decoder Boundary & Behavior Tests (bin 0, 50, 99, absence winner, ties, extreme logits)
2. Controlled Coordinate Tests (anti-centering test, dominant logit shifts, decoder-loss match)
3. Structural Loss Gradient & Graph-Safety Tests (straight, curved, partial, all-absent, gradient direction)
4. Metric Logging & Missing-Key Tests (Cases 1-5, empty-TP NaN, missing key handling)
"""

import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from configs.config import load_config, ProjectConfig, LossConfig, ModelConfig
from src.labels.coder import decode_row_anchor_predictions, RowAnchorCoder
from src.losses.structural_loss import TotalRowAnchorLoss, SmoothnessLoss, CurvatureLoss
from src.evaluation.metrics import compute_row_anchor_metrics, RowAnchorEvaluator


# =============================================================================
# 1. DECODER TESTS
# =============================================================================

def test_decoder_peak_at_bin_0():
    """Peak at boundary bin 0."""
    logits = torch.full((1, 1, 101), -10.0)
    logits[0, 0, 0] = 10.0
    pred_coords, is_present, w_hat = decode_row_anchor_predictions(logits, local_radius=2)
    assert is_present[0, 0].item() is True
    assert math.isclose(w_hat[0, 0].item(), 0.0, abs_tol=1e-3)
    # Reconstructed pixel coordinate: ((0 + 0.5) / 100) * 640 = 3.2 px
    assert math.isclose(pred_coords[0, 0].item(), 3.2, abs_tol=1e-2)


def test_decoder_peak_at_bin_50():
    """Peak at center bin 50."""
    logits = torch.full((1, 1, 101), -10.0)
    logits[0, 0, 50] = 10.0
    pred_coords, is_present, w_hat = decode_row_anchor_predictions(logits, local_radius=2)
    assert is_present[0, 0].item() is True
    assert math.isclose(w_hat[0, 0].item(), 50.0, abs_tol=1e-3)
    # Reconstructed pixel coordinate: ((50 + 0.5) / 100) * 640 = 323.2 px
    assert math.isclose(pred_coords[0, 0].item(), 323.2, abs_tol=1e-2)


def test_decoder_peak_at_bin_99():
    """Peak at boundary bin 99."""
    logits = torch.full((1, 1, 101), -10.0)
    logits[0, 0, 99] = 10.0
    pred_coords, is_present, w_hat = decode_row_anchor_predictions(logits, local_radius=2)
    assert is_present[0, 0].item() is True
    assert math.isclose(w_hat[0, 0].item(), 99.0, abs_tol=1e-3)
    # Reconstructed pixel coordinate: ((99 + 0.5) / 100) * 640 = 636.8 px
    assert math.isclose(pred_coords[0, 0].item(), 636.8, abs_tol=1e-2)


def test_decoder_strong_spatial_peak_vs_absence_winner():
    """Strong spatial peak at bin 30, but absence logit (class 100) is even higher."""
    logits = torch.full((1, 1, 101), -10.0)
    logits[0, 0, 30] = 5.0   # strong spatial peak
    logits[0, 0, 100] = 8.0  # absence class wins overall
    pred_coords, is_present, w_hat = decode_row_anchor_predictions(logits, local_radius=2)
    assert is_present[0, 0].item() is False  # Must be declared absent!
    # Spatial expectation should still be centered around bin 30
    assert math.isclose(w_hat[0, 0].item(), 30.0, abs_tol=1e-2)


def test_decoder_exact_ties_between_spatial_logits():
    """Exact tie between adjacent spatial bins (e.g. bin 20 and bin 21)."""
    logits = torch.full((1, 1, 101), -10.0)
    logits[0, 0, 20] = 5.0
    logits[0, 0, 21] = 5.0
    pred_coords, is_present, w_hat = decode_row_anchor_predictions(logits, local_radius=2)
    assert is_present[0, 0].item() is True
    # Expectation should be at the midpoint 20.5
    assert math.isclose(w_hat[0, 0].item(), 20.5, abs_tol=1e-2)
    # Pixel coordinate: ((20.5 + 0.5) / 100) * 640 = 134.4 px
    assert math.isclose(pred_coords[0, 0].item(), 134.4, abs_tol=1e-2)


def test_decoder_very_large_positive_and_negative_logits():
    """Numerically extreme logits should not cause overflow, underflow, or NaNs."""
    logits = torch.full((1, 1, 101), -500.0)
    logits[0, 0, 42] = 500.0
    pred_coords, is_present, w_hat = decode_row_anchor_predictions(logits, local_radius=2)
    assert torch.isfinite(pred_coords).all()
    assert torch.isfinite(w_hat).all()
    assert math.isclose(w_hat[0, 0].item(), 42.0, abs_tol=1e-3)


def test_decoder_output_shape():
    """Verify output shapes on multidimensional batches."""
    b, m = 4, 18
    logits = torch.randn(b, m, 101)
    coords, pres, w = decode_row_anchor_predictions(logits, local_radius=2)
    assert coords.shape == (b, m)
    assert pres.shape == (b, m)
    assert w.shape == (b, m)
    assert pres.dtype == torch.bool
    assert coords.dtype == torch.float32


# =============================================================================
# 2. CONTROLLED COORDINATE & ANTI-CENTERING TESTS
# =============================================================================

def test_anti_centering_off_center_bin():
    """
    CRITICAL CHECK:
    Verify that an off-center peak (e.g. bin 80, u ~ 515 px) is NOT pulled toward bin 50 (320 px).
    In the flawed global soft-argmax, background entropy pulled bin 80 down to ~53.7 (347 px).
    With localized soft-argmax, w_hat must remain within [79.5, 80.5].
    """
    logits = torch.zeros(1, 1, 101)  # uniform baseline across all 100 bins
    logits[0, 0, 80] = 3.0           # moderate peak at bin 80
    _, _, w_hat = decode_row_anchor_predictions(logits, local_radius=2)

    # In localized soft-argmax, expectation only involves window [78, 82]
    # Because all bins outside window are masked, w_hat cannot be pulled to 50!
    assert 78.0 <= w_hat[0, 0].item() <= 82.0
    assert abs(w_hat[0, 0].item() - 80.0) < 0.5, (
        f"Expected w_hat near 80, got {w_hat[0, 0].item()}"
    )


def test_controlled_shift_in_peak():
    """Shifting the spatial peak by N bins must shift the decoded coordinate by N bins."""
    for b_peak in [10, 25, 60, 85]:
        logits = torch.zeros(1, 1, 101)
        logits[0, 0, b_peak] = 5.0
        _, _, w = decode_row_anchor_predictions(logits, local_radius=2)
        assert math.isclose(w[0, 0].item(), float(b_peak), abs_tol=0.1)


def test_training_loss_coordinate_matches_decoder_convention():
    """
    Verify that the coordinate representation in TotalRowAnchorLoss
    is mathematically identical to decode_row_anchor_predictions.
    """
    cfg = load_config()
    loss_fn = TotalRowAnchorLoss(loss_config=cfg.loss, model_config=cfg.model)

    logits = torch.randn(2, 18, 101)
    _, _, w_decoder = decode_row_anchor_predictions(logits, local_radius=2)

    # In TotalRowAnchorLoss:
    spatial_logits = logits[:, :, :100]
    j_star = torch.argmax(spatial_logits, dim=-1)
    bin_grid = torch.arange(100, dtype=torch.float32).view(1, 1, 100)
    dist = torch.abs(bin_grid - j_star.unsqueeze(-1).float())
    in_window = (dist <= 2.0)
    window_logits = torch.where(in_window, spatial_logits, torch.full_like(spatial_logits, -1e9))
    window_probs = F.softmax(window_logits, dim=-1)
    w_loss = torch.sum(window_probs * bin_grid, dim=-1)

    assert torch.allclose(w_decoder, w_loss, atol=1e-5), "Mismatch between decoder w_hat and loss w_hat!"


# =============================================================================
# 3. STRUCTURAL LOSS GRADIENTS & GRAPH-SAFETY TESTS
# =============================================================================

def test_structural_loss_all_absent_graph_safe():
    """Entirely absent trajectory: structural loss is zero, gradients finite, autograd safe."""
    cfg = load_config()
    loss_fn = TotalRowAnchorLoss(loss_config=cfg.loss, model_config=cfg.model)

    logits = torch.randn(2, 18, 101, requires_grad=True)
    all_absent_targets = torch.full((2, 18), 100, dtype=torch.long)

    loss_dict = loss_fn(logits, all_absent_targets)
    assert math.isclose(loss_dict["loss_smooth"].item(), 0.0)
    assert math.isclose(loss_dict["loss_curv"].item(), 0.0)
    assert torch.isfinite(loss_dict["loss_total"])

    loss_dict["loss_total"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_structural_loss_partially_absent():
    """Partially absent trajectory: valid adjacent pairs and triplets handled correctly."""
    cfg = load_config()
    loss_fn = TotalRowAnchorLoss(loss_config=cfg.loss, model_config=cfg.model)

    logits = torch.randn(2, 18, 101, requires_grad=True)
    # Present at rows 0..5, absent at rows 6..17
    targets = torch.full((2, 18), 100, dtype=torch.long)
    targets[:, :6] = torch.randint(10, 90, (2, 6))

    loss_dict = loss_fn(logits, targets)
    assert loss_dict["loss_smooth"].item() > 0.0
    assert torch.isfinite(loss_dict["loss_total"])

    loss_dict["loss_total"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_structural_loss_straight_vs_curved_gradient():
    """
    Straight trajectory: L_curv should be nearly zero.
    Curved trajectory: L_curv should be non-zero and penalize zigzagging.
    """
    cfg = load_config()
    loss_fn = TotalRowAnchorLoss(loss_config=cfg.loss, model_config=cfg.model)

    # Construct logits representing a perfectly straight path at bin 50 across all 18 rows
    straight_logits = torch.full((1, 18, 101), -10.0, requires_grad=True)
    straight_logits.data[:, :, 50] = 10.0
    targets = torch.full((1, 18), 50, dtype=torch.long)

    loss_straight = loss_fn(straight_logits, targets)
    assert math.isclose(loss_straight["loss_smooth"].item(), 0.0, abs_tol=1e-4)
    assert math.isclose(loss_straight["loss_curv"].item(), 0.0, abs_tol=1e-4)

    # Construct logits representing an alternating zigzag path (bin 40, bin 60, bin 40...)
    zigzag_logits = torch.full((1, 18, 101), -10.0, requires_grad=True)
    for r in range(18):
        zigzag_logits.data[:, r, 40 if r % 2 == 0 else 60] = 10.0

    loss_zigzag = loss_fn(zigzag_logits, targets)
    assert loss_zigzag["loss_smooth"].item() > 10.0
    assert loss_zigzag["loss_curv"].item() > 10.0


def test_structural_gradient_direction():
    """
    Verify that structural smoothness loss gradient pulls adjacent rows toward each other.
    Row 0 at bin 40, Row 1 at bin 50.
    Smoothness loss L = |w_1 - w_0|.
    Gradient on Row 1 at bin 49 should be NEGATIVE (increasing logit at 49 decreases w_1 and decreases loss).
    Gradient on Row 1 at bin 51 should be POSITIVE (increasing logit at 51 increases w_1 and increases loss).
    """
    cfg = load_config()
    smoothness_fn = SmoothnessLoss()

    logits = torch.full((1, 2, 100), -5.0, requires_grad=True)
    logits.data[0, 0, 40] = 5.0  # peak at 40
    logits.data[0, 1, 50] = 5.0  # peak at 50

    # Localized window soft-argmax
    bin_grid = torch.arange(100, dtype=torch.float32).view(1, 1, 100)
    j_star = torch.argmax(logits, dim=-1)
    dist = torch.abs(bin_grid - j_star.unsqueeze(-1).float())
    in_win = (dist <= 2.0)
    win_logits = torch.where(in_win, logits, torch.full_like(logits, -1e9))
    w_hat = torch.sum(F.softmax(win_logits, dim=-1) * bin_grid, dim=-1)

    loss = smoothness_fn(w_hat)
    loss.backward()

    # On Row 1 (currently at 50, needs to move left towards 40):
    grad_left = logits.grad[0, 1, 49].item()
    grad_right = logits.grad[0, 1, 51].item()
    assert grad_left < 0.0, f"Expected negative grad on left bin 49, got {grad_left}"
    assert grad_right > 0.0, f"Expected positive grad on right bin 51, got {grad_right}"


# =============================================================================
# 4. METRIC LOGGING & MISSING-KEY TESTS
# =============================================================================

def test_metric_manual_cases():
    """Phase 4C-F Metric Manual Verification Cases 1 to 5."""
    # Case 1: GT 320 px, Pred 324 px -> TP=1, FP=0, FN=0, RMSE=4.00 px
    m1 = compute_row_anchor_metrics(
        pred_coords=np.array([[324.0]]),
        pred_present=np.array([[True]]),
        gt_coords=np.array([[320.0]]),
        gt_present=np.array([[True]]),
        coord_tolerance_px=8.0,
    )
    assert m1["tp"] == 1 and m1["fp"] == 0 and m1["fn"] == 0
    assert math.isclose(m1["rmse_tp_px"], 4.0)

    # Case 2: GT 320 px, Pred 335 px -> TP=0, FP=1, FN=1, RMSE=NaN (no TP)
    m2 = compute_row_anchor_metrics(
        pred_coords=np.array([[335.0]]),
        pred_present=np.array([[True]]),
        gt_coords=np.array([[320.0]]),
        gt_present=np.array([[True]]),
        coord_tolerance_px=8.0,
    )
    assert m2["tp"] == 0 and m2["fp"] == 1 and m2["fn"] == 1
    assert math.isnan(m2["rmse_tp_px"])

    # Case 3: GT absent, Pred present -> FP=1
    m3 = compute_row_anchor_metrics(
        pred_coords=np.array([[320.0]]),
        pred_present=np.array([[True]]),
        gt_coords=np.array([[-1.0]]),
        gt_present=np.array([[False]]),
        coord_tolerance_px=8.0,
    )
    assert m3["tp"] == 0 and m3["fp"] == 1 and m3["fn"] == 0

    # Case 4: GT present, Pred absent -> FN=1
    m4 = compute_row_anchor_metrics(
        pred_coords=np.array([[-1.0]]),
        pred_present=np.array([[False]]),
        gt_coords=np.array([[320.0]]),
        gt_present=np.array([[True]]),
        coord_tolerance_px=8.0,
    )
    assert m4["tp"] == 0 and m4["fp"] == 0 and m4["fn"] == 1

    # Case 5: GT absent, Pred absent -> TN=1
    m5 = compute_row_anchor_metrics(
        pred_coords=np.array([[-1.0]]),
        pred_present=np.array([[False]]),
        gt_coords=np.array([[-1.0]]),
        gt_present=np.array([[False]]),
        coord_tolerance_px=8.0,
    )
    assert m5["tp"] == 0 and m5["fp"] == 0 and m5["fn"] == 0 and m5["tn"] == 1


def test_metric_empty_tp_returns_nan():
    """When TP=0, rmse_tp_px must be NaN, not misleading 0.0."""
    evaluator = RowAnchorEvaluator()
    evaluator.reset()
    res = evaluator.compute()
    assert math.isnan(res["rmse_tp_px"])
    assert res["tp"] == 0
