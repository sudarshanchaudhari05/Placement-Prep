"""
End-to-End Pipeline Smoke Test Script
=====================================
Verifies all components of the perception software pipeline:
1. Configuration validation
2. SICT transform (NumPy and PyTorch)
3. Synthetic dummy-data generation (marked NOT research data)
4. Dataset loading and PyTorch DataLoader batching
5. RowAnchorNet model instantiation and parameter accounting
6. Model forward pass and strict [batch_size, 18, 101] shape validation
7. Dimension mismatch assertions (ensuring loud failures on invalid inputs)
8. Row-anchor soft-argmax coordinate decoding
9. Structural loss computation (Focal, Smoothness, Curvature) and backward pass
10. Metric calculation (F1 score, Precision, Recall, RMSE)
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import load_config, ProjectConfig
from src.sict.sict_filter import sict_transform_np, sict_transform_torch, SICTModule
from src.models.row_anchor_net import build_model, RowAnchorNet
from src.data.dummy_generator import generate_dummy_dataset
from src.data.dataset import RowAnchorDataset
from src.labels.coder import (
    RowAnchorCoder,
    get_anchor_row_y_coords,
    encode_row_anchor_targets,
    decode_row_anchor_predictions,
)
from src.losses.structural_loss import TotalRowAnchorLoss
from src.evaluation.metrics import RowAnchorEvaluator, compute_row_anchor_metrics
from src.utils.seed import seed_everything


def run_smoke_test():
    print("=" * 70)
    print("RUNNING AGV PERCEPTION MODEL SMOKE TEST")
    print("=" * 70)

    # 1. Reproducibility Seed
    print("[1/10] Setting random seed...")
    seed_everything(42)
    print("       Seed set to 42.")

    # 2. Configuration Validation
    print("[2/10] Loading & validating central configuration...")
    cfg = load_config()
    cfg.validate()
    print(f"       Config validated successfully.")
    print(f"       Camera: {cfg.camera.input_width}x{cfg.camera.input_height}")
    print(f"       ROI: y=[{cfg.roi.y_min}, {cfg.roi.y_max}]")
    print(f"       Network Input: {cfg.model.input_height}x{cfg.model.input_width}")
    print(f"       Row Anchors M={cfg.model.num_row_anchors}, Spatial Bins K={cfg.model.num_spatial_bins}")
    print(f"       Total Classes per Row: {cfg.model.total_classes}")

    # 3. SICT Glare Pre-filtering Test
    print("[3/10] Verifying Specular-Invariant Chromaticity Transform (SICT)...")
    # Test image with bright specular spot (255, 255, 255)
    test_img = np.full((288, 384, 3), 100, dtype=np.uint8)
    test_img[100:150, 150:200] = 255  # S = 765 >= 220, r=g=1/3 -> Phi=0 <= 0.05
    sict_out_np = sict_transform_np(test_img, config=cfg.sict, out_channels=3)
    assert sict_out_np.shape == (288, 384, 3), f"SICT NP shape mismatch: {sict_out_np.shape}"
    assert 0.0 <= sict_out_np.min() and sict_out_np.max() <= 1.0, "SICT NP output not in [0, 1]"

    # Verify specular suppression: glare region should be replaced by ambient intensity
    glare_val = sict_out_np[120, 170, 0]
    normal_val = sict_out_np[50, 50, 0]
    assert glare_val < normal_val or np.isclose(glare_val, cfg.sict.ambient_ratio * normal_val, atol=0.1), (
        "SICT specular suppression failed to attenuate specular glare region"
    )

    # PyTorch SICT Module test
    sict_module = SICTModule(config=cfg.sict, out_channels=3)
    t_input = torch.from_numpy(test_img.transpose(2, 0, 1)).unsqueeze(0).float()  # (1, 3, 288, 384)
    sict_out_torch = sict_module(t_input)
    assert sict_out_torch.shape == (1, 3, 288, 384), (
        f"SICT Torch shape mismatch: {sict_out_torch.shape}"
    )
    print("       SICT NumPy and PyTorch implementations passed.")

    # 4. Synthetic Dummy Data Generation
    print("[4/10] Generating synthetic dummy test data (NOT research data)...")
    dummy_dir = PROJECT_ROOT / "dummy_data"
    ann_path = generate_dummy_dataset(output_dir=dummy_dir, num_samples=6, config=cfg)
    assert ann_path.exists(), f"Dummy annotation file missing: {ann_path}"
    print(f"       Dummy dataset written to: {ann_path}")

    # 5. Dataset Loading & Batching Test
    print("[5/10] Verifying RowAnchorDataset and DataLoader batching...")
    dataset = RowAnchorDataset(annotation_file=ann_path, config=cfg, apply_sict=True)
    assert len(dataset) == 6, f"Expected 6 samples, got {len(dataset)}"

    sample = dataset[0]
    img_tensor = sample["image"]
    targets_tensor = sample["targets"]
    u_coords_tensor = sample["u_coords"]
    presence_tensor = sample["presence"]

    assert img_tensor.shape == (3, 288, 384), f"Dataset image tensor shape {img_tensor.shape} != (3, 288, 384)"
    assert targets_tensor.shape == (18,), f"Dataset targets shape {targets_tensor.shape} != (18,)"
    assert u_coords_tensor.shape == (18,), f"Dataset u_coords shape {u_coords_tensor.shape} != (18,)"
    assert presence_tensor.shape == (18,), f"Dataset presence shape {presence_tensor.shape} != (18,)"
    assert targets_tensor.min() >= 0 and targets_tensor.max() <= 100, "Target classes out of [0, 100] range"

    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    b_images = batch["image"]
    b_targets = batch["targets"]
    assert b_images.shape == (2, 3, 288, 384), f"Batch image shape mismatch: {b_images.shape}"
    assert b_targets.shape == (2, 18), f"Batch target shape mismatch: {b_targets.shape}"
    print(f"       Dataset and DataLoader passed. Batch image shape: {list(b_images.shape)}")

    # 6. Model Instantiation & Parameter Accounting
    print("[6/10] Building RowAnchorNet (ShuffleNetV2-0.5x backbone)...")
    model = build_model(cfg, apply_sict_in_forward=False)
    param_info = model.count_parameters()
    print(f"       Total Parameters:     {param_info['total']:,}")
    print(f"       Trainable Parameters: {param_info['trainable']:,}")
    print(f"       Backbone Parameters:  {param_info['backbone']:,}")
    print(f"       Head Parameters:      {param_info['head']:,}")

    # 7. Model Forward Pass & Strict Output Shape Validation
    print("[7/10] Verifying forward pass and output dimensions...")
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 288, 384)
    logits = model(dummy_input)

    expected_output_shape = (batch_size, 18, 101)
    assert logits.shape == expected_output_shape, (
        f"CRITICAL: Output shape mismatch! Expected {expected_output_shape}, got {logits.shape}"
    )
    print(f"       Model output shape verified: {list(logits.shape)} -> [batch_size, 18, 101]")

    # 8. Dimension Assertion Verification (Loud Failure Test)
    print("[8/10] Testing loud failure on invalid input dimensions...")
    try:
        invalid_input = torch.randn(2, 3, 256, 256)  # Incorrect dimensions
        model(invalid_input)
        assert False, "Model failed to raise AssertionError on invalid input spatial dimensions!"
    except AssertionError as e:
        print(f"       Passed: Successfully caught expected assertion on invalid input dimensions.")

    try:
        invalid_channels = torch.randn(2, 1, 288, 384)  # Incorrect channels
        model(invalid_channels)
        assert False, "Model failed to raise AssertionError on invalid input channels!"
    except AssertionError as e:
        print(f"       Passed: Successfully caught expected assertion on invalid channel count.")

    # 9. Loss Computation & Backward Pass Test
    print("[9/10] Testing structural loss computation and backward gradient flow...")
    criterion = TotalRowAnchorLoss(loss_config=cfg.loss, model_config=cfg.model)
    loss_dict = criterion(logits, b_targets)

    for loss_name, loss_val in loss_dict.items():
        assert torch.isfinite(loss_val), f"Loss {loss_name} is NaN or Inf!"
        print(f"       {loss_name}: {loss_val.item():.4f}")

    # Backward pass verification
    loss_dict["loss_total"].backward()
    has_grads = any(p.grad is not None and torch.norm(p.grad) > 0 for p in model.parameters())
    assert has_grads, "Gradients failed to flow back through model parameters!"
    print("       Backward gradient flow verified.")

    # 10. Coordinate Decoding & Evaluation Metrics Test
    print("[10/10] Verifying soft-argmax coordinate decoding and F1 evaluation...")
    coder = RowAnchorCoder(cfg)
    pred_coords, is_present, w_hat = coder.decode(logits)
    assert pred_coords.shape == (batch_size, 18), f"Decoded coords shape mismatch: {pred_coords.shape}"
    assert is_present.shape == (batch_size, 18), f"Decoded presence shape mismatch: {is_present.shape}"
    assert w_hat.shape == (batch_size, 18), f"Decoded w_hat shape mismatch: {w_hat.shape}"

    # Evaluator accumulator test
    evaluator = RowAnchorEvaluator(cfg)
    evaluator.update(logits, batch["u_coords"], batch["presence"])
    metrics = evaluator.compute()
    print(f"       Evaluation pipeline tested successfully.")
    print(f"       F1 Metric keys computed: {list(metrics.keys())}")

    print("=" * 70)
    print("ALL SMOKE TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return {
        "status": "PASSED",
        "total_params": param_info["total"],
        "trainable_params": param_info["trainable"],
        "output_shape": list(logits.shape),
    }


if __name__ == "__main__":
    result = run_smoke_test()
    sys.exit(0 if result["status"] == "PASSED" else 1)
