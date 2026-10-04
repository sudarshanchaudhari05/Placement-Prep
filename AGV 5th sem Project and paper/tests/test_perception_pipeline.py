"""
Comprehensive PyTorch Perception Pipeline Tests
==============================================
Validates:
A. Dataset loading
B. Tensor shapes
C. Label validity
D. Model output shape
E. RAW logits
F. Loss masking
G. Loss gradients
H. Optimizer configuration
I. Scheduler configuration
J. GPU forward/backward
K. Checkpoint save/load
L. Deterministic initialization
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import load_config, ProjectConfig
from src.data.dataset import RowAnchorDataset, build_dataloader, collate_row_anchor_batch
from src.models.row_anchor_net import build_model, RowAnchorNet
from src.losses.structural_loss import TotalRowAnchorLoss
from src.engine.trainer import build_optimizer, build_scheduler
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.seed import seed_everything


@pytest.fixture(scope="module")
def config():
    cfg = load_config()
    cfg.validate()
    return cfg


# A. Dataset loading
def test_A_dataset_loading(config):
    splits = {
        "train": (PROJECT_ROOT / "data" / "synthetic" / "train" / "annotations.json", 4200),
        "val": (PROJECT_ROOT / "data" / "synthetic" / "val" / "annotations.json", 900),
        "test": (PROJECT_ROOT / "data" / "synthetic" / "test" / "annotations.json", 900),
    }
    for split_name, (ann_path, expected_count) in splits.items():
        assert ann_path.exists(), f"Annotation file missing: {ann_path}"
        ds = RowAnchorDataset(annotation_file=ann_path, config=config)
        assert len(ds) == expected_count, f"Expected {expected_count} for {split_name}, got {len(ds)}"


# B. Tensor shapes
def test_B_tensor_shapes(config):
    ann_path = PROJECT_ROOT / "data" / "synthetic" / "val" / "annotations.json"
    ds = RowAnchorDataset(annotation_file=ann_path, config=config)
    loader = build_dataloader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    assert batch["image"].shape == (4, 3, 288, 384), f"Image shape: {batch['image'].shape}"
    assert batch["targets"].shape == (4, 18), f"Targets shape: {batch['targets'].shape}"
    assert batch["u_coords"].shape == (4, 18), f"u_coords shape: {batch['u_coords'].shape}"
    assert batch["presence"].shape == (4, 18), f"presence shape: {batch['presence'].shape}"
    assert batch["image"].dtype == torch.float32
    assert batch["targets"].dtype == torch.int64


# C. Label validity
def test_C_label_validity(config):
    ann_path = PROJECT_ROOT / "data" / "synthetic" / "val" / "annotations.json"
    ds = RowAnchorDataset(annotation_file=ann_path, config=config)
    # Check first 50 samples
    for i in range(min(50, len(ds))):
        s = ds[i]
        targets = s["targets"]
        assert (0 <= targets).all() and (targets <= 100).all(), f"Target classes out of range: {targets}"
        # Check consistency with presence mask
        presence = s["presence"]
        absent_mask = (targets == 100)
        assert torch.equal(presence, ~absent_mask), "Presence mask inconsistent with absence class 100"


# D. Model output shape
def test_D_model_output_shape(config):
    model = build_model(config, apply_sict_in_forward=False)
    model.eval()
    x = torch.randn(2, 3, 288, 384)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 18, 101), f"Expected shape (2, 18, 101), got {out.shape}"


# E. RAW logits (no softmax inside model)
def test_E_raw_logits(config):
    model = build_model(config, apply_sict_in_forward=False)
    model.eval()
    x = torch.randn(2, 3, 288, 384)
    with torch.no_grad():
        out = model(x)
    assert (out < 0.0).any(), "RAW logits must allow negative values"
    class_sums = out.sum(dim=-1)
    assert not torch.allclose(class_sums, torch.ones_like(class_sums)), "Logits must not sum to 1.0 (no softmax)"


# F. Loss masking of absent anchors
def test_F_loss_masking(config):
    criterion = TotalRowAnchorLoss(loss_config=config.loss, model_config=config.model)
    dummy_logits = torch.randn(2, 18, 101)

    # When all rows are absent (class 100), structural losses must be identically 0
    all_absent_targets = torch.full((2, 18), 100, dtype=torch.long)
    loss_dict = criterion(dummy_logits, all_absent_targets)
    assert torch.isclose(loss_dict["loss_smooth"], torch.tensor(0.0)), "Smoothness loss must be 0 for all-absent"
    assert torch.isclose(loss_dict["loss_curv"], torch.tensor(0.0)), "Curvature loss must be 0 for all-absent"


# G. Loss gradients
def test_G_loss_gradients(config):
    criterion = TotalRowAnchorLoss(loss_config=config.loss, model_config=config.model)
    model = build_model(config, apply_sict_in_forward=False)
    model.train()

    x = torch.randn(2, 3, 288, 384)
    y = torch.randint(0, 101, (2, 18))

    model.zero_grad()
    logits = model(x)
    loss_dict = criterion(logits, y)
    loss_dict["loss_total"].backward()

    # Verify gradients exist and are finite
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None, "Gradient is None"
            assert torch.isfinite(p.grad).all(), "Gradient is NaN or Inf"


# H. Optimizer configuration
def test_H_optimizer_configuration(config):
    model = build_model(config, apply_sict_in_forward=False)
    optimizer = build_optimizer(model, config.training)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == 4.0e-3
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["weight_decay"] == 1.0e-4


# I. Scheduler configuration
def test_I_scheduler_configuration(config):
    model = build_model(config, apply_sict_in_forward=False)
    optimizer = build_optimizer(model, config.training)
    scheduler = build_scheduler(optimizer, config.training, total_epochs=50)

    assert scheduler.eta_min == 1.0e-5
    for _ in range(50):
        optimizer.step()
        scheduler.step()
    final_lr = optimizer.param_groups[0]["lr"]
    assert np.isclose(final_lr, 1.0e-5, atol=1e-7), f"Expected 1e-5, got {final_lr}"


# J. GPU forward/backward
def test_J_gpu_forward_backward(config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this system")

    device = torch.device("cuda")
    model = build_model(config, apply_sict_in_forward=False).to(device)
    criterion = TotalRowAnchorLoss(loss_config=config.loss, model_config=config.model).to(device)
    optimizer = build_optimizer(model, config.training)

    x = torch.randn(4, 3, 288, 384, device=device)
    y = torch.randint(0, 101, (4, 18), device=device)

    optimizer.zero_grad()
    out = model(x)
    assert out.is_cuda
    loss = criterion(out, y)["loss_total"]
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    for p in model.parameters():
        if p.requires_grad:
            assert torch.isfinite(p.grad).all()


# K. Checkpoint save/load
def test_K_checkpoint_save_load(config, tmp_path):
    model = build_model(config, apply_sict_in_forward=False)
    optimizer = build_optimizer(model, config.training)
    scheduler = build_scheduler(optimizer, config.training, total_epochs=50)

    state = {
        "epoch": 7,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": 0.384,
        "seed": 42,
    }
    ckpt_path = save_checkpoint(state, tmp_path, filename="ckpt_test.pth", is_best=True, config=config)
    assert ckpt_path.exists()

    fresh_model = build_model(config, apply_sict_in_forward=False)
    fresh_opt = build_optimizer(fresh_model, config.training)
    fresh_sched = build_scheduler(fresh_opt, config.training, total_epochs=50)

    restored = load_checkpoint(ckpt_path, fresh_model, optimizer=fresh_opt, scheduler=fresh_sched)
    assert restored["epoch"] == 7
    assert restored["val_loss"] == 0.384

    for p1, p2 in zip(model.parameters(), fresh_model.parameters()):
        assert torch.equal(p1, p2), "Model parameters mismatch after checkpoint reload"


# L. Deterministic initialization
def test_L_deterministic_initialization(config):
    seed_everything(42, deterministic_cuda=True)
    m1 = build_model(config, apply_sict_in_forward=False)

    seed_everything(42, deterministic_cuda=True)
    m2 = build_model(config, apply_sict_in_forward=False)

    for p1, p2 in zip(m1.head.parameters(), m2.head.parameters()):
        assert torch.equal(p1, p2), "Models initialized with same seed have differing weights"
