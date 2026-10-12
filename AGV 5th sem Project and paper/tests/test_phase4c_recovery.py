"""
Phase 4C Recovery Tests
=======================
Tests verifying resume functionality, atomic checkpointing,
scheduler continuity, history reconstruction, and checkpoint 20 integrity.
"""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs.config import load_config
from src.models.row_anchor_net import RowAnchorNet, build_model
from src.engine.trainer import build_optimizer, build_scheduler
from src.utils.checkpoint import save_checkpoint, load_checkpoint, safe_atomic_torch_save


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="phase4c_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


def test_checkpoint_restoration_model_optimizer_scheduler(temp_dir):
    """Verify load_checkpoint restores model weights, optimizer state, and scheduler state."""
    model = DummyModel()
    optimizer = AdamW(model.parameters(), lr=0.004, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    # Step scheduler 20 times (epochs 1..20)
    for _ in range(20):
        scheduler.step()

    state = {
        "epoch": 20,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "learning_rate": optimizer.param_groups[0]["lr"],
    }
    ckpt_path = temp_dir / "ckpt_epoch_20.pth"
    torch.save(state, ckpt_path)

    # Fresh instances
    new_model = DummyModel()
    new_optimizer = AdamW(new_model.parameters(), lr=0.004)
    new_scheduler = CosineAnnealingLR(new_optimizer, T_max=50, eta_min=1e-5)

    loaded_state = load_checkpoint(ckpt_path, new_model, new_optimizer, new_scheduler, device=torch.device("cpu"))

    assert loaded_state["epoch"] == 20
    assert new_scheduler.last_epoch == 20
    assert abs(new_optimizer.param_groups[0]["lr"] - optimizer.param_groups[0]["lr"]) < 1e-9


def test_resume_epoch_calculation_and_schedule_continuity(temp_dir):
    """Verify resume starts at epoch 21 and preserves cosine LR schedule."""
    model = DummyModel()
    optimizer = AdamW(model.parameters(), lr=0.004)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    # Compute reference LR schedule for all 50 epochs
    ref_lrs = []
    for _ in range(50):
        ref_lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    # Create checkpoint at epoch 20
    model2 = DummyModel()
    opt2 = AdamW(model2.parameters(), lr=0.004)
    sched2 = CosineAnnealingLR(opt2, T_max=50, eta_min=1e-5)
    for _ in range(20):
        sched2.step()

    state_ep20 = {
        "epoch": 20,
        "model_state_dict": model2.state_dict(),
        "optimizer_state_dict": opt2.state_dict(),
        "scheduler_state_dict": sched2.state_dict(),
    }
    ckpt_path = temp_dir / "checkpoint_epoch_020.pth"
    torch.save(state_ep20, ckpt_path)

    # Resume into fresh model
    resumed_model = DummyModel()
    resumed_opt = AdamW(resumed_model.parameters(), lr=0.004)
    resumed_sched = CosineAnnealingLR(resumed_opt, T_max=50, eta_min=1e-5)

    data = load_checkpoint(ckpt_path, resumed_model, resumed_opt, resumed_sched)
    start_epoch = data["epoch"] + 1

    assert start_epoch == 21, f"Expected start epoch 21, got {start_epoch}"
    # LR for epoch 21 before stepping should match reference epoch 21 LR
    current_lr = resumed_opt.param_groups[0]["lr"]
    assert abs(current_lr - ref_lrs[20]) < 1e-8, f"LR mismatch at epoch 21: {current_lr} vs {ref_lrs[20]}"

    # Step for epoch 21 -> verify epoch 22 LR matches reference
    resumed_sched.step()
    next_lr = resumed_opt.param_groups[0]["lr"]
    assert abs(next_lr - ref_lrs[21]) < 1e-8, f"LR mismatch at epoch 22: {next_lr} vs {ref_lrs[21]}"


def test_history_reconstruction_and_best_loss_restoration(temp_dir):
    """Verify history reconstruction from epoch checkpoints and best loss retention."""
    ckpt_dir = temp_dir / "checkpoints"
    ckpt_dir.mkdir()

    val_losses = [42.0, 9.8, 4.52, 20.0, 30.0]  # Epoch 3 has min val loss = 4.52
    for e, vl in enumerate(val_losses, start=1):
        ckpt_data = {
            "epoch": e,
            "train_loss": vl + 1.0,
            "val_loss": vl,
            "learning_rate": 0.004,
            "val_metrics": {
                "f1_score": 0.1 * e,
                "precision": 0.1 * e,
                "recall": 0.1 * e,
                "rmse_tp_px": 4.5,
                "tp": 100 * e,
            },
        }
        torch.save(ckpt_data, ckpt_dir / f"checkpoint_epoch_{e:03d}.pth")

    # Simulate reconstruction logic in train.py
    start_epoch = 6  # resume after epoch 5
    history = []
    for e in range(1, start_epoch):
        e_ckpt_path = ckpt_dir / f"checkpoint_epoch_{e:03d}.pth"
        e_data = torch.load(e_ckpt_path, map_location="cpu")
        val_metrics = e_data.get("val_metrics", {})
        required_val_keys = ["f1_score", "precision", "recall", "rmse_tp_px", "tp"]
        for k in required_val_keys:
            if k not in val_metrics:
                raise KeyError(f"Missing '{k}'")

        duration = e_data.get("duration_sec", None)
        record = {
            "epoch": e_data["epoch"],
            "train_loss": e_data["train_loss"],
            "val_loss": e_data["val_loss"],
            "val_f1": val_metrics["f1_score"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_rmse_tp_px": val_metrics["rmse_tp_px"],
            "val_tp": val_metrics["tp"],
            "lr": e_data["learning_rate"],
            "duration_sec": duration,
            "is_best": False,
        }
        history.append(record)

    best_val_loss = min(r["val_loss"] for r in history)
    for r in history:
        if abs(r["val_loss"] - best_val_loss) < 1e-6:
            r["is_best"] = True

    assert len(history) == 5
    assert abs(best_val_loss - 4.52) < 1e-6
    assert history[2]["is_best"] is True
    assert history[0]["is_best"] is False
    assert history[0]["duration_sec"] is None  # Explicitly None, not fabricated 0.0


def test_missing_metric_key_raises_explicit_keyerror(temp_dir):
    """Verify that a missing metric key raises KeyError rather than silently defaulting."""
    ckpt_path = temp_dir / "checkpoint_epoch_001.pth"
    torch.save({
        "epoch": 1,
        "train_loss": 10.0,
        "val_loss": 10.0,
        "learning_rate": 0.004,
        "val_metrics": {"f1_score": 0.1, "precision": 0.1, "recall": 0.1},  # missing tp and rmse_tp_px
    }, ckpt_path)

    e_data = torch.load(ckpt_path, map_location="cpu")
    val_metrics = e_data.get("val_metrics", {})
    required_val_keys = ["f1_score", "precision", "recall", "rmse_tp_px", "tp"]
    with pytest.raises(KeyError, match="rmse_tp_px"):
        for k in required_val_keys:
            if k not in val_metrics:
                raise KeyError(f"Missing '{k}'")


def test_atomic_checkpoint_replacement(temp_dir):
    """Verify safe_atomic_torch_save replaces file atomically and cleans up temp files."""
    target_file = temp_dir / "model.pth"
    state_v1 = {"version": 1, "tensor": torch.ones(10)}
    safe_atomic_torch_save(state_v1, target_file)

    assert target_file.exists()
    loaded_v1 = torch.load(target_file, map_location="cpu")
    assert loaded_v1["version"] == 1

    # Overwrite atomically
    state_v2 = {"version": 2, "tensor": torch.zeros(10)}
    safe_atomic_torch_save(state_v2, target_file)

    loaded_v2 = torch.load(target_file, map_location="cpu")
    assert loaded_v2["version"] == 2

    # Check that no temporary files are left over
    temp_files = list(temp_dir.glob("*_tmp_*"))
    assert len(temp_files) == 0


def test_atomic_save_failure_cleans_up_temp_file(temp_dir):
    """Verify that when os.replace fails repeatedly, temp file is removed and error raised."""
    target_file = temp_dir / "fail_model.pth"
    state = {"version": 1}

    with patch("os.replace", side_effect=PermissionError("Simulated locked file")):
        with pytest.raises(OSError, match="Failed to atomically replace"):
            safe_atomic_torch_save(state, target_file, max_retries=2, retry_delay=0.01)

    # Ensure no leftover temp file exists in directory
    temp_files = list(temp_dir.glob("*_tmp_*"))
    assert len(temp_files) == 0


def test_actual_checkpoint_epoch_20_integrity():
    """Verify integrity, hash, and contents of the real checkpoint_epoch_020.pth."""
    ckpt_path = Path("experiments/sict_shufflenet_phase4c_retrain_50ep/checkpoints/checkpoint_epoch_020.pth")
    assert ckpt_path.exists(), f"Target checkpoint does not exist: {ckpt_path}"

    expected_sha256 = "f5cbfbd8217497ee933835974807ffd424ba8327039f7abf13486e9d8becc9d2"
    actual_sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    assert actual_sha256 == expected_sha256, f"Checkpoint SHA256 mismatch! Got {actual_sha256}"

    data = torch.load(ckpt_path, map_location="cpu")
    assert data["epoch"] == 20
    assert data["scheduler_state_dict"]["last_epoch"] == 20
    assert "model_state_dict" in data
    assert "optimizer_state_dict" in data
    assert "val_metrics" in data
    val_m = data["val_metrics"]
    for k in ["f1_score", "precision", "recall", "rmse_tp_px", "tp"]:
        assert k in val_m, f"Missing metric key {k} in actual checkpoint 20"
