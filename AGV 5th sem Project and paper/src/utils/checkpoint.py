"""
Model Checkpointing and Experiment State Management
===================================================
Saves and loads model weights, optimizer state, scheduler state,
full serialized configuration, and training metrics history.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import torch
import torch.nn as nn

from configs.config import ProjectConfig


import os
import time


def safe_atomic_torch_save(
    state: Dict[str, Any],
    target_path: Path,
    max_retries: int = 5,
    retry_delay: float = 0.2,
) -> Path:
    """
    Save PyTorch state dictionary atomically with bounded retry and cleanup.
    
    Saves to a temporary file in the same directory, then uses os.replace()
    to perform atomic update. Retries with backoff if Windows file locks occur.
    Ensures temp file cleanup on failure.
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.parent / f"{target_path.stem}_tmp_{os.getpid()}_{time.time_ns()}{target_path.suffix}"

    try:
        torch.save(state, tmp_path)
        last_err = None
        for attempt in range(max_retries):
            try:
                os.replace(tmp_path, target_path)
                return target_path
            except (PermissionError, OSError) as e:
                last_err = e
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
        raise OSError(
            f"Failed to atomically replace {target_path} after {max_retries} attempts: {last_err}"
        ) from last_err
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: Path,
    filename: str = "checkpoint.pth",
    is_best: bool = False,
    config: Optional[ProjectConfig] = None,
) -> Path:
    """
    Save training checkpoint atomically.
    
    Args:
        state: State dictionary containing 'epoch', 'model_state_dict', 'optimizer_state_dict', etc.
        checkpoint_dir: Target directory.
        filename: Checkpoint filename.
        is_best: If True, also saves a copy as 'best_model.pth'.
        config: Project configuration to serialize alongside checkpoint.
        
    Returns:
        Path to saved checkpoint.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    file_path = checkpoint_dir / filename
    safe_atomic_torch_save(state, file_path)

    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        safe_atomic_torch_save(state, best_path)

    # Save full configuration JSON for provenance
    if config is not None:
        config_path = checkpoint_dir / "config_snapshot.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)

    return file_path


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load checkpoint and restore weights into model and optimizer.
    
    Args:
        checkpoint_path: Path to .pth checkpoint.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        device: Device to map tensors to.
        
    Returns:
        Checkpoint dictionary.
    """
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
