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


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: Path,
    filename: str = "checkpoint.pth",
    is_best: bool = False,
    config: Optional[ProjectConfig] = None,
) -> Path:
    """
    Save training checkpoint.
    
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
    torch.save(state, file_path)

    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(state, best_path)

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
