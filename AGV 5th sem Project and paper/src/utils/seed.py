"""
Reproducibility Seed Utility
============================
Ensures deterministic experiments across Python, NumPy, PyTorch, and CUDA.
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic_cuda: bool = True) -> None:
    """
    Set seed for reproducibility across all random number generators.
    
    Args:
        seed: Random seed value.
        deterministic_cuda: If True, forces cuDNN deterministic mode.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_cuda and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
