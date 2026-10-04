"""
Unit Tests: Reproducibility & Determinism
========================================
Verifies that identical random seeds produce bitwise deterministic model initializations and outputs.
"""

import unittest
import torch

from configs.config import ProjectConfig
from src.utils.seed import seed_everything
from src.models.row_anchor_net import RowAnchorNet


class TestReproducibility(unittest.TestCase):
    def test_deterministic_forward_runs(self):
        cfg = ProjectConfig()

        seed_everything(42)
        m1 = RowAnchorNet(model_config=cfg.model, sict_config=cfg.sict, apply_sict_in_forward=False)
        x1 = torch.randn(2, 3, 288, 384)
        out1 = m1(x1)

        seed_everything(42)
        m2 = RowAnchorNet(model_config=cfg.model, sict_config=cfg.sict, apply_sict_in_forward=False)
        x2 = torch.randn(2, 3, 288, 384)
        out2 = m2(x2)

        diff = (out1 - out2).abs().max().item()
        self.assertEqual(diff, 0.0, f"Seeded runs differed by {diff}")


if __name__ == "__main__":
    unittest.main()
