"""
Unit Tests: Structural Row-Anchor Loss
======================================
Tests:
1. all rows absent
2. all rows valid
3. one valid row
4. two valid rows
5. alternating valid/absent rows
6. highly curved path
7. straight path
8. backward gradient flow and finiteness
"""

import unittest
import torch

from configs.config import LossConfig, ModelConfig
from src.losses.structural_loss import (
    FocalLoss,
    SmoothnessLoss,
    CurvatureLoss,
    TotalRowAnchorLoss,
)


class TestStructuralLoss(unittest.TestCase):
    def setUp(self):
        self.loss_cfg = LossConfig(focal_gamma=2.0, lambda_smooth=0.5, lambda_curv=0.75)
        self.model_cfg = ModelConfig(num_row_anchors=18, num_spatial_bins=100)
        self.total_loss = TotalRowAnchorLoss(self.loss_cfg, self.model_cfg)
        self.b = 2
        self.m = 18
        self.c = 101

    def test_case_1_all_rows_absent(self):
        """Case 1: All rows are absent (class 100)."""
        logits = torch.randn(self.b, self.m, self.c, requires_grad=True)
        targets = torch.full((self.b, self.m), 100, dtype=torch.long)
        out = self.total_loss(logits, targets)

        self.assertTrue(torch.isfinite(out["loss_total"]))
        self.assertEqual(out["loss_smooth"].item(), 0.0)
        self.assertEqual(out["loss_curv"].item(), 0.0)

        out["loss_total"].backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any())

    def test_case_2_all_rows_valid(self):
        """Case 2: All rows validly detect path."""
        logits = torch.randn(self.b, self.m, self.c, requires_grad=True)
        targets = torch.randint(0, 100, (self.b, self.m), dtype=torch.long)
        out = self.total_loss(logits, targets)

        self.assertTrue(torch.isfinite(out["loss_total"]))
        self.assertGreaterEqual(out["loss_smooth"].item(), 0.0)
        self.assertGreaterEqual(out["loss_curv"].item(), 0.0)

    def test_case_3_one_valid_row(self):
        """Case 3: Exactly one valid row present (no pairs or triplets possible)."""
        logits = torch.randn(self.b, self.m, self.c)
        targets = torch.full((self.b, self.m), 100, dtype=torch.long)
        targets[:, 7] = 50  # Only row 7 is valid
        out = self.total_loss(logits, targets)

        self.assertTrue(torch.isfinite(out["loss_total"]))
        self.assertEqual(out["loss_smooth"].item(), 0.0)
        self.assertEqual(out["loss_curv"].item(), 0.0)

    def test_case_4_two_valid_adjacent_rows(self):
        """Case 4: Two valid adjacent rows (one pair, zero triplets)."""
        logits = torch.randn(self.b, self.m, self.c)
        targets = torch.full((self.b, self.m), 100, dtype=torch.long)
        targets[:, 7] = 45
        targets[:, 8] = 55
        out = self.total_loss(logits, targets)

        self.assertTrue(torch.isfinite(out["loss_total"]))
        self.assertGreaterEqual(out["loss_smooth"].item(), 0.0)
        self.assertEqual(out["loss_curv"].item(), 0.0)  # No triplet

    def test_case_5_alternating_valid_absent_rows(self):
        """Case 5: Alternating valid and absent rows (no adjacent pairs)."""
        logits = torch.randn(self.b, self.m, self.c)
        targets = torch.full((self.b, self.m), 100, dtype=torch.long)
        targets[:, 0::2] = 50  # Rows 0, 2, 4, ... are valid, 1, 3, 5, ... are absent
        out = self.total_loss(logits, targets)

        self.assertTrue(torch.isfinite(out["loss_total"]))
        self.assertEqual(out["loss_smooth"].item(), 0.0)
        self.assertEqual(out["loss_curv"].item(), 0.0)

    def test_case_6_highly_curved_path(self):
        """Case 6: Highly curved path (significant curvature penalty)."""
        # Create w_hat simulating sharp zigzag: 10, 80, 10, 80, ...
        w_curve = torch.tensor([[10.0 if i % 2 == 0 else 80.0 for i in range(18)]])
        curv_loss = CurvatureLoss()
        val = curv_loss(w_curve)
        self.assertGreater(val.item(), 50.0)

    def test_case_7_straight_path(self):
        """Case 7: Perfectly straight linear path (curvature penalty = 0)."""
        # Linear progression: w_i = 10 + 2 * i (second derivative is exactly 0)
        w_straight = torch.tensor([[10.0 + 2.0 * i for i in range(18)]])
        curv_loss = CurvatureLoss()
        val = curv_loss(w_straight)
        self.assertAlmostEqual(val.item(), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
