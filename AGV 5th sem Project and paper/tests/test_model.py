"""
Unit Tests: Model Architecture & Parameter Verification
=======================================================
Verifies:
- Model output shape [B, 18, 101]
- Parameter count strictly 2,205,242
- CPU forward pass
- CUDA forward pass if available
- Loud assertions on invalid spatial dimensions or channel counts
"""

import unittest
import torch

from configs.config import ProjectConfig
from src.models.row_anchor_net import RowAnchorNet, build_model


class TestModelArchitecture(unittest.TestCase):
    def setUp(self):
        self.config = ProjectConfig()
        self.model = build_model(self.config, apply_sict_in_forward=False)

    def test_parameter_count(self):
        """Verify exact parameter count matching specification."""
        params = self.model.count_parameters()
        self.assertEqual(params["total"], 2205242)
        self.assertEqual(params["trainable"], 2205242)
        self.assertEqual(params["backbone"], 341792)
        self.assertEqual(params["head"], 1863450)

    def test_cpu_forward_pass(self):
        """Verify CPU forward pass and exact output shape [B, 18, 101]."""
        self.model.eval()
        x = torch.randn(2, 3, 288, 384)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(out.shape, (2, 18, 101))
        self.assertTrue(torch.isfinite(out).all())

    def test_cuda_forward_pass(self):
        """Verify CUDA forward pass if CUDA is available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        model_cuda = self.model.cuda()
        model_cuda.eval()
        x_cuda = torch.randn(2, 3, 288, 384, device="cuda")
        with torch.no_grad():
            out_cuda = model_cuda(x_cuda)
        self.assertEqual(out_cuda.shape, (2, 18, 101))
        self.assertEqual(out_cuda.device.type, "cuda")
        self.assertTrue(torch.isfinite(out_cuda).all())

    def test_loud_failure_on_invalid_dimensions(self):
        """Ensure model raises AssertionError when passed invalid dimensions."""
        self.model.eval()
        # Invalid spatial dimensions
        with self.assertRaises(AssertionError):
            self.model(torch.randn(2, 3, 256, 256))

        # Invalid channels
        with self.assertRaises(AssertionError):
            self.model(torch.randn(2, 1, 288, 384))

        # Invalid ndim
        with self.assertRaises(AssertionError):
            self.model(torch.randn(3, 288, 384))


if __name__ == "__main__":
    unittest.main()
