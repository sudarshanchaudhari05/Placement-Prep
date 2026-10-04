"""
Unit Tests: Specular-Invariant Chromaticity Transform (SICT)
============================================================
Verifies:
A. normal non-glare pixel
B. bright/specular pixel
C. dark pixel
D. constant-color image
E. zero-near RGB values
F. regression test ensuring ambient intensity uses raw RGB intensity, NOT chromaticity
NumPy and PyTorch implementations, differentiability, epsilon stability.
"""

import unittest
import numpy as np
import torch

from configs.config import SICTConfig
from src.sict.sict_filter import sict_transform_np, sict_transform_torch, SICTModule


class TestSICT(unittest.TestCase):
    def setUp(self):
        self.config = SICTConfig(
            gamma_lum=220.0,
            delta_spec=0.05,
            alpha=0.6,
            beta=0.4,
            epsilon=1e-5,
            ambient_ratio=0.2,
            frame_avg_mode="raw_intensity",
        )

    def test_case_a_normal_nonglare_pixel(self):
        """A. Normal non-glare colored pixel (e.g. yellow floor marking R=255, G=215, B=0)."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :] = [255, 215, 0]  # Strong yellow

        out = sict_transform_np(img, config=self.config, out_channels=3)
        # Expected chromaticities
        S = 255.0 + 215.0 + 0.0
        r = 255.0 / (S + 1e-5)
        g = 215.0 / (S + 1e-5)
        diff_r = r - (1.0 / 3.0)
        diff_g = g - (1.0 / 3.0)
        phi = np.sqrt(diff_r ** 2 + diff_g ** 2)
        self.assertGreater(phi, self.config.delta_spec)  # Phi > 0.05 (non-specular)

        expected_val = self.config.alpha * r + self.config.beta * g
        self.assertAlmostEqual(float(out[5, 5, 0]), expected_val, places=4)

    def test_case_b_bright_specular_pixel(self):
        """B. Bright specular glare pixel (R=255, G=255, B=255) replaced by raw I_ambient."""
        # Test with a non-specular background (e.g. green floor R=40, G=140, B=40)
        img_bg = np.zeros((10, 10, 3), dtype=np.uint8)
        img_bg[:, :] = [40, 140, 40]  # Greenish floor: S=220, r=40/220=0.18, g=140/220=0.63 -> Phi >> 0.05
        img_bg[4:6, 4:6] = 255  # Glare patch (4 pixels)

        out_bg = sict_transform_np(img_bg, config=self.config, out_channels=3)
        glare_out = float(out_bg[4, 4, 0])
        normal_out = float(out_bg[0, 0, 0])

        # Glare must be attenuated to I_ambient = ambient_ratio * mean(S / 765.0)
        img_f = img_bg.astype(np.float32)
        S_map = img_f[:, :, 0] + img_f[:, :, 1] + img_f[:, :, 2]
        expected_raw_ambient = self.config.ambient_ratio * float(np.mean(S_map / 765.0))

        self.assertLess(glare_out, normal_out)
        self.assertAlmostEqual(glare_out, expected_raw_ambient, places=4)

    def test_case_c_dark_pixel(self):
        """C. Dark pixel (R=10, G=10, B=10) with S < Gamma_lum."""
        img = np.full((10, 10, 3), 10, dtype=np.uint8)
        # S = 30 < 220. Condition S >= 220 fails, so it is treated as base response
        out = sict_transform_np(img, config=self.config, out_channels=3)
        expected = self.config.alpha * (1.0 / 3.0) + self.config.beta * (1.0 / 3.0)  # 1/3
        self.assertAlmostEqual(float(out[5, 5, 0]), expected, places=3)

    def test_case_d_constant_color_image(self):
        """D. Constant color image (e.g. uniform gray or uniform blue)."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:, :] = [0, 0, 200]  # Pure blue floor
        out = sict_transform_np(img, config=self.config, out_channels=3)
        self.assertFalse(np.isnan(out).any())
        self.assertFalse(np.isinf(out).any())
        self.assertTrue(0.0 <= out.min() and out.max() <= 1.0)

    def test_case_e_zero_near_rgb_values(self):
        """E. Zero and near-zero RGB values (numerical stability check)."""
        img = np.zeros((10, 10, 3), dtype=np.float32)
        out = sict_transform_np(img, config=self.config, out_channels=3)
        self.assertFalse(np.isnan(out).any())
        self.assertFalse(np.isinf(out).any())
        self.assertEqual(float(out.max()), 0.0)

        # Microscopic epsilon test
        img_micro = np.full((5, 5, 3), 1e-9, dtype=np.float32)
        out_micro = sict_transform_np(img_micro, config=self.config, out_channels=3)
        self.assertFalse(np.isnan(out_micro).any())
        self.assertFalse(np.isinf(out_micro).any())

    def test_regression_ambient_intensity_uses_raw_rgb_not_chromaticity(self):
        """
        REGRESSION TEST:
        Verifies that I_ambient is strictly proportional to raw RGB lighting intensity,
        and NOT invariant chromatic response mean(alpha*r + beta*g).
        
        Frame 1: Dim industrial floor (R=20, G=20, B=20), plus glare spot
        Frame 2: Bright industrial floor (R=180, G=180, B=180), plus glare spot
        Both floors have identical neutral chromaticity (r=1/3, g=1/3).
        Under chromatic averaging, both would yield identical ambient replacement (~0.0667).
        Under raw intensity averaging, Frame 2 must produce 9x higher ambient replacement.
        """
        # Dim frame (10x10)
        dim_frame = np.full((10, 10, 3), 20, dtype=np.uint8)
        dim_frame[4:6, 4:6] = 255  # Glare spot

        # Bright frame (10x10)
        bright_frame = np.full((10, 10, 3), 180, dtype=np.uint8)
        bright_frame[4:6, 4:6] = 255  # Glare spot

        out_dim = sict_transform_np(dim_frame, config=self.config, out_channels=1)
        out_bright = sict_transform_np(bright_frame, config=self.config, out_channels=1)

        dim_ambient_val = float(out_dim[5, 5, 0])
        bright_ambient_val = float(out_bright[5, 5, 0])

        # Raw intensity in bright frame must yield much higher ambient value than dim frame
        self.assertGreater(bright_ambient_val, dim_ambient_val * 4.0)

        # Explicitly verify the formula matches mean(S / 765.0) * 0.2
        expected_dim_amb = 0.2 * float(np.mean(dim_frame.astype(np.float32).sum(axis=-1) / 765.0))
        expected_bright_amb = 0.2 * float(np.mean(bright_frame.astype(np.float32).sum(axis=-1) / 765.0))

        self.assertAlmostEqual(dim_ambient_val, expected_dim_amb, places=4)
        self.assertAlmostEqual(bright_ambient_val, expected_bright_amb, places=4)

        # Confirm that if wrong chromaticity mode was used, it would fail this test
        chromatic_config = SICTConfig(frame_avg_mode="chromatic")
        wrong_dim_amb = float(sict_transform_np(dim_frame, config=chromatic_config, out_channels=1)[5, 5, 0])
        wrong_bright_amb = float(sict_transform_np(bright_frame, config=chromatic_config, out_channels=1)[5, 5, 0])
        # In chromatic mode, both floors with equal-energy neutral color yield identical wrong values
        self.assertAlmostEqual(wrong_dim_amb, wrong_bright_amb, places=2)
        # And the true bright ambient value differs significantly from the wrong chromatic value
        self.assertNotAlmostEqual(bright_ambient_val, wrong_bright_amb, places=2)

    def test_pytorch_differentiability(self):
        """Verify differentiability of PyTorch SICT implementation."""
        x = (torch.rand((2, 3, 20, 20), dtype=torch.float32) * 255.0).detach().requires_grad_(True)
        out = sict_transform_torch(x, config=self.config, out_channels=3)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any(), "NaN found in SICT gradient!")
        self.assertFalse(torch.isinf(x.grad).any(), "Inf found in SICT gradient!")
        self.assertGreater(x.grad.norm().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
