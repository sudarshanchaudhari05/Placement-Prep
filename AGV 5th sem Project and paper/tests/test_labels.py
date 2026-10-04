"""
Unit Tests: Row-Anchor Label Encoder and Decoder
=================================================
Verifies boundary conditions:
- u = 0
- u near 640 (u = 639.9 vs u = 640.0 boundary)
- negative / out-of-frame u
- no visible path
- partially visible path
- inverse consistency of soft-argmax coordinate reconstruction
- Comprehensive round-trip test on representative coordinates:
  u in {0, 1, 3.2, 100, 320, 639, 639.9, 640}
"""

import unittest
import numpy as np
import torch

from configs.config import ProjectConfig
from src.labels.coder import (
    encode_row_anchor_targets,
    decode_row_anchor_predictions,
    get_anchor_row_y_coords,
    RowAnchorCoder,
)


class TestLabelCoding(unittest.TestCase):
    def setUp(self):
        self.config = ProjectConfig()
        self.coder = RowAnchorCoder(self.config)
        self.row_y = get_anchor_row_y_coords(18, 200, 470)

    def test_u_zero_boundary(self):
        """Path at exact left boundary u = 0.0."""
        waypoints = np.array([[0.0, float(y)] for y in self.row_y], dtype=np.float32)
        classes, coords, present = encode_row_anchor_targets(
            waypoints, num_rows=18, y_min=200, y_max=470, image_width=640, num_spatial_bins=100
        )
        self.assertTrue(present.all())
        self.assertEqual(list(classes), [0] * 18)
        self.assertEqual(list(coords), [0.0] * 18)

    def test_u_near_640_boundary(self):
        """Path near right boundary: u = 639.9 (in-frame bin 99) vs u = 640.0 (out-of-frame absence)."""
        # u = 639.9 is inside column 639 -> bin 99
        waypoints_639_9 = np.array([[639.9, float(y)] for y in self.row_y], dtype=np.float32)
        classes_639_9, coords_639_9, present_639_9 = encode_row_anchor_targets(
            waypoints_639_9, num_rows=18, y_min=200, y_max=470, image_width=640, num_spatial_bins=100
        )
        self.assertTrue(present_639_9.all())
        self.assertEqual(list(classes_639_9), [99] * 18)

        # Boundary u = 640.0 has crossed the canvas edge (valid range is [0.0, 640.0))
        # Evaluates strictly to absence class 100
        waypoints_640 = np.array([[640.0, float(y)] for y in self.row_y], dtype=np.float32)
        classes_640, coords_640, present_640 = encode_row_anchor_targets(
            waypoints_640, num_rows=18, y_min=200, y_max=470, image_width=640, num_spatial_bins=100
        )
        self.assertFalse(present_640.any())
        self.assertEqual(list(classes_640), [100] * 18)  # Class 100 = Absence

    def test_negative_and_out_of_frame_u(self):
        """Path completely out of frame horizontally."""
        # Negative u (-50.0)
        waypoints_neg = np.array([[-50.0, float(y)] for y in self.row_y], dtype=np.float32)
        classes_neg, _, present_neg = encode_row_anchor_targets(
            waypoints_neg, num_rows=18, y_min=200, y_max=470, image_width=640, num_spatial_bins=100
        )
        self.assertFalse(present_neg.any())
        self.assertEqual(list(classes_neg), [100] * 18)  # All absence class 100

        # Beyond width (750.0)
        waypoints_over = np.array([[750.0, float(y)] for y in self.row_y], dtype=np.float32)
        classes_over, _, present_over = encode_row_anchor_targets(
            waypoints_over, num_rows=18, y_min=200, y_max=470, image_width=640, num_spatial_bins=100
        )
        self.assertFalse(present_over.any())
        self.assertEqual(list(classes_over), [100] * 18)

    def test_no_visible_path(self):
        """Empty waypoint array -> all rows absent."""
        waypoints_empty = np.zeros((0, 2), dtype=np.float32)
        classes, coords, present = encode_row_anchor_targets(
            waypoints_empty, num_rows=18, y_min=200, y_max=470, image_width=640, num_spatial_bins=100
        )
        self.assertFalse(present.any())
        self.assertEqual(list(classes), [100] * 18)

    def test_partially_visible_path(self):
        """Path only visible in bottom half of ROI."""
        waypoints_partial = np.array(
            [[320.0, float(self.row_y[r])] for r in range(12, 18)], dtype=np.float32
        )
        classes, coords, present = encode_row_anchor_targets(
            waypoints_partial, num_rows=18, y_min=200, y_max=470, image_width=640, num_spatial_bins=100
        )
        # Top 12 rows absent (class 100)
        self.assertFalse(present[:12].any())
        self.assertEqual(list(classes[:12]), [100] * 12)
        # Bottom 6 rows present (bin 50 for u=320)
        self.assertTrue(present[12:].all())
        self.assertEqual(list(classes[12:]), [50] * 6)

    def test_representative_coordinates_roundtrip(self):
        """
        Comprehensive round-trip test on representative coordinates:
        u in {0, 1, 3.2, 100, 320, 639, 639.9, 640}.
        
        Verifies:
        - Spatial bin vs Absence class
        - Reconstructed pixel coordinate via soft-argmax
        - Maximum reconstruction error (bounded by bin resolution: 6.4 px / 2 = 3.2 px)
        """
        test_cases = [
            # (u_input, expected_bin, expected_absence, expected_recon_u)
            (0.0, 0, False, 3.2),          # Left edge of bin 0
            (1.0, 0, False, 3.2),          # Inside bin 0
            (3.2, 0, False, 3.2),          # Exact center of bin 0
            (100.0, 15, False, 99.2),      # floor(100/640 * 100) = floor(15.625) = 15; center = 15.5*6.4 = 99.2
            (320.0, 50, False, 323.2),     # floor(320/640 * 100) = floor(50.0) = 50; center = 50.5*6.4 = 323.2
            (639.0, 99, False, 636.8),     # floor(639/640 * 100) = floor(99.84) = 99; center = 99.5*6.4 = 636.8
            (639.9, 99, False, 636.8),     # Inside bin 99
            (640.0, 100, True, None),      # Boundary: outside [0.0, 640.0) -> Absence class 100
        ]

        errors = []

        for u_val, exp_bin, exp_absence, exp_recon in test_cases:
            wp = np.array([[u_val, float(self.row_y[0])]], dtype=np.float32)
            classes, coords, present = encode_row_anchor_targets(
                wp, num_rows=18, y_min=200, y_max=470,
                image_width=640, num_spatial_bins=100
            )

            if exp_absence:
                self.assertFalse(present[0], f"Coordinate {u_val} was expected to be ABSENT")
                self.assertEqual(classes[0], 100, f"Coordinate {u_val} must map to absence class 100")
            else:
                self.assertTrue(present[0], f"Coordinate {u_val} was expected to be PRESENT")
                self.assertEqual(classes[0], exp_bin, f"Coordinate {u_val} mapped to bin {classes[0]} instead of {exp_bin}")

                # Simulate one-hot model prediction at the encoded bin to test decoder
                logits = torch.full((1, 1, 101), -20.0)
                logits[0, 0, exp_bin] = 20.0  # Sharp one-hot activation
                pred_coords, is_pres, w_hat = decode_row_anchor_predictions(
                    logits, image_width=640, num_spatial_bins=100
                )

                recon_u = pred_coords[0, 0].item()
                self.assertAlmostEqual(recon_u, exp_recon, places=2)

                err = abs(recon_u - u_val)
                errors.append((u_val, exp_bin, recon_u, err))
                # Error must be <= half-bin width (3.2 px) and strictly <= 8.0 px F1 tolerance
                self.assertLessEqual(err, 3.2001, f"Error {err:.2f} exceeded bin half-width 3.2 px for u={u_val}")
                self.assertLessEqual(err, 8.0, f"Error {err:.2f} exceeded F1 tolerance 8.0 px for u={u_val}")

        max_err = max(e[3] for e in errors)
        print(f"\n[Round-Trip Test] Max coordinate reconstruction error across in-frame points: {max_err:.2f} px (tolerance = 8.0 px)")
        for u_in, b, u_out, err in errors:
            print(f"  u={u_in:5.1f} -> Bin {b:2d} -> Recon u={u_out:5.1f} (error = {err:4.2f} px)")


if __name__ == "__main__":
    unittest.main()
