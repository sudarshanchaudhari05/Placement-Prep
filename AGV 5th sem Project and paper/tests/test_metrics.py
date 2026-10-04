"""
Unit Tests: Evaluation Metrics (F1 with 8px tolerance, RMSE, Precision, Recall)
===============================================================================
"""

import unittest
import numpy as np

from src.evaluation.metrics import compute_row_anchor_metrics


class TestMetrics(unittest.TestCase):
    def test_perfect_prediction(self):
        """All rows predicted present and exact coordinates match GT."""
        gt_coords = np.full((1, 18), 320.0, dtype=np.float32)
        gt_present = np.ones((1, 18), dtype=bool)

        pred_coords = np.full((1, 18), 320.0, dtype=np.float32)
        pred_present = np.ones((1, 18), dtype=bool)

        m = compute_row_anchor_metrics(pred_coords, pred_present, gt_coords, gt_present, coord_tolerance_px=8.0)
        self.assertEqual(m["tp"], 18)
        self.assertEqual(m["fp"], 0)
        self.assertEqual(m["fn"], 0)
        self.assertEqual(m["f1_score"], 1.0)
        self.assertEqual(m["rmse_tp_px"], 0.0)

    def test_tolerance_boundary_8px(self):
        """Test exact 8.0px tolerance boundary."""
        gt_coords = np.array([[320.0, 320.0]], dtype=np.float32)
        gt_present = np.array([[True, True]], dtype=bool)

        # First row is exactly 8.0 px off (TP), second is 8.1 px off (FP + FN)
        pred_coords = np.array([[328.0, 328.1]], dtype=np.float32)
        pred_present = np.array([[True, True]], dtype=bool)

        m = compute_row_anchor_metrics(pred_coords, pred_present, gt_coords, gt_present, coord_tolerance_px=8.0)
        self.assertEqual(m["tp"], 1)
        self.assertEqual(m["fp"], 1)
        self.assertEqual(m["fn"], 1)
        self.assertAlmostEqual(m["rmse_tp_px"], 8.0, places=4)

    def test_absence_handling(self):
        """Test true negative and false positive on absent rows."""
        gt_coords = np.array([[-1.0, -1.0]], dtype=np.float32)
        gt_present = np.array([[False, False]], dtype=bool)

        # Model correctly predicts absence on row 0 (TN), falsely predicts path on row 1 (FP)
        pred_coords = np.array([[0.0, 320.0]], dtype=np.float32)
        pred_present = np.array([[False, True]], dtype=bool)

        m = compute_row_anchor_metrics(pred_coords, pred_present, gt_coords, gt_present, coord_tolerance_px=8.0)
        self.assertEqual(m["tn"], 1)
        self.assertEqual(m["fp"], 1)
        self.assertEqual(m["tp"], 0)
        self.assertEqual(m["fn"], 0)


if __name__ == "__main__":
    unittest.main()
