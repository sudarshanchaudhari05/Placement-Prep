"""
Unit Tests: Episode-Level Geometric Sanity Gate and Retry Limits
================================================================
Verifies:
1. Invalid trajectory (zero visible anchors) is rejected by geometric gate.
2. Valid trajectory (>= 3 visible anchors) passes the geometric gate.
3. Frame retry limit enforces termination after max_frame_attempts without infinite loops.
4. Determinism: identical seed produces identical trajectory choices and validation decisions.
5. Regression test: Episode 282 (Scenario B) recovers by regenerating trajectory rather than looping.
6. Geometry vs appearance distinction properly routes failures.
"""

import unittest
from pathlib import Path
import numpy as np

from configs.config import load_config
from src.simulation.camera import CameraConfig
from src.simulation.path_generator import SplinePathGenerator, Path3D
from src.simulation.capture import SyntheticCaptureEngine
from src.simulation.generator import SyntheticDatasetGenerator
from src.simulation.validator import SampleValidator
from src.simulation.domain_randomization import ScenarioType


class TestGeneratorSanityGate(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.generator = SyntheticDatasetGenerator(base_seed=1000)
        self.path_gen = SplinePathGenerator()
        self.capture_engine = SyntheticCaptureEngine(CameraConfig())

    def test_invalid_trajectory_rejected(self):
        """
        Test A: Deliberately create a trajectory with zero visible anchors in ROI
        and verify that the geometric sanity gate rejects it.
        """
        # A path placed far off to the side (X = 10.0 m) completely outside camera FOV
        far_path = self.path_gen.generate_straight_path(
            length_m=3.5,
            lateral_offset_m=10.0,
            heading_angle_deg=0.0,
        )
        is_valid, vis_anchors = self.generator.check_trajectory_visibility(far_path, min_visible_anchors=3)
        self.assertFalse(is_valid, "Expected far-off trajectory to be rejected by geometric sanity gate")
        self.assertEqual(vis_anchors, 0, f"Expected 0 visible anchors, got {vis_anchors}")

    def test_valid_trajectory_accepted(self):
        """
        Test B: Verify a standard nominal path passes the geometric sanity gate.
        """
        nominal_path = self.path_gen.generate_straight_path(
            length_m=3.5,
            lateral_offset_m=0.0,
            heading_angle_deg=0.0,
        )
        is_valid, vis_anchors = self.generator.check_trajectory_visibility(nominal_path, min_visible_anchors=3)
        self.assertTrue(is_valid, "Expected nominal trajectory to pass geometric sanity gate")
        self.assertGreaterEqual(vis_anchors, 3, f"Expected >= 3 visible anchors, got {vis_anchors}")

    def test_frame_retry_limit_no_infinite_loop(self):
        """
        Test C: Force repeated frame validation failure and verify that
        the generator stops after max_frame_attempts without an infinite loop.
        """
        # Temporarily set max_trajectory_attempts=2 and max_frame_attempts=3
        orig_max_traj = self.generator.max_trajectory_attempts
        orig_max_frame = self.generator.max_frame_attempts
        self.generator.max_trajectory_attempts = 2
        self.generator.max_frame_attempts = 3

        # Mock validator to simulate persistent appearance rejection
        def mock_always_invalid(img, row_anchors, metadata, frame_id, scenario, seed):
            return False, "Simulated persistent appearance corrupt rejection"

        self.generator.validator.validate_sample = mock_always_invalid

        # Verify that generating an episode raises RuntimeError after max attempts instead of looping
        with self.assertRaises(RuntimeError) as ctx:
            # Generate 1 sample (will attempt 1 episode)
            self.generator.generate_dataset(num_samples=1, progress_interval=100)

        self.assertIn("Failed to generate valid trajectory and frames", str(ctx.exception))

        # Restore original limits
        self.generator.max_trajectory_attempts = orig_max_traj
        self.generator.max_frame_attempts = orig_max_frame

    def test_determinism_identical_decisions(self):
        """
        Test D: Verify that identical master seed produces identical trajectory decisions.
        """
        gen1 = SyntheticDatasetGenerator(base_seed=42)
        gen2 = SyntheticDatasetGenerator(base_seed=42)

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        path1 = gen1._generate_scenario_trajectory(ScenarioType.SCENARIO_B, rng1)
        path2 = gen2._generate_scenario_trajectory(ScenarioType.SCENARIO_B, rng2)

        self.assertTrue(np.allclose(path1.waypoints_world, path2.waypoints_world))
        val1, cnt1 = gen1.check_trajectory_visibility(path1)
        val2, cnt2 = gen2.check_trajectory_visibility(path2)

        self.assertEqual(val1, val2)
        self.assertEqual(cnt1, cnt2)

    def test_episode_282_regression_recovery(self):
        """
        Test F: Specifically reproduce Episode-282-style geometry failure and verify
        that the generator rejects Attempt 0 and recovers by generating a valid trajectory on Attempt 1.
        """
        ep_id = 282
        scenario = ScenarioType.SCENARIO_B
        ep_seed = self.generator.base_seed + ep_id * 1000

        # Attempt 0: known to generate curved trajectory that exits FOV
        seed_attempt0 = ep_seed + 0 * 10007
        rng0 = np.random.default_rng(seed_attempt0)
        path0 = self.generator._generate_scenario_trajectory(scenario, rng0)
        valid0, vis0 = self.generator.check_trajectory_visibility(path0, min_visible_anchors=3)

        self.assertFalse(valid0, "Episode 282 Attempt 0 should have failed geometric sanity gate")
        self.assertEqual(vis0, 0, "Episode 282 Attempt 0 should have 0 visible anchors")

        # Attempt 1: next deterministic trajectory attempt
        seed_attempt1 = ep_seed + 1 * 10007
        rng1 = np.random.default_rng(seed_attempt1)
        path1 = self.generator._generate_scenario_trajectory(scenario, rng1)
        valid1, vis1 = self.generator.check_trajectory_visibility(path1, min_visible_anchors=3)

        self.assertTrue(valid1, "Episode 282 Attempt 1 should pass geometric sanity gate")
        self.assertGreaterEqual(vis1, 3, "Episode 282 Attempt 1 should have >= 3 visible anchors")

    def test_geometry_vs_appearance_rejection_detection(self):
        """
        Verify that SampleValidator correctly categorizes geometry vs appearance rejections.
        """
        self.assertTrue(SampleValidator.is_geometry_rejection(
            "All 18 row anchors are unexpectedly absent in a scene designed to have a visible path"
        ))
        self.assertTrue(SampleValidator.is_geometry_rejection(
            "Row 0 marked present but coordinate u=645.0 is out-of-bounds [0, 640)"
        ))
        self.assertTrue(SampleValidator.is_geometry_rejection(
            "Candidate trajectory has 1 visible anchors (< 3)"
        ))
        self.assertFalse(SampleValidator.is_geometry_rejection(
            "Invalid dimensions: (288, 384, 3) != (480, 640, 3)"
        ))
        self.assertFalse(SampleValidator.is_geometry_rejection(
            "Missing required metadata key: 'specular_exponent'"
        ))


if __name__ == "__main__":
    unittest.main()
