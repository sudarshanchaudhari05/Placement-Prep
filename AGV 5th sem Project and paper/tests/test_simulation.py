"""
Unit Tests: Simulation and Synthetic Generation Pipeline
========================================================
Verifies:
1. Camera optical projection and inverse ground mapping
2. Spline path generator (straight, curved, S-curves R >= 0.75 m)
3. Domain randomization sampling within research specification limits
4. SampleValidator quality control gate
5. CoppeliaSimClient diagnostics and remote API interface
6. Deterministic generation reproducibility
"""

import unittest
from pathlib import Path
import numpy as np

from src.simulation.camera import CameraConfig, PinholeCameraModel
from src.simulation.path_generator import SplinePathGenerator
from src.simulation.domain_randomization import DomainRandomizer, ScenarioType
from src.simulation.validator import SampleValidator
from src.simulation.client import CoppeliaSimClient
from src.simulation.capture import SyntheticCaptureEngine


class TestSimulationPipeline(unittest.TestCase):
    def setUp(self):
        self.camera_cfg = CameraConfig()
        self.camera_model = PinholeCameraModel(self.camera_cfg)
        self.path_gen = SplinePathGenerator()
        self.randomizer = DomainRandomizer()
        self.capture_engine = SyntheticCaptureEngine(self.camera_cfg)

    def test_camera_projection_and_unprojection(self):
        """Verify ground projection and unprojection roundtrip."""
        # Test point on floor: X = 0.1 m, Y = 1.0 m, Z = 0 m
        pts_ground = np.array([[0.1, 1.0]], dtype=np.float64)
        uv, valid = self.camera_model.project_ground_points_to_image(pts_ground, pitch_offset_deg=0.0)

        self.assertTrue(valid[0])
        u, v = uv[0]
        self.assertTrue(0.0 <= u < 640.0)
        self.assertTrue(0.0 <= v < 480.0)

        # Unproject back to ground
        pts_recon = self.camera_model.unproject_image_to_ground(uv, pitch_offset_deg=0.0)
        self.assertAlmostEqual(pts_recon[0, 0], 0.1, places=3)
        self.assertAlmostEqual(pts_recon[0, 1], 1.0, places=3)

    def test_s_curve_minimum_radius(self):
        """Verify S-curve generator strictly respects R >= 0.75 m (kappa <= 1.333 1/m)."""
        s_path = self.path_gen.generate_s_curve_path(min_radius_m=0.75, length_m=3.0)
        max_curv = float(np.max(s_path.curvatures))
        min_radius = 1.0 / max_curv if max_curv > 0 else float("inf")
        self.assertGreaterEqual(min_radius, 0.74, f"S-curve radius {min_radius:.3f} m below 0.75 m")

    def test_domain_randomization_bounds(self):
        """Verify domain variables fall within research specification bounds."""
        for sc in ScenarioType:
            params = self.randomizer.sample_parameters(scenario=sc, seed=42)
            # Specular exponent eta in [15, 120]
            self.assertTrue(15.0 <= params.specular_exponent <= 120.0)
            # Diffuse albedo in [0.25, 0.80]
            self.assertTrue(0.25 <= params.diffuse_albedo <= 0.80)
            # Pitch offset in [-3.0, 3.0] deg
            self.assertTrue(-3.0 <= params.pitch_offset_deg <= 3.0)

    def test_sample_validator_gate(self):
        """Verify SampleValidator rejects invalid frames and accepts valid ones."""
        val = SampleValidator(log_path=Path("tests_rejected_log.json"))

        valid_img = np.zeros((480, 640, 3), dtype=np.uint8)
        valid_anchors = [
            {"row_idx": i, "present": True, "u_coord": 320.0, "class_id": 50}
            for i in range(18)
        ]
        meta = {
            "seed": 42,
            "scenario": "nominal",
            "specular_exponent": 40.0,
            "diffuse_albedo": 0.52,
            "lighting_lux": 400.0,
        }

        # Valid sample
        is_val, reason = val.validate_sample(valid_img, valid_anchors, meta, 0, "nominal", 42)
        self.assertTrue(is_val, f"Valid sample rejected: {reason}")

        # Invalid dimensions
        invalid_img = np.zeros((288, 384, 3), dtype=np.uint8)
        is_val, reason = val.validate_sample(invalid_img, valid_anchors, meta, 1, "nominal", 42)
        self.assertFalse(is_val)

        # Invalid class ID (> 99 for present row)
        bad_anchors = list(valid_anchors)
        bad_anchors[0] = {"row_idx": 0, "present": True, "u_coord": 320.0, "class_id": 105}
        is_val, reason = val.validate_sample(valid_img, bad_anchors, meta, 2, "nominal", 42)
        self.assertFalse(is_val)

    def test_coppeliasim_client_interface(self):
        """Verify CoppeliaSim client diagnostic methods without crashing."""
        client = CoppeliaSimClient(host="localhost", port=23000)
        self.assertFalse(client.is_connected)
        # connect() should return False gracefully if server is offline or package uninstalled
        conn = client.connect()
        self.assertFalse(conn)

    def test_deterministic_frame_generation(self):
        """Verify identical seed generates bitwise identical frames."""
        path = self.path_gen.generate_straight_path(length_m=3.0)
        params1 = self.randomizer.sample_parameters(ScenarioType.SCENARIO_A, seed=123)
        params2 = self.randomizer.sample_parameters(ScenarioType.SCENARIO_A, seed=123)

        img1, uv1 = self.capture_engine.render_frame_offline(path, params1)
        img2, uv2 = self.capture_engine.render_frame_offline(path, params2)

        diff = np.max(np.abs(img1.astype(np.int16) - img2.astype(np.int16)))
        self.assertEqual(diff, 0, f"Frame generation non-deterministic! Diff: {diff}")
        self.assertTrue(np.allclose(uv1, uv2))


if __name__ == "__main__":
    unittest.main()
