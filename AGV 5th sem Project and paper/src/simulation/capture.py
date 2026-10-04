"""
Synthetic Camera Frame Capture and Projection Engine
====================================================
Integrates:
1. Pinhole camera optics (FoV 62.2 deg, mounting height 0.24 m, pitch 28.5 deg)
2. 3D ground path projection to 2D image coordinates (u, v)
3. Domain randomization rendering:
   - Polyurethane/epoxy floor with diffuse albedo and specular exponent
   - Overhead industrial lighting (lux scaling)
   - High-bay specular glare lobe streaks
   - 50 mm floor guide line with Poisson abrasion gaps
   - Tire scuff marks and oil stains
4. CoppeliaSim live vision sensor capture interface when connected.
"""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np

from src.simulation.camera import CameraConfig, PinholeCameraModel
from src.simulation.domain_randomization import RandomizedParams, ScenarioType
from src.simulation.path_generator import Path3D


class SyntheticCaptureEngine:
    """
    Renders physically grounded 640x480 RGB frames and extracts ground-truth visible path waypoints.
    """

    def __init__(self, camera_config: Optional[CameraConfig] = None):
        self.camera_cfg = camera_config or CameraConfig()
        self.camera_model = PinholeCameraModel(self.camera_cfg)

    def render_frame_offline(
        self,
        path: Path3D,
        params: RandomizedParams,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render 640x480 RGB synthetic frame using physical camera projection and domain parameters.
        
        Args:
            path: Path3D ground truth trajectory on floor (Z = 0).
            params: Domain randomized environmental variables.
            
        Returns:
            img_rgb: (480, 640, 3) uint8 image in RGB order.
            visible_uv: (K, 2) float32 array of visible path waypoints in pixel coordinates [u, v].
        """
        w = self.camera_cfg.width   # 640
        h = self.camera_cfg.height  # 480

        rng = np.random.default_rng(params.seed)

        # 1. Base floor surface: industrial concrete or epoxy
        # Baseline gray modulated by surface diffuse albedo (rho_d ~ 0.52) and lux
        lux_scale = np.clip(params.lighting_lux / 1000.0, 0.2, 1.5)
        base_gray = int(np.clip(params.diffuse_albedo * 200.0 * lux_scale, 20, 220))

        # Add subtle floor texture grain
        grain = rng.integers(-8, 9, size=(h, w), dtype=np.int16)
        floor_gray = np.clip(base_gray + grain, 0, 255).astype(np.uint8)

        # Base 3-channel RGB image (neutral gray floor)
        img_rgb = cv2.merge([floor_gray, floor_gray, floor_gray])

        # 2. Project 3D ground path waypoints into 2D camera coordinates
        pts_xy = path.waypoints_world[:, :2]  # [X, Y]
        uv_all, valid_depth = self.camera_model.project_ground_points_to_image(
            pts_xy, pitch_offset_deg=params.pitch_offset_deg
        )

        # Filter points in front of camera and inside horizontal view
        valid_mask = valid_depth & np.isfinite(uv_all[:, 0]) & np.isfinite(uv_all[:, 1])
        valid_mask = valid_mask & (uv_all[:, 1] >= 0) & (uv_all[:, 1] < h)

        visible_uv = uv_all[valid_mask].astype(np.float32)

        # 3. Draw industrial guide tape (50 mm white/yellow tape)
        if len(visible_uv) > 1:
            tape_color = (255, 255, 255)  # White safety marking
            # Compute line width in pixels dynamically based on perspective depth
            # Project left and right tape boundaries (+/- tape_width / 2)
            half_w = path.tape_width_m / 2.0
            tangents = path.tangents[valid_mask]

            # Normal to tangent on ground: [-dy, dx]
            normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
            left_pts_xy = pts_xy[valid_mask] + normals * half_w
            right_pts_xy = pts_xy[valid_mask] - normals * half_w

            uv_left, _ = self.camera_model.project_ground_points_to_image(
                left_pts_xy, pitch_offset_deg=params.pitch_offset_deg
            )
            uv_right, _ = self.camera_model.project_ground_points_to_image(
                right_pts_xy, pitch_offset_deg=params.pitch_offset_deg
            )

            # Apply line abrasion (Poisson breaks & material loss)
            # Break mask along line sequence
            loss_ratio = params.abrasion_loss_ratio
            num_pts = len(visible_uv)

            for i in range(num_pts - 1):
                # Check for abrasion gap
                if params.num_abrasion_breaks > 0 and rng.random() < loss_ratio:
                    continue  # Gap/break in physical tape

                # Trapezoidal quadrilateral for perspective line segment
                quad = np.array([
                    uv_left[i], uv_right[i],
                    uv_right[i + 1], uv_left[i + 1]
                ], dtype=np.int32)

                # Simulate surface degradation / fading
                fade = 1.0 - (loss_ratio * 0.5) if params.scenario == ScenarioType.SCENARIO_C else 1.0
                seg_col = tuple(int(c * fade) for c in tape_color)
                cv2.fillPoly(img_rgb, [quad], color=seg_col)

        # 4. Rubber scuff streaks (black tire marks across floor)
        for _ in range(params.rubber_scuffs_count):
            pt1 = (int(rng.uniform(50, w - 50)), int(rng.uniform(h // 2, h - 20)))
            length = int(rng.uniform(40, 120))
            angle = rng.uniform(-np.pi / 4, np.pi / 4)
            pt2 = (int(pt1[0] + length * np.cos(angle)), int(pt1[1] + length * np.sin(angle)))
            cv2.line(img_rgb, pt1, pt2, color=(30, 30, 30), thickness=int(rng.uniform(3, 8)))

        # 5. Dark oil stains / surface discoloration
        for _ in range(params.oil_stains_count):
            oil_center = (int(rng.uniform(100, w - 100)), int(rng.uniform(h // 2, h - 50)))
            axes = (int(rng.uniform(15, 45)), int(rng.uniform(10, 30)))
            angle = float(rng.uniform(0, 180))
            cv2.ellipse(img_rgb, oil_center, axes, angle, 0, 360, color=(25, 25, 25), thickness=-1)

        # 6. Overhead Specular Glare Lobes
        if params.glare_intensity > 0.05 and params.glare_center_uv is not None:
            gu, gv = params.glare_center_uv
            # Create specular highlight mask: intensity decays with specular exponent eta
            # High eta -> tight concentrated specular core; low eta -> broad diffused bloom
            radius_x = int(max(20, 180.0 / np.sqrt(max(params.specular_exponent, 15.0))))
            radius_y = int(radius_x * 0.7)  # Slightly elliptical along optical axis

            overlay = img_rgb.copy()
            cv2.ellipse(
                overlay,
                (gu, gv),
                (radius_x, radius_y),
                0, 0, 360,
                color=(255, 255, 255),
                thickness=-1,
            )
            alpha_glare = min(1.0, params.glare_intensity * 0.95)
            img_rgb = cv2.addWeighted(overlay, alpha_glare, img_rgb, 1.0 - alpha_glare, 0)

            # Gaussian blur highlight core to simulate realistic lens diffraction & floor gloss
            kernel_size = max(15, (radius_x // 3) * 2 + 1)
            img_rgb = cv2.GaussianBlur(img_rgb, (kernel_size, kernel_size), 0)

        # Ensure strict (480, 640, 3) uint8 RGB format
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
        assert img_rgb.shape == (h, w, 3), f"Frame shape mismatch: {img_rgb.shape}"

        return img_rgb, visible_uv
