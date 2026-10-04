"""
Camera Optical Model and Coordinate Projection
==============================================
Governed by the physical AGV specifications from research blueprint:
- Sensor: Sony IMX219 CSI camera module (640x480 resolution)
- Field of View (FoV): 62.2 deg horizontal
- Mounting Height: h = 0.24 m above shop floor
- Downward Pitch Angle: theta_pitch = 28.5 deg
- Dynamic Pitch Noise: Delta theta_p ~ N(0, 1.2 deg)
- Lookahead Tracking Preview Horizon: 1.2 m ahead of chassis bumper
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fov_h_deg: float = 62.2          # Horizontal field of view in degrees
    mounting_height_m: float = 0.24  # Height above floor in meters
    nominal_pitch_deg: float = 28.5  # Downward tilt angle in degrees
    bumper_offset_m: float = 0.20    # Forward offset of camera from wheel axle
    lookahead_horizon_m: float = 1.2 # Active forward tracking horizon

    @property
    def f_px(self) -> float:
        """Focal length in pixels derived from horizontal FoV."""
        fov_rad = np.radians(self.fov_h_deg)
        return float((self.width / 2.0) / np.tan(fov_rad / 2.0))

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """3x3 Camera Intrinsic Calibration Matrix K."""
        f = self.f_px
        return np.array([
            [f,   0.0, self.cx],
            [0.0, f,   self.cy],
            [0.0, 0.0, 1.0    ]
        ], dtype=np.float64)


class PinholeCameraModel:
    """
    Pinhole camera model with downward pitch tilt for ground-plane projections.
    World Coordinate Frame:
      X: Lateral (Right +)
      Y: Longitudinal (Forward + along floor)
      Z: Normal to ground (Up +)
      Ground plane: Z = 0
    """

    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()

    def get_rotation_matrix(self, pitch_offset_deg: float = 0.0) -> np.ndarray:
        """
        Compute camera rotation matrix from world/robot frame to camera optical frame.
        Total pitch angle = nominal_pitch + pitch_offset
        """
        total_pitch_rad = np.radians(self.config.nominal_pitch_deg + pitch_offset_deg)
        c = np.cos(total_pitch_rad)
        s = np.sin(total_pitch_rad)

        # Standard optical frame: X_c right, Y_c down, Z_c forward along optical axis
        # Robot frame: X right, Y forward, Z up
        # Rotation taking [X, Y, Z] to [X_c, Y_c, Z_c]:
        # X_c = X
        # Y_c = -Y*s - Z*c
        # Z_c =  Y*c - Z*s
        R = np.array([
            [ 1.0,  0.0,  0.0],
            [ 0.0,   -s,   -c],
            [ 0.0,    c,   -s]
        ], dtype=np.float64)
        return R

    def project_ground_points_to_image(
        self,
        points_xy: np.ndarray,
        pitch_offset_deg: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 2D ground coordinates (X, Y) where Z=0 into image (u, v) coordinates.
        
        Args:
            points_xy: Array of shape (N, 2), columns [X_world, Y_world] in meters.
            pitch_offset_deg: Simulated dynamic pitch vibration offset in degrees.
            
        Returns:
            uv_coords: Array of shape (N, 2), columns [u, v] in pixel space.
            valid_mask: Boolean array of shape (N,), True if point is in front of camera.
        """
        assert points_xy.ndim == 2 and points_xy.shape[1] == 2, "Expected (N, 2) array"
        n = points_xy.shape[0]

        # Robot/Camera translation: camera is at X=0, Y=bumper_offset, Z=mounting_height
        X = points_xy[:, 0]
        Y = points_xy[:, 1] - self.config.bumper_offset_m
        Z = -self.config.mounting_height_m  # Ground relative to camera optical center

        R = self.get_rotation_matrix(pitch_offset_deg)

        # Transform to camera optical frame: P_c = R * [X, Y, Z]^T
        pts_world_rel = np.stack([X, Y, np.full(n, Z)], axis=0)  # (3, N)
        pts_cam = R @ pts_world_rel  # (3, N)

        Xc = pts_cam[0, :]
        Yc = pts_cam[1, :]
        Zc = pts_cam[2, :]

        # Check positive depth (in front of camera lens)
        valid_depth = Zc > 0.05

        f = self.config.f_px
        cx = self.config.cx
        cy = self.config.cy

        u = np.full(n, np.nan, dtype=np.float32)
        v = np.full(n, np.nan, dtype=np.float32)

        u[valid_depth] = (f * (Xc[valid_depth] / Zc[valid_depth]) + cx).astype(np.float32)
        v[valid_depth] = (f * (Yc[valid_depth] / Zc[valid_depth]) + cy).astype(np.float32)

        uv_coords = np.stack([u, v], axis=1)
        return uv_coords, valid_depth

    def unproject_image_to_ground(
        self,
        uv_coords: np.ndarray,
        pitch_offset_deg: float = 0.0,
    ) -> np.ndarray:
        """
        Inverse perspective map: unprojects image pixel coordinates (u, v) back to ground (X, Y, Z=0).
        """
        assert uv_coords.ndim == 2 and uv_coords.shape[1] == 2, "Expected (N, 2) array"
        n = uv_coords.shape[0]

        f = self.config.f_px
        cx = self.config.cx
        cy = self.config.cy

        u = uv_coords[:, 0]
        v = uv_coords[:, 1]

        # Normalized camera ray: d_c = [(u - cx)/f, (v - cy)/f, 1]^T
        dx_c = (u - cx) / f
        dy_c = (v - cy) / f
        dz_c = np.ones(n, dtype=np.float64)

        rays_c = np.stack([dx_c, dy_c, dz_c], axis=0)  # (3, N)

        R = self.get_rotation_matrix(pitch_offset_deg)
        R_inv = R.T  # Orthogonal rotation inverse

        # Direction rays in robot frame
        rays_r = R_inv @ rays_c  # (3, N)
        Rx = rays_r[0, :]
        Ry = rays_r[1, :]
        Rz = rays_r[2, :]

        # Camera origin in robot frame: [0, bumper_offset, mounting_height]
        h = self.config.mounting_height_m
        # Ground plane is Z = 0 -> h + s * Rz = 0 -> s = -h / Rz
        # Valid when ray points down towards floor (Rz < 0)
        valid_intersect = Rz < -1e-4

        X_ground = np.full(n, np.nan, dtype=np.float32)
        Y_ground = np.full(n, np.nan, dtype=np.float32)

        s = -h / Rz[valid_intersect]
        X_ground[valid_intersect] = (s * Rx[valid_intersect]).astype(np.float32)
        Y_ground[valid_intersect] = (self.config.bumper_offset_m + s * Ry[valid_intersect]).astype(np.float32)

        return np.stack([X_ground, Y_ground], axis=1)
