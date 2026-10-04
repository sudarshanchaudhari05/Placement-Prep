"""
Ground-Plane Spline and Waypoint Path Generator
===============================================
Generates 3D ground truth paths compatible with camera projection and row-anchor labeling:
- Straight segments
- Gentle and tight curves
- Compound S-curves with minimum radius of curvature R >= 0.75 m (Scenario D)
- Tape width: 50 mm (0.05 m)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Path3D:
    waypoints_world: np.ndarray  # (N, 3): columns [X, Y, Z=0] in meters
    tangents: np.ndarray         # (N, 2): unit tangent vectors [dx, dy]
    curvatures: np.ndarray       # (N,): curvature kappa (1/m)
    tape_width_m: float = 0.05   # Standard industrial guide tape width (50 mm)
    path_type: str = "straight"


class SplinePathGenerator:
    """
    Generates continuous geometric paths on the shop-floor ground plane (Z = 0).
    Coordinates in robot/world frame:
      X: Lateral (meters)
      Y: Forward (meters, extending from 0.0 to preview horizon >= 2.5 m)
    """

    def __init__(self, tape_width_m: float = 0.05, step_m: float = 0.02):
        self.tape_width_m = tape_width_m
        self.step_m = step_m  # 2 cm waypoint resolution

    def generate_straight_path(
        self,
        length_m: float = 3.0,
        lateral_offset_m: float = 0.0,
        heading_angle_deg: float = 0.0,
    ) -> Path3D:
        """Generate straight path with optional lateral offset and heading angle."""
        y_vals = np.arange(0.1, length_m, self.step_m, dtype=np.float64)
        heading_rad = np.radians(heading_angle_deg)
        x_vals = lateral_offset_m + y_vals * np.tan(heading_rad)
        z_vals = np.zeros_like(y_vals)

        pts = np.stack([x_vals, y_vals, z_vals], axis=1)
        n = len(y_vals)
        tangents = np.tile([np.sin(heading_rad), np.cos(heading_rad)], (n, 1))
        curvatures = np.zeros(n, dtype=np.float64)

        return Path3D(
            waypoints_world=pts,
            tangents=tangents,
            curvatures=curvatures,
            tape_width_m=self.tape_width_m,
            path_type="straight",
        )

    def generate_curved_path(
        self,
        radius_m: float = 1.5,
        turn_direction: str = "left",  # 'left' or 'right'
        length_m: float = 3.0,
        lateral_offset_m: float = 0.0,
    ) -> Path3D:
        """
        Generate continuous constant-curvature circular arc.
        Curvature kappa = 1 / radius.
        """
        assert radius_m >= 0.5, "Radius must be at least 0.5 m"
        sign = -1.0 if turn_direction == "left" else 1.0

        y_vals = np.arange(0.1, length_m, self.step_m, dtype=np.float64)
        # Circular arc approximation: x(y) = lateral_offset + sign * (radius - sqrt(radius^2 - y^2))
        y_clamped = np.clip(y_vals, 0.0, radius_m * 0.95)
        x_vals = lateral_offset_m + sign * (radius_m - np.sqrt(radius_m ** 2 - y_clamped ** 2))
        z_vals = np.zeros_like(y_vals)

        pts = np.stack([x_vals, y_vals, z_vals], axis=1)
        # Tangents
        dy = np.gradient(y_vals)
        dx = np.gradient(x_vals)
        norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-8
        tangents = np.stack([dx / norm, dy / norm], axis=1)
        curvatures = np.full(len(y_vals), 1.0 / radius_m, dtype=np.float64)

        return Path3D(
            waypoints_world=pts,
            tangents=tangents,
            curvatures=curvatures,
            tape_width_m=self.tape_width_m,
            path_type=f"curve_{turn_direction}",
        )

    def generate_s_curve_path(
        self,
        min_radius_m: float = 0.75,
        length_m: float = 3.0,
        amplitude_m: float = 0.25,
        lateral_offset_m: float = 0.0,
    ) -> Path3D:
        """
        Generate compound S-curve with verified minimum radius of curvature R >= min_radius_m (0.75 m).
        x(y) = lateral_offset + A * sin(2 * pi * y / wavelength)
        Max curvature: kappa_max = A * (2*pi / L)^2 = 1 / R_min
        """
        # Determine wavelength to guarantee kappa_max <= 1 / min_radius_m
        # L = 2 * pi * sqrt(A * min_radius)
        wavelength_m = 2.0 * np.pi * np.sqrt(amplitude_m * min_radius_m)
        wavelength_m = max(wavelength_m, 1.8)  # Minimum feasible wavelength

        y_vals = np.arange(0.1, length_m, self.step_m, dtype=np.float64)
        omega = 2.0 * np.pi / wavelength_m
        x_vals = lateral_offset_m + amplitude_m * np.sin(omega * y_vals)
        z_vals = np.zeros_like(y_vals)

        pts = np.stack([x_vals, y_vals, z_vals], axis=1)

        # First and second derivatives
        dx = amplitude_m * omega * np.cos(omega * y_vals)
        dy = np.ones_like(y_vals)
        ddx = -amplitude_m * (omega ** 2) * np.sin(omega * y_vals)
        ddy = np.zeros_like(y_vals)

        # Exact curvature: kappa = |dx*ddy - dy*ddx| / (dx^2 + dy^2)^(3/2)
        curvatures = np.abs(-ddx) / np.power(1.0 + dx ** 2, 1.5)

        norm = np.sqrt(dx ** 2 + dy ** 2)
        tangents = np.stack([dx / norm, dy / norm], axis=1)

        return Path3D(
            waypoints_world=pts,
            tangents=tangents,
            curvatures=curvatures,
            tape_width_m=self.tape_width_m,
            path_type="s_curve",
        )
