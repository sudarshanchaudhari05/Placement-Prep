"""
Row-Anchor Coordinate Encoder and Decoder
=========================================
Encodes continuous horizontal coordinates into spatial grid cells and absence tokens.
Decodes network logits into continuous path coordinates via soft-argmax.

Formulations:
Grid cell mapping (horizontal):
    k = floor((u / 640) * 100)
    where u in [0, 640), k in {0, ..., 99}.
    Absence class index = 100 (in 0-indexed PyTorch classification).
    (Corresponds to class 101 in 1-based indexing).

Absence condition:
    If no path waypoint falls within +/- 7.5 pixels of an anchor row,
    assign absence class 100.

Soft-argmax coordinate decoding:
    w_hat_i = sum_{j=0}^{K-1} j * sigma(P_{i,j})
    u_hat_i = ((w_hat_i + 0.5) / 100) * 640
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch

from configs.config import ProjectConfig, ModelConfig, ROIConfig, CameraConfig


def get_anchor_row_y_coords(
    num_rows: int = 18,
    y_min: int = 200,
    y_max: int = 470,
) -> np.ndarray:
    """
    Generate the y-pixel coordinates of the M horizontal anchor rows in the original frame.
    
    Args:
        num_rows: Number of anchor rows (M = 18).
        y_min: Top boundary of ROI (200).
        y_max: Bottom boundary of ROI (470).
        
    Returns:
        np.ndarray of shape (num_rows,) with integer y-coordinates.
    """
    assert num_rows > 1, "Must have at least 2 anchor rows"
    return np.linspace(y_min, y_max, num_rows).round().astype(np.int32)


def encode_row_anchor_targets(
    waypoints_uv: np.ndarray,
    num_rows: int = 18,
    y_min: int = 200,
    y_max: int = 470,
    image_width: int = 640,
    num_spatial_bins: int = 100,
    waypoint_tolerance_px: float = 7.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode path waypoints (u, v) into row-anchor grid cell targets.
    
    Args:
        waypoints_uv: Array of path points of shape (N, 2), where columns are [u, v].
                      u in [0, image_width], v in [0, image_height].
        num_rows: M = 18 horizontal anchor rows.
        y_min: ROI top boundary (200).
        y_max: ROI bottom boundary (470).
        image_width: Original image width (640).
        num_spatial_bins: K = 100 spatial bins.
        waypoint_tolerance_px: 7.5 px tolerance for waypoint association.
        
    Returns:
        target_classes: np.ndarray of shape (M,), int64. Values in [0, 99] or 100 (absence).
        target_u_coords: np.ndarray of shape (M,), float32. Exact u or NaN if absent.
        is_present: np.ndarray of shape (M,), bool. True if path present at this row.
    """
    row_y_coords = get_anchor_row_y_coords(num_rows, y_min, y_max)
    absence_class = num_spatial_bins  # 100

    target_classes = np.full(num_rows, absence_class, dtype=np.int64)
    target_u_coords = np.full(num_rows, np.nan, dtype=np.float32)
    is_present = np.zeros(num_rows, dtype=bool)

    if waypoints_uv is None or len(waypoints_uv) == 0:
        return target_classes, target_u_coords, is_present

    u_points = waypoints_uv[:, 0]
    v_points = waypoints_uv[:, 1]

    for i, row_y in enumerate(row_y_coords):
        # Distance to all waypoints
        vert_dist = np.abs(v_points - row_y)
        valid_indices = np.where(vert_dist <= waypoint_tolerance_px)[0]

        if len(valid_indices) > 0:
            # Pick closest waypoint vertically
            best_idx = valid_indices[np.argmin(vert_dist[valid_indices])]
            u_val = float(u_points[best_idx])

            # Valid continuous horizontal image domain is [0.0, image_width) (pixel columns 0..639).
            # Any coordinate u >= image_width or u < 0.0 has crossed the sensor canvas boundary
            # and is assigned the absence class 100 (token 101).
            if 0.0 <= u_val < image_width:
                # Discretize: k = floor((u / 640) * 100) -> valid range {0, ..., 99}
                grid_k = int(np.floor((u_val / image_width) * num_spatial_bins))
                grid_k = min(max(grid_k, 0), num_spatial_bins - 1)

                target_classes[i] = grid_k
                target_u_coords[i] = u_val
                is_present[i] = True

    return target_classes, target_u_coords, is_present


def decode_row_anchor_predictions(
    logits: torch.Tensor,
    image_width: int = 640,
    num_spatial_bins: int = 100,
    local_radius: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode model logits into predicted presence and localized soft-argmax horizontal coordinates.
    
    Phase 4C Formulation:
    - Peak spatial bin: j* = argmax_{0..K-1}(z)
    - Local window: W = [max(0, j* - R), min(K-1, j* + R)] with radius R = 2
    - Local softmax over window W: p_j = exp(z_j - max(z_W)) / sum_{k in W} exp(z_k - max(z_W))
    - Continuous coordinate expectation: w_hat = sum_{j in W} j * p_j
    - Continuous pixel coordinate: u_hat = ((w_hat + 0.5) / K) * image_width
    - Presence determination: is_present = (argmax_{0..K}(z) != absence_class)
    
    Args:
        logits: Tensor of shape (B, M, K + 1), e.g. (B, 18, 101).
        image_width: 640 px.
        num_spatial_bins: K = 100.
        local_radius: R = 2 (5-bin window).
        
    Returns:
        pred_coords: Tensor of shape (B, M), continuous u coordinates in [0, 640].
        is_present: Tensor of shape (B, M), boolean mask (True if not absence class).
        w_hat: Tensor of shape (B, M), localized soft-argmax bin index in [0, K-1].
    """
    # Strict shape checks
    assert logits.ndim == 3, f"Expected 3D logits (B, M, K+1), got {logits.shape}"
    b, m, total_c = logits.shape
    assert total_c == num_spatial_bins + 1, (
        f"Expected {num_spatial_bins + 1} classes, got {total_c}"
    )

    absence_class_idx = num_spatial_bins  # 100

    # 1. Presence determination: preserves class-argmax presence rule
    argmax_classes = torch.argmax(logits, dim=-1)  # (B, M)
    is_present = (argmax_classes != absence_class_idx)  # (B, M)

    # 2. Peak spatial bin selection
    spatial_logits = logits[:, :, :num_spatial_bins]  # (B, M, K)
    j_star = torch.argmax(spatial_logits, dim=-1)  # (B, M)

    # 3. Vectorized local window masking: W = [max(0, j* - R), min(K-1, j* + R)]
    bin_grid = torch.arange(
        num_spatial_bins, dtype=torch.float32, device=logits.device
    ).view(1, 1, num_spatial_bins)
    dist = torch.abs(bin_grid - j_star.unsqueeze(-1).float())
    in_window = (dist <= float(local_radius))

    # 4. Numerically stable local softmax over window W
    # Bins outside window are masked with -1e9 so exp(-1e9) -> 0.0
    window_logits = torch.where(
        in_window,
        spatial_logits,
        torch.full_like(spatial_logits, -1e9),
    )
    window_probs = torch.softmax(window_logits, dim=-1)  # (B, M, K)

    # 5. Local coordinate expectation
    w_hat = torch.sum(window_probs * bin_grid, dim=-1)  # (B, M)

    # 6. Reconstruct horizontal pixel coordinate
    pred_coords = ((w_hat + 0.5) / float(num_spatial_bins)) * float(image_width)

    return pred_coords, is_present, w_hat


class RowAnchorCoder:
    """Convenience class encapsulating row-anchor encoding and decoding logic."""

    def __init__(self, config: Optional[ProjectConfig] = None):
        if config is None:
            config = ProjectConfig()
        self.num_rows = config.model.num_row_anchors  # 18
        self.num_spatial_bins = config.model.num_spatial_bins  # 100
        self.total_classes = config.model.total_classes  # 101
        self.image_width = config.camera.input_width  # 640
        self.image_height = config.camera.input_height  # 480
        self.y_min = config.roi.y_min  # 200
        self.y_max = config.roi.y_max  # 470
        self.waypoint_tolerance = config.dataset.waypoint_tolerance_px  # 7.5
        self.eval_tolerance = config.dataset.eval_tolerance_px  # 8.0

        self.row_y_coords = get_anchor_row_y_coords(
            self.num_rows, self.y_min, self.y_max
        )

    def encode(self, waypoints_uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return encode_row_anchor_targets(
            waypoints_uv=waypoints_uv,
            num_rows=self.num_rows,
            y_min=self.y_min,
            y_max=self.y_max,
            image_width=self.image_width,
            num_spatial_bins=self.num_spatial_bins,
            waypoint_tolerance_px=self.waypoint_tolerance,
        )

    def decode(
        self,
        logits: torch.Tensor,
        local_radius: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return decode_row_anchor_predictions(
            logits=logits,
            image_width=self.image_width,
            num_spatial_bins=self.num_spatial_bins,
            local_radius=local_radius,
        )
