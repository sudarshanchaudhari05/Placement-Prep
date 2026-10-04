"""
Synthetic Dummy Data Generator (FOR PIPELINE TESTING ONLY)
=========================================================
WARNING:
This dummy data is NOT research data and is strictly intended for verifying
the software pipeline, data loaders, tensor shapes, and smoke tests.
The full 6000-image research dataset is NOT generated here.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import cv2
import numpy as np

from configs.config import ProjectConfig
from src.labels.coder import encode_row_anchor_targets, get_anchor_row_y_coords


def generate_dummy_dataset(
    output_dir: Optional[Path] = None,
    num_samples: int = 5,
    config: Optional[ProjectConfig] = None,
) -> Path:
    """
    Generate minimal dummy dataset strictly for unit/smoke tests.
    
    Args:
        output_dir: Output directory where dummy images and annotations will be saved.
        num_samples: Number of dummy samples to generate (default 5).
        config: Project configuration.
        
    Returns:
        Path to the generated annotations JSON file.
    """
    if config is None:
        config = ProjectConfig()

    if output_dir is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
        output_dir = base_dir / "dummy_data"

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    w = config.camera.input_width   # 640
    h = config.camera.input_height  # 480
    num_rows = config.model.num_row_anchors  # 18
    num_bins = config.model.num_spatial_bins  # 100
    y_min = config.roi.y_min  # 200
    y_max = config.roi.y_max  # 470
    row_y_coords = get_anchor_row_y_coords(num_rows, y_min, y_max)

    samples = []

    for idx in range(num_samples):
        # 1. Create simulated floor background (industrial gray epoxy)
        img = np.full((h, w, 3), 70 + idx * 5, dtype=np.uint8)

        # 2. Draw a curved path (yellow or white industrial floor marking)
        # Parameterized line: u = center + amplitude * sin(...)
        center_u = 320.0 + (idx - 2) * 30.0
        curve_amp = 40.0 + idx * 10.0

        waypoints = []
        for y_pt in range(y_min - 20, y_max + 20, 5):
            t = (y_pt - y_min) / float(y_max - y_min)
            u_pt = center_u + curve_amp * np.sin(t * np.pi)
            # Simulate occasional path discontinuity (abrasion) on sample 2
            if idx == 2 and 280 <= y_pt <= 330:
                continue
            waypoints.append([u_pt, float(y_pt)])

        waypoints_arr = np.array(waypoints, dtype=np.float32)

        # Draw path line onto image
        if len(waypoints_arr) > 1:
            pts = waypoints_arr.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], isClosed=False, color=(0, 215, 255), thickness=8)

        # 3. Simulate high-bay specular glare spot
        glare_center = (int(center_u + 20), int(y_min + 70))
        cv2.circle(img, glare_center, 45, (255, 255, 255), -1)
        # Blur the glare
        img = cv2.GaussianBlur(img, (21, 21), 0)

        # 4. Mandatory prominent watermark indicating dummy test status
        cv2.putText(
            img,
            "DUMMY TEST DATA - NOT FOR RESEARCH EVALUATION",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        img_filename = f"dummy_{idx:04d}.png"
        img_path = images_dir / img_filename
        cv2.imwrite(str(img_path), img)

        # 5. Encode row-anchor targets
        target_classes, target_u_coords, is_present = encode_row_anchor_targets(
            waypoints_uv=waypoints_arr,
            num_rows=num_rows,
            y_min=y_min,
            y_max=y_max,
            image_width=w,
            num_spatial_bins=num_bins,
            waypoint_tolerance_px=config.dataset.waypoint_tolerance_px,
        )

        # Format row-anchors list
        row_anchors = []
        for r_i in range(num_rows):
            pres = bool(is_present[r_i])
            row_anchors.append(
                {
                    "row_idx": r_i,
                    "row_y_orig": int(row_y_coords[r_i]),
                    "u_coord": float(target_u_coords[r_i]) if pres else None,
                    "grid_bin": int(target_classes[r_i]) if pres else None,
                    "class_id": int(target_classes[r_i]),  # 0..99 or 100 (absence)
                    "present": pres,
                }
            )

        sample_record = {
            "image_path": f"images/{img_filename}",
            "width": w,
            "height": h,
            "roi": {"y_min": y_min, "y_max": y_max, "x_min": 0, "x_max": w},
            "row_anchors": row_anchors,
            "metadata": {
                "is_dummy": True,
                "purpose": "pipeline_verification_only",
                "simulated_specular_exponent": 45.0,
                "simulated_lighting_lux": 800.0,
            },
        }
        samples.append(sample_record)

    annotations_path = output_dir / "dummy_annotations.json"
    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump({"version": "1.0", "samples": samples}, f, indent=2)

    return annotations_path


if __name__ == "__main__":
    p = generate_dummy_dataset()
    print(f"Generated dummy dataset at: {p}")
