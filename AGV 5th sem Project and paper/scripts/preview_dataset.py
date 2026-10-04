"""
Synthetic Dataset Visual Validation Tool
========================================
Renders:
1. Original 640x480 frame with projected ground-truth path & 18 row anchors (present vs absent distinguished)
2. SICT pre-filtered representation (Option A: 640x480) with projected path overlay
Generates and saves visual verification figures for representative samples from each scenario.
"""

import sys
from pathlib import Path
from typing import Optional, List
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import load_config
from src.sict.sict_filter import sict_transform_np
from src.labels.coder import get_anchor_row_y_coords


def preview_dataset(
    manifest_paths: Optional[List[str]] = None,
    output_dir: str = "data/metadata/previews",
    samples_per_scenario: int = 1,
) -> None:
    if manifest_paths is None:
        manifest_paths = [
            "data/synthetic/train/annotations.json",
            "data/synthetic/val/annotations.json",
            "data/synthetic/test/annotations.json",
        ]

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    by_scenario = {}
    cfg = load_config()
    row_y = get_anchor_row_y_coords(18, 200, 470)

    for m_path in manifest_paths:
        manifest_p = Path(m_path)
        if not manifest_p.exists():
            continue
        with open(manifest_p, "r", encoding="utf-8") as f:
            data = json.load(f)
        root_dir = manifest_p.parent
        for s in data.get("samples", []):
            sc = s["scenario"]
            if sc not in by_scenario or len(by_scenario[sc]) < samples_per_scenario:
                by_scenario.setdefault(sc, []).append((s, root_dir))

    for sc, sc_samples in by_scenario.items():
        chosen = sc_samples[:samples_per_scenario]
        for idx, (sample, root_dir) in enumerate(chosen):
            img_rel = sample["image_path"]
            img_path = root_dir / img_rel
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Option A: SICT applied directly to full 640x480 frame
            sict_rgb = sict_transform_np(img_rgb, config=cfg.sict, out_channels=3)

            # Render Overlays
            overlay_rgb = img_rgb.copy()
            overlay_sict = (sict_rgb * 255.0).astype(np.uint8).copy()

            # 1. Draw ROI bounds
            for canv in [overlay_rgb, overlay_sict]:
                cv2.line(canv, (0, 200), (640, 200), (255, 0, 0), 1)
                cv2.line(canv, (0, 470), (640, 470), (255, 0, 0), 1)

            # 2. Draw row anchors and path
            row_anchors = sample["row_anchors"]
            pts_present = []

            for item in row_anchors:
                r_idx = item["row_idx"]
                y_coord = int(row_y[r_idx])
                is_pres = item["present"]
                u_coord = item["u_coord"]

                for canv in [overlay_rgb, overlay_sict]:
                    if is_pres and u_coord is not None:
                        pt = (int(u_coord), y_coord)
                        pts_present.append(pt)
                        # Green circle for valid anchor
                        cv2.circle(canv, pt, 5, (0, 255, 0), -1)
                    else:
                        # Red 'X' marker on left margin for absent row
                        cv2.drawMarker(canv, (15, y_coord), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 8, 2)

            # Draw spline line connecting valid anchors
            if len(pts_present) > 1:
                for k in range(len(pts_present) - 1):
                    for canv in [overlay_rgb, overlay_sict]:
                        cv2.line(canv, pts_present[k], pts_present[k + 1], (0, 255, 0), 2)

            # Create side-by-side comparative figure
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            axes[0].imshow(overlay_rgb)
            axes[0].set_title(f"Scenario: {sc} | Frame: {sample['image_id']}\nRaw 640x480 + Ground Truth Path & Row Anchors")
            axes[0].axis("off")

            axes[1].imshow(overlay_sict)
            axes[1].set_title(f"Option A: Full-Frame SICT Pre-filter (640x480)\nSpecular-Suppressed Chromaticity Representation")
            axes[1].axis("off")

            plt.tight_layout()
            preview_filename = f"preview_{sc}_{idx}.png"
            fig_path = out_p / preview_filename
            plt.savefig(fig_path, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"Saved visual validation figure: {fig_path}")


if __name__ == "__main__":
    preview_dataset()
