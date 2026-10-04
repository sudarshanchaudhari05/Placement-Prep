"""
Generate Comprehensive 20-Sample Validation Contact Sheets
=========================================================
Generates two readable 4x5 grid contact sheets for all 20 validation samples:
1. data/validation_preview/validation_contact_sheet.png (Raw RGB + Ground Truth Overlay)
2. data/validation_preview/validation_sict_contact_sheet.png (SICT Transformed + Ground Truth Overlay)

Each sample cell displays:
- Scenario Name (A: Nominal, B: Specular Glare, C: Floor Abrasion, D: S-Curve)
- Image ID and Split
- 640x480 frame with ROI boundary lines (y=200, y=470)
- Ground-truth path spline overlay (green line)
- 18 row-anchor locations (green filled circles for present)
- Absence rows clearly distinguished (distinct red 'X' markers and row indicators)
"""

import sys
import os
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import load_config
from src.sict.sict_filter import sict_transform_np
from src.labels.coder import get_anchor_row_y_coords


def load_all_20_samples(base_dir: Path):
    """Load all 20 validation samples from train, val, and test manifests."""
    manifest_paths = [
        ("train", base_dir / "data" / "synthetic" / "train" / "annotations.json"),
        ("val", base_dir / "data" / "synthetic" / "val" / "annotations.json"),
        ("test", base_dir / "data" / "synthetic" / "test" / "annotations.json"),
    ]

    all_samples = []
    for split_name, m_path in manifest_paths:
        if not m_path.exists():
            continue
        with open(m_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        root_dir = m_path.parent
        for s in data.get("samples", []):
            s_copy = dict(s)
            s_copy["split"] = split_name
            s_copy["abs_image_path"] = root_dir / s["image_path"]
            all_samples.append(s_copy)

    # Sort deterministically by image_id
    all_samples.sort(key=lambda x: x["image_id"])
    return all_samples


def draw_sample_overlay(
    img_canvas: np.ndarray,
    sample: dict,
    row_y_coords: np.ndarray,
    is_sict: bool = False,
) -> np.ndarray:
    """
    Renders path, ROI lines, and 18 row anchors (present vs absent) onto canvas.
    Canvas must be uint8 RGB of shape (480, 640, 3).
    """
    canvas = img_canvas.copy()

    # 1. Draw ROI boundaries (y=200 and y=470) in cyan/blue
    roi_color = (0, 180, 255) if not is_sict else (50, 150, 255)
    cv2.line(canvas, (0, 200), (639, 200), roi_color, 1, cv2.LINE_AA)
    cv2.line(canvas, (0, 470), (639, 470), roi_color, 1, cv2.LINE_AA)

    # 2. Draw ground truth path spline if available from path_points_2d or anchors
    raw_pts = sample.get("path_points_2d", [])
    if len(raw_pts) > 1:
        pts = np.array(raw_pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=False, color=(0, 255, 120), thickness=2, lineType=cv2.LINE_AA)

    # 3. Draw 18 Row Anchors
    row_anchors = sample.get("row_anchors", [])
    for item in row_anchors:
        r_idx = item["row_idx"]
        y_c = int(row_y_coords[r_idx])
        is_pres = item["present"]
        u_c = item["u_coord"]

        if is_pres and u_c is not None:
            # Valid / Present anchor
            pt = (int(round(u_c)), y_c)
            # Outer black ring for visibility against any background
            cv2.circle(canvas, pt, 6, (0, 0, 0), -1, cv2.LINE_AA)
            # Inner bright lime green circle
            cv2.circle(canvas, pt, 4, (0, 255, 0), -1, cv2.LINE_AA)
        else:
            # Absent anchor: prominently mark at left margin (x=24, y=y_c)
            # Red box background
            cv2.rectangle(canvas, (10, y_c - 6), (36, y_c + 6), (20, 20, 20), -1)
            # Red X marker
            cv2.drawMarker(
                canvas,
                (23, y_c),
                (255, 30, 30),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=10,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            # Subtle dashed indicator extending horizontally
            for dash_x in range(45, 120, 15):
                cv2.line(canvas, (dash_x, y_c), (dash_x + 8, y_c), (220, 40, 40), 1)

    return canvas


def create_contact_sheet(
    samples: list,
    row_y_coords: np.ndarray,
    output_path: Path,
    is_sict: bool = False,
    sict_config = None,
):
    """Creates a 4x5 contact sheet of all 20 samples."""
    num_samples = len(samples)
    assert num_samples == 20, f"Expected 20 samples, found {num_samples}"

    nrows = 4
    ncols = 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 18), dpi=140)

    scenario_labels = {
        "scenario_a_nominal": "Scenario A: Nominal",
        "scenario_b_glare": "Scenario B: Specular Glare",
        "scenario_c_abrasion": "Scenario C: Floor Abrasion",
        "scenario_d_s_curve": "Scenario D: S-Curve Curvature",
    }

    for idx, sample in enumerate(samples):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        img_bgr = cv2.imread(str(sample["abs_image_path"]))
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read {sample['abs_image_path']}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if is_sict:
            # Full-frame Option A SICT
            sict_map = sict_transform_np(img_rgb, config=sict_config, out_channels=3)
            base_canvas = (sict_map * 255.0).astype(np.uint8)
        else:
            base_canvas = img_rgb

        # Draw overlays
        overlay = draw_sample_overlay(base_canvas, sample, row_y_coords, is_sict=is_sict)

        ax.imshow(overlay)
        ax.set_xticks([])
        ax.set_yticks([])

        # Compute present/absent counts
        pres_count = sum(1 for a in sample["row_anchors"] if a["present"])
        abs_count = 18 - pres_count

        sc_name = scenario_labels.get(sample["scenario"], sample["scenario"])
        img_id = sample["image_id"]
        split = sample["split"].upper()

        title_text = f"[{img_id}] {sc_name} ({split})\nPresent: {pres_count}/18 | Absent: {abs_count}/18"
        ax.set_title(title_text, fontsize=9.5, fontweight="bold", pad=4)

        # Border color per scenario
        border_colors = {
            0: "#2b8a3e",  # Green for Nominal
            1: "#d9480f",  # Orange for Glare
            2: "#862e9c",  # Purple for Abrasion
            3: "#1864ab",  # Blue for S-Curve
        }
        for spine in ax.spines.values():
            spine.set_edgecolor(border_colors.get(r, "#555555"))
            spine.set_linewidth(2.5)

    # Main Figure Title
    sheet_title = (
        "VALIDATION DATASET INSPECTION: OPTION A SICT OUTPUT (20 SAMPLES)\n"
        "Full-Frame Specular-Invariant Chromaticity Transform (640x480) with Ground-Truth Row Anchors"
        if is_sict
        else "VALIDATION DATASET INSPECTION: RAW 640x480 RGB (20 SAMPLES)\n"
        "Floor Marking Camera Frames with Ground-Truth Path & 18 Row Anchors (Present vs Absent)"
    )
    fig.suptitle(sheet_title, fontsize=16, fontweight="bold", y=0.995)

    # Custom Legend
    legend_elements = [
        mpatches.Patch(color=(0/255, 255/255, 120/255), label="Ground-Truth Path Spline"),
        plt.Line2D([0], [0], marker="o", color="w", label="Present Anchor (Class 0-99)",
                   markerfacecolor=(0/255, 255/255, 0/255), markeredgecolor="black", markersize=9),
        plt.Line2D([0], [0], marker="X", color="w", label="Absent Anchor (Class 100)",
                   markerfacecolor=(255/255, 30/255, 30/255), markeredgecolor="darkred", markersize=9),
        plt.Line2D([0], [0], color=(0/255, 180/255, 255/255), lw=1.5, label="ROI Bounds (y=200, y=470)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.005))

    plt.tight_layout(rect=[0.01, 0.03, 0.99, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved contact sheet: {output_path} ({os.path.getsize(output_path):,} bytes)")


def main():
    cfg = load_config()
    row_y = get_anchor_row_y_coords(cfg.model.num_row_anchors, cfg.roi.y_min, cfg.roi.y_max)
    samples = load_all_20_samples(PROJECT_ROOT)

    out_dir = PROJECT_ROOT / "data" / "validation_preview"
    raw_sheet = out_dir / "validation_contact_sheet.png"
    sict_sheet = out_dir / "validation_sict_contact_sheet.png"

    print(f"Generating raw RGB contact sheet for {len(samples)} samples...")
    create_contact_sheet(samples, row_y, raw_sheet, is_sict=False)

    print(f"Generating SICT contact sheet for {len(samples)} samples...")
    create_contact_sheet(samples, row_y, sict_sheet, is_sict=True, sict_config=cfg.sict)

    print("Completed successfully.")


if __name__ == "__main__":
    main()
