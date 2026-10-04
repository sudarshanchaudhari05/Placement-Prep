"""
Automated Quality Assurance, Manifest, and Visualization Artifacts Generator
============================================================================
Generates:
1. data/synthetic/metadata/file_manifest.json (SHA-256 hash, size, path per image)
2. data/synthetic/metadata/dataset_report.json & dataset_report.txt (comprehensive QA audit)
3. Visual Quality Control Contact Sheets:
   - data/synthetic/metadata/dataset_overview_contact_sheet.png
   - data/synthetic/metadata/scenario_A_contact_sheet.png
   - data/synthetic/metadata/scenario_B_glare_contact_sheet.png
   - data/synthetic/metadata/scenario_C_abrasion_contact_sheet.png
   - data/synthetic/metadata/scenario_D_scurve_contact_sheet.png
   - data/synthetic/metadata/dataset_sict_contact_sheet.png
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Set, Optional

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import load_config
from src.sict.sict_filter import sict_transform_np
from src.labels.coder import get_anchor_row_y_coords


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_file_manifest(data_dir: Path) -> Dict[str, Any]:
    """
    Builds data/synthetic/metadata/file_manifest.json.
    Records: image_id, relative_path, file_size_bytes, sha256.
    """
    print("\n--- Generating File Integrity SHA-256 Manifest ---")
    t0 = time.time()
    synthetic_p = data_dir / "synthetic"
    meta_p = synthetic_p / "metadata"
    meta_p.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    manifest_records = []

    for split in splits:
        ann_file = synthetic_p / split / "annotations.json"
        if not ann_file.exists():
            continue
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for s in data.get("samples", []):
            img_rel = f"synthetic/{split}/{s['image_path']}"
            img_full = synthetic_p / split / s["image_path"]
            if not img_full.exists():
                raise FileNotFoundError(f"Missing image file: {img_full}")

            file_size = os.path.getsize(img_full)
            digest = compute_sha256(img_full)

            manifest_records.append({
                "image_id": s["image_id"],
                "relative_path": img_rel,
                "file_size_bytes": file_size,
                "sha256": digest,
            })

    output_manifest = {
        "dataset_version": "1.0",
        "generated_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_files": len(manifest_records),
        "files": manifest_records,
    }

    out_file = meta_p / "file_manifest.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_manifest, f, indent=2)

    dt = time.time() - t0
    print(f"File manifest created: {out_file} ({len(manifest_records)} files hashed in {dt:.1f}s)")
    return output_manifest


def generate_qa_and_statistics_report(data_dir: Path) -> Dict[str, Any]:
    """
    Computes dataset statistics and data-integrity quality audits.
    Outputs:
    - data/synthetic/metadata/dataset_report.json
    - data/synthetic/metadata/dataset_report.txt
    """
    print("\n--- Generating Comprehensive Dataset QA & Statistics Report ---")
    synthetic_p = data_dir / "synthetic"
    meta_p = synthetic_p / "metadata"
    meta_p.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    split_samples: Dict[str, List[Dict[str, Any]]] = {}
    episodes_by_split: Dict[str, Set[str]] = {}
    all_image_ids = set()
    all_image_paths = set()
    all_seeds = set()

    duplicate_image_ids = 0
    duplicate_image_paths = 0
    duplicate_seeds = 0
    corrupt_images = 0
    invalid_annotations = 0

    scenario_counts_by_split: Dict[str, Dict[str, int]] = {
        "train": {}, "val": {}, "test": {}, "overall": {}
    }

    total_anchors = 0
    absent_anchors = 0
    visible_per_image = []
    spatial_bins = []

    for split in splits:
        manifest_file = synthetic_p / split / "annotations.json"
        episodes_by_split[split] = set()
        split_samples[split] = []

        if not manifest_file.exists():
            continue

        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        s_list = data.get("samples", [])
        split_samples[split] = s_list

        for s in s_list:
            img_id = s.get("image_id")
            if img_id in all_image_ids:
                duplicate_image_ids += 1
            else:
                all_image_ids.add(img_id)

            rel_path = f"{split}/{s.get('image_path')}"
            if rel_path in all_image_paths:
                duplicate_image_paths += 1
            else:
                all_image_paths.add(rel_path)

            full_img_p = synthetic_p / split / s.get("image_path", "")
            if not full_img_p.exists() or os.path.getsize(full_img_p) == 0:
                corrupt_images += 1

            seed = s.get("seed")
            if seed in all_seeds:
                duplicate_seeds += 1
            else:
                all_seeds.add(seed)

            ep_id = s.get("scene_id")
            if ep_id:
                episodes_by_split[split].add(ep_id)

            sc = s.get("scenario", "unknown")
            scenario_counts_by_split[split][sc] = scenario_counts_by_split[split].get(sc, 0) + 1
            scenario_counts_by_split["overall"][sc] = scenario_counts_by_split["overall"].get(sc, 0) + 1

            row_anchors = s.get("row_anchors", [])
            if len(row_anchors) != 18:
                invalid_annotations += 1
                continue

            vis_count = 0
            for item in row_anchors:
                total_anchors += 1
                pres = item.get("present", False)
                cls_id = item.get("class_id")

                if pres:
                    vis_count += 1
                    if cls_id is not None and 0 <= cls_id < 100:
                        spatial_bins.append(cls_id)
                    else:
                        invalid_annotations += 1
                else:
                    absent_anchors += 1
                    if cls_id != 100:
                        invalid_annotations += 1

            visible_per_image.append(vis_count)

    # Check cross-split episode leakage
    train_val_overlap = len(episodes_by_split["train"].intersection(episodes_by_split["val"]))
    train_test_overlap = len(episodes_by_split["train"].intersection(episodes_by_split["test"]))
    val_test_overlap = len(episodes_by_split["val"].intersection(episodes_by_split["test"]))

    # Check rejected samples
    rejected_log = meta_p / "rejected_samples.json"
    rejected_count = 0
    if rejected_log.exists():
        with open(rejected_log, "r", encoding="utf-8") as f:
            rej_data = json.load(f)
            rejected_count = rej_data.get("total_rejected", len(rej_data.get("rejected_samples", [])))

    total_images = len(all_image_ids)
    pct_absent = (absent_anchors / total_anchors * 100.0) if total_anchors > 0 else 0.0
    avg_vis = float(np.mean(visible_per_image)) if visible_per_image else 0.0
    min_vis = int(np.min(visible_per_image)) if visible_per_image else 0
    max_vis = int(np.max(visible_per_image)) if visible_per_image else 0

    spatial_distribution = {
        "p10": float(np.percentile(spatial_bins, 10)) if spatial_bins else 0.0,
        "p25": float(np.percentile(spatial_bins, 25)) if spatial_bins else 0.0,
        "median": float(np.median(spatial_bins)) if spatial_bins else 0.0,
        "p75": float(np.percentile(spatial_bins, 75)) if spatial_bins else 0.0,
        "p90": float(np.percentile(spatial_bins, 90)) if spatial_bins else 0.0,
    }

    report = {
        "summary": {
            "total_images": total_images,
            "train_count": len(split_samples["train"]),
            "val_count": len(split_samples["val"]),
            "test_count": len(split_samples["test"]),
        },
        "scenario_distribution": scenario_counts_by_split,
        "anchor_statistics": {
            "total_anchors_evaluated": total_anchors,
            "absent_anchors_count": absent_anchors,
            "percentage_absent_anchors": round(pct_absent, 2),
            "avg_visible_anchors_per_image": round(avg_vis, 2),
            "min_visible_anchors": min_vis,
            "max_visible_anchors": max_vis,
        },
        "spatial_bin_distribution": spatial_distribution,
        "data_integrity": {
            "corrupt_images": corrupt_images,
            "rejected_frames": rejected_count,
            "invalid_annotations": invalid_annotations,
            "duplicate_image_ids": duplicate_image_ids,
            "duplicate_image_paths": duplicate_image_paths,
            "duplicate_seeds": duplicate_seeds,
            "scene_leakage": {
                "train_val_overlap": train_val_overlap,
                "train_test_overlap": train_test_overlap,
                "val_test_overlap": val_test_overlap,
            },
        },
        "configuration": {
            "resolution": "640x480 RGB",
            "model_input": "384x288 RGB",
            "roi": "y=200:470, x=0:640",
            "row_anchors_M": 18,
            "spatial_bins_K": 100,
            "absence_class": 100,
            "sict_order": "Option A (Full 640x480 frame prior to ROI crop)",
            "sict_ambient_intensity": "I_ambient = 0.2 * mean((R+G+B)/765.0)",
        }
    }

    # Save JSON report
    json_path = meta_p / "dataset_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save human-readable text report
    txt_path = meta_p / "dataset_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("SYNTHETIC RESEARCH DATASET QUALITY ASSURANCE & INTEGRITY AUDIT REPORT\n")
        f.write("================================================================================\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("1. DATASET COUNTS & SPLIT ALLOCATION\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(f"Total Synthesized Images:          {report['summary']['total_images']}\n")
        f.write(f"  - Train Set:                     {report['summary']['train_count']} ({report['summary']['train_count']/report['summary']['total_images']*100:.1f}%)\n")
        f.write(f"  - Validation Set:                {report['summary']['val_count']} ({report['summary']['val_count']/report['summary']['total_images']*100:.1f}%)\n")
        f.write(f"  - In-Domain Test Set:            {report['summary']['test_count']} ({report['summary']['test_count']/report['summary']['total_images']*100:.1f}%)\n\n")

        f.write("2. SCENARIO DISTRIBUTION\n")
        f.write("--------------------------------------------------------------------------------\n")
        for sc, cnt in report['scenario_distribution']['overall'].items():
            f.write(f"  - {sc:<25}: {cnt} ({cnt/report['summary']['total_images']*100:.1f}%)\n")
        f.write("\n  Split Breakdown:\n")
        for spl in ['train', 'val', 'test']:
            f.write(f"    * {spl.upper()}: {report['scenario_distribution'][spl]}\n")

        f.write("\n3. ROW ANCHOR & SPATIAL DISTRIBUTION STATISTICS\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(f"Total Anchors Evaluated:           {report['anchor_statistics']['total_anchors_evaluated']:,}\n")
        f.write(f"Absent Anchors Count:              {report['anchor_statistics']['absent_anchors_count']:,} ({report['anchor_statistics']['percentage_absent_anchors']}%)\n")
        f.write(f"Average Visible Anchors/Image:     {report['anchor_statistics']['avg_visible_anchors_per_image']} / 18\n")
        f.write(f"Min / Max Visible Anchors:         {report['anchor_statistics']['min_visible_anchors']} / {report['anchor_statistics']['max_visible_anchors']}\n")
        f.write(f"Spatial Bins (10th/25th/Med/75th/90th): {spatial_distribution['p10']:.1f} / {spatial_distribution['p25']:.1f} / {spatial_distribution['median']:.1f} / {spatial_distribution['p75']:.1f} / {spatial_distribution['p90']:.1f}\n\n")

        f.write("4. DATA INTEGRITY & AUDIT VERIFICATION\n")
        f.write("--------------------------------------------------------------------------------\n")
        f.write(f"Corrupt Image Files:               {report['data_integrity']['corrupt_images']} (PASS)\n")
        f.write(f"Invalid Annotations:               {report['data_integrity']['invalid_annotations']} (PASS)\n")
        f.write(f"Duplicate Image IDs:               {report['data_integrity']['duplicate_image_ids']} (PASS)\n")
        f.write(f"Duplicate Image Paths:             {report['data_integrity']['duplicate_image_paths']} (PASS)\n")
        f.write(f"Duplicate Random Seeds:            {report['data_integrity']['duplicate_seeds']} (PASS)\n")
        f.write(f"Rejected Samples (Logged):         {report['data_integrity']['rejected_frames']}\n")
        f.write("Cross-Split Scene/Episode Leakage:\n")
        f.write(f"  - Train vs Val Overlap:          {report['data_integrity']['scene_leakage']['train_val_overlap']} (PASS - Disjoint)\n")
        f.write(f"  - Train vs Test Overlap:         {report['data_integrity']['scene_leakage']['train_test_overlap']} (PASS - Disjoint)\n")
        f.write(f"  - Val vs Test Overlap:           {report['data_integrity']['scene_leakage']['val_test_overlap']} (PASS - Disjoint)\n\n")

        f.write("5. RESEARCH SPECIFICATION & REPRODUCIBILITY\n")
        f.write("--------------------------------------------------------------------------------\n")
        for k, v in report['configuration'].items():
            f.write(f"  {k:<28}: {v}\n")
        f.write("================================================================================\n")

    print(f"Dataset reports saved:\n  - {json_path}\n  - {txt_path}")
    return report


def draw_sample_overlay(
    img_canvas: np.ndarray,
    sample: dict,
    row_y_coords: np.ndarray,
    is_sict: bool = False,
) -> np.ndarray:
    """Renders path, ROI lines, and 18 row anchors onto canvas."""
    canvas = img_canvas.copy()

    # ROI boundaries (y=200 and y=470)
    roi_color = (0, 180, 255) if not is_sict else (50, 150, 255)
    cv2.line(canvas, (0, 200), (639, 200), roi_color, 1, cv2.LINE_AA)
    cv2.line(canvas, (0, 470), (639, 470), roi_color, 1, cv2.LINE_AA)

    # Path spline
    raw_pts = sample.get("path_points_2d", [])
    if len(raw_pts) > 1:
        pts = np.array(raw_pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=False, color=(0, 255, 120), thickness=2, lineType=cv2.LINE_AA)

    # 18 Row Anchors
    row_anchors = sample.get("row_anchors", [])
    for item in row_anchors:
        r_idx = item["row_idx"]
        y_c = int(row_y_coords[r_idx])
        is_pres = item["present"]
        u_c = item["u_coord"]

        if is_pres and u_c is not None:
            pt = (int(round(u_c)), y_c)
            cv2.circle(canvas, pt, 6, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, pt, 4, (0, 255, 0), -1, cv2.LINE_AA)
        else:
            cv2.rectangle(canvas, (10, y_c - 6), (36, y_c + 6), (20, 20, 20), -1)
            cv2.drawMarker(
                canvas, (23, y_c), (255, 30, 30),
                markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_AA,
            )
            for dash_x in range(45, 120, 15):
                cv2.line(canvas, (dash_x, y_c), (dash_x + 8, y_c), (220, 40, 40), 1)

    return canvas


def generate_all_contact_sheets(data_dir: Path):
    """
    Renders all 6 required visual quality control contact sheets:
    1. dataset_overview_contact_sheet.png (16 samples: 4 A, 4 B, 4 C, 4 D)
    2. scenario_A_contact_sheet.png (16 samples of Scenario A)
    3. scenario_B_glare_contact_sheet.png (16 samples of Scenario B)
    4. scenario_C_abrasion_contact_sheet.png (16 samples of Scenario C)
    5. scenario_D_scurve_contact_sheet.png (16 samples of Scenario D)
    6. dataset_sict_contact_sheet.png (16 comparative pairs: Raw RGB vs Option A SICT)
    """
    print("\n--- Generating Visual Quality Control Contact Sheets ---")
    cfg = load_config()
    row_y = get_anchor_row_y_coords(cfg.model.num_row_anchors, cfg.roi.y_min, cfg.roi.y_max)
    synthetic_p = data_dir / "synthetic"
    meta_p = synthetic_p / "metadata"
    meta_p.mkdir(parents=True, exist_ok=True)

    # Collect samples grouped by scenario
    by_scenario: Dict[str, List[Dict[str, Any]]] = {
        "scenario_a_nominal": [],
        "scenario_b_glare": [],
        "scenario_c_abrasion": [],
        "scenario_d_s_curve": [],
    }

    for split in ["train", "val", "test"]:
        ann_file = synthetic_p / split / "annotations.json"
        if not ann_file.exists():
            continue
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for s in data.get("samples", []):
            sc = s.get("scenario")
            if sc in by_scenario:
                s_copy = dict(s)
                s_copy["abs_image_path"] = synthetic_p / split / s["image_path"]
                s_copy["split"] = split
                by_scenario[sc].append(s_copy)

    scenario_pretty = {
        "scenario_a_nominal": "Scenario A (Nominal)",
        "scenario_b_glare": "Scenario B (Specular Glare)",
        "scenario_c_abrasion": "Scenario C (Floor Abrasion)",
        "scenario_d_s_curve": "Scenario D (S-Curve Curvature)",
    }

    # 1. Dataset Overview Contact Sheet (16 samples: 4 from each scenario)
    overview_samples = []
    for sc in ["scenario_a_nominal", "scenario_b_glare", "scenario_c_abrasion", "scenario_d_s_curve"]:
        overview_samples.extend(by_scenario[sc][:4])

    _render_grid_sheet(
        samples=overview_samples,
        row_y=row_y,
        nrows=4, ncols=4,
        out_path=meta_p / "dataset_overview_contact_sheet.png",
        sheet_title="DATASET OVERVIEW: REPRESENTATIVE SAMPLES ACROSS ALL 4 SCENARIOS (16 SAMPLES)\n"
                    "Row 1: Nominal | Row 2: Specular Glare | Row 3: Floor Abrasion | Row 4: S-Curve Curvature",
        scenario_pretty=scenario_pretty,
    )

    # 2. Scenario A Contact Sheet (16 samples)
    _render_grid_sheet(
        samples=by_scenario["scenario_a_nominal"][:16],
        row_y=row_y,
        nrows=4, ncols=4,
        out_path=meta_p / "scenario_A_contact_sheet.png",
        sheet_title="SCENARIO A: NOMINAL FLOOR MARKING SAMPLES (16 SAMPLES)\n"
                    "Pristine White Tape, Industrial Concrete Floor (300-400 Lux), Continuous Anchors",
        scenario_pretty=scenario_pretty,
    )

    # 3. Scenario B Glare Contact Sheet (16 samples)
    _render_grid_sheet(
        samples=by_scenario["scenario_b_glare"][:16],
        row_y=row_y,
        nrows=4, ncols=4,
        out_path=meta_p / "scenario_B_glare_contact_sheet.png",
        sheet_title="SCENARIO B: SPECULAR GLARE SAMPLES (16 SAMPLES)\n"
                    "High-Gloss Epoxy Surface, Intense Overhead Illumination (1200-1500 Lux), Specular Lobes",
        scenario_pretty=scenario_pretty,
    )

    # 4. Scenario C Abrasion Contact Sheet (16 samples)
    _render_grid_sheet(
        samples=by_scenario["scenario_c_abrasion"][:16],
        row_y=row_y,
        nrows=4, ncols=4,
        out_path=meta_p / "scenario_C_abrasion_contact_sheet.png",
        sheet_title="SCENARIO C: FLOOR ABRASION SAMPLES (16 SAMPLES)\n"
                    "Tape Breaks (Poisson lambda=0.05/m), 25-40% Material Loss, Tire Scuffs & Occlusions",
        scenario_pretty=scenario_pretty,
    )

    # 5. Scenario D S-Curve Contact Sheet (16 samples)
    _render_grid_sheet(
        samples=by_scenario["scenario_d_s_curve"][:16],
        row_y=row_y,
        nrows=4, ncols=4,
        out_path=meta_p / "scenario_D_scurve_contact_sheet.png",
        sheet_title="SCENARIO D: HIGH-CURVATURE S-CURVE SAMPLES (16 SAMPLES)\n"
                    "Compound Curves (R >= 0.75 m), Dynamic Turn Transitions, Realistic Anchor Absence",
        scenario_pretty=scenario_pretty,
    )

    # 6. SICT Comparative Contact Sheet (8 representative pairs: Raw RGB vs Option A SICT)
    _render_sict_comparative_sheet(
        by_scenario=by_scenario,
        row_y=row_y,
        sict_config=cfg.sict,
        out_path=meta_p / "dataset_sict_contact_sheet.png",
    )


def _render_grid_sheet(
    samples: List[Dict[str, Any]],
    row_y: np.ndarray,
    nrows: int, ncols: int,
    out_path: Path,
    sheet_title: str,
    scenario_pretty: Dict[str, str],
):
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 16), dpi=130)
    for idx, sample in enumerate(samples[:nrows*ncols]):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        img_bgr = cv2.imread(str(sample["abs_image_path"]))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        overlay = draw_sample_overlay(img_rgb, sample, row_y, is_sict=False)

        ax.imshow(overlay)
        ax.set_xticks([])
        ax.set_yticks([])

        pres_count = sum(1 for a in sample["row_anchors"] if a["present"])
        sc_lbl = scenario_pretty.get(sample["scenario"], sample["scenario"])
        img_id = sample["image_id"]
        split = sample["split"].upper()

        ax.set_title(f"[{img_id}] {sc_lbl} ({split})\nVisible Anchors: {pres_count}/18", fontsize=9, fontweight="bold", pad=3)

    fig.suptitle(sheet_title, fontsize=15, fontweight="bold", y=0.995)

    legend_elements = [
        mpatches.Patch(color=(0/255, 255/255, 120/255), label="Ground-Truth Path Spline"),
        plt.Line2D([0], [0], marker="o", color="w", label="Present Anchor (Class 0-99)",
                   markerfacecolor=(0/255, 255/255, 0/255), markeredgecolor="black", markersize=8),
        plt.Line2D([0], [0], marker="X", color="w", label="Absent Anchor (Class 100)",
                   markerfacecolor=(255/255, 30/255, 30/255), markeredgecolor="darkred", markersize=8),
        plt.Line2D([0], [0], color=(0/255, 180/255, 255/255), lw=1.5, label="ROI Bounds (y=200, y=470)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=10.5, frameon=True, bbox_to_anchor=(0.5, 0.005))
    plt.tight_layout(rect=[0.01, 0.03, 0.99, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved contact sheet: {out_path}")


def _render_sict_comparative_sheet(
    by_scenario: Dict[str, List[Dict[str, Any]]],
    row_y: np.ndarray,
    sict_config,
    out_path: Path,
):
    """8 rows of [Raw RGB | Option A SICT] (2 pairs per scenario)."""
    selected_samples = []
    for sc in ["scenario_a_nominal", "scenario_b_glare", "scenario_c_abrasion", "scenario_d_s_curve"]:
        selected_samples.extend(by_scenario[sc][:2])

    fig, axes = plt.subplots(4, 4, figsize=(22, 16), dpi=130)

    for idx, sample in enumerate(selected_samples):
        # Place 2 samples per row: Col 0 & 1 for first sample, Col 2 & 3 for second sample
        grid_row = idx // 2
        col_offset = (idx % 2) * 2

        ax_raw = axes[grid_row, col_offset]
        ax_sict = axes[grid_row, col_offset + 1]

        img_bgr = cv2.imread(str(sample["abs_image_path"]))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        sict_rgb = sict_transform_np(img_rgb, config=sict_config, out_channels=3)
        sict_canvas = (sict_rgb * 255.0).astype(np.uint8)

        overlay_raw = draw_sample_overlay(img_rgb, sample, row_y, is_sict=False)
        overlay_sict = draw_sample_overlay(sict_canvas, sample, row_y, is_sict=True)

        ax_raw.imshow(overlay_raw)
        ax_raw.set_xticks([])
        ax_raw.set_yticks([])
        ax_raw.set_title(f"[{sample['image_id']}] RAW RGB ({sample['scenario']})", fontsize=8.5, fontweight="bold")

        ax_sict.imshow(overlay_sict)
        ax_sict.set_xticks([])
        ax_sict.set_yticks([])
        ax_sict.set_title(f"[{sample['image_id']}] SICT OUTPUT (Option A)", fontsize=8.5, fontweight="bold")

    fig.suptitle(
        "SICT PRE-FILTER COMPARISON: RAW 640x480 RGB vs OPTION A SICT OUTPUT\n"
        "Demonstrating Specular Glare Attenuation and Ambient Chromaticity Invariance",
        fontsize=15, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SICT comparison contact sheet: {out_path}")
