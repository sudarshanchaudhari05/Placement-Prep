"""
Descriptive Dataset Quality Assurance (QA) Report Tool
======================================================
Generates QA metrics across synthesized dataset splits:
- Total images and distributions across scenarios (Nominal, Glare, Abrasion, S-Curve)
- Split allocation (Train, Val, Test)
- Row anchor statistics (absence percentage, average visible anchors, min/max)
- Spatial bin distribution
- Rejected frames and invalid annotation counts
- Duplicate image ID and seed verification
- Cross-split episode/scene leakage verification

NOTE: This is strictly a data-integrity quality audit, NOT a model evaluation metric.
"""

from pathlib import Path
from typing import Dict, Any, List, Set
import json
import numpy as np


def generate_dataset_qa_report(
    data_dir: str = "data",
) -> Dict[str, Any]:
    base_p = Path(data_dir)
    synthetic_p = base_p / "synthetic"

    splits = ["train", "val", "test"]
    all_samples = []
    split_samples = {}
    episodes_by_split: Dict[str, Set[str]] = {}

    all_image_ids = set()
    duplicate_image_ids = 0
    all_seeds = set()
    duplicate_seeds = 0

    invalid_annotations = 0
    total_anchors = 0
    absent_anchors = 0
    visible_per_image = []
    spatial_bins = []

    scenario_counts = {}

    for split in splits:
        manifest_file = synthetic_p / split / "annotations.json"
        episodes_by_split[split] = set()
        if not manifest_file.exists():
            split_samples[split] = []
            continue

        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        s_list = data.get("samples", [])
        split_samples[split] = s_list

        for s in s_list:
            all_samples.append(s)

            # Duplicate checks
            img_id = s.get("image_id")
            if img_id in all_image_ids:
                duplicate_image_ids += 1
            else:
                all_image_ids.add(img_id)

            seed = s.get("seed")
            if seed in all_seeds:
                duplicate_seeds += 1
            else:
                all_seeds.add(seed)

            # Episode leakage tracking
            ep_id = s.get("scene_id")
            if ep_id:
                episodes_by_split[split].add(ep_id)

            # Scenario tracking
            sc = s.get("scenario", "unknown")
            scenario_counts[sc] = scenario_counts.get(sc, 0) + 1

            # Anchor analysis
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

    # Cross-split episode leakage check
    leakage_train_val = len(episodes_by_split["train"].intersection(episodes_by_split["val"]))
    leakage_train_test = len(episodes_by_split["train"].intersection(episodes_by_split["test"]))
    leakage_val_test = len(episodes_by_split["val"].intersection(episodes_by_split["test"]))

    # Check rejected samples log
    rejected_log = base_p / "metadata" / "rejected_samples.json"
    rejected_count = 0
    if rejected_log.exists():
        with open(rejected_log, "r", encoding="utf-8") as f:
            rej_data = json.load(f)
            rejected_count = rej_data.get("total_rejected", 0)

    total_imgs = len(all_samples)
    pct_absent = (absent_anchors / total_anchors * 100.0) if total_anchors > 0 else 0.0
    avg_vis = float(np.mean(visible_per_image)) if visible_per_image else 0.0
    min_vis = int(np.min(visible_per_image)) if visible_per_image else 0
    max_vis = int(np.max(visible_per_image)) if visible_per_image else 0

    bin_percentiles = (
        {
            "p10": float(np.percentile(spatial_bins, 10)),
            "p25": float(np.percentile(spatial_bins, 25)),
            "median": float(np.median(spatial_bins)),
            "p75": float(np.percentile(spatial_bins, 75)),
            "p90": float(np.percentile(spatial_bins, 90)),
        }
        if spatial_bins
        else {}
    )

    report = {
        "total_images": total_imgs,
        "images_per_split": {s: len(split_samples[s]) for s in splits},
        "images_per_scenario": scenario_counts,
        "anchor_statistics": {
            "total_anchors_evaluated": total_anchors,
            "absent_anchors_count": absent_anchors,
            "percentage_absent_anchors": round(pct_absent, 2),
            "avg_visible_anchors_per_image": round(avg_vis, 2),
            "min_visible_anchors": min_vis,
            "max_visible_anchors": max_vis,
        },
        "spatial_bin_distribution": bin_percentiles,
        "data_integrity": {
            "rejected_frames": rejected_count,
            "invalid_annotations": invalid_annotations,
            "duplicate_image_ids": duplicate_image_ids,
            "duplicate_seeds": duplicate_seeds,
            "scene_leakage": {
                "train_val_overlap": leakage_train_val,
                "train_test_overlap": leakage_train_test,
                "val_test_overlap": leakage_val_test,
            },
        },
    }

    # Pretty print
    print("=" * 70)
    print("SYNTHETIC DATASET QUALITY ASSURANCE (QA) REPORT")
    print("=" * 70)
    print(f"Total Synthesized Images:          {report['total_images']}")
    print(f"Split Distribution:                {report['images_per_split']}")
    print(f"Scenario Distribution:             {report['images_per_scenario']}")
    print("-" * 70)
    print(f"Absence Anchor Ratio:              {report['anchor_statistics']['percentage_absent_anchors']}%")
    print(f"Average Visible Anchors/Image:     {report['anchor_statistics']['avg_visible_anchors_per_image']} / 18")
    print(f"Min / Max Visible Anchors:         {report['anchor_statistics']['min_visible_anchors']} / {report['anchor_statistics']['max_visible_anchors']}")
    print(f"Spatial Bin Distribution (Median): {bin_percentiles.get('median', 'N/A')}")
    print("-" * 70)
    print(f"Rejected Frames:                   {report['data_integrity']['rejected_frames']}")
    print(f"Invalid Annotations:               {report['data_integrity']['invalid_annotations']}")
    print(f"Duplicate Image IDs:               {report['data_integrity']['duplicate_image_ids']}")
    print(f"Duplicate Seeds:                   {report['data_integrity']['duplicate_seeds']}")
    print(f"Scene Leakage Between Splits:      {report['data_integrity']['scene_leakage']}")
    print("=" * 70)

    # Persist report
    report_file = base_p / "metadata" / "dataset_qa_report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    generate_dataset_qa_report()
