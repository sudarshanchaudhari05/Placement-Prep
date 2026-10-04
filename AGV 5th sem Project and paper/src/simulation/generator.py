"""
Master Synthetic Dataset Generator
==================================
Orchestrates:
1. Scenario distribution (Nominal, Specular Glare, Abrasion, S-Curve)
2. Episode-level train / val / test split allocation (70% / 15% / 15%)
3. Stratified scenario balance across all splits without cross-split episode leakage
4. Episode-Level Geometric Sanity Gate:
   - Evaluates candidate 3D trajectories before frame generation
   - Enforces >= min_visible_anchors_trajectory (default: 3) under nominal camera geometry
   - Deterministically regenerates invalid trajectories up to max_trajectory_attempts (default: 50)
5. Bounded Frame-Level Retry Gate:
   - Bounded retries up to max_frame_attempts (default: 10)
   - Strict distinction between geometry-related and appearance-related rejections
   - Geometry rejections trigger trajectory regeneration rather than wasteful appearance resampling
6. Deterministic per-frame seeds and metadata tracking
7. Row-anchor label encoding via src/labels/coder.py
8. Strict sample validation with rejection logging and manifest export
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import time
import cv2
import numpy as np

from configs.config import ProjectConfig, load_config
from src.simulation.camera import CameraConfig
from src.simulation.path_generator import SplinePathGenerator, Path3D
from src.simulation.domain_randomization import DomainRandomizer, ScenarioType, RandomizedParams
from src.simulation.capture import SyntheticCaptureEngine
from src.simulation.validator import SampleValidator
from src.labels.coder import encode_row_anchor_targets, get_anchor_row_y_coords


class SyntheticDatasetGenerator:
    """
    Coordinates synthetic dataset generation according to frozen research specification.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        config: Optional[ProjectConfig] = None,
        base_seed: int = 1000,
    ):
        self.config = config or load_config()
        self.base_dir = Path(base_dir) if base_dir else Path("data")
        self.metadata_dir = self.base_dir / "synthetic" / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.base_seed = base_seed

        self.camera_cfg = CameraConfig()
        self.path_gen = SplinePathGenerator()
        self.randomizer = DomainRandomizer()
        self.capture_engine = SyntheticCaptureEngine(self.camera_cfg)
        self.validator = SampleValidator(self.metadata_dir / "rejected_samples.json")

        self.row_y = get_anchor_row_y_coords(
            self.config.model.num_row_anchors,
            self.config.roi.y_min,
            self.config.roi.y_max,
        )

        # Configurable retry and sanity gate parameters
        self.max_trajectory_attempts: int = getattr(
            self.config.dataset, "max_trajectory_attempts", 50
        )
        self.max_frame_attempts: int = getattr(
            self.config.dataset, "max_frame_attempts", 10
        )
        self.min_visible_anchors: int = getattr(
            self.config.dataset, "min_visible_anchors_trajectory", 3
        )

    def check_trajectory_visibility(
        self,
        path: Path3D,
        min_visible_anchors: Optional[int] = None,
    ) -> Tuple[bool, int]:
        """
        Episode-Level Geometric Sanity Gate.
        Projects candidate ground trajectory under nominal camera orientation (pitch offset = 0.0 deg)
        and verifies that it yields at least min_visible_anchors in the camera ROI.

        Args:
            path: Candidate 3D ground trajectory.
            min_visible_anchors: Minimum required visible row anchors (default: self.min_visible_anchors).

        Returns:
            Tuple of (is_valid: bool, visible_anchors_count: int).
        """
        threshold = min_visible_anchors if min_visible_anchors is not None else self.min_visible_anchors
        pts_xy = path.waypoints_world[:, :2]

        uv_all, valid_depth = self.capture_engine.camera_model.project_ground_points_to_image(
            pts_xy, pitch_offset_deg=0.0
        )
        valid_mask = (
            valid_depth
            & np.isfinite(uv_all[:, 0])
            & np.isfinite(uv_all[:, 1])
            & (uv_all[:, 1] >= 0)
            & (uv_all[:, 1] < self.config.camera.input_height)
        )
        vis_uv = uv_all[valid_mask]

        _, _, is_pres = encode_row_anchor_targets(
            waypoints_uv=vis_uv,
            num_rows=self.config.model.num_row_anchors,
            y_min=self.config.roi.y_min,
            y_max=self.config.roi.y_max,
            image_width=self.config.camera.input_width,
            num_spatial_bins=self.config.model.num_spatial_bins,
            waypoint_tolerance_px=self.config.dataset.waypoint_tolerance_px,
        )
        visible_count = int(np.sum(is_pres))
        return (visible_count >= threshold), visible_count

    def _generate_scenario_trajectory(
        self,
        scenario: ScenarioType,
        rng: np.random.Generator,
    ) -> Path3D:
        """Instantiate scenario-specific ground path trajectory."""
        if scenario == ScenarioType.SCENARIO_D:
            return self.path_gen.generate_s_curve_path(
                min_radius_m=0.75,
                length_m=3.5,
                amplitude_m=float(rng.uniform(0.15, 0.30)),
                lateral_offset_m=float(rng.uniform(-0.15, 0.15)),
            )
        elif scenario == ScenarioType.SCENARIO_A:
            return self.path_gen.generate_straight_path(
                length_m=3.5,
                lateral_offset_m=float(rng.uniform(-0.20, 0.20)),
                heading_angle_deg=float(rng.uniform(-4.0, 4.0)),
            )
        else:
            if rng.random() < 0.5:
                return self.path_gen.generate_curved_path(
                    radius_m=float(rng.uniform(1.2, 2.5)),
                    turn_direction="left" if rng.random() < 0.5 else "right",
                    lateral_offset_m=float(rng.uniform(-0.15, 0.15)),
                )
            else:
                return self.path_gen.generate_straight_path(
                    lateral_offset_m=float(rng.uniform(-0.20, 0.20)),
                    heading_angle_deg=float(rng.uniform(-5.0, 5.0)),
                )

    def generate_dataset(
        self,
        num_samples: int = 6000,
        output_subdirs: bool = True,
        progress_interval: int = 500,
    ) -> Dict[str, Any]:
        """
        Generate synthetic dataset with episode-level split isolation and geometric sanity gate.
        
        Args:
            num_samples: Total number of frames to generate (4200 train, 900 val, 900 test for 6000).
            output_subdirs: If True, writes into data/synthetic/{train, val, test}.
            progress_interval: Print progress every N generated frames.
            
        Returns:
            Dictionary with generation statistics.
        """
        scenarios = [
            ScenarioType.SCENARIO_A,
            ScenarioType.SCENARIO_B,
            ScenarioType.SCENARIO_C,
            ScenarioType.SCENARIO_D,
        ]

        frames_per_episode = 5

        # Split proportions: 70% train, 15% val, 15% test
        if num_samples == 6000:
            target_splits = {"train": 4200, "val": 900, "test": 900}
        else:
            n_train = int(round(0.70 * num_samples))
            n_val = int(round(0.15 * num_samples))
            n_test = num_samples - n_train - n_val
            target_splits = {"train": n_train, "val": n_val, "test": n_test}

        # Build stratified episode plans for each split to ensure perfect scenario balance
        # and strictly disjoint episode sets (zero cross-split scene leakage).
        split_episodes_plan = {"train": [], "val": [], "test": []}
        global_ep_counter = 0

        for split_name, total_split_samples in target_splits.items():
            samples_per_sc = total_split_samples // len(scenarios)
            remainder = total_split_samples % len(scenarios)

            sc_alloc = {sc: samples_per_sc for sc in scenarios}
            for sc in scenarios[:remainder]:
                sc_alloc[sc] += 1

            for sc in scenarios:
                rem_frames = sc_alloc[sc]
                while rem_frames > 0:
                    f_in_ep = min(rem_frames, frames_per_episode)
                    split_episodes_plan[split_name].append((global_ep_counter, sc, f_in_ep))
                    rem_frames -= f_in_ep
                    global_ep_counter += 1

        print(f"Total planned episodes: {global_ep_counter} across splits: "
              f"train={len(split_episodes_plan['train'])}, "
              f"val={len(split_episodes_plan['val'])}, "
              f"test={len(split_episodes_plan['test'])}")

        manifests = {"train": [], "val": [], "test": []}
        global_frame_id = 0
        start_time = time.time()

        for split_name in ["train", "val", "test"]:
            split_img_dir = self.base_dir / "synthetic" / split_name / "images"
            split_img_dir.mkdir(parents=True, exist_ok=True)
            episodes = split_episodes_plan[split_name]

            print(f"\n--- Starting generation for split: {split_name.upper()} "
                  f"({target_splits[split_name]} frames in {len(episodes)} episodes) ---")

            for ep_id, scenario, frame_count in episodes:
                ep_seed = self.base_seed + ep_id * 1000
                episode_success = False
                traj_attempt = 0

                while traj_attempt < self.max_trajectory_attempts:
                    traj_seed = ep_seed + traj_attempt * 10007
                    traj_rng = np.random.default_rng(traj_seed)
                    candidate_path = self._generate_scenario_trajectory(scenario, traj_rng)

                    # 1. Episode-Level Geometric Sanity Gate
                    is_geom_valid, vis_anchors = self.check_trajectory_visibility(candidate_path)
                    if not is_geom_valid:
                        self.validator.record_geometry_rejection(
                            ep_id=ep_id,
                            scenario=scenario.value,
                            seed=traj_seed,
                            reason=f"Candidate trajectory has {vis_anchors} visible anchors (< {self.min_visible_anchors})",
                        )
                        traj_attempt += 1
                        continue

                    path = candidate_path

                    # 2. Render frames within this verified episode
                    ep_samples = []
                    episode_frames_valid = True

                    for f_in_ep in range(frame_count):
                        base_frame_seed = ep_seed + f_in_ep * 1000 + traj_attempt * 1000003
                        frame_valid = False

                        for frame_attempt in range(self.max_frame_attempts):
                            frame_seed = base_frame_seed + frame_attempt * 31
                            params = self.randomizer.sample_parameters(
                                scenario=scenario,
                                seed=frame_seed,
                                path_length_m=3.5,
                            )

                            img_rgb, visible_uv = self.capture_engine.render_frame_offline(path, params)

                            target_classes, target_u_coords, is_present = encode_row_anchor_targets(
                                waypoints_uv=visible_uv,
                                num_rows=self.config.model.num_row_anchors,
                                y_min=self.config.roi.y_min,
                                y_max=self.config.roi.y_max,
                                image_width=self.config.camera.input_width,
                                num_spatial_bins=self.config.model.num_spatial_bins,
                                waypoint_tolerance_px=self.config.dataset.waypoint_tolerance_px,
                            )

                            row_anchors = []
                            for r_i in range(self.config.model.num_row_anchors):
                                pres = bool(is_present[r_i])
                                row_anchors.append({
                                    "row_idx": r_i,
                                    "row_y_orig": int(self.row_y[r_i]),
                                    "u_coord": float(target_u_coords[r_i]) if pres else None,
                                    "grid_bin": int(target_classes[r_i]) if pres else None,
                                    "class_id": int(target_classes[r_i]),
                                    "present": pres,
                                })

                            is_valid, reason = self.validator.validate_sample(
                                img=img_rgb,
                                row_anchors=row_anchors,
                                metadata=params.to_dict(),
                                frame_id=global_frame_id + len(ep_samples),
                                scenario=scenario.value,
                                seed=frame_seed,
                            )

                            if is_valid:
                                frame_valid = True
                                ep_samples.append((img_rgb, row_anchors, params, frame_seed, visible_uv))
                                break
                            else:
                                # Geometry failure during frame rendering -> abort to regenerate trajectory
                                if SampleValidator.is_geometry_rejection(reason):
                                    break
                                # Otherwise, continue appearance retries up to max_frame_attempts

                        if not frame_valid:
                            episode_frames_valid = False
                            break

                    if episode_frames_valid:
                        # Episode successfully generated; persist images and manifest records
                        for img_rgb, row_anchors, params, frame_seed, visible_uv in ep_samples:
                            img_filename = f"syn_{global_frame_id:06d}.png"
                            img_path = split_img_dir / img_filename
                            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(str(img_path), img_bgr)

                            sample_record = {
                                "image_id": f"syn_{global_frame_id:06d}",
                                "image_path": f"images/{img_filename}",
                                "scenario": scenario.value,
                                "scene_id": f"ep_{ep_id:04d}",
                                "frame_id": global_frame_id,
                                "seed": frame_seed,
                                "width": self.config.camera.input_width,
                                "height": self.config.camera.input_height,
                                "roi": {
                                    "y_min": self.config.roi.y_min,
                                    "y_max": self.config.roi.y_max,
                                    "x_min": self.config.roi.x_min,
                                    "x_max": self.config.roi.x_max,
                                },
                                "row_anchors": row_anchors,
                                "path_points_2d": visible_uv[::2].tolist() if len(visible_uv) > 0 else [],
                                "metadata": params.to_dict(),
                            }
                            manifests[split_name].append(sample_record)
                            global_frame_id += 1

                            if global_frame_id % progress_interval == 0:
                                elapsed = time.time() - start_time
                                fps = global_frame_id / elapsed if elapsed > 0 else 0
                                print(f"  Progress: {global_frame_id}/{num_samples} frames generated "
                                      f"({global_frame_id/num_samples*100:.1f}%) | "
                                      f"Elapsed: {elapsed:.1f}s | Speed: {fps:.1f} FPS")

                        episode_success = True
                        break
                    else:
                        traj_attempt += 1

                if not episode_success:
                    raise RuntimeError(
                        f"Failed to generate valid trajectory and frames for episode {ep_id} "
                        f"({scenario.value}) after {self.max_trajectory_attempts} trajectory attempts."
                    )

        # Save manifests for all splits
        for split_name, samples_list in manifests.items():
            manifest_file = self.base_dir / "synthetic" / split_name / "annotations.json"
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump({
                    "version": "1.0",
                    "dataset_type": "synthetic",
                    "split": split_name,
                    "total_samples": len(samples_list),
                    "samples": samples_list,
                }, f, indent=2)
            print(f"Saved manifest: {manifest_file} ({len(samples_list)} samples)")

        # Persist rejection log in metadata
        self.validator.save_rejected_log()
        # Also persist to data/metadata for backwards compatibility
        compat_rej = self.base_dir / "metadata" / "rejected_samples.json"
        compat_rej.parent.mkdir(parents=True, exist_ok=True)
        with open(compat_rej, "w", encoding="utf-8") as f:
            json.dump({
                "total_rejected": len(self.validator.rejected_records),
                "rejected_samples": self.validator.rejected_records,
            }, f, indent=2)

        elapsed_total = time.time() - start_time
        print(f"\nAll {global_frame_id} frames generated in {elapsed_total:.1f}s "
              f"({global_frame_id/elapsed_total:.1f} FPS average).")

        stats = {
            "total_requested": num_samples,
            "total_generated": global_frame_id,
            "train_samples": len(manifests["train"]),
            "val_samples": len(manifests["val"]),
            "test_samples": len(manifests["test"]),
            "rejected_samples": len(self.validator.rejected_records),
            "manifests": {
                s: str(self.base_dir / "synthetic" / s / "annotations.json")
                for s in manifests
            },
        }
        return stats
