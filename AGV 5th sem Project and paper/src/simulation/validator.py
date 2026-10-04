"""
Automated Sample and Annotation Validator
=========================================
Strict quality control gate for synthesized dataset frames.
Rejects and logs frames with:
- Corrupted image bytes or incorrect dimensions (must be strictly 640x480x3)
- Invalid class IDs (must be in [0, 100])
- Impossible coordinates (< 0 or >= 640 for present anchors)
- Missing metadata or domain attributes
- Unexpected total absence when visible path exists in ROI
Rejected samples are logged to metadata/rejected_samples.json without silent repairing.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import numpy as np


class SampleValidator:
    """
    Validates synthetic sample records against research schema before writing to dataset.
    """

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or (Path("data") / "metadata" / "rejected_samples.json")
        self.rejected_records: List[Dict[str, Any]] = []

    def validate_sample(
        self,
        img: np.ndarray,
        row_anchors: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        frame_id: int,
        scenario: str,
        seed: int,
        require_visible_path: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate single sample.
        
        Returns:
            (is_valid, rejection_reason)
        """
        # 1. Image dimension check
        if img is None:
            return self._reject(frame_id, scenario, seed, "Image array is None or corrupt")
        if img.shape != (480, 640, 3):
            return self._reject(frame_id, scenario, seed, f"Invalid dimensions: {img.shape} != (480, 640, 3)")
        if img.dtype != np.uint8:
            return self._reject(frame_id, scenario, seed, f"Invalid dtype: {img.dtype} != uint8")

        # 2. Row anchor length check
        if len(row_anchors) != 18:
            return self._reject(frame_id, scenario, seed, f"Expected 18 row anchors, got {len(row_anchors)}")

        # 3. Anchor coordinate and class validation
        present_count = 0
        for item in row_anchors:
            r_idx = item.get("row_idx")
            if r_idx is None or not (0 <= r_idx < 18):
                return self._reject(frame_id, scenario, seed, f"Invalid row_idx: {r_idx}")

            is_pres = item.get("present", False)
            class_id = item.get("class_id")

            if is_pres:
                present_count += 1
                u = item.get("u_coord")
                if u is None or not (0.0 <= u < 640.0):
                    return self._reject(
                        frame_id, scenario, seed,
                        f"Row {r_idx} marked present but coordinate u={u} is out-of-bounds [0, 640)"
                    )
                if class_id is None or not (0 <= class_id < 100):
                    return self._reject(
                        frame_id, scenario, seed,
                        f"Row {r_idx} marked present but class_id={class_id} not in spatial bins [0, 99]"
                    )
            else:
                if class_id != 100:
                    return self._reject(
                        frame_id, scenario, seed,
                        f"Row {r_idx} marked absent but class_id={class_id} != 100"
                    )

        # 4. Total unexpected absence check
        if require_visible_path and present_count == 0:
            return self._reject(
                frame_id, scenario, seed,
                "All 18 row anchors are unexpectedly absent in a scene designed to have a visible path"
            )

        # 5. Metadata verification
        required_meta_keys = ["seed", "scenario", "specular_exponent", "diffuse_albedo", "lighting_lux"]
        for k in required_meta_keys:
            if k not in metadata:
                return self._reject(frame_id, scenario, seed, f"Missing required metadata key: '{k}'")

        return True, None

    def _reject(
        self,
        frame_id: int,
        scenario: str,
        seed: int,
        reason: str,
    ) -> Tuple[bool, str]:
        record = {
            "frame_id": frame_id,
            "scenario": scenario,
            "seed": seed,
            "rejection_reason": reason,
        }
        self.rejected_records.append(record)
        return False, reason

    def save_rejected_log(self) -> None:
        """Persist rejected samples log to JSON file."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_rejected": len(self.rejected_records),
                "rejected_samples": self.rejected_records,
            }, f, indent=2)
