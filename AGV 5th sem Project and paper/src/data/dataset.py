"""
PyTorch Dataset for Row-Anchor Path Tracking
============================================
Handles annotation loading, verification, and preprocessing.
Specification format supports:
  - image path
  - 18 row-anchor labels
  - absence labels (class 100 / token 101)
  - original image dimensions (640x480)
  - optional physical metadata (specular exponent, lighting, abrasion, etc.)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from configs.config import ProjectConfig
from src.data.transforms import PreprocessingPipeline


class RowAnchorDataset(Dataset):
    """
    Row-Anchor AGV Dataset conforming to frozen research specification.
    """

    def __init__(
        self,
        annotation_file: Union[str, Path],
        config: Optional[ProjectConfig] = None,
        split: Optional[str] = None,
        apply_sict: bool = True,
        transform: Optional[PreprocessingPipeline] = None,
    ):
        self.annotation_file = Path(annotation_file)
        assert self.annotation_file.exists(), (
            f"Annotation file not found: {self.annotation_file}"
        )

        self.config = config or ProjectConfig()
        self.split = split

        if transform is None:
            self.transform = PreprocessingPipeline(
                roi_config=self.config.roi,
                model_config=self.config.model,
                sict_config=self.config.sict,
                apply_sict=apply_sict,
                normalize=True,
            )
        else:
            self.transform = transform

        with open(self.annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            self.samples = data
        elif isinstance(data, dict) and "samples" in data:
            self.samples = data["samples"]
        else:
            raise ValueError(f"Invalid annotation format in {self.annotation_file}")

        if self.split is not None:
            filtered = [s for s in self.samples if s.get("split") == self.split]
            assert len(filtered) > 0, (
                f"Split '{self.split}' requested but no matching samples found in {self.annotation_file}"
            )
            self.samples = filtered

        self.root_dir = self.annotation_file.parent
        self._validate_schema()

    def _validate_schema(self) -> None:
        """Validate sample entries against specification."""
        assert len(self.samples) > 0, "Annotation dataset contains 0 samples!"
        first = self.samples[0]
        assert "image_path" in first, "Sample missing 'image_path'"
        assert "row_anchors" in first, "Sample missing 'row_anchors'"
        assert len(first["row_anchors"]) == self.config.model.num_row_anchors, (
            f"Expected {self.config.model.num_row_anchors} row anchors per sample, "
            f"got {len(first['row_anchors'])}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Resolve image path relative to dataset root or absolute
        img_p = Path(sample["image_path"])
        if not img_p.is_absolute():
            img_p = self.root_dir / img_p

        assert img_p.exists(), f"Image file not found: {img_p}"

        # Load image via OpenCV
        img_bgr = cv2.imread(str(img_p))
        assert img_bgr is not None, f"Failed to load image from: {img_p}"
        h, w, c = img_bgr.shape
        assert (w, h) == (self.config.camera.input_width, self.config.camera.input_height), (
            f"Image dimensions ({w}, {h}) do not match specification "
            f"({self.config.camera.input_width}, {self.config.camera.input_height})"
        )

        # Preprocess into (3, 288, 384) tensor
        tensor_img = self.transform(img_bgr, is_bgr=True)

        # Parse row-anchor labels
        row_anchors = sample["row_anchors"]
        num_rows = self.config.model.num_row_anchors
        absence_class = self.config.model.num_spatial_bins  # 100

        target_classes = np.full(num_rows, absence_class, dtype=np.int64)
        target_coords = np.full(num_rows, -1.0, dtype=np.float32)
        presence = np.zeros(num_rows, dtype=bool)

        for item in row_anchors:
            r_idx = item["row_idx"]
            assert 0 <= r_idx < num_rows, f"Invalid row index: {r_idx}"
            is_pres = item.get("present", True)
            presence[r_idx] = is_pres
            if is_pres:
                cls_id = item["class_id"]
                assert 0 <= cls_id < absence_class, (
                    f"Present class index {cls_id} out of bounds [0, {absence_class-1}]"
                )
                target_classes[r_idx] = cls_id
                target_coords[r_idx] = float(item["u_coord"])
            else:
                target_classes[r_idx] = absence_class
                target_coords[r_idx] = -1.0

        return {
            "image": tensor_img,  # (3, 288, 384)
            "targets": torch.from_numpy(target_classes).long(),  # (18,)
            "u_coords": torch.from_numpy(target_coords).float(),  # (18,)
            "presence": torch.from_numpy(presence).bool(),  # (18,)
            "metadata": sample.get("metadata", {}),
            "image_path": str(img_p),
        }
