"""
Central Configuration Loader & Validation Schema
=================================================
Ensures every research parameter is traceable, validated, and strictly adheres to
the frozen research specification.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


@dataclass
class ExperimentConfig:
    name: str = "sict_shufflenet_row_anchor_baseline"
    seed: int = 42
    output_dir: str = "experiments"
    device: str = "cuda"


@dataclass
class CameraConfig:
    input_width: int = 640
    input_height: int = 480
    channels: int = 3


@dataclass
class ROIConfig:
    y_min: int = 200
    y_max: int = 470
    x_min: int = 0
    x_max: int = 640

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def width(self) -> int:
        return self.x_max - self.x_min


@dataclass
class ModelConfig:
    name: str = "shufflenet_v2_x0_5"
    pretrained: bool = True
    input_channels: int = 3
    input_height: int = 288
    input_width: int = 384
    num_row_anchors: int = 18       # M = 18
    num_spatial_bins: int = 100     # K = 100
    num_absence_classes: int = 1    # 1 absence class

    @property
    def total_classes(self) -> int:
        return self.num_spatial_bins + self.num_absence_classes  # 101


@dataclass
class SICTConfig:
    enabled: bool = True
    gamma_lum: float = 220.0
    delta_spec: float = 0.05
    alpha: float = 0.6
    beta: float = 0.4
    epsilon: float = 1.0e-5
    ambient_ratio: float = 0.2
    frame_avg_mode: str = "raw_intensity"
    out_channels: int = 3


@dataclass
class TrainingConfig:
    optimizer: str = "AdamW"
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1.0e-4
    initial_lr: float = 4.0e-3
    final_lr: float = 1.0e-5
    scheduler: str = "cosine_annealing"
    batch_size: int = 32
    max_epochs: int = 50
    num_workers: int = 0


@dataclass
class LossConfig:
    focal_gamma: float = 2.0
    lambda_smooth: float = 0.5
    lambda_curv: float = 0.75


@dataclass
class DatasetConfig:
    synthetic_target_total: int = 6000
    synthetic_train_count: int = 4200
    synthetic_val_count: int = 900
    synthetic_test_count: int = 900
    real_world_target_total: int = 1000
    waypoint_tolerance_px: float = 7.5
    eval_tolerance_px: float = 8.0


@dataclass
class ProjectConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sict: SICTConfig = field(default_factory=SICTConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    def validate(self) -> None:
        """Validate all parameters against frozen research specification."""
        assert self.camera.input_width == 640 and self.camera.input_height == 480, (
            f"Camera resolution must be 640x480, got {self.camera.input_width}x{self.camera.input_height}"
        )
        assert self.roi.y_min == 200 and self.roi.y_max == 470, (
            f"ROI y-range must be [200, 470], got [{self.roi.y_min}, {self.roi.y_max}]"
        )
        assert self.model.input_height == 288 and self.model.input_width == 384, (
            f"Network input must be (288, 384), got ({self.model.input_height}, {self.model.input_width})"
        )
        assert self.model.num_row_anchors == 18, (
            f"M must be 18 anchor rows, got {self.model.num_row_anchors}"
        )
        assert self.model.num_spatial_bins == 100, (
            f"K must be 100 spatial bins, got {self.model.num_spatial_bins}"
        )
        assert self.model.total_classes == 101, (
            f"Total classes per row must be 101, got {self.model.total_classes}"
        )
        assert self.sict.gamma_lum == 220.0, f"Gamma_lum must be 220, got {self.sict.gamma_lum}"
        assert self.sict.delta_spec == 0.05, f"delta_spec must be 0.05, got {self.sict.delta_spec}"
        assert self.sict.alpha == 0.6 and self.sict.beta == 0.4, (
            f"alpha/beta must be 0.6/0.4, got {self.sict.alpha}/{self.sict.beta}"
        )
        assert self.sict.ambient_ratio == 0.2, (
            f"ambient_ratio must be 0.2, got {self.sict.ambient_ratio}"
        )
        assert self.loss.focal_gamma == 2.0, f"focal_gamma must be 2.0, got {self.loss.focal_gamma}"
        assert self.loss.lambda_smooth == 0.5, f"lambda_smooth must be 0.5, got {self.loss.lambda_smooth}"
        assert self.loss.lambda_curv == 0.75, f"lambda_curv must be 0.75, got {self.loss.lambda_curv}"
        assert self.dataset.waypoint_tolerance_px == 7.5, (
            f"waypoint_tolerance_px must be 7.5, got {self.dataset.waypoint_tolerance_px}"
        )
        assert self.dataset.eval_tolerance_px == 8.0, (
            f"eval_tolerance_px must be 8.0, got {self.dataset.eval_tolerance_px}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(config_path: Optional[str] = None) -> ProjectConfig:
    """Load configuration from YAML file, with strict validation."""
    if config_path is None:
        # Default to configs/default_config.yaml relative to project root
        base_dir = Path(__file__).resolve().parent.parent
        config_path = base_dir / "configs" / "default_config.yaml"

    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    cfg = ProjectConfig(
        experiment=ExperimentConfig(**data.get("experiment", {})),
        camera=CameraConfig(**data.get("camera", {})),
        roi=ROIConfig(**data.get("roi", {})),
        model=ModelConfig(**data.get("model", {})),
        sict=SICTConfig(**data.get("sict", {})),
        training=TrainingConfig(**data.get("training", {})),
        loss=LossConfig(**data.get("loss", {})),
        dataset=DatasetConfig(**data.get("dataset", {})),
    )
    cfg.validate()
    return cfg
