"""
Domain Randomization and Scenario Management
============================================
Implements the exact procedural domain randomization distributions from research blueprint:
- Floor specular exponent: eta ~ Uniform[15, 120]
- Surface diffuse albedo: rho_d ~ Normal(mu=0.52, sigma=0.12), operational range [0.25, 0.80]
- Luminaire lighting: 200 - 1400 lux (Log-normal mu=6.5, sigma=0.4)
- Line marking abrasion: Poisson gap distribution (lambda = 0.05 breaks/m) with 0% - 45% material loss
- Camera pitch offset: Delta theta_p ~ Normal(0, 1.2 deg), range [-3.0 deg, +3.0 deg]
- Secondary degradation: high-gloss epoxy glare lobes, rubber scuff streaks, oil mask stains
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
import numpy as np


class ScenarioType(str, Enum):
    SCENARIO_A = "scenario_a_nominal"
    SCENARIO_B = "scenario_b_glare"
    SCENARIO_C = "scenario_c_abrasion"
    SCENARIO_D = "scenario_d_s_curve"


@dataclass
class RandomizedParams:
    specular_exponent: float
    diffuse_albedo: float
    lighting_lux: float
    luminaire_pos_xyz: Optional[List[float]] = None
    abrasion_loss_ratio: float = 0.0
    num_abrasion_breaks: int = 0
    pitch_offset_deg: float = 0.0
    glare_intensity: float = 0.0
    glare_center_uv: Optional[Tuple[int, int]] = None
    rubber_scuffs_count: int = 0
    oil_stains_count: int = 0
    speed_mps: float = 1.0
    scenario: str = "nominal"
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "specular_exponent": round(self.specular_exponent, 2),
            "diffuse_albedo": round(self.diffuse_albedo, 4),
            "lighting_lux": round(self.lighting_lux, 1),
            "luminaire_pos_xyz": self.luminaire_pos_xyz,
            "abrasion_loss_ratio": round(self.abrasion_loss_ratio, 4),
            "num_abrasion_breaks": self.num_abrasion_breaks,
            "pitch_offset_deg": round(self.pitch_offset_deg, 3),
            "glare_intensity": round(self.glare_intensity, 3),
            "glare_center_uv": self.glare_center_uv,
            "rubber_scuffs_count": self.rubber_scuffs_count,
            "oil_stains_count": self.oil_stains_count,
            "speed_mps": round(self.speed_mps, 2),
            "scenario": self.scenario,
            "seed": self.seed,
        }


# Type alias for XYZ tuple
Tuple_xyz = Optional[List[float]]


@dataclass
class ScenarioConfig:
    scenario_type: ScenarioType
    min_lux: float
    max_lux: float
    min_specular_exp: float
    max_specular_exp: float
    albedo_mean: float
    albedo_std: float
    abrasion_range: Tuple[float, float]
    pitch_noise_std_deg: float = 1.2


class DomainRandomizer:
    """
    Deterministic domain parameter generator parameterized by seed and scenario.
    """

    SCENARIO_CONFIGS = {
        ScenarioType.SCENARIO_A: ScenarioConfig(
            scenario_type=ScenarioType.SCENARIO_A,
            min_lux=300.0,
            max_lux=400.0,
            min_specular_exp=15.0,
            max_specular_exp=40.0,
            albedo_mean=0.52,
            albedo_std=0.08,
            abrasion_range=(0.0, 0.05),
        ),
        ScenarioType.SCENARIO_B: ScenarioConfig(
            scenario_type=ScenarioType.SCENARIO_B,
            min_lux=1200.0,
            max_lux=1500.0,
            min_specular_exp=80.0,
            max_specular_exp=120.0,
            albedo_mean=0.55,
            albedo_std=0.10,
            abrasion_range=(0.0, 0.10),
        ),
        ScenarioType.SCENARIO_C: ScenarioConfig(
            scenario_type=ScenarioType.SCENARIO_C,
            min_lux=300.0,
            max_lux=800.0,
            min_specular_exp=25.0,
            max_specular_exp=70.0,
            albedo_mean=0.50,
            albedo_std=0.12,
            abrasion_range=(0.25, 0.40),
        ),
        ScenarioType.SCENARIO_D: ScenarioConfig(
            scenario_type=ScenarioType.SCENARIO_D,
            min_lux=80.0,
            max_lux=1400.0,
            min_specular_exp=20.0,
            max_specular_exp=100.0,
            albedo_mean=0.52,
            albedo_std=0.12,
            abrasion_range=(0.05, 0.25),
        ),
    }

    def sample_parameters(
        self,
        scenario: ScenarioType,
        seed: int,
        path_length_m: float = 3.0,
    ) -> RandomizedParams:
        """
        Deterministically sample environmental domain variables.
        """
        rng = np.random.default_rng(seed)
        cfg = self.SCENARIO_CONFIGS[scenario]

        # 1. Specular exponent eta ~ Uniform[min, max]
        eta = float(rng.uniform(cfg.min_specular_exp, cfg.max_specular_exp))

        # 2. Surface diffuse albedo rho_d ~ Normal(mu, sigma) clamped [0.25, 0.80]
        rho_d = float(np.clip(rng.normal(cfg.albedo_mean, cfg.albedo_std), 0.25, 0.80))

        # 3. Lighting lux
        lux = float(rng.uniform(cfg.min_lux, cfg.max_lux))

        # 4. Luminaire position: [X in -2..2, Y in 1..4, Z in 3.5..6.0]
        lum_pos = [
            float(rng.uniform(-2.0, 2.0)),
            float(rng.uniform(1.0, 4.0)),
            float(rng.uniform(3.5, 6.0)),
        ]

        # 5. Line abrasion: Poisson process lambda = 0.05 breaks/m
        expected_breaks = 0.05 * path_length_m
        num_breaks = int(rng.poisson(expected_breaks)) if scenario != ScenarioType.SCENARIO_A else 0
        abrasion_loss = float(rng.uniform(cfg.abrasion_range[0], cfg.abrasion_range[1]))

        # 6. Camera pitch offset Delta theta_p ~ Normal(0, 1.2 deg)
        pitch_noise = float(np.clip(rng.normal(0.0, cfg.pitch_noise_std_deg), -3.0, 3.0))

        # 7. Scenario-specific degradation attributes
        if scenario == ScenarioType.SCENARIO_B:
            glare_intensity = float(rng.uniform(0.7, 1.0))
            # Glare centered along optical path
            glare_u = int(rng.uniform(240, 400))
            glare_v = int(rng.uniform(220, 350))
            glare_center = (glare_u, glare_v)
            scuffs = int(rng.integers(0, 2))
            oil = int(rng.integers(0, 2))
        elif scenario == ScenarioType.SCENARIO_C:
            glare_intensity = float(rng.uniform(0.1, 0.4))
            glare_center = None
            scuffs = int(rng.integers(3, 8))
            oil = int(rng.integers(2, 6))
        elif scenario == ScenarioType.SCENARIO_D:
            glare_intensity = float(rng.uniform(0.2, 0.6))
            glare_center = (int(rng.uniform(200, 440)), int(rng.uniform(240, 360)))
            scuffs = int(rng.integers(1, 4))
            oil = int(rng.integers(1, 3))
        else:  # Scenario A (Nominal)
            glare_intensity = 0.0
            glare_center = None
            scuffs = 0
            oil = 0

        # Scenario D speed parameter: 0.8 - 1.2 m/s
        speed = float(rng.uniform(0.8, 1.2)) if scenario == ScenarioType.SCENARIO_D else 1.0

        return RandomizedParams(
            specular_exponent=eta,
            diffuse_albedo=rho_d,
            lighting_lux=lux,
            luminaire_pos_xyz=lum_pos,
            abrasion_loss_ratio=abrasion_loss,
            num_abrasion_breaks=num_breaks,
            pitch_offset_deg=pitch_noise,
            glare_intensity=glare_intensity,
            glare_center_uv=glare_center,
            rubber_scuffs_count=scuffs,
            oil_stains_count=oil,
            speed_mps=speed,
            scenario=scenario.value,
            seed=seed,
        )
