"""
Simulation and Synthetic Data Generation Module
===============================================
Encapsulates CoppeliaSim ZeroMQ Remote API integration, camera optics,
procedural path generation, domain randomization, and dataset synthesis.
"""

from src.simulation.camera import CameraConfig, PinholeCameraModel
from src.simulation.domain_randomization import DomainRandomizer, ScenarioConfig, ScenarioType
from src.simulation.path_generator import SplinePathGenerator, Path3D
from src.simulation.client import CoppeliaSimClient
from src.simulation.validator import SampleValidator
from src.simulation.generator import SyntheticDatasetGenerator

__all__ = [
    "CameraConfig",
    "PinholeCameraModel",
    "DomainRandomizer",
    "ScenarioConfig",
    "ScenarioType",
    "SplinePathGenerator",
    "Path3D",
    "CoppeliaSimClient",
    "SampleValidator",
    "SyntheticDatasetGenerator",
]
