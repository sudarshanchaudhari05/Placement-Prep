"""
SunFarm Smart Irrigation System - Configuration Module
======================================================
Centralized configuration parameters for sensor boundaries, motor control
hysteresis thresholds, ML model hyper-parameters, and file system paths.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"

DATASET_PATH = DATA_DIR / "synthetic_irrigation_data.csv"
MODEL_PATH = MODELS_DIR / "irrigation_model.pkl"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

# Feature Names (Ordered expected input for ML model)
FEATURE_NAMES = [
    "soil_moisture",     # Percentage (%) [0.0 - 100.0]
    "soil_temperature",  # Degrees Celsius (°C) [-10.0 - 60.0]
    "air_temperature",   # Degrees Celsius (°C) [-10.0 - 60.0]
    "air_humidity",      # Percentage (%) [0.0 - 100.0]
    "wind_speed",        # Kilometers per hour (km/h) [0.0 - 120.0]
    "wind_direction"     # Degrees (°) [0.0 - 360.0]
]

# Physical Sensor Boundaries & Plausibility Limits
SENSOR_BOUNDS = {
    "soil_moisture": {"min": 0.0, "max": 100.0, "unit": "%"},
    "soil_temperature": {"min": -10.0, "max": 60.0, "unit": "°C"},
    "air_temperature": {"min": -10.0, "max": 60.0, "unit": "°C"},
    "air_humidity": {"min": 0.0, "max": 100.0, "unit": "%"},
    "wind_speed": {"min": 0.0, "max": 120.0, "unit": "km/h"},
    "wind_direction": {"min": 0.0, "max": 360.0, "unit": "°"}
}

# Deterministic Motor Controller Hysteresis Thresholds
# Motor turns ON when soil moisture drops strictly below LOWER_THRESHOLD.
# Motor turns OFF when soil moisture rises strictly above UPPER_THRESHOLD.
# When soil moisture is between [LOWER_THRESHOLD, UPPER_THRESHOLD],
# the controller holds the previous state to prevent rapid relay cycling (hysteresis).
MOTOR_CONFIG = {
    "SOIL_MOISTURE_LOWER_THRESHOLD": 30.0,  # (%) Turn ON below this
    "SOIL_MOISTURE_UPPER_THRESHOLD": 55.0,  # (%) Turn OFF above this
    "DEFAULT_INITIAL_STATE": "OFF"
}

# ML Irrigation Classification Levels
IRRIGATION_CLASSES = {
    0: {
        "code": "NO_IRRIGATION",
        "label": "No irrigation required",
        "short_label": "NO IRRIGATION",
        "description": "Soil has ample moisture. Crop water stress is minimal."
    },
    1: {
        "code": "LOW_IRRIGATION",
        "label": "Low irrigation requirement",
        "short_label": "LOW IRRIGATION",
        "description": "Mild water depletion. Light or scheduled maintenance watering suggested."
    },
    2: {
        "code": "MEDIUM_IRRIGATION",
        "label": "Medium irrigation requirement",
        "short_label": "MEDIUM IRRIGATION",
        "description": "Moderate moisture deficit or high evaporative weather demand. Irrigation advised."
    },
    3: {
        "code": "HIGH_IRRIGATION",
        "label": "High irrigation requirement",
        "short_label": "HIGH IRRIGATION",
        "description": "Severe soil dryness and/or extreme evaporative stress. Urgent watering required."
    }
}

# Synthetic Dataset Generation Parameters
DATASET_CONFIG = {
    "NUM_SAMPLES": 10000,
    "RANDOM_SEED": 42,
    "TEST_SPLIT_RATIO": 0.20
}
