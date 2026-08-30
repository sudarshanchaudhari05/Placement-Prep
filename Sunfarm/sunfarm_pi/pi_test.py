#!/usr/bin/env python3
"""
SunFarm Smart Irrigation - Raspberry Pi Edge Verification Script
================================================================
Verifies that the pre-trained ML model and deterministic motor controller
execute correctly in the Raspberry Pi edge environment completely offline.

Tests three standard agronomic test cases:
  1. CASE 1: Very Dry (Critical Water Deficit)
  2. CASE 2: Adequately Wet (Post-Rain / Saturated)
  3. CASE 3: Moderate (Adequate / Maintenance)
"""

import sys
import platform
import time
from pathlib import Path

# Enable UTF-8 encoding on standard output for cross-platform degree symbol display
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.config import MODEL_PATH, MOTOR_CONFIG
from src.irrigation_predictor import IrrigationPredictor
from src.motor_controller import MotorController


def print_system_diagnostics():
    """Prints edge system environment and Python runtime diagnostics."""
    print("=" * 65)
    print("      SunFarm Raspberry Pi Edge System Diagnostics")
    print("=" * 65)
    print(f"Python Version:   {platform.python_version()} ({platform.python_implementation()})")
    print(f"Platform:         {platform.platform()}")
    print(f"Architecture:     {platform.machine()}")
    print(f"Model Artifact:   {MODEL_PATH}")
    print(f"Model Exists:     {MODEL_PATH.exists()} ({MODEL_PATH.stat().st_size / 1024:.1f} KB)" if MODEL_PATH.exists() else f"Model Exists: False")
    print("-" * 65)


def display_case_result(case_title: str, sensors: dict, motor_res: dict, ml_res: dict, suggestion: str, latency_ms: float):
    """Prints structured test case evaluation."""
    print("\n" + "=" * 65)
    print(f" {case_title}")
    print("=" * 65)
    print("Sensor Data:")
    print(f"  Soil Moisture:    {sensors['soil_moisture']:>5.1f} %")
    print(f"  Soil Temperature: {sensors['soil_temperature']:>5.1f} °C")
    print(f"  Air Temperature:  {sensors['air_temperature']:>5.1f} °C")
    print(f"  Air Humidity:     {sensors['air_humidity']:>5.1f} %")
    print(f"  Wind Speed:       {sensors['wind_speed']:>5.1f} km/h")
    print(f"  Wind Direction:   {sensors['wind_direction']:>5.1f} °")
    print()
    print("--- MOTOR CONTROL ---")
    print(f"  Motor Status:     {motor_res['motor_status']}")
    print(f"  Motor Reason:     {motor_res['reason']}")
    print()
    print("--- ML IRRIGATION PREDICTION ---")
    print(f"  Irrigation Class: {ml_res['irrigation_class']}")
    print(f"  Recommendation:   {ml_res['irrigation_label']}")
    print(f"  Confidence:       {ml_res['confidence'] * 100:.1f} %")
    print(f"  Inference Time:   {latency_ms:.2f} ms")
    print()
    print("--- FARMER SUGGESTION ---")
    print(f"  \"{suggestion}\"")
    print("=" * 65)


def run_pi_verification():
    """Loads the model and evaluates standard predefined benchmark scenarios."""
    print_system_diagnostics()

    # 1. Load ML Model
    print("[1/2] Loading trained Random Forest model artifact...")
    try:
        t0 = time.perf_counter()
        predictor = IrrigationPredictor(MODEL_PATH)
        load_time = (time.perf_counter() - t0) * 1000.0
        print(f"[OK] Model successfully loaded in {load_time:.2f} ms! Model Name: {predictor.model_name}")
    except Exception as e:
        print(f"[FAIL] Error loading model from {MODEL_PATH}: {e}")
        sys.exit(1)

    # 2. Initialize Motor Controller
    print("[2/2] Initializing Deterministic Motor Controller (Safety Layer)...")
    motor = MotorController(
        lower_threshold=MOTOR_CONFIG["SOIL_MOISTURE_LOWER_THRESHOLD"],
        upper_threshold=MOTOR_CONFIG["SOIL_MOISTURE_UPPER_THRESHOLD"]
    )
    print(f"[OK] Motor Controller Active: Hysteresis Thresholds [{motor.lower_threshold}% - {motor.upper_threshold}%]")

    # Predefined scenarios
    test_cases = [
        {
            "title": "CASE 1 — Very dry (Critical Deficit)",
            "sensors": {
                "soil_moisture": 15.0,
                "soil_temperature": 32.0,
                "air_temperature": 38.0,
                "air_humidity": 30.0,
                "wind_speed": 15.0,
                "wind_direction": 180.0
            }
        },
        {
            "title": "CASE 2 — Adequately wet (Post-Rain / Saturated)",
            "sensors": {
                "soil_moisture": 75.0,
                "soil_temperature": 27.0,
                "air_temperature": 28.0,
                "air_humidity": 80.0,
                "wind_speed": 4.0,
                "wind_direction": 90.0
            }
        },
        {
            "title": "CASE 3 — Moderate (Adequate / Maintenance)",
            "sensors": {
                "soil_moisture": 45.0,
                "soil_temperature": 28.0,
                "air_temperature": 30.0,
                "air_humidity": 60.0,
                "wind_speed": 5.0,
                "wind_direction": 270.0
            }
        }
    ]

    print("\n>>> Executing Predefined Scenario Evaluations on Raspberry Pi Pipeline...")

    for case in test_cases:
        sensors = case["sensors"]

        # Time the inference
        t_start = time.perf_counter()
        ml_res = predictor.predict(sensors)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        motor_res = motor.evaluate(sensors["soil_moisture"])
        suggestion = predictor.generate_farmer_suggestion(sensors, ml_res, motor_res)

        display_case_result(
            case_title=case["title"],
            sensors=sensors,
            motor_res=motor_res,
            ml_res=ml_res,
            suggestion=suggestion,
            latency_ms=latency_ms
        )

    print("\n[SUCCESS] All Raspberry Pi test cases executed successfully completely offline!")


if __name__ == "__main__":
    run_pi_verification()
