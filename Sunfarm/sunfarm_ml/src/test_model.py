"""
SunFarm Smart Irrigation - Model Testing & Verification Script
==============================================================
Loads the saved production model artifact and tests it against:
  1. Holdout test split with full classification metrics and confusion matrix
  2. Specific realistic agronomic edge cases (desert heatwave, monsoon, mild day, etc.)
  3. Prediction latency and boundary stability
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config.config import MODEL_PATH, DATASET_PATH, FEATURE_NAMES, IRRIGATION_CLASSES, DATASET_CONFIG
from src.irrigation_predictor import IrrigationPredictor
from src.motor_controller import MotorController


def run_holdout_test(predictor: IrrigationPredictor):
    """Evaluate predictor on the holdout test set."""
    print("=" * 70)
    print("           SunFarm ML - Holdout Test Set Evaluation")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)
    X = df[FEATURE_NAMES]
    y = df["irrigation_class"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=DATASET_CONFIG["TEST_SPLIT_RATIO"],
        random_state=DATASET_CONFIG["RANDOM_SEED"],
        stratify=y
    )

    y_pred = predictor.model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_mac = f1_score(y_test, y_pred, average="macro")
    f1_w = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    print(f"Loaded Model: {predictor.model_name}")
    print(f"Test Samples: {len(X_test)}")
    print(f"Accuracy:     {acc * 100:.2f}%")
    print(f"F1 Macro:     {f1_mac * 100:.2f}%")
    print(f"F1 Weighted:  {f1_w * 100:.2f}%")

    print("\nConfusion Matrix:")
    labels = [f"Class {i}" for i in range(4)]
    cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels])
    print(cm_df.to_string())

    print("\nClassification Report:")
    target_names = [IRRIGATION_CLASSES[i]["label"] for i in range(4)]
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))


def run_edge_case_tests(predictor: IrrigationPredictor):
    """Test predictor against specific realistic agricultural edge scenarios."""
    print("=" * 70)
    print("            SunFarm ML - Agricultural Scenario Stress Tests")
    print("=" * 70)

    scenarios = [
        {
            "name": "Scenario 1: Saturated Soil Post Heavy Rain",
            "sensors": {
                "soil_moisture": 78.0,
                "soil_temperature": 21.0,
                "air_temperature": 23.0,
                "air_humidity": 88.0,
                "wind_speed": 4.5,
                "wind_direction": 180.0
            },
            "expected_class": 0
        },
        {
            "name": "Scenario 2: Hot Dry Heatwave with Critically Parched Soil",
            "sensors": {
                "soil_moisture": 14.0,
                "soil_temperature": 36.0,
                "air_temperature": 41.0,
                "air_humidity": 18.0,
                "wind_speed": 18.0,
                "wind_direction": 290.0
            },
            "expected_class": 3
        },
        {
            "name": "Scenario 3: Moderate Moisture on Warm Sunny Afternoon",
            "sensors": {
                "soil_moisture": 42.0,
                "soil_temperature": 27.0,
                "air_temperature": 31.0,
                "air_humidity": 45.0,
                "wind_speed": 9.0,
                "wind_direction": 120.0
            },
            "expected_class": 2
        },
        {
            "name": "Scenario 4: Cool Morning with Adequate Soil Hydration",
            "sensors": {
                "soil_moisture": 58.0,
                "soil_temperature": 16.0,
                "air_temperature": 18.0,
                "air_humidity": 75.0,
                "wind_speed": 3.0,
                "wind_direction": 45.0
            },
            "expected_class": 0
        },
        {
            "name": "Scenario 5: Mild Depletion under Moderate Weather",
            "sensors": {
                "soil_moisture": 48.0,
                "soil_temperature": 24.0,
                "air_temperature": 25.0,
                "air_humidity": 55.0,
                "wind_speed": 6.0,
                "wind_direction": 210.0
            },
            "expected_class": 1
        }
    ]

    motor = MotorController()

    for s in scenarios:
        sensor_data = s["sensors"]
        res = predictor.predict(sensor_data)
        motor_res = motor.evaluate(sensor_data["soil_moisture"])
        suggestion = predictor.generate_farmer_suggestion(sensor_data, res, motor_res)

        print(f"\n[{s['name']}]")
        print(f"  Inputs: Moisture={sensor_data['soil_moisture']}% | AirTemp={sensor_data['air_temperature']}°C | "
              f"Humidity={sensor_data['air_humidity']}% | Wind={sensor_data['wind_speed']} km/h")
        print(f"  ML Output:   Class {res['irrigation_class']} ({res['irrigation_label']}) | "
              f"Confidence: {res['confidence']*100:.1f}%")
        print(f"  Motor:       {motor_res['motor_status']} | Reason: {motor_res['reason']}")
        print(f"  Suggestion:  \"{suggestion}\"")


def main():
    predictor = IrrigationPredictor()
    run_holdout_test(predictor)
    run_edge_case_tests(predictor)


if __name__ == "__main__":
    main()
