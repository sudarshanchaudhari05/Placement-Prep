"""
SunFarm Smart Irrigation - Raspberry Pi Inference Engine & Advisor
==================================================================
Lightweight inference engine for Raspberry Pi edge execution. Loads the
pre-trained Random Forest model and generates irrigation recommendations,
confidence scores, and actionable farmer advice.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import joblib

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import MODEL_PATH, FEATURE_NAMES, IRRIGATION_CLASSES, SENSOR_BOUNDS


class IrrigationPredictor:
    """
    Inference wrapper for the SunFarm ML irrigation recommendation model on Raspberry Pi.
    """

    def __init__(self, model_path: Union[str, Path] = MODEL_PATH):
        """
        Initialize predictor by loading the trained model artifact.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model artifact not found at '{self.model_path}'."
            )

        artifact = joblib.load(self.model_path)
        if isinstance(artifact, dict) and "model" in artifact:
            self.model = artifact["model"]
            self.model_name = artifact.get("model_name", "Trained ML Model")
            self.feature_names = artifact.get("features", FEATURE_NAMES)
            self.metrics = artifact.get("metrics", {})
        else:
            self.model = artifact
            self.model_name = "Trained ML Model"
            self.feature_names = FEATURE_NAMES
            self.metrics = {}

    def validate_sensor_payload(self, sensor_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Validates that all 6 required sensor parameters exist and are valid numbers.
        """
        cleaned = {}
        for feature in self.feature_names:
            if feature not in sensor_data:
                raise ValueError(f"Missing required sensor parameter: '{feature}'")

            val = sensor_data[feature]
            if val is None or not isinstance(val, (int, float, np.number)):
                raise ValueError(f"Invalid non-numeric value for sensor '{feature}': {val}")

            cleaned[feature] = float(val)

        return cleaned

    def predict(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts irrigation requirement given a dictionary of 6 sensor inputs.

        Expected input format:
        {
            "soil_moisture": float,
            "soil_temperature": float,
            "air_temperature": float,
            "air_humidity": float,
            "wind_speed": float,
            "wind_direction": float
        }

        Returns:
        {
            "irrigation_class": int (0-3),
            "irrigation_label": str,
            "confidence": float (0.0 to 1.0),
            "probabilities": dict
        }
        """
        clean_inputs = self.validate_sensor_payload(sensor_data)
        input_df = pd.DataFrame([clean_inputs], columns=self.feature_names)

        pred_class = int(self.model.predict(input_df)[0])

        # Compute probability distribution and confidence score
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(input_df)[0]
            confidence = float(probs[pred_class])
            prob_dict = {
                IRRIGATION_CLASSES[i]["short_label"]: round(float(probs[i]), 4)
                for i in range(len(probs))
            }
        else:
            confidence = 1.0
            prob_dict = {IRRIGATION_CLASSES[pred_class]["short_label"]: 1.0}

        class_meta = IRRIGATION_CLASSES.get(
            pred_class,
            {"label": "Unknown", "short_label": "UNKNOWN", "description": ""}
        )

        return {
            "irrigation_class": pred_class,
            "irrigation_label": class_meta["short_label"],
            "irrigation_description": class_meta["label"],
            "confidence": round(confidence, 4),
            "probabilities": prob_dict
        }

    def generate_farmer_suggestion(
        self,
        sensor_data: Dict[str, Any],
        prediction: Dict[str, Any],
        motor_result: Dict[str, Any]
    ) -> str:
        """
        Generates actionable human-readable advisory text for display on the LCD.
        """
        moisture = sensor_data.get("soil_moisture", 0.0)
        air_temp = sensor_data.get("air_temperature", 0.0)
        humidity = sensor_data.get("air_humidity", 0.0)
        wind_speed = sensor_data.get("wind_speed", 0.0)
        irr_class = prediction["irrigation_class"]
        motor_status = motor_result.get("motor_status", "OFF")

        # Atmospheric evaporative context
        if air_temp >= 32.0 and humidity <= 35.0 and wind_speed >= 12.0:
            evap_context = "Atmospheric conditions show severe evaporative heat and wind stress."
        elif air_temp >= 30.0 and humidity <= 45.0:
            evap_context = "Warm and dry weather is accelerating crop transpiration."
        elif humidity >= 80.0:
            evap_context = "High humidity is keeping soil evaporation minimal."
        else:
            evap_context = "Atmospheric evaporative demand is moderate."

        if irr_class == 3:
            advice = (
                f"Soil moisture is critically low ({moisture:.1f}%). {evap_context} "
                f"Urgent high-volume irrigation is required to prevent crop wilting."
            )
        elif irr_class == 2:
            advice = (
                f"Soil moisture is in the depletion zone ({moisture:.1f}%). {evap_context} "
                f"Irrigation is recommended during the current watering window."
            )
        elif irr_class == 1:
            advice = (
                f"Soil moisture is adequate ({moisture:.1f}%). {evap_context} "
                f"Only light or scheduled maintenance irrigation is needed."
            )
        else:  # Class 0
            advice = (
                f"Soil is well-hydrated ({moisture:.1f}%). {evap_context} "
                f"No irrigation required at this time. Conserving water."
            )

        if motor_status == "ON":
            advice += " Automated water pump is active."

        return advice


# Module-level singleton
_predictor_instance = None


def get_predictor() -> IrrigationPredictor:
    """Singleton getter for the IrrigationPredictor on Raspberry Pi."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = IrrigationPredictor()
    return _predictor_instance


def predict_irrigation(sensor_data: Dict[str, Any]) -> Dict[str, Any]:
    """Functional helper for Raspberry Pi inference."""
    return get_predictor().predict(sensor_data)
