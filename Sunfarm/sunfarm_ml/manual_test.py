"""
SunFarm Smart Irrigation - Manual Interactive Testing Interface
===============================================================
Allows interactive testing of the SunFarm hybrid decision system by entering
real-time or hypothetical sensor values via terminal.

Reuses the existing:
  - config/config.py
  - src/irrigation_predictor.py
  - src/motor_controller.py
  - models/irrigation_model.pkl
"""

import sys
from pathlib import Path
from typing import Optional, Dict

# Set UTF-8 encoding on stdout for cross-platform degree symbol display
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.config import SENSOR_BOUNDS, MOTOR_CONFIG
from src.irrigation_predictor import IrrigationPredictor
from src.motor_controller import MotorController


def prompt_sensor_value(prompt_label: str, feature_key: str) -> Optional[float]:
    """
    Prompts the user for a single float sensor value with range validation.
    Returns the parsed float value, or None if the user requested to quit ('q').
    """
    bounds = SENSOR_BOUNDS[feature_key]
    min_val = bounds["min"]
    max_val = bounds["max"]
    unit = bounds["unit"]

    while True:
        try:
            raw_input = input(f"{prompt_label}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive test mode.")
            return None

        if raw_input.lower() in ("q", "quit", "exit"):
            return None

        if not raw_input:
            print("  [Error] Input cannot be empty. Please enter a numeric value or 'q' to quit.")
            continue

        try:
            val = float(raw_input)
        except ValueError:
            print(f"  [Error] '{raw_input}' is not a valid number. Please enter a valid number or 'q' to quit.")
            continue

        if val < min_val or val > max_val:
            print(
                f"  [Range Error] {prompt_label} must be between {min_val}{unit} and {max_val}{unit}. "
                f"You entered: {val}{unit}."
            )
            continue

        return val


def display_result(sensor_data: Dict[str, float], motor_res: Dict[str, str], ml_res: Dict[str, any], suggestion: str):
    """
    Displays the formatted evaluation result matching the user specification.
    """
    print("\n" + "=" * 50)
    print("SunFarm Decision Result")
    print("=======================")
    print()
    print("SENSOR DATA")
    print(f"Soil Moisture:     {sensor_data['soil_moisture']:>5.1f} %")
    print(f"Soil Temperature:  {sensor_data['soil_temperature']:>5.1f} °C")
    print(f"Air Temperature:   {sensor_data['air_temperature']:>5.1f} °C")
    print(f"Air Humidity:      {sensor_data['air_humidity']:>5.1f} %")
    print(f"Wind Speed:        {sensor_data['wind_speed']:>5.1f} km/h")
    print(f"Wind Direction:    {sensor_data['wind_direction']:>5.1f} °")
    print()
    print("---")
    print()
    print("## MOTOR CONTROL")
    print()
    print(f"Motor Status:       {motor_res['motor_status']}")
    print(f"Reason:             {motor_res['reason']}")
    print()
    print("---")
    print()
    print("## ML IRRIGATION PREDICTION")
    print()
    print(f"Recommendation:     {ml_res['irrigation_label']}")
    print(f"Class:              {ml_res['irrigation_class']}")
    print(f"Confidence:         {ml_res['confidence'] * 100:.1f} %")
    print()
    print("---")
    print()
    print("## FARMER SUGGESTION")
    print()
    print(f"{suggestion}")
    print()
    print("=" * 50 + "\n")


def run_interactive_tester():
    """Main interactive loop for manual test cases."""
    print("=" * 65)
    print("     SunFarm Smart Irrigation - Manual Interactive Tester")
    print("=" * 65)
    print("Enter the six sensor values to test the hybrid decision system.")
    print("Type 'q' or 'quit' at any prompt to exit.\n")

    # Initialize existing components
    try:
        predictor = IrrigationPredictor()
        motor = MotorController(
            lower_threshold=MOTOR_CONFIG["SOIL_MOISTURE_LOWER_THRESHOLD"],
            upper_threshold=MOTOR_CONFIG["SOIL_MOISTURE_UPPER_THRESHOLD"]
        )
    except Exception as e:
        print(f"[Fatal Error] Failed to initialize system components: {e}")
        return

    test_case_num = 1

    while True:
        print(f"--- [Test Case #{test_case_num}] ---")

        # 1. Soil Moisture
        moisture = prompt_sensor_value("Enter Soil Moisture (%)", "soil_moisture")
        if moisture is None:
            break

        # 2. Soil Temperature
        soil_temp = prompt_sensor_value("Enter Soil Temperature (°C)", "soil_temperature")
        if soil_temp is None:
            break

        # 3. Air Temperature
        air_temp = prompt_sensor_value("Enter Air Temperature (°C)", "air_temperature")
        if air_temp is None:
            break

        # 4. Air Humidity
        humidity = prompt_sensor_value("Enter Air Humidity (%)", "air_humidity")
        if humidity is None:
            break

        # 5. Wind Speed
        wind_speed = prompt_sensor_value("Enter Wind Speed (km/h)", "wind_speed")
        if wind_speed is None:
            break

        # 6. Wind Direction
        wind_dir = prompt_sensor_value("Enter Wind Direction (°)", "wind_direction")
        if wind_dir is None:
            break

        sensor_data = {
            "soil_moisture": moisture,
            "soil_temperature": soil_temp,
            "air_temperature": air_temp,
            "air_humidity": humidity,
            "wind_speed": wind_speed,
            "wind_direction": wind_dir
        }

        # Evaluate using existing components
        motor_res = motor.evaluate(sensor_data["soil_moisture"])
        ml_res = predictor.predict(sensor_data)
        suggestion = predictor.generate_farmer_suggestion(sensor_data, ml_res, motor_res)

        # Display formatted output
        display_result(sensor_data, motor_res, ml_res, suggestion)

        test_case_num += 1

    print("\nThank you for using the SunFarm Interactive Tester. Goodbye!")


if __name__ == "__main__":
    run_interactive_tester()
