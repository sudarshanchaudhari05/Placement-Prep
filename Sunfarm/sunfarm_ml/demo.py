"""
SunFarm Smart Irrigation - End-to-End System Demo & LoRa Simulator
==================================================================
Simulates incoming LoRa telemetry packets from the STM32 sensor node and passes
them through the hybrid decision pipeline (Deterministic Motor Control + ML Predictor).

Demonstrates:
  1. Sensor payload decoding
  2. Motor ON/OFF decision with hysteresis safety rules
  3. ML irrigation recommendation classification & confidence
  4. Human-readable farmer advice generation for RP2350 LCD display
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config.config import SENSOR_BOUNDS, MOTOR_CONFIG
from src.motor_controller import MotorController
from src.irrigation_predictor import IrrigationPredictor


def display_telemetry_decision(
    packet_index: int,
    sensor_data: dict,
    motor_result: dict,
    ml_result: dict,
    farmer_suggestion: str
):
    """Prints formatted output matching the SunFarm system specification."""
    print("=" * 65)
    print(f"       SunFarm Telemetry Decision System - LoRa Packet #{packet_index}")
    print("=" * 65)
    print("Sensor Data:")
    print(f"Soil Moisture:    {sensor_data['soil_moisture']:.1f}%")
    print(f"Soil Temperature: {sensor_data['soil_temperature']:.1f}°C")
    print(f"Air Temperature:  {sensor_data['air_temperature']:.1f}°C")
    print(f"Air Humidity:     {sensor_data['air_humidity']:.1f}%")
    print(f"Wind Speed:       {sensor_data['wind_speed']:.1f} km/h")
    print(f"Wind Direction:   {sensor_data['wind_direction']:.1f}°")
    print()
    print(f"Motor Status:     {motor_result['motor_status']}")
    print(f"Motor Reason:     {motor_result['reason']}")
    print()
    print(f"ML Recommendation:{ml_result['irrigation_label']} (Class {ml_result['irrigation_class']})")
    print(f"Confidence:       {ml_result['confidence'] * 100:.1f}%")
    print()
    print("Farmer Suggestion:")
    print(f"\"{farmer_suggestion}\"")
    print("=" * 65)
    print()


def run_demo(interactive: bool = False, delay_seconds: float = 1.0):
    """Executes the end-to-end multi-scenario LoRa demonstration."""
    print("\n[INIT] Initializing SunFarm Smart Irrigation Decision Engine...")
    predictor = IrrigationPredictor()
    motor = MotorController(
        lower_threshold=MOTOR_CONFIG["SOIL_MOISTURE_LOWER_THRESHOLD"],
        upper_threshold=MOTOR_CONFIG["SOIL_MOISTURE_UPPER_THRESHOLD"]
    )
    print(f"[OK] ML Model Loaded: {predictor.model_name}")
    print(f"[OK] Motor Controller Initialized: Hysteresis [{motor.lower_threshold}% - {motor.upper_threshold}%]\n")

    # Realistic simulated LoRa packet sequence tracing an irrigation cycle
    lora_packets = [
        {
            "description": "Packet #1: Hot dry afternoon with low soil moisture (Pump turns ON)",
            "sensors": {
                "soil_moisture": 24.0,
                "soil_temperature": 29.0,
                "air_temperature": 34.0,
                "air_humidity": 42.0,
                "wind_speed": 8.0,
                "wind_direction": 210.0
            }
        },
        {
            "description": "Packet #2: Active irrigation in progress - moisture entering hysteresis zone (Pump stays ON)",
            "sensors": {
                "soil_moisture": 38.5,
                "soil_temperature": 27.2,
                "air_temperature": 33.1,
                "air_humidity": 45.0,
                "wind_speed": 7.5,
                "wind_direction": 215.0
            }
        },
        {
            "description": "Packet #3: Irrigation continuing - moisture approaching saturation (Pump stays ON)",
            "sensors": {
                "soil_moisture": 52.0,
                "soil_temperature": 25.5,
                "air_temperature": 31.8,
                "air_humidity": 48.0,
                "wind_speed": 6.8,
                "wind_direction": 220.0
            }
        },
        {
            "description": "Packet #4: Soil reached upper capacity threshold (Pump turns OFF)",
            "sensors": {
                "soil_moisture": 58.0,
                "soil_temperature": 24.0,
                "air_temperature": 30.5,
                "air_humidity": 55.0,
                "wind_speed": 5.2,
                "wind_direction": 200.0
            }
        },
        {
            "description": "Packet #5: Post-irrigation cool evening - moisture settling (Pump stays OFF)",
            "sensors": {
                "soil_moisture": 53.5,
                "soil_temperature": 22.0,
                "air_temperature": 24.0,
                "air_humidity": 68.0,
                "wind_speed": 4.0,
                "wind_direction": 180.0
            }
        },
        {
            "description": "Packet #6: Next morning - moderate moisture with rising heat (Pump stays OFF, Medium Rec)",
            "sensors": {
                "soil_moisture": 41.0,
                "soil_temperature": 26.5,
                "air_temperature": 31.0,
                "air_humidity": 40.0,
                "wind_speed": 11.5,
                "wind_direction": 240.0
            }
        },
        {
            "description": "Packet #7: Extreme windy heatwave - soil drying rapidly towards lower threshold",
            "sensors": {
                "soil_moisture": 28.5,
                "soil_temperature": 35.0,
                "air_temperature": 39.5,
                "air_humidity": 19.0,
                "wind_speed": 22.0,
                "wind_direction": 285.0
            }
        }
    ]

    for idx, packet in enumerate(lora_packets, start=1):
        sensors = packet["sensors"]
        print(f">>> Processing {packet['description']}")

        # 1. Deterministic Motor Control Evaluation
        motor_res = motor.evaluate(sensors["soil_moisture"])

        # 2. ML Recommendation Inference
        ml_res = predictor.predict(sensors)

        # 3. Farmer Suggestion Generation
        suggestion = predictor.generate_farmer_suggestion(sensors, ml_res, motor_res)

        # 4. Display Formatted Output
        display_telemetry_decision(
            packet_index=idx,
            sensor_data=sensors,
            motor_result=motor_res,
            ml_result=ml_res,
            farmer_suggestion=suggestion
        )

        if idx < len(lora_packets) and delay_seconds > 0:
            time.sleep(delay_seconds)

    print("[DONE] LoRa telemetry simulation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SunFarm Smart Irrigation Demo & LoRa Simulator")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay in seconds between simulated LoRa packets")
    args = parser.parse_args()

    run_demo(delay_seconds=args.delay)
