"""
SunFarm Smart Irrigation - Deterministic Motor Controller (Safety Layer)
========================================================================
Implements a safety-critical, deterministic rule-based motor controller
with configurable hysteresis to prevent relay chatter.

Control Rules:
- Soil Moisture < Lower Threshold (default 30%): Turn Motor ON
- Soil Moisture > Upper Threshold (default 55%): Turn Motor OFF
- In-between (30% <= Moisture <= 55%): Maintain previous motor state (Hysteresis)
- Invalid/Out-of-range sensor readings: Turn Motor OFF (Safety Shutdown)
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import MOTOR_CONFIG, SENSOR_BOUNDS


class MotorController:
    """
    Stateful rule-based motor controller with configurable hysteresis thresholds.
    """

    def __init__(
        self,
        lower_threshold: float = MOTOR_CONFIG["SOIL_MOISTURE_LOWER_THRESHOLD"],
        upper_threshold: float = MOTOR_CONFIG["SOIL_MOISTURE_UPPER_THRESHOLD"],
        initial_state: str = MOTOR_CONFIG["DEFAULT_INITIAL_STATE"]
    ):
        """
        Initialize the Motor Controller.

        :param lower_threshold: Soil moisture percentage below which motor turns ON (e.g. 30.0)
        :param upper_threshold: Soil moisture percentage above which motor turns OFF (e.g. 55.0)
        :param initial_state: Default starting state ('ON' or 'OFF')
        """
        if lower_threshold >= upper_threshold:
            raise ValueError(
                f"Invalid hysteresis bounds: lower_threshold ({lower_threshold}) "
                f"must be strictly less than upper_threshold ({upper_threshold})"
            )

        self.lower_threshold = float(lower_threshold)
        self.upper_threshold = float(upper_threshold)
        self.state = str(initial_state).upper()

        if self.state not in ("ON", "OFF"):
            self.state = "OFF"

    def evaluate(self, soil_moisture: float) -> Dict[str, str]:
        """
        Evaluate soil moisture and update motor state based on hysteresis rules.

        :param soil_moisture: Current soil moisture percentage (0.0 - 100.0)
        :return: Dict containing 'motor_status' ('ON' or 'OFF') and 'reason'
        """
        # 1. Sensor Plausibility / Safety Validation
        min_bound = SENSOR_BOUNDS["soil_moisture"]["min"]
        max_bound = SENSOR_BOUNDS["soil_moisture"]["max"]

        if soil_moisture is None or not isinstance(soil_moisture, (int, float)):
            self.state = "OFF"
            return {
                "motor_status": "OFF",
                "reason": "Safety Alert: Invalid soil moisture sensor data received. Motor held OFF for safety."
            }

        if soil_moisture < min_bound or soil_moisture > max_bound:
            self.state = "OFF"
            return {
                "motor_status": "OFF",
                "reason": (
                    f"Safety Alert: Out-of-bounds soil moisture reading ({soil_moisture}%). "
                    f"Expected [{min_bound}%, {max_bound}%]. Motor shut OFF for fail-safe protection."
                )
            }

        # 2. Deterministic Hysteresis Control Logic
        previous_state = self.state

        if soil_moisture < self.lower_threshold:
            self.state = "ON"
            reason = (
                f"Soil moisture ({soil_moisture:.1f}%) is below lower threshold "
                f"({self.lower_threshold:.1f}%). Motor turned ON to replenish root zone water."
            )
        elif soil_moisture > self.upper_threshold:
            self.state = "OFF"
            reason = (
                f"Soil moisture ({soil_moisture:.1f}%) is above upper threshold "
                f"({self.upper_threshold:.1f}%). Motor turned OFF to prevent waterlogging."
            )
        else:
            # Within Hysteresis Deadband [lower_threshold, upper_threshold]
            if previous_state == "ON":
                self.state = "ON"
                reason = (
                    f"Soil moisture ({soil_moisture:.1f}%) is in hysteresis deadband "
                    f"[{self.lower_threshold:.1f}% - {self.upper_threshold:.1f}%]. "
                    f"Motor maintained ON to complete irrigation cycle."
                )
            else:
                self.state = "OFF"
                reason = (
                    f"Soil moisture ({soil_moisture:.1f}%) is in hysteresis deadband "
                    f"[{self.lower_threshold:.1f}% - {self.upper_threshold:.1f}%]. "
                    f"Motor maintained OFF (soil has adequate baseline moisture)."
                )

        return {
            "motor_status": self.state,
            "reason": reason
        }

    def reset_state(self, new_state: str = "OFF") -> None:
        """Reset internal motor state."""
        self.state = new_state.upper()


# Module-level default singleton controller instance
_default_controller = MotorController()


def evaluate_motor(
    soil_moisture: float,
    lower_threshold: Optional[float] = None,
    upper_threshold: Optional[float] = None,
    current_state: Optional[str] = None
) -> Dict[str, str]:
    """
    Convenience function to evaluate motor state. Can be used either statelessly
    by passing current_state, or using the persistent singleton.

    :param soil_moisture: Soil moisture value
    :param lower_threshold: Optional override for lower hysteresis threshold
    :param upper_threshold: Optional override for upper hysteresis threshold
    :param current_state: Optional explicit current state ('ON' or 'OFF')
    :return: {"motor_status": "ON" | "OFF", "reason": "..."}
    """
    global _default_controller

    if lower_threshold is not None or upper_threshold is not None:
        low = lower_threshold if lower_threshold is not None else MOTOR_CONFIG["SOIL_MOISTURE_LOWER_THRESHOLD"]
        high = upper_threshold if upper_threshold is not None else MOTOR_CONFIG["SOIL_MOISTURE_UPPER_THRESHOLD"]
        init = current_state if current_state is not None else _default_controller.state
        ctrl = MotorController(lower_threshold=low, upper_threshold=high, initial_state=init)
        return ctrl.evaluate(soil_moisture)

    if current_state is not None:
        _default_controller.state = current_state

    return _default_controller.evaluate(soil_moisture)


if __name__ == "__main__":
    # Self-test hysteresis sequence
    print("=== Testing Motor Controller Hysteresis Sequence ===")
    ctrl = MotorController(lower_threshold=30.0, upper_threshold=55.0, initial_state="OFF")

    test_readings = [60.0, 45.0, 28.0, 35.0, 50.0, 56.0, 50.0, 35.0, 25.0]
    for reading in test_readings:
        res = ctrl.evaluate(reading)
        print(f"Moisture: {reading:4.1f}% -> Motor: {res['motor_status']:3s} | Reason: {res['reason']}")
