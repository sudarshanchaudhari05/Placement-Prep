# SunFarm - Smart Agriculture Irrigation ML Recommendation & Control System

> **A lightweight, edge-compatible hybrid irrigation decision system for Raspberry Pi.**

---

## 1. System Overview & Hardware Architecture

The **SunFarm** smart irrigation system is designed to optimize water usage, protect crops from water stress, and prevent over-irrigation.

```
[STM32 + 6 Sensors] ---> [LoRa Transceiver] ---> [Raspberry Pi Decision Engine] ---> [RP2350 LCD Display]
                                                              |
                                                              +---> [Water Pump Relay (ON/OFF)]
```

### The Hybrid Architecture Decision
A critical principle in mission-critical agriculture is **safety and determinism**:
1. **Safety Layer (Deterministic Motor Controller)**: The ML model does **NOT** directly control the water pump. Instead, a deterministic rule-based controller regulates the pump based on root-zone soil moisture with configurable **hysteresis** (default: `< 30%` turns ON, `> 55%` turns OFF). This prevents relay chatter and guarantees fail-safe operation.
2. **Advisory Layer (Lightweight ML Model)**: A multi-class machine learning model analyzes all 6 environmental sensors to classify irrigation urgency (Levels 0–3), evaluate atmospheric evaporative demand (VPD, heat, wind), and generate actionable recommendations for farmers on the LCD display.

---

## 2. Sensor Inputs & Specifications

The STM32 sensor node broadcasts 6 physical telemetry values over LoRa:

| # | Sensor Parameter | Physical Unit | Operational Range | Role in Decision System |
|---|-------------------|---------------|-------------------|-------------------------|
| 1 | **Soil Moisture** | `%` (0–100) | 0.0 – 100.0% | Primary safety parameter for motor control & ML |
| 2 | **Soil Temperature** | `°C` | -10.0 – 60.0°C | Root zone thermal activity & evaporation factor |
| 3 | **Air Temperature** | `°C` | -10.0 – 60.0°C | Vapor Pressure Deficit (VPD) driver |
| 4 | **Air Humidity** | `%` (0–100) | 0.0 – 100.0% | Atmospheric moisture & transpiration demand |
| 5 | **Wind Speed** | `km/h` | 0.0 – 120.0 km/h | Boundary layer heat & moisture dissipation |
| 6 | **Wind Direction** | `°` (0–360) | 0.0 – 360.0° | Microclimate advection & regional wind tracking |

---

## 3. Irrigation Classification Levels

The ML model classifies irrigation urgency into 4 distinct classes:

| Class | Label | Short Label | Description | Typical Agronomic State |
|:-----:|:------|:------------|:------------|:------------------------|
| **0** | No irrigation required | `NO IRRIGATION` | Adequate root hydration; low evaporative loss | Soil > 60%, cool/humid weather |
| **1** | Low irrigation requirement | `LOW IRRIGATION` | Mild depletion; optional maintenance watering | Soil 45–60%, mild weather |
| **2** | Medium irrigation requirement | `MEDIUM IRRIGATION` | Depletion zone or high evaporative heat demand | Soil 30–45%, warm/sunny |
| **3** | High irrigation requirement | `HIGH IRRIGATION` | Severe water stress; rapid crop wilting danger | Soil < 30%, dry heatwave/wind |

---

## 4. Rule-Based Motor Controller (Hysteresis)

```
        Soil Moisture Drops Below 30%
   ---------------------------------------->  [ MOTOR ON ]
  |                                                |
  |  Adequate moisture                             | Soil moisture
  |  (holds OFF in 30% - 55%)                      | increases during
  |                                                | watering
  |                                                v
[ MOTOR OFF ] <-------------------------------------
                  Soil Moisture Rises Above 55%
```

- **Lower Threshold (`SOIL_MOISTURE_LOWER_THRESHOLD = 30.0%`)**: When moisture drops strictly below 30%, pump turns **ON**.
- **Upper Threshold (`SOIL_MOISTURE_UPPER_THRESHOLD = 55.0%`)**: When moisture rises strictly above 55%, pump turns **OFF**.
- **Hysteresis Deadband (`30.0% - 55.0%`)**: Retains the previous motor state. If watering, pump stays ON until 55% is reached; if drying, pump stays OFF until 30% is reached.
- **Fail-Safe Override**: If sensor readings are disconnected or out of physical bounds (`< 0%` or `> 100%`), the pump immediately triggers a safety shutdown (`OFF`).

---

## 5. Project Directory Structure

```
sunfarm_ml/
├── config/
│   ├── __init__.py
│   └── config.py                   # Central thresholds, sensor bounds & paths
├── data/
│   └── synthetic_irrigation_data.csv # 10,000 realistic agricultural samples
├── models/
│   ├── irrigation_model.pkl        # Exported lightweight model artifact
│   └── model_metrics.json          # Benchmark metrics & comparison results
├── src/
│   ├── __init__.py
│   ├── generate_dataset.py         # Physics-based synthetic data generator
│   ├── train_model.py              # Model training & benchmarking pipeline
│   ├── test_model.py               # Evaluation & stress testing script
│   ├── irrigation_predictor.py     # Clean predictor class & farmer advisor
│   └── motor_controller.py         # Hysteresis safety layer & motor evaluator
├── demo.py                         # End-to-end LoRa packet simulator & demo
├── manual_test.py                  # Interactive CLI manual tester for live input
├── requirements.txt                # Lightweight dependencies
└── README.md                       # Complete documentation
```

---

## 6. Installation & Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Agronomic Dataset
```bash
python src/generate_dataset.py
```

### 3. Train & Benchmark ML Models
```bash
python src/train_model.py
```

### 4. Run Model Verification & Stress Tests
```bash
python src/test_model.py
```

### 5. Run Live Integration Demo (LoRa Simulation)
```bash
python demo.py
```

### 6. Run Interactive Manual Tester
```bash
python manual_test.py
```

---

## 7. Prediction & Motor Interface Example

### ML Predictor Usage
```python
from src.irrigation_predictor import IrrigationPredictor

predictor = IrrigationPredictor()
telemetry = {
    "soil_moisture": 24.0,
    "soil_temperature": 29.0,
    "air_temperature": 34.0,
    "air_humidity": 42.0,
    "wind_speed": 8.0,
    "wind_direction": 210.0
}

result = predictor.predict(telemetry)
print(result)
# Output:
# {
#   "irrigation_class": 3,
#   "irrigation_label": "HIGH IRRIGATION",
#   "irrigation_description": "High irrigation requirement",
#   "confidence": 0.942,
#   "probabilities": {...}
# }
```

### Motor Controller Usage
```python
from src.motor_controller import MotorController

motor = MotorController(lower_threshold=30.0, upper_threshold=55.0)
status = motor.evaluate(soil_moisture=24.0)
print(status)
# Output:
# {
#   "motor_status": "ON",
#   "reason": "Soil moisture (24.0%) is below lower threshold (30.0%). Motor turned ON to replenish root zone water."
# }
```

---

## 8. Raspberry Pi & Hardware Integration Roadmap

1. **LoRa Receiver Integration**: Replace the packet generator in `demo.py` with serial / SPI reading from SX1278 / SX1262 LoRa module (`pyserial` or `spidev`).
2. **RP2350 LCD Bridge**: Send JSON/UART packets containing `{"moisture", "motor", "ml_rec", "confidence", "advice"}` directly to the RP2350 microcontroller via UART `/dev/serial0`.
3. **Relay GPIO**: Connect `motor_result["motor_status"] == "ON"` to a GPIO pin driving an optocoupled relay module (`RPi.GPIO` or `gpiod`).
