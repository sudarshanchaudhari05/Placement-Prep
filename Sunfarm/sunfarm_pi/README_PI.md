# SunFarm - Raspberry Pi Edge Deployment Guide

> **Deploying the SunFarm Hybrid Irrigation Decision Engine on Raspberry Pi.**

---

## 1. Overview

`sunfarm_pi` is the lightweight, inference-only deployment package of the **SunFarm Smart Irrigation System**. It runs **100% offline** on Raspberry Pi OS (32-bit or 64-bit) with no internet access required.

### Deployment Directory Structure

```
sunfarm_pi/
├── models/
│   └── irrigation_model.pkl    # Pre-trained Random Forest model artifact (2.1 MB)
├── src/
│   ├── __init__.py
│   ├── irrigation_predictor.py # Offline inference engine & farmer advisor
│   └── motor_controller.py     # Deterministic hysteresis safety layer
├── config/
│   ├── __init__.py
│   └── config.py               # Hysteresis thresholds & sensor boundaries
├── pi_test.py                  # Offline verification test suite
├── requirements.txt            # Minimal edge dependencies
└── README_PI.md                # Raspberry Pi deployment instructions
```

---

## 2. Step-by-Step Raspberry Pi Setup Instructions

### Step 1: Copy `sunfarm_pi` to Raspberry Pi

#### Option A: Over Local Network via SCP (Recommended)
From your laptop terminal (PowerShell, macOS, or Linux):
```bash
# Replace 'pi' with your username and 'raspberrypi.local' or Pi IP address
scp -r sunfarm_pi pi@raspberrypi.local:~/sunfarm_pi
```

#### Option B: Via USB Drive
1. Copy the `sunfarm_pi` folder to a USB drive.
2. Insert the USB drive into your Raspberry Pi.
3. Open a terminal on the Pi and copy to home directory:
   ```bash
   cp -r /media/pi/<USB_NAME>/sunfarm_pi ~/sunfarm_pi
   ```

---

### Step 2: SSH into Raspberry Pi & Navigate to Project
```bash
ssh pi@raspberrypi.local
cd ~/sunfarm_pi
```

---

### Step 3: Create a Python Virtual Environment
Creating a virtual environment ensures clean dependency isolation on Raspberry Pi OS:

```bash
# Ensure Python 3 and venv package are installed
sudo apt update
sudo apt install -y python3 python3-venv python3-pip

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

---

### Step 4: Install Inference Dependencies
Install only the lightweight packages needed for offline inference:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note**: Installation takes about 1–2 minutes on a Raspberry Pi 4 / 5 (pre-built wheels for ARM are downloaded from PyPI).

---

### Step 5: Run Edge Verification Test (`pi_test.py`)
Run the verification script to confirm model loading, deterministic motor hysteresis, and offline predictions:

```bash
python3 pi_test.py
```

---

### Step 6: Verify Successful Output

You should see output similar to this:

```text
=================================================================
      SunFarm Raspberry Pi Edge System Diagnostics
=================================================================
Python Version:   3.11.2 (CPython)
Platform:         Linux-6.1.21-v8+-aarch64
Architecture:     aarch64
Model Artifact:   /home/pi/sunfarm_pi/models/irrigation_model.pkl
Model Exists:     True (2147.3 KB)
-----------------------------------------------------------------
[1/2] Loading trained Random Forest model artifact...
[OK] Model successfully loaded in 142.30 ms! Model Name: Random Forest
[2/2] Initializing Deterministic Motor Controller (Safety Layer)...
[OK] Motor Controller Active: Hysteresis Thresholds [30.0% - 55.0%]

>>> Executing Predefined Scenario Evaluations on Raspberry Pi Pipeline...

=================================================================
 CASE 1 — Very dry (Critical Deficit)
=================================================================
Sensor Data:
  Soil Moisture:     15.0 %
  Soil Temperature:  32.0 °C
  Air Temperature:   38.0 °C
  Air Humidity:      30.0 %
  Wind Speed:        15.0 km/h
  Wind Direction:   180.0 °

--- MOTOR CONTROL ---
  Motor Status:     ON
  Motor Reason:     Soil moisture (15.0%) is below lower threshold (30.0%). Motor turned ON to replenish root zone water.

--- ML IRRIGATION PREDICTION ---
  Irrigation Class: 3
  Recommendation:   HIGH IRRIGATION
  Confidence:       100.0 %
  Inference Time:   18.42 ms

--- FARMER SUGGESTION ---
  "Soil moisture is critically low (15.0%). Atmospheric conditions show severe evaporative heat and wind stress. Urgent high-volume irrigation is required to prevent crop wilting. Automated water pump is active."
=================================================================

[SUCCESS] All Raspberry Pi test cases executed successfully completely offline!
```

---

## 3. Hardware Integration Roadmap (Next Phase)

```
[STM32 + Sensors] --(LoRa)--> [SX1278 SPI/UART on Pi] 
                                    |
                            [sunfarm_pi Engine]
                             /              \
                            v                v
                 [GPIO Relay -> Pump]   [UART Serial -> RP2350 LCD]
```

1. **LoRa Receiver**: Read packets from SX1278 / SX1262 LoRa module using `pyserial` (`/dev/ttyAMA0` or `/dev/serial0`) or `spidev`.
2. **RP2350 LCD Output**: Transmit formatted telemetry & suggestions over UART at 115200 baud to the RP2350 display node.
3. **Pump Relay Actuation**: Connect `motor_status == "ON"` to Raspberry Pi GPIO (e.g. GPIO 17 / Pin 11) using `RPi.GPIO` or `gpiod`.
