"""
SunFarm Smart Irrigation - Synthetic Dataset Generator
======================================================
Generates realistic agronomic sensor datasets based on atmospheric physics,
Vapor Pressure Deficit (VPD), soil moisture dynamics, and controlled noise.

Class Labels:
  0: No irrigation required
  1: Low irrigation requirement
  2: Medium irrigation requirement
  3: High irrigation requirement
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to sys.path to allow absolute imports from config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import DATASET_PATH, DATASET_CONFIG, FEATURE_NAMES, IRRIGATION_CLASSES


def calculate_vpd(air_temp_c: np.ndarray, air_humidity_pct: np.ndarray) -> np.ndarray:
    """
    Calculate Vapor Pressure Deficit (VPD) in kPa using Tetens formula.
    VPD is a primary driver of crop transpiration and surface evaporation.
    """
    # Saturated vapor pressure (kPa)
    svp = 0.61078 * np.exp((17.27 * air_temp_c) / (air_temp_c + 237.3))
    # Actual vapor pressure (kPa)
    avp = svp * (air_humidity_pct / 100.0)
    # Vapor Pressure Deficit (kPa)
    vpd = np.maximum(0.0, svp - avp)
    return vpd


def generate_synthetic_data(num_samples: int = 10000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic agricultural data across 5 distinct microclimate/weather regimes
    to guarantee diverse, realistic distributions.
    """
    np.random.seed(random_seed)

    # Allocate samples across realistic agronomic weather profiles
    proportions = [0.25, 0.25, 0.20, 0.15, 0.15]  # sum = 1.0
    counts = [int(p * num_samples) for p in proportions]
    counts[-1] = num_samples - sum(counts[:-1])  # Ensure exact total

    records = []

    # Profile 1: Hot, dry sunny afternoon (High evaporative stress)
    n = counts[0]
    soil_moisture_p1 = np.random.uniform(12.0, 52.0, n)
    air_temp_p1 = np.random.normal(33.0, 4.0, n).clip(26.0, 44.0)
    soil_temp_p1 = (air_temp_p1 * 0.85 + np.random.normal(3.0, 1.5, n)).clip(22.0, 42.0)
    air_humidity_p1 = np.random.normal(32.0, 8.0, n).clip(12.0, 50.0)
    wind_speed_p1 = np.random.gamma(shape=2.5, scale=4.0, size=n).clip(2.0, 35.0)
    wind_dir_p1 = np.random.uniform(0.0, 360.0, n)
    records.append(np.column_stack([soil_moisture_p1, soil_temp_p1, air_temp_p1, air_humidity_p1, wind_speed_p1, wind_dir_p1]))

    # Profile 2: Mild, pleasant sunny day (Moderate evaporative stress)
    n = counts[1]
    soil_moisture_p2 = np.random.uniform(25.0, 68.0, n)
    air_temp_p2 = np.random.normal(26.0, 3.0, n).clip(19.0, 32.0)
    soil_temp_p2 = (air_temp_p2 * 0.90 + np.random.normal(1.0, 1.2, n)).clip(18.0, 30.0)
    air_humidity_p2 = np.random.normal(52.0, 8.0, n).clip(38.0, 72.0)
    wind_speed_p2 = np.random.gamma(shape=2.0, scale=3.5, size=n).clip(1.0, 22.0)
    wind_dir_p2 = np.random.uniform(0.0, 360.0, n)
    records.append(np.column_stack([soil_moisture_p2, soil_temp_p2, air_temp_p2, air_humidity_p2, wind_speed_p2, wind_dir_p2]))

    # Profile 3: Cool, humid morning / overcast (Low evaporative stress)
    n = counts[2]
    soil_moisture_p3 = np.random.uniform(35.0, 78.0, n)
    air_temp_p3 = np.random.normal(18.0, 3.0, n).clip(10.0, 24.0)
    soil_temp_p3 = (air_temp_p3 * 0.95 + np.random.normal(0.0, 1.0, n)).clip(12.0, 22.0)
    air_humidity_p3 = np.random.normal(78.0, 7.0, n).clip(62.0, 96.0)
    wind_speed_p3 = np.random.gamma(shape=1.5, scale=2.5, size=n).clip(0.5, 15.0)
    wind_dir_p3 = np.random.uniform(0.0, 360.0, n)
    records.append(np.column_stack([soil_moisture_p3, soil_temp_p3, air_temp_p3, air_humidity_p3, wind_speed_p3, wind_dir_p3]))

    # Profile 4: Post-rain / monsoon / wet soil (Very low demand, saturated)
    n = counts[3]
    soil_moisture_p4 = np.random.uniform(58.0, 90.0, n)
    air_temp_p4 = np.random.normal(24.0, 3.0, n).clip(16.0, 31.0)
    soil_temp_p4 = (air_temp_p4 * 0.92 + np.random.normal(-0.5, 1.0, n)).clip(15.0, 28.0)
    air_humidity_p4 = np.random.normal(86.0, 5.0, n).clip(72.0, 99.0)
    wind_speed_p4 = np.random.gamma(shape=2.2, scale=4.0, size=n).clip(1.0, 28.0)
    wind_dir_p4 = np.random.uniform(0.0, 360.0, n)
    records.append(np.column_stack([soil_moisture_p4, soil_temp_p4, air_temp_p4, air_humidity_p4, wind_speed_p4, wind_dir_p4]))

    # Profile 5: Scorching heatwave with windy dry air (Extreme stress)
    n = counts[4]
    soil_moisture_p5 = np.random.uniform(8.0, 40.0, n)
    air_temp_p5 = np.random.normal(39.0, 3.5, n).clip(34.0, 48.0)
    soil_temp_p5 = (air_temp_p5 * 0.90 + np.random.normal(4.0, 1.5, n)).clip(30.0, 46.0)
    air_humidity_p5 = np.random.normal(20.0, 6.0, n).clip(8.0, 35.0)
    wind_speed_p5 = np.random.gamma(shape=3.5, scale=4.5, size=n).clip(6.0, 45.0)
    wind_dir_p5 = np.random.uniform(0.0, 360.0, n)
    records.append(np.column_stack([soil_moisture_p5, soil_temp_p5, air_temp_p5, air_humidity_p5, wind_speed_p5, wind_dir_p5]))

    # Combine all profiles
    raw_data = np.vstack(records)
    df = pd.DataFrame(raw_data, columns=FEATURE_NAMES)

    # Shuffle rows
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Calculate Agronomic Indices
    soil_moist = df["soil_moisture"].values
    soil_t = df["soil_temperature"].values
    air_t = df["air_temperature"].values
    air_h = df["air_humidity"].values
    wind_s = df["wind_speed"].values

    # 1. Moisture Stress Deficit Score (0 = completely wet >= 65%, 1 = critically dry <= 15%)
    # Plant available water drops significantly below 50%
    moist_deficit = np.clip((65.0 - soil_moist) / 50.0, 0.0, 1.0)

    # 2. Atmospheric Evaporative Demand Score
    vpd = calculate_vpd(air_t, air_h)
    vpd_norm = np.clip(vpd / 3.2, 0.0, 1.0)
    wind_norm = np.clip(wind_s / 30.0, 0.0, 1.0)
    soil_t_norm = np.clip((soil_t - 16.0) / 24.0, 0.0, 1.0)
    air_t_norm = np.clip((air_t - 16.0) / 24.0, 0.0, 1.0)

    evap_demand = (
        0.45 * vpd_norm +
        0.25 * wind_norm +
        0.15 * air_t_norm +
        0.15 * soil_t_norm
    )

    # 3. Controlled Noise & Interaction
    # Interaction: dry soil + high evap demand multiplies the urgency of irrigation
    interaction = moist_deficit * evap_demand
    controlled_noise = np.random.normal(0.0, 0.035, len(df))

    # Overall Irrigation Urgency Score (0.0 to 1.0+)
    irrigation_score = (
        0.60 * moist_deficit +
        0.28 * evap_demand +
        0.12 * interaction +
        controlled_noise
    )

    # Physical agricultural overrides:
    # If soil is saturated (> 65%), irrigation is not required regardless of weather
    irrigation_score = np.where(soil_moist >= 65.0, np.minimum(irrigation_score, 0.18), irrigation_score)
    # If soil is extremely dry (< 18%), urgent irrigation is mandatory
    irrigation_score = np.where(soil_moist <= 18.0, np.maximum(irrigation_score, 0.78), irrigation_score)

    # Class boundaries:
    # 0: No Irrigation   (score < 0.28)
    # 1: Low Irrigation  (0.28 <= score < 0.50)
    # 2: Med Irrigation  (0.50 <= score < 0.72)
    # 3: High Irrigation (score >= 0.72)
    classes = np.zeros(len(df), dtype=int)
    classes[irrigation_score >= 0.28] = 1
    classes[irrigation_score >= 0.50] = 2
    classes[irrigation_score >= 0.72] = 3

    df["irrigation_class"] = classes
    df["irrigation_label"] = df["irrigation_class"].map(lambda c: IRRIGATION_CLASSES[c]["short_label"])

    # Round feature columns to realistic sensor decimal precision
    df["soil_moisture"] = df["soil_moisture"].round(1)
    df["soil_temperature"] = df["soil_temperature"].round(1)
    df["air_temperature"] = df["air_temperature"].round(1)
    df["air_humidity"] = df["air_humidity"].round(1)
    df["wind_speed"] = df["wind_speed"].round(1)
    df["wind_direction"] = df["wind_direction"].round(1)

    return df


def main():
    print(f"Generating realistic synthetic irrigation dataset ({DATASET_CONFIG['NUM_SAMPLES']} samples)...")
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_data(
        num_samples=DATASET_CONFIG["NUM_SAMPLES"],
        random_seed=DATASET_CONFIG["RANDOM_SEED"]
    )

    df.to_csv(DATASET_PATH, index=False)
    print(f"Dataset successfully saved to: {DATASET_PATH}")
    print("\nDataset Summary:")
    print(f"Total Samples: {len(df)}")
    print("\nClass Distribution:")
    class_counts = df["irrigation_class"].value_counts().sort_index()
    for class_id, count in class_counts.items():
        pct = (count / len(df)) * 100
        desc = IRRIGATION_CLASSES[class_id]["label"]
        print(f"  Class {class_id} ({desc}): {count} samples ({pct:.1f}%)")

    print("\nFeature Summary Statistics:")
    print(df[FEATURE_NAMES].describe().round(2))


if __name__ == "__main__":
    main()
