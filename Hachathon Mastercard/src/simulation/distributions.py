"""Statistical distributions and customer profiling for payment simulation."""

from typing import Dict, Tuple, List
import numpy as np

# Diurnal hourly probabilities (0:00 to 23:00) reflecting natural human activity
HOURLY_ACTIVITY_PROB: np.ndarray = np.array([
    0.015, 0.010, 0.008, 0.006, 0.008, 0.015,  # 00:00 - 05:00 (Night)
    0.030, 0.045, 0.060, 0.065, 0.070, 0.075,  # 06:00 - 11:00 (Morning)
    0.080, 0.085, 0.075, 0.070, 0.075, 0.080,  # 12:00 - 17:00 (Afternoon)
    0.085, 0.070, 0.055, 0.045, 0.030, 0.020,  # 18:00 - 23:00 (Evening)
])
HOURLY_ACTIVITY_PROB = HOURLY_ACTIVITY_PROB / HOURLY_ACTIVITY_PROB.sum()

# Off-hours hourly probabilities (typically 23:00 to 05:00)
OFF_HOURS_PROB: np.ndarray = np.array([
    0.16, 0.18, 0.20, 0.18, 0.14, 0.08,  # 00:00 - 05:00
    0.01, 0.005, 0.005, 0.005, 0.005, 0.005,
    0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
    0.005, 0.005, 0.005, 0.005, 0.015, 0.045,
])
OFF_HOURS_PROB = OFF_HOURS_PROB / OFF_HOURS_PROB.sum()

# Base Merchant Category Risks (Baseline mean risk rating)
CATEGORY_BASE_RISK: Dict[str, float] = {
    "groceries": 0.05,
    "utilities": 0.06,
    "dining": 0.10,
    "retail": 0.15,
    "travel": 0.25,
    "marketplace": 0.28,
    "electronics": 0.35,
    "digital_goods": 0.40,
    "gaming": 0.45,
    "money_transfer": 0.50,
    "crypto_exchange": 0.65,
    "luxury": 0.40,
}

# Category transaction amount log-normal parameters (mean_log, sigma_log)
CATEGORY_AMOUNT_PARAMS: Dict[str, Tuple[float, float]] = {
    "groceries": (3.8, 0.5),         # ~$45, spread $15-$120
    "utilities": (4.4, 0.4),         # ~$80, spread $40-$180
    "dining": (3.5, 0.6),            # ~$35, spread $12-$90
    "retail": (4.2, 0.7),            # ~$65, spread $20-$250
    "travel": (5.7, 0.8),            # ~$300, spread $80-$1200
    "marketplace": (4.0, 0.7),       # ~$55, spread $15-$220
    "electronics": (5.5, 0.8),       # ~$250, spread $60-$1000
    "digital_goods": (2.8, 0.6),     # ~$16, spread $5-$60
    "gaming": (2.9, 0.7),            # ~$18, spread $5-$80
    "money_transfer": (5.2, 0.9),    # ~$180, spread $30-$900
    "crypto_exchange": (6.0, 1.0),   # ~$400, spread $50-$2500
    "luxury": (6.5, 0.9),            # ~$650, spread $150-$3500
}

# Category frequency distribution for legitimate transactions
LEGITIMATE_CATEGORY_WEIGHTS: Dict[str, float] = {
    "groceries": 0.26,
    "dining": 0.20,
    "retail": 0.18,
    "marketplace": 0.10,
    "utilities": 0.08,
    "digital_goods": 0.06,
    "electronics": 0.04,
    "travel": 0.03,
    "gaming": 0.02,
    "money_transfer": 0.015,
    "luxury": 0.01,
    "crypto_exchange": 0.005,
}

# Category frequency distribution for fraudulent transactions (broad, realistic spread across everyday merchants)
FRAUD_CATEGORY_WEIGHTS: Dict[str, float] = {
    "retail": 0.22,
    "digital_goods": 0.18,
    "groceries": 0.14,
    "marketplace": 0.12,
    "dining": 0.10,
    "electronics": 0.08,
    "gaming": 0.05,
    "travel": 0.04,
    "money_transfer": 0.03,
    "utilities": 0.02,
    "luxury": 0.015,
    "crypto_exchange": 0.005,
}

# Payment channel distribution for legitimate transactions
LEGITIMATE_CHANNEL_WEIGHTS: Dict[str, float] = {
    "pos_contactless": 0.32,
    "pos_chip": 0.25,
    "e-commerce": 0.22,
    "mobile_app": 0.12,
    "recurring_subscription": 0.06,
    "p2p_transfer": 0.025,
    "api_gateway": 0.005,
}

# Authentication method distribution for legitimate transactions
LEGITIMATE_AUTH_WEIGHTS: Dict[str, float] = {
    "biometric": 0.35,
    "3ds_v2": 0.25,
    "sms_otp": 0.18,
    "password": 0.12,
    "none": 0.07,
    "hardware_token": 0.02,
    "push_notification": 0.01,
}

# Country population/activity weighting
COUNTRY_WEIGHTS: Dict[str, float] = {
    "US": 0.35,
    "GB": 0.15,
    "DE": 0.10,
    "FR": 0.08,
    "CA": 0.08,
    "IN": 0.08,
    "SG": 0.04,
    "AU": 0.04,
    "JP": 0.03,
    "NL": 0.02,
    "BR": 0.02,
    "AE": 0.01,
}


def sample_categorical(weights: Dict[str, float], rng: np.random.Generator) -> str:
    """Sample a categorical value given a dictionary of label -> weight."""
    labels = list(weights.keys())
    probs = np.array(list(weights.values()), dtype=np.float64)
    probs = probs / probs.sum()
    return str(rng.choice(labels, p=probs))


def clip_score(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Safely clip risk and deviation scores to [0.0, 1.0]."""
    return float(np.clip(value, min_val, max_val))
