"""Configuration constants, schema definitions, and paths for FraudForge AI."""

from pathlib import Path
from typing import List

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
GENERATED_DATA_DIR = DATA_DIR / "generated"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Default Random Seed
DEFAULT_SEED = 42

# Column Schema Definitions
NUMERICAL_FEATURES: List[str] = [
    "transaction_amount",
    "transaction_hour",
    "account_age_days",
    "device_age_days",
    "device_change",
    "IP_risk_score",
    "merchant_risk_score",
    "transaction_velocity_1h",
    "transaction_velocity_24h",
    "average_customer_amount",
    "amount_deviation",
    "geographic_deviation",
    "behavioral_deviation",
    "failed_authentication_count",
    "identity_risk_score",
]

CATEGORICAL_FEATURES: List[str] = [
    "merchant_category",
    "payment_channel",
    "authentication_method",
    "transaction_country",
    "customer_country",
]

TARGET_COLUMNS: List[str] = [
    "attack_type",
    "fraud_label",
]

ALL_COLUMNS: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + TARGET_COLUMNS

# Domain Vocabularies
MERCHANT_CATEGORIES: List[str] = [
    "groceries",
    "retail",
    "dining",
    "travel",
    "digital_goods",
    "gaming",
    "luxury",
    "crypto_exchange",
    "money_transfer",
    "utilities",
    "electronics",
    "marketplace",
]

PAYMENT_CHANNELS: List[str] = [
    "e-commerce",
    "pos_contactless",
    "pos_chip",
    "p2p_transfer",
    "mobile_app",
    "recurring_subscription",
    "api_gateway",
]

AUTHENTICATION_METHODS: List[str] = [
    "none",
    "sms_otp",
    "biometric",
    "3ds_v2",
    "hardware_token",
    "password",
    "push_notification",
]

SUPPORTED_COUNTRIES: List[str] = [
    "US",
    "GB",
    "DE",
    "FR",
    "CA",
    "IN",
    "SG",
    "JP",
    "AU",
    "BR",
    "AE",
    "NL",
]
