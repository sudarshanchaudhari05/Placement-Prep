"""Unit tests for Synthetic Transaction Generator and validation."""

import pytest
import pandas as pd
import numpy as np

from src.simulation.transaction_generator import TransactionGenerator
from src.utils.config import ALL_COLUMNS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMNS


def test_generator_determinism():
    """Verify identical random seeds produce identical generated transaction datasets."""
    gen1 = TransactionGenerator(seed=123)
    df1 = gen1.generate_dataset(n_samples=200, fraud_ratio=0.20, shuffle=False)

    gen2 = TransactionGenerator(seed=123)
    df2 = gen2.generate_dataset(n_samples=200, fraud_ratio=0.20, shuffle=False)

    pd.testing.assert_frame_equal(df1, df2)


def test_dataset_schema_and_columns():
    """Verify generated dataset adheres strictly to expected column schema."""
    generator = TransactionGenerator(seed=42)
    df = generator.generate_dataset(n_samples=500, fraud_ratio=0.15)

    assert list(df.columns) == ALL_COLUMNS
    assert len(df) == 500
    assert df["fraud_label"].sum() == 75  # 15% of 500


def test_no_null_values_or_nans():
    """Ensure zero NaN, Null, or infinite values in generated data."""
    generator = TransactionGenerator(seed=42)
    df = generator.generate_dataset(n_samples=1000, fraud_ratio=0.15)

    assert df.isnull().sum().sum() == 0
    assert not np.isinf(df[NUMERICAL_FEATURES].to_numpy()).any()


def test_invariants_and_ranges():
    """Check mathematical and business domain invariants on generated records."""
    generator = TransactionGenerator(seed=42)
    df = generator.generate_dataset(n_samples=1000, fraud_ratio=0.20)

    # Positive transaction amounts
    assert (df["transaction_amount"] > 0).all()

    # Valid hours (0 to 23)
    assert df["transaction_hour"].between(0, 23).all()

    # Risk scores within [0.0, 1.0]
    assert df["IP_risk_score"].between(0.0, 1.0).all()
    assert df["merchant_risk_score"].between(0.0, 1.0).all()
    assert df["behavioral_deviation"].between(0.0, 1.0).all()
    assert df["identity_risk_score"].between(0.0, 1.0).all()

    # Velocity invariants
    assert (df["transaction_velocity_24h"] >= df["transaction_velocity_1h"]).all()
    assert (df["transaction_velocity_1h"] >= 1).all()

    # Binary flags
    assert df["device_change"].isin([0, 1]).all()
    assert df["geographic_deviation"].isin([0, 1]).all()
    assert df["fraud_label"].isin([0, 1]).all()


def test_label_and_attack_type_consistency():
    """Ensure label alignment between fraud_label and attack_type."""
    generator = TransactionGenerator(seed=42)
    df = generator.generate_dataset(n_samples=1000, fraud_ratio=0.20)

    legit = df[df["fraud_label"] == 0]
    fraud = df[df["fraud_label"] == 1]

    assert (legit["attack_type"] == "LEGITIMATE").all()
    assert (fraud["attack_type"] != "LEGITIMATE").all()
    assert fraud["attack_type"].nunique() > 20


def test_distribution_divergence_between_fraud_and_legit():
    """Verify fraudulent transactions have measurably distinct behavioral distributions."""
    generator = TransactionGenerator(seed=42)
    df = generator.generate_dataset(n_samples=3000, fraud_ratio=0.20)

    legit = df[df["fraud_label"] == 0]
    fraud = df[df["fraud_label"] == 1]

    # Fraud should on average exhibit higher IP risk score than legitimate
    assert fraud["IP_risk_score"].mean() > legit["IP_risk_score"].mean()

    # Fraud should on average exhibit higher behavioral deviation
    assert fraud["behavioral_deviation"].mean() > legit["behavioral_deviation"].mean()

    # Fraud should on average exhibit higher identity risk score
    assert fraud["identity_risk_score"].mean() > legit["identity_risk_score"].mean()


def test_realistic_feature_overlap_and_bounds():
    """Verify realistic distribution properties and non-extreme separation."""
    generator = TransactionGenerator(seed=42)
    df = generator.generate_dataset(n_samples=5000, fraud_ratio=0.20)

    legit = df[df["fraud_label"] == 0]
    fraud = df[df["fraud_label"] == 1]

    # Legitimate device change rate should be realistic (~10% - 25%)
    legit_dc_rate = legit["device_change"].mean()
    assert 0.10 <= legit_dc_rate <= 0.25

    # Fraud device change rate should leave substantial device_change == 0 (~35% - 65%)
    fraud_dc_rate = fraud["device_change"].mean()
    assert 0.35 <= fraud_dc_rate <= 0.65

    # Legitimate users should occasionally make high-ticket purchases (> $500)
    assert (legit["transaction_amount"] > 500).any()

    # Fraud includes micro / normal carding transactions (< $25)
    assert (fraud["transaction_amount"] < 25).any()


def test_validate_dataset_method():
    """Test built-in validate_dataset static utility."""
    generator = TransactionGenerator(seed=42)
    df = generator.generate_dataset(n_samples=500, fraud_ratio=0.10)

    validation = TransactionGenerator.validate_dataset(df)
    assert validation["invariants_passed"] is True
    assert validation["total_records"] == 500
    assert validation["total_nulls"] == 0
