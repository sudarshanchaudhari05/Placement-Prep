"""Unit tests for AttackMutator engine."""

import pytest
import numpy as np
import pandas as pd

from src.attacks.attack_library import get_default_attack_library
from src.attacks.attack_mutator import AttackMutator
from src.simulation.transaction_generator import TransactionGenerator
from src.utils.config import ALL_COLUMNS, NUMERICAL_FEATURES


@pytest.fixture
def base_fraud_tx():
    library = get_default_attack_library()
    generator = TransactionGenerator(seed=42)
    atk = library.get_by_id("ATK-021")  # Adversarial Perturbation
    return generator.generate_fraud_transaction(archetype=atk), atk


def test_mutator_determinism(base_fraud_tx):
    tx, atk = base_fraud_tx
    mutator1 = AttackMutator(seed=123)
    mut_tx1, _ = mutator1.mutate_transaction(tx, archetype=atk, mutation_intensity=0.7)

    mutator2 = AttackMutator(seed=123)
    mut_tx2, _ = mutator2.mutate_transaction(tx, archetype=atk, mutation_intensity=0.7)

    assert mut_tx1 == mut_tx2


def test_mutation_preserves_label_and_schema(base_fraud_tx):
    tx, atk = base_fraud_tx
    mutator = AttackMutator(seed=42)
    mut_tx, audit = mutator.mutate_transaction(tx, archetype=atk, mutation_intensity=0.8)

    # Invariants
    assert mut_tx["fraud_label"] == 1
    assert mut_tx["transaction_amount"] > 0
    assert 0 <= mut_tx["transaction_hour"] <= 23
    assert 0.0 <= mut_tx["IP_risk_score"] <= 1.0
    assert 0.0 <= mut_tx["behavioral_deviation"] <= 1.0
    assert mut_tx["transaction_velocity_24h"] >= mut_tx["transaction_velocity_1h"] >= 1
    assert len(audit) > 0


def test_signal_masking_device_change():
    mutator = AttackMutator(seed=42)
    fake_tx = {
        "device_change": 1,
        "device_age_days": 1,
        "account_age_days": 100,
        "fraud_label": 1,
        "attack_type": "Test Attack",
        "transaction_amount": 50.0,
        "average_customer_amount": 50.0,
        "amount_deviation": 0.0,
        "IP_risk_score": 0.2,
        "behavioral_deviation": 0.2,
        "transaction_velocity_1h": 1,
        "transaction_velocity_24h": 1,
    }
    mut_tx, audit = mutator.mutate_transaction(fake_tx, detector_weaknesses=["device_change"])
    assert mut_tx["device_change"] == 0
    assert mut_tx["device_age_days"] >= 25
    assert any(a["feature"] == "device_change" for a in audit)


def test_behavioral_and_velocity_smoothing():
    mutator = AttackMutator(seed=42)
    fake_tx = {
        "device_change": 0,
        "device_age_days": 50,
        "account_age_days": 100,
        "fraud_label": 1,
        "attack_type": "Test Attack",
        "transaction_amount": 50.0,
        "average_customer_amount": 50.0,
        "amount_deviation": 0.0,
        "IP_risk_score": 0.5,
        "behavioral_deviation": 0.85,
        "transaction_velocity_1h": 6,
        "transaction_velocity_24h": 12,
    }
    mut_tx, audit = mutator.mutate_transaction(
        fake_tx,
        detector_weaknesses=["behavioral_deviation", "transaction_velocity_1h", "IP_risk_score"],
        mutation_intensity=0.8,
    )
    assert mut_tx["behavioral_deviation"] < 0.85
    assert mut_tx["transaction_velocity_1h"] < 6
    assert mut_tx["IP_risk_score"] < 0.5


def test_mutate_dataframe():
    gen = TransactionGenerator(seed=42)
    df = gen.generate_dataset(n_samples=100, fraud_ratio=0.50)
    df_fraud = df[df["fraud_label"] == 1].copy()

    mutator = AttackMutator(seed=42)
    mut_df, audit_df = mutator.mutate_dataframe(df_fraud, mutation_intensity=0.6)

    assert len(mut_df) == len(df_fraud)
    assert (mut_df["fraud_label"] == 1).all()
    assert not audit_df.empty
    assert "reason" in audit_df.columns
