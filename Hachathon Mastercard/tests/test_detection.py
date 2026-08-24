"""Unit tests for ML detector training, inference, and evaluation."""

import pytest
import numpy as np
import pandas as pd

from src.simulation.transaction_generator import TransactionGenerator
from src.detection.train import train_baseline_detector
from src.detection.predict import FraudDetector
from src.detection.evaluate import (
    evaluate_global_metrics,
    evaluate_attack_specific_metrics,
    get_weakest_attacks,
)


@pytest.fixture(scope="module")
def trained_detector_bundle():
    gen = TransactionGenerator(seed=42)
    df = gen.generate_dataset(n_samples=600, fraud_ratio=0.20)
    artifact, train_df, test_df = train_baseline_detector(df, test_size=0.25, seed=42)
    detector = FraudDetector(artifact=artifact)
    return detector, test_df


def test_detector_training(trained_detector_bundle):
    detector, test_df = trained_detector_bundle
    assert detector.model is not None
    assert detector.pipeline is not None
    assert len(detector.feature_names) > 0


def test_detector_inference(trained_detector_bundle):
    detector, test_df = trained_detector_bundle

    # Batch probabilities
    probs = detector.predict_proba(test_df)
    assert len(probs) == len(test_df)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Batch binary predictions
    preds = detector.predict(test_df)
    assert len(preds) == len(test_df)
    assert set(np.unique(preds)).issubset({0, 1})


def test_single_transaction_scoring(trained_detector_bundle):
    detector, test_df = trained_detector_bundle
    sample_tx = test_df.iloc[0].to_dict()

    result = detector.score_transaction(sample_tx)
    assert "fraud_probability" in result
    assert "is_fraud" in result
    assert "decision" in result
    assert result["decision"] in ["APPROVE", "DECLINE"]
    assert "risk_tier" in result


def test_evaluation_metrics(trained_detector_bundle):
    detector, test_df = trained_detector_bundle
    y_true = test_df["fraud_label"].to_numpy()
    y_pred = detector.predict(test_df)
    y_prob = detector.predict_proba(test_df)

    metrics = evaluate_global_metrics(y_true, y_pred, y_prob)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["false_positive_rate"] <= 1.0
    assert metrics["true_negatives"] + metrics["false_positives"] + metrics["false_negatives"] + metrics["true_positives"] == len(test_df)


def test_attack_specific_breakdown(trained_detector_bundle):
    detector, test_df = trained_detector_bundle
    attack_eval = evaluate_attack_specific_metrics(test_df, detector)

    assert not attack_eval.empty
    assert "detection_rate" in attack_eval.columns
    assert "false_negative_rate" in attack_eval.columns
    assert "attack_type" in attack_eval.columns

    weakest = get_weakest_attacks(attack_eval, bottom_n=3)
    assert len(weakest) <= 3
    assert (weakest["is_fraud"] == True).all()
