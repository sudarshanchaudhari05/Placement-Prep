"""Unit tests for Feature Engineering pipeline."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.features.feature_engineering import (
    FraudFeaturePipeline,
    extract_features_and_targets,
)
from src.simulation.transaction_generator import TransactionGenerator


@pytest.fixture
def sample_data():
    gen = TransactionGenerator(seed=42)
    return gen.generate_dataset(n_samples=250, fraud_ratio=0.15)


def test_extract_features_and_targets(sample_data):
    X, y, attacks = extract_features_and_targets(sample_data)

    assert "fraud_label" not in X.columns
    assert "attack_type" not in X.columns
    assert len(X) == 250
    assert len(y) == 250
    assert len(attacks) == 250
    assert (y.isin([0, 1])).all()


def test_pipeline_fit_transform(sample_data):
    X, y, _ = extract_features_and_targets(sample_data)
    pipeline = FraudFeaturePipeline()

    X_transformed = pipeline.fit_transform(X, y)

    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[0] == 250
    assert X_transformed.shape[1] > 20  # Numerical + OneHot encoded categorical features
    assert not np.isnan(X_transformed).any()
    assert not np.isinf(X_transformed).any()


def test_pipeline_feature_names(sample_data):
    X, y, _ = extract_features_and_targets(sample_data)
    pipeline = FraudFeaturePipeline()
    pipeline.fit(X, y)

    feature_names = pipeline.get_feature_names_out()
    assert len(feature_names) == pipeline.transform(X).shape[1]
    assert any("merchant_category_" in name for name in feature_names)
    assert "transaction_amount" in feature_names


def test_pipeline_serialization(sample_data, tmp_path):
    X, y, _ = extract_features_and_targets(sample_data)
    pipeline = FraudFeaturePipeline()
    X_orig = pipeline.fit_transform(X, y)

    save_path = tmp_path / "pipeline.joblib"
    pipeline.save(save_path)

    loaded_pipeline = FraudFeaturePipeline.load(save_path)
    X_loaded = loaded_pipeline.transform(X)

    np.testing.assert_array_almost_equal(X_orig, X_loaded)
