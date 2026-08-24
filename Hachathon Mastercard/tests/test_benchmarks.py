"""Unit tests for Robustness Benchmarks, Model Comparisons, and Leakage Auditing."""

import pytest
import pandas as pd
import numpy as np

from src.detection.benchmarks import RobustnessBenchmark
from src.simulation.transaction_generator import TransactionGenerator
from src.utils.config import ALL_COLUMNS, NUMERICAL_FEATURES


@pytest.fixture(scope="module")
def benchmark_data():
    gen = TransactionGenerator(seed=42)
    df_train = gen.generate_dataset(n_samples=400, fraud_ratio=0.20)
    df_norm = gen.generate_dataset(n_samples=150, fraud_ratio=0.20)
    df_adv = gen.generate_dataset(n_samples=150, fraud_ratio=0.20)
    return df_train, df_norm, df_adv


def test_target_leakage_audit_clean_data(benchmark_data):
    df_train, _, _ = benchmark_data
    audit = RobustnessBenchmark.audit_target_leakage(df_train)

    assert audit["leakage_detected"] is False
    assert "PASS" in audit["status"]
    assert len(audit["leakage_violations"]) == 0


def test_generator_separation_analysis(benchmark_data):
    df_train, _, _ = benchmark_data
    analysis = RobustnessBenchmark.analyze_generator_separation(df_train)

    assert analysis["total_numerical_features_analyzed"] == len(NUMERICAL_FEATURES)
    assert len(analysis["feature_separation_rankings"]) == len(NUMERICAL_FEATURES)

    for item in analysis["feature_separation_rankings"]:
        assert "cohen_d" in item
        assert "ks_statistic" in item
        assert item["ks_statistic"] >= 0.0


def test_ablation_study_execution(benchmark_data, tmp_path):
    df_train, df_norm, df_adv = benchmark_data
    benchmark = RobustnessBenchmark(seed=42, output_dir=tmp_path)

    ablation_df = benchmark.run_ablation_study(df_train, df_norm, df_adv)

    assert not ablation_df.empty
    assert len(ablation_df) == 6  # 2 models * 3 ablation configs
    assert set(ablation_df["model"].unique()) == {"XGBOOST", "RANDOM_FOREST"}
    assert "Without device_change" in ablation_df["ablation_setting"].values
    assert "Without Top-3 Risk Signals" in ablation_df["ablation_setting"].values

    csv_path = tmp_path / "ablation_study.csv"
    assert csv_path.exists()
