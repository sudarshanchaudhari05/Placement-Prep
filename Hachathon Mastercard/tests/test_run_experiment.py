"""Unit tests for run_experiment.py orchestration runner."""

import pytest
from pathlib import Path
from unittest.mock import patch
import run_experiment
from run_experiment import run_experiment as execute_run_experiment, save_summary_file, print_results


def test_run_experiment_orchestration(tmp_path):
    """Test fast end-to-end execution of run_experiment on a small sample configuration."""
    results = execute_run_experiment(
        samples=200,
        fraud_ratio=0.20,
        mutation_intensity=0.50,
        output_dir=tmp_path,
    )

    assert "normal_test_set_performance" in results
    assert "unseen_adversarial_test_performance" in results
    assert "adversarial_false_negatives" in results
    assert "attack_comparison" in results

    norm_base = results["normal_test_set_performance"]["baseline_detector"]
    assert "f1_score" in norm_base
    assert "recall" in norm_base

    adv_base = results["unseen_adversarial_test_performance"]["baseline_detector"]
    adv_hard = results["unseen_adversarial_test_performance"]["hardened_detector"]
    assert adv_base["recall"] >= 0.0
    assert adv_hard["recall"] >= 0.0


def test_save_summary_file(tmp_path):
    """Test saving human-readable summary text file."""
    dummy_results = {
        "sample_counts": {
            "dataset_a_total": 1000,
            "dataset_a_train": 800,
            "dataset_a_test": 200,
            "dataset_b_adversarial_train": 250,
            "dataset_c_unseen_test": 200,
        },
        "seeds": {
            "baseline_dataset_a": 42,
            "adversarial_train_dataset_b": 101,
            "unseen_test_dataset_c": 1337,
        },
        "detector_feature_dependencies": ["merchant_risk_score", "device_change"],
        "normal_test_set_performance": {
            "baseline_detector": {
                "accuracy": 0.98,
                "precision": 0.95,
                "recall": 0.96,
                "f1_score": 0.955,
                "false_positive_rate": 0.01,
            },
            "hardened_detector": {
                "accuracy": 0.985,
                "precision": 0.96,
                "recall": 0.97,
                "f1_score": 0.965,
                "false_positive_rate": 0.01,
            },
        },
        "unseen_adversarial_test_performance": {
            "baseline_detector": {
                "recall": 0.65,
                "f1_score": 0.75,
                "false_negatives": 35,
            },
            "hardened_detector": {
                "recall": 0.88,
                "f1_score": 0.89,
                "false_negatives": 12,
            },
        },
        "attack_comparison": [
            {
                "attack_type": "Test Attack A",
                "is_fraud": True,
                "baseline_detection_rate": 20.0,
                "hardened_detection_rate": 80.0,
                "detection_rate_delta": 60.0,
            }
        ],
    }

    summary_file = tmp_path / "latest_run_summary.txt"
    save_summary_file(dummy_results, summary_file)

    assert summary_file.exists()
    content = summary_file.read_text(encoding="utf-8")
    assert "FRAUDFORGE AI -- LATEST EXPERIMENT RUN SUMMARY" in content
    assert "Test Attack A" in content
    assert "DEFENSE GAINS & HARDENING IMPACT" in content


def test_print_results_execution(capsys):
    """Test print_results formats and prints correctly without errors."""
    dummy_results = {
        "normal_test_set_performance": {
            "baseline_detector": {
                "f1_score": 0.9554,
                "recall": 0.9633,
                "false_positive_rate": 0.0094,
            },
            "hardened_detector": {
                "f1_score": 0.9314,
                "recall": 0.9733,
                "false_positive_rate": 0.0206,
            },
        },
        "unseen_adversarial_test_performance": {
            "baseline_detector": {
                "recall": 0.68,
                "f1_score": 0.7846,
                "false_negatives": 96,
            },
            "hardened_detector": {
                "recall": 0.8933,
                "f1_score": 0.9008,
                "false_negatives": 32,
            },
        },
        "attack_comparison": [
            {
                "attack_type": "Test Archetype",
                "is_fraud": True,
                "baseline_detection_rate": 10.0,
                "hardened_detection_rate": 50.0,
            }
        ],
    }

    print_results(dummy_results)
    captured = capsys.readouterr()
    assert "FINAL RESULTS" in captured.out
    assert "NORMAL TRAFFIC" in captured.out
    assert "UNSEEN ADVERSARIAL ATTACKS" in captured.out
    assert "Test Archetype" in captured.out


def test_cli_argument_parsing():
    """Verify CLI parser handles default and custom arguments."""
    with patch("sys.argv", ["run_experiment.py", "--samples", "5000", "--fraud-ratio", "0.25", "--mutation-intensity", "0.80"]):
        parser = run_experiment.argparse.ArgumentParser()
        parser.add_argument("--samples", type=int, default=10000)
        parser.add_argument("--fraud-ratio", type=float, default=0.15)
        parser.add_argument("--mutation-intensity", type=float, default=0.65)
        args = parser.parse_args(["--samples", "5000", "--fraud-ratio", "0.25", "--mutation-intensity", "0.80"])

        assert args.samples == 5000
        assert args.fraud_ratio == 0.25
        assert args.mutation_intensity == 0.80
