"""
SunFarm Smart Irrigation - Model Training & Evaluation Suite
============================================================
Trains and compares lightweight machine learning classifiers suitable for
edge inference on Raspberry Pi:
  1. Decision Tree Classifier
  2. Random Forest Classifier
  3. Gradient Boosting Classifier

Evaluates models using Accuracy, Precision, Recall, F1 Score, Confusion Matrix,
and Inference Latency. Selects and exports the best model to joblib/pickle format.
"""

import sys
import time
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Suppress sklearn/joblib verbose threading warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import (
    DATASET_PATH,
    MODEL_PATH,
    METRICS_PATH,
    FEATURE_NAMES,
    DATASET_CONFIG,
    IRRIGATION_CLASSES
)


def load_and_split_data():
    """Load the synthetic dataset and split into train/test sets."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Please run generate_dataset.py first."
        )

    df = pd.read_csv(DATASET_PATH)
    X = df[FEATURE_NAMES]
    y = df["irrigation_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=DATASET_CONFIG["TEST_SPLIT_RATIO"],
        random_state=DATASET_CONFIG["RANDOM_SEED"],
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def benchmark_inference_latency(model, X_sample, n_iterations: int = 1000) -> float:
    """Benchmark inference latency (milliseconds per sample)."""
    single_sample = X_sample.iloc[[0]]
    # Warmup
    for _ in range(50):
        _ = model.predict(single_sample)

    start_time = time.perf_counter()
    for _ in range(n_iterations):
        _ = model.predict(single_sample)
    end_time = time.perf_counter()

    latency_ms = ((end_time - start_time) / n_iterations) * 1000.0
    return latency_ms


def train_and_evaluate():
    """Train candidates, compare metrics, and select best lightweight model."""
    print("=" * 70)
    print("      SunFarm Irrigation ML - Model Training & Evaluation Suite")
    print("=" * 70)

    X_train, X_test, y_train, y_test = load_and_split_data()
    print(f"Dataset Split: {len(X_train)} training samples | {len(X_test)} test samples")
    print(f"Features: {', '.join(FEATURE_NAMES)}")
    print("-" * 70)

    # Candidate Lightweight Models
    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=DATASET_CONFIG["RANDOM_SEED"]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=DATASET_CONFIG["RANDOM_SEED"],
            n_jobs=1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=60,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.85,
            random_state=DATASET_CONFIG["RANDOM_SEED"]
        )
    }

    results = {}
    best_model_name = None
    best_f1 = -1.0

    temp_model_path = MODEL_PATH.parent / "temp_eval.pkl"
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        print(f"\n>>> Training Candidate: {name} ...")
        train_start = time.perf_counter()
        model.fit(X_train, y_train)
        train_duration = time.perf_counter() - train_start

        # Predictions on Test Set
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        # Edge latency and model size benchmark
        latency_ms = benchmark_inference_latency(model, X_test)
        joblib.dump(model, temp_model_path)
        model_size_kb = temp_model_path.stat().st_size / 1024.0

        results[name] = {
            "model_object": model,
            "accuracy": float(acc),
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "latency_ms": float(latency_ms),
            "model_size_kb": float(model_size_kb),
            "train_duration_s": float(train_duration),
            "confusion_matrix": cm.tolist()
        }

        print(f"  Training Time:      {train_duration:.3f} s")
        print(f"  Accuracy:           {acc * 100:.2f}%")
        print(f"  Precision (Macro):  {prec_macro * 100:.2f}%")
        print(f"  Recall (Macro):     {rec_macro * 100:.2f}%")
        print(f"  F1 Score (Macro):   {f1_macro * 100:.2f}%")
        print(f"  F1 Score (Weighted):{f1_weighted * 100:.2f}%")
        print(f"  Inference Latency:  {latency_ms:.4f} ms/sample")
        print(f"  Model Size:         {model_size_kb:.1f} KB")
        print("\n  Confusion Matrix:")
        print("  " + "\n  ".join([str(row) for row in cm]))

        # Model selection heuristic: Priority on F1-Macro with reasonable size and edge latency
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_model_name = name

    # Cleanup temp file
    if temp_model_path.exists():
        temp_model_path.unlink()

    # Comparison Table
    print("\n" + "=" * 70)
    print("                    MODEL COMPARISON SUMMARY")
    print("=" * 70)
    header = f"{'Model':<20} | {'Accuracy':<10} | {'F1-Macro':<10} | {'F1-Weighted':<12} | {'Latency(ms)':<12} | {'Size(KB)':<8}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"{name:<20} | {m['accuracy']*100:>8.2f}% | {m['f1_macro']*100:>8.2f}% | "
            f"{m['f1_weighted']*100:>10.2f}% | {m['latency_ms']:>10.4f} ms | {m['model_size_kb']:>6.1f} KB"
        )
    print("=" * 70)

    print(f"\n[WINNER] Selected Best Lightweight Model: >>> {best_model_name} <<<")
    selected_model = results[best_model_name]["model_object"]

    # Save Winning Model Artifact
    artifact = {
        "model": selected_model,
        "model_name": best_model_name,
        "features": FEATURE_NAMES,
        "classes": IRRIGATION_CLASSES,
        "metrics": {
            "accuracy": results[best_model_name]["accuracy"],
            "f1_macro": results[best_model_name]["f1_macro"],
            "f1_weighted": results[best_model_name]["f1_weighted"],
            "precision_macro": results[best_model_name]["precision_macro"],
            "recall_macro": results[best_model_name]["recall_macro"],
            "latency_ms": results[best_model_name]["latency_ms"],
            "model_size_kb": results[best_model_name]["model_size_kb"]
        }
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"Exported final model to: {MODEL_PATH}")

    # Save metrics JSON for documentation
    metrics_export = {
        k: {m_k: m_v for m_k, m_v in v.items() if m_k != "model_object"}
        for k, v in results.items()
    }
    metrics_export["selected_model"] = best_model_name
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_export, f, indent=2)
    print(f"Exported metrics report to: {METRICS_PATH}")

    # Detailed Classification Report for Selected Model
    y_pred_best = selected_model.predict(X_test)
    target_names = [IRRIGATION_CLASSES[i]["label"] for i in sorted(IRRIGATION_CLASSES.keys())]
    print(f"\n--- Detailed Classification Report for {best_model_name} ---")
    print(classification_report(y_test, y_pred_best, target_names=target_names, digits=4))

    # Feature Importances if available
    if hasattr(selected_model, "feature_importances_"):
        print("Feature Importances:")
        importances = pd.Series(selected_model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
        for feat, imp in importances.items():
            bar = "#" * int(imp * 40)
            print(f"  {feat:<18}: {imp*100:>5.2f}% {bar}")

    return best_model_name, results


if __name__ == "__main__":
    train_and_evaluate()
