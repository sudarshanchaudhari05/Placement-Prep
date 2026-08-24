"""FraudForge AI: Detector Evaluation and Attack Vulnerability Analysis.

Computes comprehensive global classification metrics and granular attack-specific
detection and false-negative rates for all GenAI fraud archetypes.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)

from src.detection.predict import FraudDetector
from src.features.feature_engineering import extract_features_and_targets
from src.utils.config import MODELS_DIR, GENERATED_DATA_DIR, PROCESSED_DATA_DIR


def evaluate_global_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """Calculate standard global classification and fraud detection metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = float(fp / max(1, (fp + tn)))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "false_positive_rate": fpr,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_evaluated": int(len(y_true)),
    }


def evaluate_attack_specific_metrics(
    test_df: pd.DataFrame,
    detector: FraudDetector,
) -> pd.DataFrame:
    """Evaluate detector performance broken down across every attack archetype.

    Calculates:
    - total_transactions
    - detected (TP)
    - missed (FN)
    - detection_rate (%)
    - false_negative_rate (%)
    """
    preds = detector.predict(test_df)
    probs = detector.predict_proba(test_df)

    eval_df = test_df[["attack_type", "fraud_label"]].copy()
    eval_df["predicted_label"] = preds
    eval_df["fraud_probability"] = probs

    records: List[Dict[str, Any]] = []

    # Process each attack type (including LEGITIMATE)
    for attack_type, group in eval_df.groupby("attack_type"):
        total = len(group)
        is_legit = (group["fraud_label"].iloc[0] == 0)

        if is_legit:
            # For legitimate, "detected" means correctly classified as 0 (TN)
            correct = int((group["predicted_label"] == 0).sum())
            incorrect = int((group["predicted_label"] == 1).sum())  # False alarms
            det_rate = correct / total
            fn_rate = incorrect / total
        else:
            # For fraud attacks, "detected" means correctly caught as 1 (TP)
            detected = int((group["predicted_label"] == 1).sum())
            missed = int((group["predicted_label"] == 0).sum())  # False negatives
            det_rate = detected / total
            fn_rate = missed / total

        records.append({
            "attack_type": attack_type,
            "is_fraud": not is_legit,
            "total_transactions": total,
            "detected": detected if not is_legit else correct,
            "missed": missed if not is_legit else incorrect,
            "detection_rate": float(np.round(det_rate * 100, 2)),
            "false_negative_rate": float(np.round(fn_rate * 100, 2)),
            "avg_predicted_risk": float(np.round(group["fraud_probability"].mean(), 4)),
        })

    result_df = pd.DataFrame(records)
    # Sort fraud attacks by detection rate ascending (weakest defense first)
    fraud_slice = result_df[result_df["is_fraud"]].sort_values(by="detection_rate", ascending=True)
    legit_slice = result_df[~result_df["is_fraud"]]
    
    return pd.concat([fraud_slice, legit_slice], ignore_index=True)


def get_weakest_attacks(
    attack_eval_df: pd.DataFrame,
    bottom_n: int = 10,
) -> pd.DataFrame:
    """Extract the top N weakest attack vectors where the detector has highest miss rates."""
    fraud_only = attack_eval_df[attack_eval_df["is_fraud"]].copy()
    return fraud_only.sort_values(by="detection_rate", ascending=True).head(bottom_n)


def format_evaluation_report(
    global_metrics: Dict[str, Any],
    attack_eval_df: pd.DataFrame,
) -> str:
    """Format an end-to-end readable terminal evaluation report."""
    lines = []
    lines.append("=" * 80)
    lines.append("                FRAUDFORGE AI -- BLUE-TEAM DETECTOR EVALUATION REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append("> GLOBAL CLASSIFICATION METRICS (HELD-OUT TEST SET):")
    lines.append(f"  * Accuracy              : {global_metrics['accuracy']:.4f} ({global_metrics['accuracy']*100:.2f}%)")
    lines.append(f"  * Precision             : {global_metrics['precision']:.4f} ({global_metrics['precision']*100:.2f}%)")
    lines.append(f"  * Recall                : {global_metrics['recall']:.4f} ({global_metrics['recall']*100:.2f}%)")
    lines.append(f"  * F1 Score              : {global_metrics['f1_score']:.4f}")
    lines.append(f"  * ROC-AUC               : {global_metrics['roc_auc']:.4f}")
    lines.append(f"  * False Positive Rate   : {global_metrics['false_positive_rate']:.4f} ({global_metrics['false_positive_rate']*100:.2f}%)")
    lines.append("")
    lines.append("> CONFUSION MATRIX:")
    lines.append(f"  * True Negatives (TN)   : {global_metrics['true_negatives']:,}")
    lines.append(f"  * False Positives (FP)  : {global_metrics['false_positives']:,}")
    lines.append(f"  * False Negatives (FN)  : {global_metrics['false_negatives']:,}")
    lines.append(f"  * True Positives (TP)   : {global_metrics['true_positives']:,}")
    lines.append(f"  * Total Evaluated       : {global_metrics['total_evaluated']:,}")
    lines.append("")
    lines.append("-" * 80)
    lines.append("> ATTACK VULNERABILITY ANALYSIS (FRAUD DETECTION RATES BY ARCHETYPE):")
    lines.append(f"{'Attack Archetype':<55} | {'Total':<6} | {'Caught':<6} | {'Missed':<6} | {'Det. Rate':<9}")
    lines.append("-" * 80)

    fraud_df = attack_eval_df[attack_eval_df["is_fraud"]].sort_values(by="detection_rate", ascending=True)
    for _, row in fraud_df.iterrows():
        name = str(row["attack_type"])
        if len(name) > 53:
            name = name[:50] + "..."
        lines.append(
            f"{name:<55} | {int(row['total_transactions']):<6} | {int(row['detected']):<6} | {int(row['missed']):<6} | {row['detection_rate']:>7.1f}%"
        )

    lines.append("-" * 80)
    legit_row = attack_eval_df[~attack_eval_df["is_fraud"]].iloc[0]
    lines.append(
        f"{'LEGITIMATE TRANSACTIONS (Specificity)':<55} | {int(legit_row['total_transactions']):<6} | {int(legit_row['detected']):<6} | {int(legit_row['missed']):<6} | {legit_row['detection_rate']:>7.1f}%"
    )
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    """CLI evaluation entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate FraudForge AI Baseline Detector")
    parser.add_argument("--data", type=str, default="", help="Path to evaluation test CSV dataset")
    parser.add_argument("--model", type=str, default=str(MODELS_DIR / "baseline_detector.joblib"))
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first via src.detection.train")

    detector = FraudDetector(artifact_path=model_path, threshold=args.threshold)

    if args.data and Path(args.data).exists():
        test_df = pd.read_csv(args.data)
    else:
        # Load test split if available, otherwise stratified split from generated 10k dataset
        test_split_path = PROCESSED_DATA_DIR / "test_split.csv"
        gen_file = GENERATED_DATA_DIR / "synthetic_transactions_v1.csv"
        if test_split_path.exists():
            test_df = pd.read_csv(test_split_path)
        elif gen_file.exists():
            full_df = pd.read_csv(gen_file)
            from sklearn.model_selection import train_test_split
            _, test_df = train_test_split(full_df, test_size=0.20, stratify=full_df["fraud_label"], random_state=42)
        else:
            raise FileNotFoundError("Evaluation dataset not found. Run src.detection.train or provide --data.")

    y_true = test_df["fraud_label"].to_numpy()
    y_pred = detector.predict(test_df)
    y_prob = detector.predict_proba(test_df)

    global_metrics = evaluate_global_metrics(y_true, y_pred, y_prob)
    attack_eval_df = evaluate_attack_specific_metrics(test_df, detector)

    report = format_evaluation_report(global_metrics, attack_eval_df)
    print(report)


if __name__ == "__main__":
    main()
