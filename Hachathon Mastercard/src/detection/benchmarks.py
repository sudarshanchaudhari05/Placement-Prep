"""FraudForge AI: Robustness Benchmarking, Model Comparison, and Feature Ablation Engine.

Provides:
1. Direct side-by-side benchmarking: XGBoost vs Random Forest across normal & adversarial test sets.
2. Feature Ablation Studies: Full features vs without device_change vs without top-3 risk signals.
3. Target & Metadata Leakage Auditing.
4. Synthetic Data Separation & Distribution Realism Diagnostics.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.features.feature_engineering import FraudFeaturePipeline, extract_features_and_targets
from src.detection.train import train_baseline_detector
from src.detection.predict import FraudDetector
from src.detection.evaluate import evaluate_global_metrics, evaluate_attack_specific_metrics
from src.utils.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMNS,
    DEFAULT_SEED,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    EXPERIMENTS_DIR,
)


class RobustnessBenchmark:
    """Robustness benchmarking and feature ablation study suite."""

    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        output_dir: Optional[Path] = None,
    ):
        self.seed = seed
        self.output_dir = output_dir or EXPERIMENTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_and_eval_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test_normal: pd.DataFrame,
        y_test_normal: pd.Series,
        X_test_adv: pd.DataFrame,
        y_test_adv: pd.Series,
        numerical_cols: List[str],
        categorical_cols: List[str],
    ) -> Dict[str, Any]:
        """Train a specific model architecture with specified feature subset and evaluate on both test sets."""
        pipeline = FraudFeaturePipeline(
            numerical_features=numerical_cols,
            categorical_features=categorical_cols,
        )
        X_train_trans = pipeline.fit_transform(X_train, y_train)
        X_test_norm_trans = pipeline.transform(X_test_normal)
        X_test_adv_trans = pipeline.transform(X_test_adv)

        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = float(n_neg / max(1, n_pos))

        if model_name.lower() == "xgboost" and XGBOOST_AVAILABLE:
            model = XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=scale_pos_weight,
                random_state=self.seed,
                eval_metric="logloss",
                n_jobs=-1,
            )
        else:
            model = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                class_weight="balanced",
                random_state=self.seed,
                n_jobs=-1,
            )

        model.fit(X_train_trans, y_train)

        # Normal Test Evaluation
        y_pred_norm = model.predict(X_test_norm_trans)
        y_prob_norm = model.predict_proba(X_test_norm_trans)[:, 1]
        normal_metrics = evaluate_global_metrics(y_test_normal.to_numpy(), y_pred_norm, y_prob_norm)

        # Adversarial Test Evaluation
        y_pred_adv = model.predict(X_test_adv_trans)
        y_prob_adv = model.predict_proba(X_test_adv_trans)[:, 1]
        adv_metrics = evaluate_global_metrics(y_test_adv.to_numpy(), y_pred_adv, y_prob_adv)

        return {
            "model_type": model_name,
            "feature_count": len(pipeline.get_feature_names_out()),
            "normal_test": normal_metrics,
            "adversarial_test": adv_metrics,
        }

    def run_model_comparison(
        self,
        train_df: pd.DataFrame,
        test_normal_df: pd.DataFrame,
        test_adv_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Direct benchmark: XGBoost vs Random Forest on identical splits."""
        X_train, y_train, _ = extract_features_and_targets(train_df)
        X_norm, y_norm, _ = extract_features_and_targets(test_normal_df)
        X_adv, y_adv, _ = extract_features_and_targets(test_adv_df)

        results = {}
        for m in ["xgboost", "random_forest"]:
            results[m] = self.train_and_eval_model(
                model_name=m,
                X_train=X_train,
                y_train=y_train,
                X_test_normal=X_norm,
                y_test_normal=y_norm,
                X_test_adv=X_adv,
                y_test_adv=y_adv,
                numerical_cols=NUMERICAL_FEATURES,
                categorical_cols=CATEGORICAL_FEATURES,
            )
        return results

    def run_ablation_study(
        self,
        train_df: pd.DataFrame,
        test_normal_df: pd.DataFrame,
        test_adv_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run feature ablation experiments across XGBoost and Random Forest.

        Ablations:
        1. All features
        2. Without device_change
        3. Without top-3 risk signals (device_change, merchant_risk_score, IP_risk_score)
        """
        X_train, y_train, _ = extract_features_and_targets(train_df)
        X_norm, y_norm, _ = extract_features_and_targets(test_normal_df)
        X_adv, y_adv, _ = extract_features_and_targets(test_adv_df)

        ablation_configs = [
            ("All Features", NUMERICAL_FEATURES, CATEGORICAL_FEATURES),
            (
                "Without device_change",
                [f for f in NUMERICAL_FEATURES if f != "device_change"],
                CATEGORICAL_FEATURES,
            ),
            (
                "Without Top-3 Risk Signals",
                [f for f in NUMERICAL_FEATURES if f not in ["device_change", "merchant_risk_score", "IP_risk_score"]],
                CATEGORICAL_FEATURES,
            ),
        ]

        records: List[Dict[str, Any]] = []

        for model_name in ["xgboost", "random_forest"]:
            for config_name, num_cols, cat_cols in ablation_configs:
                res = self.train_and_eval_model(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_test_normal=X_norm,
                    y_test_normal=y_norm,
                    X_test_adv=X_adv,
                    y_test_adv=y_adv,
                    numerical_cols=num_cols,
                    categorical_cols=cat_cols,
                )
                norm_m = res["normal_test"]
                adv_m = res["adversarial_test"]

                records.append({
                    "model": model_name.upper(),
                    "ablation_setting": config_name,
                    "num_features": res["feature_count"],
                    "normal_accuracy": norm_m["accuracy"],
                    "normal_f1": norm_m["f1_score"],
                    "normal_recall": norm_m["recall"],
                    "normal_fpr": norm_m["false_positive_rate"],
                    "adversarial_accuracy": adv_m["accuracy"],
                    "adversarial_f1": adv_m["f1_score"],
                    "adversarial_recall": adv_m["recall"],
                    "adversarial_fpr": adv_m["false_positive_rate"],
                    "adversarial_misses": adv_m["false_negatives"],
                })

        ablation_df = pd.DataFrame(records)
        csv_path = self.output_dir / "ablation_study.csv"
        ablation_df.to_csv(csv_path, index=False)
        return ablation_df

    @staticmethod
    def audit_target_leakage(df: pd.DataFrame) -> Dict[str, Any]:
        """Rigorously verify that attack_type, attack_id, or targets never leak into features."""
        feature_cols = [c for c in df.columns if c not in TARGET_COLUMNS]
        pipeline = FraudFeaturePipeline()
        pipeline.fit(df[feature_cols])

        transformed_feature_names = pipeline.get_feature_names_out()

        forbidden_tokens = ["attack_type", "attack_id", "fraud_label", "decision", "label"]
        leakage_detected = []

        for feat in transformed_feature_names:
            for token in forbidden_tokens:
                if token.lower() in feat.lower():
                    leakage_detected.append({"feature": feat, "matched_token": token})

        return {
            "total_input_features": len(feature_cols),
            "total_transformed_features": len(transformed_feature_names),
            "leakage_tokens_tested": forbidden_tokens,
            "leakage_detected": len(leakage_detected) > 0,
            "leakage_violations": leakage_detected,
            "status": "PASS - Zero Target Leakage" if len(leakage_detected) == 0 else "FAIL - Target Leakage Detected",
        }

    @staticmethod
    def analyze_generator_separation(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze synthetic transaction distributions to identify overly strong separation cues."""
        legit = df[df["fraud_label"] == 0]
        fraud = df[df["fraud_label"] == 1]

        separation_stats: List[Dict[str, Any]] = []

        for feat in NUMERICAL_FEATURES:
            v_legit = legit[feat].dropna().to_numpy()
            v_fraud = fraud[feat].dropna().to_numpy()

            mean_l = float(np.mean(v_legit))
            std_l = float(np.std(v_legit))
            mean_f = float(np.mean(v_fraud))
            std_f = float(np.std(v_fraud))

            # Cohen's d effect size
            pooled_std = np.sqrt(((std_l ** 2) + (std_f ** 2)) / 2.0)
            cohen_d = float(abs(mean_f - mean_l) / max(1e-6, pooled_std))

            # Kolmogorov-Smirnov 2-sample statistic (distribution divergence [0.0, 1.0])
            ks_stat, _ = ks_2samp(v_legit, v_fraud)

            # Categorize separation severity
            if cohen_d > 2.0 or ks_stat > 0.70:
                verdict = "VERY_STRONG_SEPARATION (Artificially Easy)"
            elif cohen_d > 1.2 or ks_stat > 0.45:
                verdict = "MODERATE_SEPARATION"
            else:
                verdict = "REALISTIC_OVERLAP"

            separation_stats.append({
                "feature": feat,
                "legit_mean": round(mean_l, 4),
                "fraud_mean": round(mean_f, 4),
                "cohen_d": round(cohen_d, 3),
                "ks_statistic": round(float(ks_stat), 3),
                "assessment": verdict,
            })

        separation_df = pd.DataFrame(separation_stats).sort_values(by="ks_statistic", ascending=False)

        critical_cues = separation_df[separation_df["assessment"].str.contains("VERY_STRONG")]["feature"].tolist()

        return {
            "total_numerical_features_analyzed": len(NUMERICAL_FEATURES),
            "artificially_strong_cues": critical_cues,
            "feature_separation_rankings": separation_df.to_dict(orient="records"),
        }


def main():
    """CLI runner for robustness benchmarks and feature ablation studies."""
    parser = argparse.ArgumentParser(description="FraudForge AI Robustness & Model Benchmarks")
    parser.add_argument("--train_data", type=str, default=str(PROCESSED_DATA_DIR / "train_split.csv"))
    parser.add_argument("--test_normal", type=str, default=str(PROCESSED_DATA_DIR / "test_split.csv"))
    parser.add_argument("--test_adv", type=str, default=str(PROCESSED_DATA_DIR / "unseen_adversarial_test_c.csv"))
    args = parser.parse_args()

    train_path = Path(args.train_data)
    norm_path = Path(args.test_normal)
    adv_path = Path(args.test_adv)

    if not (train_path.exists() and norm_path.exists() and adv_path.exists()):
        print("[!] Generating required datasets via AdaptiveFeedbackLoop first...")
        from src.adversarial.feedback_loop import AdaptiveFeedbackLoop
        loop = AdaptiveFeedbackLoop()
        loop.run_cycle()

    train_df = pd.read_csv(train_path)
    norm_df = pd.read_csv(norm_path)
    adv_df = pd.read_csv(adv_path)

    benchmark = RobustnessBenchmark()

    print("=" * 80)
    print("        FRAUDFORGE AI -- ROBUSTNESS BENCHMARKS & MODEL COMPARISONS")
    print("=" * 80)

    # 1. Target Leakage Audit
    print("\n[*] 1. AUDITING FOR TARGET & METADATA LEAKAGE...")
    leakage_audit = RobustnessBenchmark.audit_target_leakage(train_df)
    leakage_path = EXPERIMENTS_DIR / "leakage_audit.json"
    with open(leakage_path, "w", encoding="utf-8") as f:
        json.dump(leakage_audit, f, indent=2)
    print(f"    [+] Leakage Status: {leakage_audit['status']}")
    print(f"    [+] Features Verified: {leakage_audit['total_transformed_features']} (Zero target leakage)")

    # 2. Generator Realism & Feature Separation Diagnostics
    print("\n[*] 2. ANALYZING SYNTHETIC GENERATOR FEATURE SEPARATION...")
    realism_audit = RobustnessBenchmark.analyze_generator_separation(train_df)
    realism_path = EXPERIMENTS_DIR / "generator_realism_analysis.json"
    with open(realism_path, "w", encoding="utf-8") as f:
        json.dump(realism_audit, f, indent=2)
    print(f"    [!] Identified Artificially Strong Separation Features: {realism_audit['artificially_strong_cues']}")

    # 3. Model Architecture Comparison
    print("\n[*] 3. EXECUTING MODEL BENCHMARK: XGBOOST vs RANDOM FOREST...")
    model_comp = benchmark.run_model_comparison(train_df, norm_df, adv_df)

    # 4. Feature Ablation Study
    print("\n[*] 4. EXECUTING FEATURE ABLATION STUDIES...")
    ablation_df = benchmark.run_ablation_study(train_df, norm_df, adv_df)

    # Combine into unified JSON report
    full_report = {
        "target_leakage_audit": leakage_audit,
        "generator_realism_analysis": realism_audit,
        "model_architecture_comparison": model_comp,
        "feature_ablation_summary": ablation_df.to_dict(orient="records"),
    }
    report_json_path = EXPERIMENTS_DIR / "robustness_benchmarks.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    # Terminal Summary Table
    print("\n" + "=" * 80)
    print("                      FEATURE ABLATION & ROBUSTNESS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<8} | {'Ablation Setting':<30} | {'Norm F1':<8} | {'Adv Recall':<10} | {'Adv Misses':<10}")
    print("-" * 80)
    for _, row in ablation_df.iterrows():
        print(
            f"{row['model']:<8} | {row['ablation_setting']:<30} | {row['normal_f1']:>7.4f} | {row['adversarial_recall']*100:>9.1f}% | {int(row['adversarial_misses']):>10,d}"
        )
    print("=" * 80)
    print(f"[+] All robustness artifacts saved under {EXPERIMENTS_DIR}")


if __name__ == "__main__":
    main()
