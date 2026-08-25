"""FraudForge AI: Adaptive Red-Team / Blue-Team Closed-Loop Engine.

Orchestrates the closed-loop adversarial feedback cycle:
1. Train Baseline Detector on Dataset A
2. Evaluate and extract False Negatives & Vulnerable Attack Vectors
3. Analyze Detector Feature Dependencies
4. Mutate vulnerable attacks via AttackMutator (Dataset B)
5. Retrain Hardened Detector on Augmented Training Data
6. Evaluate both models on Unseen Adversarial Test Set (Dataset C)
7. Export comprehensive comparison reports and metrics
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable
import numpy as np
import pandas as pd

from src.attacks.attack_library import AttackLibrary, get_default_attack_library
from src.attacks.attack_mutator import AttackMutator
from src.simulation.transaction_generator import TransactionGenerator
from src.features.feature_engineering import FraudFeaturePipeline, extract_features_and_targets
from src.detection.train import train_baseline_detector, save_detector
from src.detection.predict import FraudDetector
from src.detection.evaluate import (
    evaluate_global_metrics,
    evaluate_attack_specific_metrics,
    get_weakest_attacks,
)
from src.utils.config import (
    DEFAULT_SEED,
    MODELS_DIR,
    GENERATED_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXPERIMENTS_DIR,
)


class AdaptiveFeedbackLoop:
    """Closed-loop Red Team / Blue Team training and hardening orchestrator."""

    def __init__(
        self,
        baseline_seed: int = 42,
        adversarial_train_seed: int = 101,
        unseen_test_seed: int = 1337,
        output_dir: Optional[Path] = None,
    ):
        self.baseline_seed = baseline_seed
        self.adversarial_train_seed = adversarial_train_seed
        self.unseen_test_seed = unseen_test_seed
        self.output_dir = output_dir or EXPERIMENTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.attack_library = get_default_attack_library()

    def run_cycle(
        self,
        n_samples: int = 10000,
        fraud_ratio: float = 0.15,
        mutation_intensity: float = 0.65,
        verbose: bool = True,
        step_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Execute full closed-loop experiment across Datasets A, B, and C."""

        if verbose:
            print("=" * 80)
            print("    FRAUDFORGE AI -- CLOSED-LOOP ADAPTIVE RED TEAM / BLUE TEAM EXPERIMENT")
            print("=" * 80)

        if step_callback:
            step_callback(1, "Generating synthetic payment environment...")
        elif verbose:
            print("\n[*] STEP 1: Generating Dataset A (Baseline Training & Held-Out Test Set)...")
        gen_a = TransactionGenerator(seed=self.baseline_seed)
        df_a = gen_a.generate_dataset(n_samples=n_samples, fraud_ratio=fraud_ratio)

        if step_callback:
            step_callback(2, "Training baseline fraud detector...")
        elif verbose:
            print("\n[*] STEP 2: Training Baseline Fraud Detector...")

        artifact_baseline, train_df_a, test_df_a = train_baseline_detector(
            df=df_a,
            test_size=0.20,
            seed=self.baseline_seed,
            model_type="xgboost",
        )
        baseline_model_path = MODELS_DIR / "baseline_detector.joblib"
        save_detector(artifact_baseline, baseline_model_path)
        detector_baseline = FraudDetector(artifact=artifact_baseline)

        # Save Dataset A and splits
        gen_a_path = GENERATED_DATA_DIR / "synthetic_transactions_v1.csv"
        train_a_path = PROCESSED_DATA_DIR / "train_split.csv"
        test_a_path = PROCESSED_DATA_DIR / "test_split.csv"
        GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df_a.to_csv(gen_a_path, index=False)
        train_df_a.to_csv(train_a_path, index=False)
        test_df_a.to_csv(test_a_path, index=False)

        if verbose:
            print(f"    [+] Baseline model trained on {len(train_df_a):,} samples.")
            print(f"    [+] Saved baseline detector to {baseline_model_path}")

        # ---------------------------------------------------------------------
        # STEP 3: Evaluate Baseline Detector & Identify Vulnerabilities
        # ---------------------------------------------------------------------
        if step_callback:
            step_callback(3, "Identifying detector weaknesses...")
        elif verbose:
            print("\n[*] STEP 3: Evaluating Baseline Detector on Held-Out Test Set A...")
        y_true_a = test_df_a["fraud_label"].to_numpy()
        y_pred_a_base = detector_baseline.predict(test_df_a)
        y_prob_a_base = detector_baseline.predict_proba(test_df_a)

        baseline_normal_metrics = evaluate_global_metrics(y_true_a, y_pred_a_base, y_prob_a_base)
        baseline_attack_eval = evaluate_attack_specific_metrics(test_df_a, detector_baseline)
        weakest_attacks = get_weakest_attacks(baseline_attack_eval, bottom_n=5)

        if verbose:
            print(f"    [+] Baseline Normal Test F1: {baseline_normal_metrics['f1_score']:.4f} | Recall: {baseline_normal_metrics['recall']:.4f}")
            print(f"    [!] Identified Weakest Attack Archetypes:")
            for _, row in weakest_attacks.iterrows():
                print(f"        • {row['attack_type']}: {row['detection_rate']}% detection ({int(row['missed'])} missed)")

        feat_imp_df = detector_baseline.get_feature_importances(top_n=15)
        top_weakness_features = feat_imp_df.head(6)["feature"].tolist()
        clean_features = []
        for f in top_weakness_features:
            for base_feat in ["device_change", "merchant_risk_score", "IP_risk_score", "transaction_velocity_24h", "transaction_velocity_1h", "behavioral_deviation", "amount_deviation"]:
                if base_feat in f and base_feat not in clean_features:
                    clean_features.append(base_feat)
        if not clean_features:
            clean_features = ["device_change", "merchant_risk_score", "IP_risk_score", "behavioral_deviation"]

        feat_imp_path = self.output_dir / "feature_importance.csv"
        feat_imp_df.to_csv(feat_imp_path, index=False)
        if verbose:
            print(f"    [+] Top Relied-Upon Features by Detector: {clean_features}")
            print(f"    [+] Exported feature importances to {feat_imp_path}")

        # ---------------------------------------------------------------------
        # STEP 4: Generate Dataset B (Adversarial Training Set via AttackMutator)
        # ---------------------------------------------------------------------
        if step_callback:
            step_callback(4, "Generating adaptive adversarial attacks...")
        elif verbose:
            print("\n[*] STEP 4: Generating Dataset B (Adversarial Training Set)...")
        gen_b = TransactionGenerator(seed=self.adversarial_train_seed)
        df_b_raw = gen_b.generate_dataset(n_samples=2500, fraud_ratio=0.50)
        df_b_fraud = df_b_raw[df_b_raw["fraud_label"] == 1].copy()
        df_b_legit = df_b_raw[df_b_raw["fraud_label"] == 0].copy()

        mutator_train = AttackMutator(seed=self.adversarial_train_seed)
        mutated_fraud_train, audit_log_train = mutator_train.mutate_dataframe(
            df_fraud=df_b_fraud,
            attack_library=self.attack_library,
            detector_weaknesses=clean_features,
            mutation_intensity=mutation_intensity,
        )

        dataset_b = pd.concat([mutated_fraud_train, df_b_legit], ignore_index=True)
        dataset_b_path = GENERATED_DATA_DIR / "adversarial_train_dataset_b.csv"
        dataset_b.to_csv(dataset_b_path, index=False)
        if verbose:
            print(f"    [+] Generated Dataset B: {len(dataset_b):,} samples ({len(mutated_fraud_train):,} mutated attacks).")

        # ---------------------------------------------------------------------
        # STEP 5: Train Hardened Detector on Augmented Training Data (A + B)
        # ---------------------------------------------------------------------
        if step_callback:
            step_callback(5, "Training hardened detector...")
        elif verbose:
            print("\n[*] STEP 5: Retraining Hardened Blue-Team Detector on Augmented Data...")
        augmented_train_df = pd.concat([train_df_a, dataset_b], ignore_index=True).sample(
            frac=1.0, random_state=self.adversarial_train_seed
        ).reset_index(drop=True)

        X_aug, y_aug, _ = extract_features_and_targets(augmented_train_df)
        hardened_pipeline = FraudFeaturePipeline()
        X_aug_trans = hardened_pipeline.fit_transform(X_aug, y_aug)

        n_neg = int((y_aug == 0).sum())
        n_pos = int((y_aug == 1).sum())
        scale_pos_weight = float(n_neg / max(1, n_pos))

        from xgboost import XGBClassifier
        hardened_model = XGBClassifier(
            n_estimators=180,
            max_depth=6,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            random_state=self.adversarial_train_seed,
            eval_metric="logloss",
            n_jobs=-1,
        )
        hardened_model.fit(X_aug_trans, y_aug)

        artifact_hardened = {
            "pipeline": hardened_pipeline,
            "model": hardened_model,
            "model_type": "xgboost_hardened",
            "feature_names": hardened_pipeline.get_feature_names_out(),
            "seed": self.adversarial_train_seed,
        }
        hardened_model_path = MODELS_DIR / "hardened_detector.joblib"
        save_detector(artifact_hardened, hardened_model_path)
        detector_hardened = FraudDetector(artifact=artifact_hardened)
        if verbose:
            print(f"    [+] Hardened Detector successfully trained on {len(augmented_train_df):,} total samples.")
            print(f"    [+] Saved hardened model to {hardened_model_path}")

        # ---------------------------------------------------------------------
        # STEP 6: Generate Dataset C (Unseen Adversarial Test Set) & Test
        # ---------------------------------------------------------------------
        if step_callback:
            step_callback(6, "Testing on unseen adversarial attacks...")
        elif verbose:
            print("\n[*] STEP 6: Generating Dataset C (Unseen Adversarial Test Set - seed=1337)...")
        gen_c = TransactionGenerator(seed=self.unseen_test_seed)
        df_c_raw = gen_c.generate_dataset(n_samples=2000, fraud_ratio=0.15)
        df_c_fraud = df_c_raw[df_c_raw["fraud_label"] == 1].copy()
        df_c_legit = df_c_raw[df_c_raw["fraud_label"] == 0].copy()

        mutator_test = AttackMutator(seed=self.unseen_test_seed)
        mutated_fraud_test, audit_log_test = mutator_test.mutate_dataframe(
            df_fraud=df_c_fraud,
            attack_library=self.attack_library,
            detector_weaknesses=clean_features,
            mutation_intensity=min(1.0, mutation_intensity + 0.10),  # Slightly harder test mutations
        )

        dataset_c = pd.concat([mutated_fraud_test, df_c_legit], ignore_index=True).sample(
            frac=1.0, random_state=self.unseen_test_seed
        ).reset_index(drop=True)
        dataset_c_path = PROCESSED_DATA_DIR / "unseen_adversarial_test_c.csv"
        dataset_c.to_csv(dataset_c_path, index=False)
        if verbose:
            print(f"    [+] Saved Dataset C to {dataset_c_path} ({len(dataset_c):,} samples: {len(df_c_legit):,} legit, {len(mutated_fraud_test):,} unseen mutated attacks).")

        # ---------------------------------------------------------------------
        # STEP 7: Side-by-Side Comparative Evaluation
        # ---------------------------------------------------------------------
        if verbose:
            print("\n[*] STEP 7: Performing Side-by-Side Defense Evaluation...")

        # 1. Evaluate on Normal Test Set A
        y_pred_a_hard = detector_hardened.predict(test_df_a)
        y_prob_a_hard = detector_hardened.predict_proba(test_df_a)
        hardened_normal_metrics = evaluate_global_metrics(y_true_a, y_pred_a_hard, y_prob_a_hard)

        # 2. Evaluate on Unseen Adversarial Test Set C
        y_true_c = dataset_c["fraud_label"].to_numpy()

        y_pred_c_base = detector_baseline.predict(dataset_c)
        y_prob_c_base = detector_baseline.predict_proba(dataset_c)
        baseline_adv_metrics = evaluate_global_metrics(y_true_c, y_pred_c_base, y_prob_c_base)

        y_pred_c_hard = detector_hardened.predict(dataset_c)
        y_prob_c_hard = detector_hardened.predict_proba(dataset_c)
        hardened_adv_metrics = evaluate_global_metrics(y_true_c, y_pred_c_hard, y_prob_c_hard)

        # Attack-specific evaluation on Dataset C
        attack_eval_c_base = evaluate_attack_specific_metrics(dataset_c, detector_baseline)
        attack_eval_c_hard = evaluate_attack_specific_metrics(dataset_c, detector_hardened)

        # Build attack comparison DataFrame
        comparison_records = []
        base_dict = {row["attack_type"]: row for _, row in attack_eval_c_base.iterrows()}
        hard_dict = {row["attack_type"]: row for _, row in attack_eval_c_hard.iterrows()}

        for attack_name, base_row in base_dict.items():
            hard_row = hard_dict.get(attack_name, {})
            is_fraud = base_row.get("is_fraud", True)
            b_det = base_row.get("detection_rate", 0.0)
            h_det = hard_row.get("detection_rate", 0.0)
            improvement = float(np.round(h_det - b_det, 2))

            comparison_records.append({
                "attack_type": attack_name,
                "is_fraud": is_fraud,
                "total_samples": base_row.get("total_transactions", 0),
                "baseline_caught": base_row.get("detected", 0),
                "baseline_missed": base_row.get("missed", 0),
                "baseline_detection_rate": b_det,
                "hardened_caught": hard_row.get("detected", 0),
                "hardened_missed": hard_row.get("missed", 0),
                "hardened_detection_rate": h_det,
                "detection_rate_delta": improvement,
            })

        comparison_df = pd.DataFrame(comparison_records)
        comparison_csv_path = self.output_dir / "adversarial_report.csv"
        comparison_df.to_csv(comparison_csv_path, index=False)
        if verbose:
            print(f"    [+] Saved attack comparison breakdown to {comparison_csv_path}")

        # ---------------------------------------------------------------------
        # STEP 8: Export Complete Experiment JSON
        # ---------------------------------------------------------------------
        experiment_summary = {
            "seeds": {
                "baseline_dataset_a": self.baseline_seed,
                "adversarial_train_dataset_b": self.adversarial_train_seed,
                "unseen_test_dataset_c": self.unseen_test_seed,
            },
            "sample_counts": {
                "dataset_a_total": len(df_a),
                "dataset_a_train": len(train_df_a),
                "dataset_a_test": len(test_df_a),
                "dataset_b_adversarial_train": len(dataset_b),
                "dataset_c_unseen_test": len(dataset_c),
            },
            "detector_feature_dependencies": clean_features,
            "normal_test_set_performance": {
                "baseline_detector": baseline_normal_metrics,
                "hardened_detector": hardened_normal_metrics,
            },
            "unseen_adversarial_test_performance": {
                "baseline_detector": baseline_adv_metrics,
                "hardened_detector": hardened_adv_metrics,
            },
            "adversarial_false_negatives": {
                "baseline_detector_misses": baseline_adv_metrics["false_negatives"],
                "hardened_detector_misses": hardened_adv_metrics["false_negatives"],
                "miss_reduction": baseline_adv_metrics["false_negatives"] - hardened_adv_metrics["false_negatives"],
            },
            "attack_comparison": comparison_df.to_dict(orient="records"),
            "feature_importances": feat_imp_df.to_dict(orient="records"),
        }

        exp_json_path = self.output_dir / "baseline_vs_adversarial.json"
        with open(exp_json_path, "w", encoding="utf-8") as f:
            json.dump(experiment_summary, f, indent=2)
        if verbose:
            print(f"    [+] Saved full experiment summary to {exp_json_path}")

        # ---------------------------------------------------------------------
        # STEP 9: Print Comparative Report Table
        # ---------------------------------------------------------------------
        if verbose:
            self._print_terminal_report(
                baseline_normal_metrics=baseline_normal_metrics,
                hardened_normal_metrics=hardened_normal_metrics,
                baseline_adv_metrics=baseline_adv_metrics,
                hardened_adv_metrics=hardened_adv_metrics,
                comparison_df=comparison_df,
            )

        return experiment_summary

    def _print_terminal_report(
        self,
        baseline_normal_metrics: Dict[str, Any],
        hardened_normal_metrics: Dict[str, Any],
        baseline_adv_metrics: Dict[str, Any],
        hardened_adv_metrics: Dict[str, Any],
        comparison_df: pd.DataFrame,
    ) -> None:
        """Format and print full terminal before/after report."""
        print("\n" + "=" * 80)
        print("                 FRAUDFORGE AI -- BEFORE / AFTER DEFENSE BENCHMARK")
        print("=" * 80)
        print("\n> 1. PERFORMANCE ON UNSEEN ADVERSARIAL ATTACKS (DATASET C):")
        print(f"{'Metric':<28} | {'Baseline Detector':<20} | {'Hardened Detector':<20} | {'Delta':<10}")
        print("-" * 80)

        for m_key, m_name in [
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Adversarial Recall"),
            ("f1_score", "F1 Score"),
            ("roc_auc", "ROC-AUC"),
            ("false_positive_rate", "False Positive Rate"),
        ]:
            b_val = baseline_adv_metrics[m_key]
            h_val = hardened_adv_metrics[m_key]
            delta = h_val - b_val
            sign = "+" if delta >= 0 else ""
            if "rate" in m_key or "accuracy" in m_key or "precision" in m_key or "recall" in m_key:
                print(f"{m_name:<28} | {b_val*100:>18.2f}% | {h_val*100:>18.2f}% | {sign}{delta*100:>8.2f}%")
            else:
                print(f"{m_name:<28} | {b_val:>20.4f} | {h_val:>20.4f} | {sign}{delta:>9.4f}")

        print("-" * 80)
        print(f"{'False Negatives (Missed Attacks)':<28} | {baseline_adv_metrics['false_negatives']:>20,d} | {hardened_adv_metrics['false_negatives']:>20,d} | {hardened_adv_metrics['false_negatives'] - baseline_adv_metrics['false_negatives']:>10,d}")
        print(f"{'False Positives (False Alarms)':<28} | {baseline_adv_metrics['false_positives']:>20,d} | {hardened_adv_metrics['false_positives']:>20,d} | {hardened_adv_metrics['false_positives'] - baseline_adv_metrics['false_positives']:>10,d}")

        print("\n> 2. ATTACK-SPECIFIC DETECTION RATE IMPROVEMENTS (UNSEEN ATTACKS):")
        print(f"{'Attack Archetype':<46} | {'Baseline':<10} | {'Hardened':<10} | {'Improvement':<12}")
        print("-" * 80)

        fraud_comp = comparison_df[comparison_df["is_fraud"]].sort_values(by="baseline_detection_rate", ascending=True)
        for _, row in fraud_comp.head(10).iterrows():
            name = str(row["attack_type"])
            if len(name) > 44:
                name = name[:41] + "..."
            delta_str = f"+{row['detection_rate_delta']:.1f}%" if row['detection_rate_delta'] >= 0 else f"{row['detection_rate_delta']:.1f}%"
            print(f"{name:<46} | {row['baseline_detection_rate']:>9.1f}% | {row['hardened_detection_rate']:>9.1f}% | {delta_str:>11}")

        print("-" * 80)
        legit_row = comparison_df[~comparison_df["is_fraud"]].iloc[0]
        legit_delta = f"+{legit_row['detection_rate_delta']:.1f}%" if legit_row['detection_rate_delta'] >= 0 else f"{legit_row['detection_rate_delta']:.1f}%"
        print(f"{'LEGITIMATE TRANSACTIONS (Specificity)':<46} | {legit_row['baseline_detection_rate']:>9.1f}% | {legit_row['hardened_detection_rate']:>9.1f}% | {legit_delta:>11}")
        print("=" * 80)


def main():
    """CLI runner for full closed-loop feedback experiment."""
    parser = argparse.ArgumentParser(description="FraudForge AI Closed-Loop Feedback Experiment")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--fraud_ratio", type=float, default=0.15)
    parser.add_argument("--mutation_intensity", type=float, default=0.65)
    parser.add_argument("--baseline_seed", type=int, default=42)
    parser.add_argument("--adv_train_seed", type=int, default=101)
    parser.add_argument("--unseen_test_seed", type=int, default=1337)
    args = parser.parse_args()

    loop = AdaptiveFeedbackLoop(
        baseline_seed=args.baseline_seed,
        adversarial_train_seed=args.adv_train_seed,
        unseen_test_seed=args.unseen_test_seed,
    )
    loop.run_cycle(
        n_samples=args.n_samples,
        fraud_ratio=args.fraud_ratio,
        mutation_intensity=args.mutation_intensity,
    )


if __name__ == "__main__":
    main()
