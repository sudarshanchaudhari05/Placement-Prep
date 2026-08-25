"""FraudForge AI: Closed-Loop Experiment Orchestrator.

A clean, single entry point to execute the complete adaptive cycle:
RED TEAM -> BLUE TEAM -> ADAPT -> HARDEN -> TEST

Usage:
    python run_experiment.py
    python run_experiment.py --samples 10000 --fraud-ratio 0.15 --mutation-intensity 0.65
"""

import argparse
import datetime
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.adversarial.feedback_loop import AdaptiveFeedbackLoop
from src.utils.config import EXPERIMENTS_DIR


# ============================================
# FRAUDFORGE AI
# Red Team -> Blue Team -> Adaptive Defense
# ============================================

# 1. Configuration & Default Settings
DEFAULT_SAMPLES = 10000
DEFAULT_FRAUD_RATIO = 0.15
DEFAULT_MUTATION_INTENSITY = 0.65


# 2. Run the experiment
def run_experiment(
    samples: int = DEFAULT_SAMPLES,
    fraud_ratio: float = DEFAULT_FRAUD_RATIO,
    mutation_intensity: float = DEFAULT_MUTATION_INTENSITY,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute the full 6-stage adaptive feedback loop and return results dictionary."""
    print("=" * 60)
    print("                 FRAUDFORGE AI")
    print("        RED TEAM -> BLUE TEAM -> ADAPTIVE DEFENSE")
    print("=" * 60)
    print()

    def stage_progress_callback(stage_num: int, stage_desc: str) -> None:
        print(f"[{stage_num}/6] {stage_desc}")
        print("      [OK] Complete\n")

    loop = AdaptiveFeedbackLoop(
        baseline_seed=42,
        adversarial_train_seed=101,
        unseen_test_seed=1337,
        output_dir=output_dir or EXPERIMENTS_DIR,
    )

    results = loop.run_cycle(
        n_samples=samples,
        fraud_ratio=fraud_ratio,
        mutation_intensity=mutation_intensity,
        verbose=False,
        step_callback=stage_progress_callback,
    )

    return results


# 3. Print the results
def print_results(results: Dict[str, Any]) -> None:
    """Format and print a concise, human-readable terminal summary of the experiment."""
    norm_base = results["normal_test_set_performance"]["baseline_detector"]
    norm_hard = results["normal_test_set_performance"]["hardened_detector"]

    adv_base = results["unseen_adversarial_test_performance"]["baseline_detector"]
    adv_hard = results["unseen_adversarial_test_performance"]["hardened_detector"]

    base_misses = adv_base["false_negatives"]
    hard_misses = adv_hard["false_negatives"]
    miss_reduction = base_misses - hard_misses
    fn_reduction_pct = (miss_reduction / max(1, base_misses)) * 100.0
    rec_gain_pts = (adv_hard["recall"] - adv_base["recall"]) * 100.0

    print("=" * 60)
    print("                    FINAL RESULTS")
    print("=" * 60)
    print("\nNORMAL TRAFFIC")
    print("-" * 60)
    print(f"{'Metric':<25} {'Baseline':<15} {'Hardened':<15}")
    base_f1_str = f"{norm_base['f1_score']:.4f}"
    hard_f1_str = f"{norm_hard['f1_score']:.4f}"
    base_rec_str = f"{norm_base['recall']*100:.2f}%"
    hard_rec_str = f"{norm_hard['recall']*100:.2f}%"
    base_fpr_str = f"{norm_base['false_positive_rate']*100:.2f}%"
    hard_fpr_str = f"{norm_hard['false_positive_rate']*100:.2f}%"
    print(f"{'F1 Score':<25} {base_f1_str:<15} {hard_f1_str:<15}")
    print(f"{'Recall':<25} {base_rec_str:<15} {hard_rec_str:<15}")
    print(f"{'FPR':<25} {base_fpr_str:<15} {hard_fpr_str:<15}")

    print("\nUNSEEN ADVERSARIAL ATTACKS")
    print("-" * 60)
    print(f"{'Metric':<25} {'Baseline':<15} {'Hardened':<15}")
    adv_base_rec_str = f"{adv_base['recall']*100:.2f}%"
    adv_hard_rec_str = f"{adv_hard['recall']*100:.2f}%"
    adv_base_f1_str = f"{adv_base['f1_score']:.4f}"
    adv_hard_f1_str = f"{adv_hard['f1_score']:.4f}"
    print(f"{'Recall':<25} {adv_base_rec_str:<15} {adv_hard_rec_str:<15}")
    print(f"{'F1 Score':<25} {adv_base_f1_str:<15} {adv_hard_f1_str:<15}")
    print(f"{'Missed Attacks':<25} {base_misses:<15d} {hard_misses:<15d}")

    print("\nIMPROVEMENT")
    print("-" * 60)
    print(f"{'Adversarial Recall':<25} +{rec_gain_pts:.2f} percentage points")
    print(f"{'Missed Attacks':<25} -{miss_reduction:d}")
    print(f"{'False Negative Reduction':<25} {fn_reduction_pct:.1f}%")

    # Top vulnerable attacks
    attack_comp = results.get("attack_comparison", [])
    if attack_comp:
        fraud_attacks = [a for a in attack_comp if a.get("is_fraud", True)]
        # Sort by baseline detection rate ascending (weakest first)
        fraud_attacks = sorted(fraud_attacks, key=lambda x: x.get("baseline_detection_rate", 0.0))
        top_weak = fraud_attacks[:5]

        print("\nTOP RED-TEAM WEAKNESSES")
        print("-" * 60)
        print(f"{'Attack Archetype':<40} {'Baseline -> Hardened'}")
        for atk in top_weak:
            name = atk["attack_type"]
            if len(name) > 38:
                name = name[:35] + "..."
            b_rate = atk.get("baseline_detection_rate", 0.0)
            h_rate = atk.get("hardened_detection_rate", 0.0)
            print(f"{name:<40} {b_rate:>5.1f}% -> {h_rate:>5.1f}%")

    print("\n" + "=" * 60)
    print("             RED TEAM -> BLUE TEAM LOOP COMPLETE")
    print("=" * 60)


# 4. Save human-readable summary
def save_summary_file(results: Dict[str, Any], output_path: Path) -> None:
    """Export human-readable text summary of the experiment run."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    norm_base = results["normal_test_set_performance"]["baseline_detector"]
    norm_hard = results["normal_test_set_performance"]["hardened_detector"]
    adv_base = results["unseen_adversarial_test_performance"]["baseline_detector"]
    adv_hard = results["unseen_adversarial_test_performance"]["hardened_detector"]

    base_misses = adv_base["false_negatives"]
    hard_misses = adv_hard["false_negatives"]
    miss_reduction = base_misses - hard_misses
    fn_reduction_pct = (miss_reduction / max(1, base_misses)) * 100.0
    rec_gain_pts = (adv_hard["recall"] - adv_base["recall"]) * 100.0

    counts = results.get("sample_counts", {})
    seeds = results.get("seeds", {})
    deps = results.get("detector_feature_dependencies", [])

    lines = [
        "=" * 70,
        "FRAUDFORGE AI -- LATEST EXPERIMENT RUN SUMMARY",
        "=" * 70,
        f"Timestamp: {now}",
        "",
        "CONFIGURATION:",
        f"  - Dataset A Samples: {counts.get('dataset_a_total', 'N/A'):,}",
        f"  - Train / Test Split: {counts.get('dataset_a_train', 'N/A'):,} train / {counts.get('dataset_a_test', 'N/A'):,} test",
        f"  - Dataset B (Adversarial Train): {counts.get('dataset_b_adversarial_train', 'N/A'):,} samples",
        f"  - Dataset C (Unseen Adversarial Test): {counts.get('dataset_c_unseen_test', 'N/A'):,} samples",
        f"  - Random Seeds: Baseline={seeds.get('baseline_dataset_a', 'N/A')}, Adversarial Train={seeds.get('adversarial_train_dataset_b', 'N/A')}, Unseen Test={seeds.get('unseen_test_dataset_c', 'N/A')}",
        "",
        "NORMAL HELD-OUT TEST PERFORMANCE:",
        f"  - Baseline Detector : Accuracy: {norm_base['accuracy']*100:.2f}%, Precision: {norm_base['precision']*100:.2f}%, Recall: {norm_base['recall']*100:.2f}%, F1: {norm_base['f1_score']:.4f}, FPR: {norm_base['false_positive_rate']*100:.2f}%",
        f"  - Hardened Detector : Accuracy: {norm_hard['accuracy']*100:.2f}%, Precision: {norm_hard['precision']*100:.2f}%, Recall: {norm_hard['recall']*100:.2f}%, F1: {norm_hard['f1_score']:.4f}, FPR: {norm_hard['false_positive_rate']*100:.2f}%",
        "",
        "UNSEEN ADVERSARIAL TEST PERFORMANCE (DATASET C):",
        f"  - Baseline Detector : Recall: {adv_base['recall']*100:.2f}%, F1: {adv_base['f1_score']:.4f}, Missed: {base_misses} attacks",
        f"  - Hardened Detector : Recall: {adv_hard['recall']*100:.2f}%, F1: {adv_hard['f1_score']:.4f}, Missed: {hard_misses} attacks",
        "",
        "DEFENSE GAINS & HARDENING IMPACT:",
        f"  - Adversarial Recall Improvement : +{rec_gain_pts:.2f} percentage points",
        f"  - Missed Attacks Reduction       : -{miss_reduction:d} fewer false negatives",
        f"  - False Negative Reduction Rate  : {fn_reduction_pct:.1f}%",
        "",
        "TOP IDENTIFIED WEAKNESSES (DATASET C):",
    ]

    attack_comp = results.get("attack_comparison", [])
    if attack_comp:
        fraud_attacks = [a for a in attack_comp if a.get("is_fraud", True)]
        fraud_attacks = sorted(fraud_attacks, key=lambda x: x.get("baseline_detection_rate", 0.0))[:8]
        for atk in fraud_attacks:
            lines.append(f"  - {atk['attack_type']}: {atk.get('baseline_detection_rate', 0.0):.1f}% -> {atk.get('hardened_detection_rate', 0.0):.1f}% (Delta: {atk.get('detection_rate_delta', 0.0):+.1f}%)")

    lines.extend([
        "",
        "TOP RELIED-UPON DETECTOR FEATURE DEPENDENCIES:",
        f"  - {', '.join(deps) if deps else 'None'}",
        "=" * 70,
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# 5. Main entry point
def main() -> None:
    """Parse CLI arguments and execute the experiment."""
    parser = argparse.ArgumentParser(
        description="FraudForge AI: Closed-Loop Red Team / Blue Team Experiment Runner"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"Total synthetic samples for Dataset A (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--fraud-ratio",
        type=float,
        default=DEFAULT_FRAUD_RATIO,
        help=f"Fraction of transactions that are fraudulent (default: {DEFAULT_FRAUD_RATIO})",
    )
    parser.add_argument(
        "--mutation-intensity",
        type=float,
        default=DEFAULT_MUTATION_INTENSITY,
        help=f"Adversarial mutation intensity [0.0 - 1.0] (default: {DEFAULT_MUTATION_INTENSITY})",
    )

    args = parser.parse_args()

    try:
        results = run_experiment(
            samples=args.samples,
            fraud_ratio=args.fraud_ratio,
            mutation_intensity=args.mutation_intensity,
        )
        print_results(results)
        summary_path = EXPERIMENTS_DIR / "latest_run_summary.txt"
        save_summary_file(results, summary_path)
        print(f"\n[+] Human-readable summary saved to: {summary_path}")

    except Exception as exc:
        print("\n" + "=" * 60)
        print("EXPERIMENT FAILED")
        print("=" * 60)
        print("\nStage:")
        print("Adaptive Feedback Loop Execution")
        print(f"\nError:\n{exc}")
        print("\nCheck the traceback below for technical details.\n")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
