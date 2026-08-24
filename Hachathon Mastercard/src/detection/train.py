"""FraudForge AI: Baseline ML Detector Trainer.

Trains an XGBoost (or Random Forest fallback) fraud detection model
on synthetic payment transaction data using stratified train/test splits.
"""

import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.features.feature_engineering import FraudFeaturePipeline, extract_features_and_targets
from src.simulation.transaction_generator import TransactionGenerator
from src.utils.config import DEFAULT_SEED, MODELS_DIR, GENERATED_DATA_DIR, PROCESSED_DATA_DIR


def train_baseline_detector(
    df: pd.DataFrame,
    test_size: float = 0.20,
    seed: int = DEFAULT_SEED,
    model_type: str = "xgboost",
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Train baseline fraud classifier and return fitted artifacts and train/test splits.

    Returns:
        artifact: Dict containing 'pipeline', 'model', 'model_type', 'feature_names'
        train_df: Full train dataframe (with targets and attack_type)
        test_df: Full test dataframe (with targets and attack_type)
    """
    # Stratified Train/Test split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["fraud_label"],
        random_state=seed,
    )

    X_train, y_train, _ = extract_features_and_targets(train_df)
    X_test, y_test, _ = extract_features_and_targets(test_df)

    # Fit Feature Pipeline
    pipeline = FraudFeaturePipeline()
    X_train_transformed = pipeline.fit_transform(X_train, y_train)

    # Class imbalance ratio for scale_pos_weight
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = float(n_neg / max(1, n_pos))

    # Model Selection & Training
    if model_type.lower() == "xgboost" and XGBOOST_AVAILABLE:
        model = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            eval_metric="logloss",
            n_jobs=-1,
        )
        selected_model_type = "xgboost"
    else:
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        selected_model_type = "random_forest"

    model.fit(X_train_transformed, y_train)

    artifact = {
        "pipeline": pipeline,
        "model": model,
        "model_type": selected_model_type,
        "feature_names": pipeline.get_feature_names_out(),
        "seed": seed,
    }

    return artifact, train_df, test_df


def save_detector(artifact: Dict[str, Any], filepath: Path) -> None:
    """Save trained detector artifact to disk."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, filepath)


def main():
    """CLI runner to generate data (or load existing), train detector, and persist model."""
    parser = argparse.ArgumentParser(description="Train FraudForge AI Baseline Detector")
    parser.add_argument("--data", type=str, default="", help="Path to input CSV dataset. If empty, synthetic data will be generated.")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of samples if generating data")
    parser.add_argument("--fraud_ratio", type=float, default=0.15, help="Fraud ratio if generating data")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic random seed")
    parser.add_argument("--model_type", type=str, default="xgboost", choices=["xgboost", "random_forest"])
    parser.add_argument("--output_model", type=str, default=str(MODELS_DIR / "baseline_detector.joblib"))
    args = parser.parse_args()

    print("=" * 70)
    print("FraudForge AI — Baseline ML Detector Training")
    print("=" * 70)

    if args.data and Path(args.data).exists():
        print(f"[*] Loading dataset from: {args.data}")
        df = pd.read_csv(args.data)
    else:
        print(f"[*] Generating {args.n_samples:,} synthetic transactions (fraud_ratio={args.fraud_ratio:.2%}, seed={args.seed})...")
        generator = TransactionGenerator(seed=args.seed)
        df = generator.generate_dataset(n_samples=args.n_samples, fraud_ratio=args.fraud_ratio)
        gen_path = GENERATED_DATA_DIR / "synthetic_transactions_v1.csv"
        df.to_csv(gen_path, index=False)
        print(f"[+] Saved generated dataset to {gen_path}")

    print(f"[*] Training baseline {args.model_type.upper()} detector...")
    artifact, train_df, test_df = train_baseline_detector(
        df=df,
        test_size=0.20,
        seed=args.seed,
        model_type=args.model_type,
    )

    out_path = Path(args.output_model)
    save_detector(artifact, out_path)

    # Save train and test splits to processed directory
    train_split_path = PROCESSED_DATA_DIR / "train_split.csv"
    test_split_path = PROCESSED_DATA_DIR / "test_split.csv"
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_split_path, index=False)
    test_df.to_csv(test_split_path, index=False)

    print(f"[+] Model artifact successfully saved to {out_path}")
    print(f"    - Model type: {artifact['model_type']}")
    print(f"    - Total features: {len(artifact['feature_names'])}")
    print(f"    - Training samples: {len(train_df):,} (saved to {train_split_path})")
    print(f"    - Test samples: {len(test_df):,} (saved to {test_split_path})")
    print("=" * 70)


if __name__ == "__main__":
    main()
