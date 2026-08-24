"""FraudForge AI: Synthetic Payment Transaction Generator.

Generates realistic payment transaction datasets with correlated behavioral features,
statistical distributions, and simulated GenAI fraud attack archetypes.
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

from src.attacks.attack_library import AttackLibrary, AttackArchetype, get_default_attack_library
from src.simulation.distributions import (
    HOURLY_ACTIVITY_PROB,
    OFF_HOURS_PROB,
    CATEGORY_BASE_RISK,
    CATEGORY_AMOUNT_PARAMS,
    LEGITIMATE_CATEGORY_WEIGHTS,
    FRAUD_CATEGORY_WEIGHTS,
    LEGITIMATE_CHANNEL_WEIGHTS,
    LEGITIMATE_AUTH_WEIGHTS,
    COUNTRY_WEIGHTS,
    sample_categorical,
    clip_score,
)
from src.utils.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMNS,
    ALL_COLUMNS,
    DEFAULT_SEED,
    GENERATED_DATA_DIR,
)


class TransactionGenerator:
    """Realistic Synthetic Transaction Generator with GenAI attack injection."""

    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        attack_library: Optional[AttackLibrary] = None,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.attack_library = attack_library or get_default_attack_library()

    def _sample_customer_profile(self) -> Dict[str, Any]:
        """Generate a baseline customer profile representing historical habits."""
        cust_country = sample_categorical(COUNTRY_WEIGHTS, self.rng)
        account_age_days = int(np.clip(self.rng.exponential(scale=350) + 14, 1, 1800))
        device_age_days = int(np.clip(self.rng.uniform(1, min(account_age_days, 730)), 1, account_age_days))
        avg_amount = float(np.round(np.exp(self.rng.normal(loc=4.1, scale=0.65)), 2))
        avg_amount = max(5.0, avg_amount)

        return {
            "customer_country": cust_country,
            "account_age_days": account_age_days,
            "device_age_days": device_age_days,
            "average_customer_amount": avg_amount,
        }

    def generate_legitimate_transaction(
        self,
        customer_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synthesize a single realistic legitimate transaction."""
        profile = customer_profile or self._sample_customer_profile()

        # Category and Channel
        category = sample_categorical(LEGITIMATE_CATEGORY_WEIGHTS, self.rng)
        channel = sample_categorical(LEGITIMATE_CHANNEL_WEIGHTS, self.rng)
        auth_method = sample_categorical(LEGITIMATE_AUTH_WEIGHTS, self.rng)

        # Diurnal Hour
        hour = int(self.rng.choice(24, p=HOURLY_ACTIVITY_PROB))

        # Amount conditioned on customer baseline and category typical scale with heavy-tail events
        cat_mean, cat_sigma = CATEGORY_AMOUNT_PARAMS[category]
        cat_baseline = np.exp(cat_mean)
        blend_factor = 0.65
        expected_amount = (blend_factor * profile["average_customer_amount"]) + ((1.0 - blend_factor) * cat_baseline)
        
        # ~6% of legitimate transactions are large purchases (e.g. travel, gifts, electronics, large groceries)
        is_high_ticket = self.rng.random() < 0.06
        if is_high_ticket:
            amount_factor = np.exp(self.rng.normal(1.2, 0.45))
        else:
            amount_factor = np.exp(self.rng.normal(0.0, 0.38))
        amount = float(np.round(np.clip(expected_amount * amount_factor, 1.5, 3500.0), 2))

        # Amount deviation relative to customer's historical average
        amount_dev = float(np.round((amount - profile["average_customer_amount"]) / profile["average_customer_amount"], 4))

        # Geographic consistency (domestic ~95% of the time)
        is_cross_border = self.rng.random() < 0.04
        if is_cross_border:
            other_countries = [c for c in COUNTRY_WEIGHTS if c != profile["customer_country"]]
            tx_country = str(self.rng.choice(other_countries))
            geo_dev = 1
        else:
            tx_country = profile["customer_country"]
            geo_dev = 0

        # Realistic probabilistic device change for legitimate users: ~15% baseline
        p_device_change = 0.14
        if channel in ["e-commerce", "mobile_app"]:
            p_device_change += 0.03
        if profile["account_age_days"] < 90:
            p_device_change += 0.03
        device_change = int(self.rng.random() < p_device_change)
        device_age = int(self.rng.integers(1, min(profile["account_age_days"] + 1, 30))) if device_change else profile["device_age_days"]

        # Risk Scores (Beta distributions skewed low for benign transactions)
        ip_risk = clip_score(float(self.rng.beta(1.5, 8.5)))
        cat_base_risk = CATEGORY_BASE_RISK[category]
        merchant_risk = clip_score(float(cat_base_risk + self.rng.normal(0.0, 0.06)))
        behavioral_dev = clip_score(float(self.rng.beta(1.4, 7.5)))
        identity_risk = clip_score(float(self.rng.beta(1.2, 8.5)))

        # Velocity
        v1h = int(1 + self.rng.poisson(0.15))
        v24h = int(v1h + self.rng.poisson(1.2))

        # Failed Auth (98% of legitimate transactions have 0 failed attempts)
        failed_auth = int(self.rng.choice([0, 1, 2], p=[0.975, 0.02, 0.005]))

        return {
            "transaction_amount": amount,
            "transaction_hour": hour,
            "merchant_category": category,
            "payment_channel": channel,
            "authentication_method": auth_method,
            "transaction_country": tx_country,
            "customer_country": profile["customer_country"],
            "account_age_days": profile["account_age_days"],
            "device_age_days": device_age,
            "device_change": device_change,
            "IP_risk_score": float(np.round(ip_risk, 4)),
            "merchant_risk_score": float(np.round(merchant_risk, 4)),
            "transaction_velocity_1h": v1h,
            "transaction_velocity_24h": v24h,
            "average_customer_amount": profile["average_customer_amount"],
            "amount_deviation": amount_dev,
            "geographic_deviation": geo_dev,
            "behavioral_deviation": float(np.round(behavioral_dev, 4)),
            "failed_authentication_count": failed_auth,
            "identity_risk_score": float(np.round(identity_risk, 4)),
            "attack_type": "LEGITIMATE",
            "fraud_label": 0,
        }

    def generate_fraud_transaction(
        self,
        archetype: Optional[AttackArchetype] = None,
        customer_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synthesize a fraudulent transaction driven by a specific GenAI attack archetype."""
        if archetype is None:
            archetypes = self.attack_library.get_all()
            archetype = self.rng.choice(archetypes)

        profile = customer_profile or self._sample_customer_profile()
        params = archetype.simulation_parameters

        # Base transaction defaults (blend archetype preferred category with everyday fraud categories)
        if "merchant_category" in params and self.rng.random() < 0.50:
            category = params["merchant_category"]
        else:
            category = sample_categorical(FRAUD_CATEGORY_WEIGHTS, self.rng)

        channel = params.get("payment_channel") or archetype.affected_payment_surface
        auth_method = params.get("auth_method_override") or sample_categorical(LEGITIMATE_AUTH_WEIGHTS, self.rng)

        # Account & Device age adjustments for synthetic identity / ATO
        account_age = profile["account_age_days"]
        if "account_age_max" in params:
            account_age = int(self.rng.integers(1, params["account_age_max"] + 1))
        elif "account_age_range" in params:
            low, high = params["account_age_range"]
            account_age = int(self.rng.integers(low, high + 1))

        # Probabilistic device change in fraud: ~45% overall (preserves meaningful fraction of device_change = 0)
        if "device_change" in params:
            p_dc = 0.70 if params["device_change"] == 1 else 0.15
            device_change = int(self.rng.random() < p_dc)
        else:
            device_change = int(self.rng.random() < 0.45)

        device_age = profile["device_age_days"]
        if "device_age_max" in params:
            device_age = int(self.rng.integers(1, params["device_age_max"] + 1))
        elif device_change == 1:
            device_age = int(self.rng.integers(1, 15))

        # Hour Distribution
        hour_dist = params.get("hour_distribution", "any")
        if hour_dist == "off_hours":
            hour = int(self.rng.choice(24, p=OFF_HOURS_PROB))
        elif hour_dist == "daytime":
            hour = int(self.rng.integers(9, 21))
        elif hour_dist == "business_hours":
            hour = int(self.rng.integers(9, 18))
        else:
            hour = int(self.rng.integers(0, 24))

        # Transaction Amount Calculation (Realistic scaling)
        if "fixed_amount_range" in params:
            low, high = params["fixed_amount_range"]
            amount = float(np.round(self.rng.uniform(low, high), 2))
        elif "amount_multiplier" in params:
            low_mult, high_mult = params["amount_multiplier"]
            scaled_low = max(0.9, low_mult * 0.55)
            scaled_high = max(1.2, high_mult * 0.55)
            mult = self.rng.uniform(scaled_low, scaled_high)
            base_ref = (0.5 * profile["average_customer_amount"]) + (0.5 * np.exp(CATEGORY_AMOUNT_PARAMS[category][0]))
            amount = float(np.round(base_ref * mult, 2))
        else:
            cat_mean, _ = CATEGORY_AMOUNT_PARAMS[category]
            amount = float(np.round(np.exp(cat_mean) * self.rng.uniform(1.1, 2.4), 2))

        amount = max(1.0, min(amount, 3500.0))
        amount_dev = float(np.round((amount - profile["average_customer_amount"]) / profile["average_customer_amount"], 4))

        # Geographic Deviation
        if "geographic_deviation" in params:
            geo_dev = int(params["geographic_deviation"])
        else:
            geo_dev = int(self.rng.random() < 0.35)

        if geo_dev == 1:
            other_countries = [c for c in COUNTRY_WEIGHTS if c != profile["customer_country"]]
            tx_country = str(self.rng.choice(other_countries))
        else:
            tx_country = profile["customer_country"]

        # Risk & Deviation Scores with Archetype-Specific Shifts
        base_ip = float(self.rng.beta(1.5, 8.5))
        ip_risk = clip_score(base_ip + params.get("ip_risk_shift", 0.0))

        base_merchant_risk = CATEGORY_BASE_RISK[category]
        merchant_risk = clip_score(base_merchant_risk + params.get("merchant_risk_shift", 0.0) + float(self.rng.normal(0, 0.04)))

        base_behavioral = float(self.rng.beta(1.4, 7.5))
        behavioral_dev = clip_score(base_behavioral + params.get("behavioral_dev_shift", 0.0))

        base_id_risk = float(self.rng.beta(1.2, 8.5))
        identity_risk = clip_score(base_id_risk + params.get("identity_risk_shift", 0.0))

        # Velocity Shifts
        v1h = int(1 + self.rng.poisson(0.15) + params.get("velocity_1h_boost", 0))
        v24h = int(v1h + self.rng.poisson(1.2) + params.get("velocity_24h_boost", 0))

        # Failed Auth Count
        if "failed_auth_count" in params:
            failed_auth = int(params["failed_auth_count"])
        else:
            failed_auth = int(self.rng.choice([0, 1, 2], p=[0.90, 0.08, 0.02]))

        return {
            "transaction_amount": amount,
            "transaction_hour": hour,
            "merchant_category": category,
            "payment_channel": channel,
            "authentication_method": auth_method,
            "transaction_country": tx_country,
            "customer_country": profile["customer_country"],
            "account_age_days": account_age,
            "device_age_days": device_age,
            "device_change": device_change,
            "IP_risk_score": float(np.round(ip_risk, 4)),
            "merchant_risk_score": float(np.round(merchant_risk, 4)),
            "transaction_velocity_1h": v1h,
            "transaction_velocity_24h": v24h,
            "average_customer_amount": profile["average_customer_amount"],
            "amount_deviation": amount_dev,
            "geographic_deviation": geo_dev,
            "behavioral_deviation": float(np.round(behavioral_dev, 4)),
            "failed_authentication_count": failed_auth,
            "identity_risk_score": float(np.round(identity_risk, 4)),
            "attack_type": archetype.name,
            "fraud_label": 1,
        }

    def generate_dataset(
        self,
        n_samples: int = 10000,
        fraud_ratio: float = 0.15,
        shuffle: bool = True,
    ) -> pd.DataFrame:
        """Generate a complete synthetic dataset of legitimate and attack transactions."""
        n_fraud = int(np.round(n_samples * fraud_ratio))
        n_legit = n_samples - n_fraud

        records: List[Dict[str, Any]] = []

        # Generate Legitimate Transactions
        for _ in range(n_legit):
            records.append(self.generate_legitimate_transaction())

        # Generate Fraudulent Transactions across all archetypes
        archetypes = self.attack_library.get_all()
        n_archetypes = len(archetypes)

        for i in range(n_fraud):
            archetype = archetypes[i % n_archetypes]
            records.append(self.generate_fraud_transaction(archetype=archetype))

        df = pd.DataFrame.from_records(records)
        df = df[ALL_COLUMNS]

        if shuffle:
            df = df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

        return df

    @staticmethod
    def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate structural integrity, schema, distributions, and invariants."""
        # 1. Missing columns
        missing_cols = [col for col in ALL_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset is missing required columns: {missing_cols}")

        # 2. Null values
        null_counts = df[ALL_COLUMNS].isnull().sum().to_dict()
        total_nulls = sum(null_counts.values())

        # 3. Value Range & Invariant Checks
        invariants = {
            "valid_fraud_labels": bool(df["fraud_label"].isin([0, 1]).all()),
            "positive_amounts": bool((df["transaction_amount"] > 0).all()),
            "valid_hours": bool(df["transaction_hour"].between(0, 23).all()),
            "valid_ip_risk_range": bool(df["IP_risk_score"].between(0.0, 1.0).all()),
            "valid_merchant_risk_range": bool(df["merchant_risk_score"].between(0.0, 1.0).all()),
            "valid_behavioral_dev_range": bool(df["behavioral_deviation"].between(0.0, 1.0).all()),
            "valid_identity_risk_range": bool(df["identity_risk_score"].between(0.0, 1.0).all()),
            "velocity_consistency": bool((df["transaction_velocity_24h"] >= df["transaction_velocity_1h"]).all()),
            "legit_attack_alignment": bool((df[df["fraud_label"] == 0]["attack_type"] == "LEGITIMATE").all()),
            "fraud_attack_alignment": bool((df[df["fraud_label"] == 1]["attack_type"] != "LEGITIMATE").all()),
        }

        # 4. Summary Statistics
        stats = {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "class_distribution": df["fraud_label"].value_counts().to_dict(),
            "fraud_percentage": float(np.round((df["fraud_label"].mean() * 100), 2)),
            "attack_type_count": int(df["attack_type"].nunique()),
            "total_nulls": total_nulls,
            "duplicate_count": int(df.duplicated(subset=NUMERICAL_FEATURES + CATEGORICAL_FEATURES).sum()),
            "invariants_passed": all(invariants.values()),
            "invariant_details": invariants,
        }

        return stats


def main():
    """CLI entrypoint to generate and validate synthetic dataset."""
    parser = argparse.ArgumentParser(description="FraudForge AI Synthetic Data Generator")
    parser.add_argument("--n_samples", type=int, default=10000, help="Total number of transactions to generate")
    parser.add_argument("--fraud_ratio", type=float, default=0.15, help="Proportion of fraudulent transactions (0.0 - 1.0)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default=str(GENERATED_DATA_DIR / "synthetic_transactions_v1.csv"), help="Output CSV path")
    args = parser.parse_args()

    print("=" * 70)
    print("FraudForge AI — Synthetic Payment Transaction Generator")
    print("=" * 70)
    print(f"Target sample count: {args.n_samples:,}")
    print(f"Target fraud ratio : {args.fraud_ratio:.2%}")
    print(f"Deterministic seed : {args.seed}")
    print(f"Output destination : {args.output}")
    print("-" * 70)

    generator = TransactionGenerator(seed=args.seed)
    df = generator.generate_dataset(n_samples=args.n_samples, fraud_ratio=args.fraud_ratio)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[+] Dataset saved successfully to {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

    # Validation
    validation = TransactionGenerator.validate_dataset(df)
    print("-" * 70)
    print("DATASET VALIDATION SUMMARY:")
    print(f"• Total Rows: {validation['total_records']:,}")
    print(f"• Total Columns: {validation['total_columns']}")
    print(f"• Class Breakdown: Legitimate (0) = {validation['class_distribution'].get(0, 0):,}, Fraud (1) = {validation['class_distribution'].get(1, 0):,} ({validation['fraud_percentage']}%)")
    print(f"• Unique Attack Types: {validation['attack_type_count']}")
    print(f"• Missing/Null Values: {validation['total_nulls']}")
    print(f"• Duplicates: {validation['duplicate_count']}")
    print(f"• All Invariants Passed: {validation['invariants_passed']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
