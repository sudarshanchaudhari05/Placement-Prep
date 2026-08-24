"""FraudForge AI: Attack Mutation Engine.

Provides targeted, feature-aware adversarial mutations on payment fraud archetypes
to generate harder synthetic variants based on blue-team detector vulnerabilities.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.attacks.attack_library import AttackArchetype, AttackLibrary, get_default_attack_library
from src.simulation.distributions import clip_score
from src.utils.config import DEFAULT_SEED, ALL_COLUMNS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES


class AttackMutator:
    """Adversarial mutator producing harder, evasive synthetic payment fraud variants."""

    def __init__(self, seed: int = DEFAULT_SEED):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.audit_log: List[Dict[str, Any]] = []

    def mutate_transaction(
        self,
        tx: Dict[str, Any],
        archetype: Optional[AttackArchetype] = None,
        detector_weaknesses: Optional[List[str]] = None,
        mutation_intensity: float = 0.50,
        strategy: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Apply targeted adversarial mutations to a single fraud transaction.

        Args:
            tx: Original baseline fraud transaction dictionary.
            archetype: Attack archetype metadata.
            detector_weaknesses: List of top relied-upon features by detector.
            mutation_intensity: Float in (0.0, 1.0] dictating strength of stealth shift.
            strategy: Optional specific strategy name ('signal_masking', 'behavioral_blending',
                      'velocity_smoothing', 'amount_camouflage', 'risk_suppression', 'stealth_blend').

        Returns:
            mutated_tx: Dictionary with mutated features preserving domain invariants.
            tx_audit: List of audit records explaining changes made to specific features.
        """
        mutated = tx.copy()
        tx_audit: List[Dict[str, Any]] = []
        intensity = float(np.clip(mutation_intensity, 0.1, 1.0))
        weaknesses = detector_weaknesses or ["device_change", "IP_risk_score", "behavioral_deviation", "amount_deviation"]

        attack_name = archetype.name if archetype else tx.get("attack_type", "GENERIC_FRAUD")

        # 1. Strategy: Signal Masking (Device change camouflage)
        # If detector heavily relies on device_change and the attack triggered it:
        if ("device_change" in weaknesses or strategy == "signal_masking") and mutated.get("device_change", 0) == 1:
            orig_dc = mutated["device_change"]
            orig_da = mutated["device_age_days"]
            # Adversary uses cloned session / familiar device profile
            mutated["device_change"] = 0
            mutated["device_age_days"] = max(25, int(mutated["account_age_days"] * 0.45))

            record = {
                "attack_type": attack_name,
                "feature": "device_change",
                "original_value": orig_dc,
                "mutated_value": mutated["device_change"],
                "reason": "Signal Masking: Cloned trusted device fingerprint to bypass novel device rules",
                "mutation_intensity": intensity,
            }
            tx_audit.append(record)
            self.audit_log.append(record)

        # 2. Strategy: Behavioral Blending (Biometric & typing cadence smoothing)
        if ("behavioral_deviation" in weaknesses or strategy in ["behavioral_blending", "stealth_blend"]):
            orig_bd = mutated.get("behavioral_deviation", 0.5)
            if orig_bd > 0.25:
                # Blend closer to legitimate baseline (~0.15)
                blend_factor = 1.0 - (0.55 * intensity)
                new_bd = clip_score(float(np.round(max(0.12, orig_bd * blend_factor), 4)))
                mutated["behavioral_deviation"] = new_bd

                record = {
                    "attack_type": attack_name,
                    "feature": "behavioral_deviation",
                    "original_value": orig_bd,
                    "mutated_value": new_bd,
                    "reason": "Behavioral Blending: Simulating natural keystroke/navigation curves",
                    "mutation_intensity": intensity,
                }
                tx_audit.append(record)
                self.audit_log.append(record)

        # 3. Strategy: Velocity Smoothing (Skirting burst rate limits)
        if ("transaction_velocity_1h" in weaknesses or "transaction_velocity_24h" in weaknesses or strategy == "velocity_smoothing"):
            orig_v1 = mutated.get("transaction_velocity_1h", 1)
            orig_v24 = mutated.get("transaction_velocity_24h", 1)
            if orig_v1 > 1 or orig_v24 > 2:
                new_v1 = max(1, int(np.round(orig_v1 * (1.0 - 0.50 * intensity))))
                new_v24 = max(new_v1, int(np.round(orig_v24 * (1.0 - 0.45 * intensity))))
                mutated["transaction_velocity_1h"] = new_v1
                mutated["transaction_velocity_24h"] = new_v24

                record = {
                    "attack_type": attack_name,
                    "feature": "transaction_velocity_1h",
                    "original_value": orig_v1,
                    "mutated_value": new_v1,
                    "reason": "Velocity Smoothing: Throttling bot tempo below burst velocity triggers",
                    "mutation_intensity": intensity,
                }
                tx_audit.append(record)
                self.audit_log.append(record)

        # 4. Strategy: Amount Camouflage (Structuring transactions closer to baseline)
        if ("amount_deviation" in weaknesses or "transaction_amount" in weaknesses or strategy == "amount_camouflage"):
            orig_amt = mutated.get("transaction_amount", 100.0)
            orig_dev = mutated.get("amount_deviation", 0.0)
            avg_amt = mutated.get("average_customer_amount", 60.0)

            # Move amount closer to customer average
            if abs(orig_dev) > 0.50:
                dampener = 1.0 - (0.50 * intensity)
                new_dev = float(np.round(orig_dev * dampener, 4))
                new_amt = float(np.round(max(3.0, avg_amt * (1.0 + new_dev)), 2))
                mutated["transaction_amount"] = new_amt
                mutated["amount_deviation"] = new_dev

                record = {
                    "attack_type": attack_name,
                    "feature": "amount_deviation",
                    "original_value": orig_dev,
                    "mutated_value": new_dev,
                    "reason": "Amount Camouflage: Structuring spend closer to cardholder historical mean",
                    "mutation_intensity": intensity,
                }
                tx_audit.append(record)
                self.audit_log.append(record)

        # 5. Strategy: Risk-Signal Suppression (Residential Proxy & IP Reputation)
        if ("IP_risk_score" in weaknesses or "merchant_risk_score" in weaknesses or strategy == "risk_suppression"):
            orig_ip = mutated.get("IP_risk_score", 0.5)
            if orig_ip > 0.30:
                # Residential proxy routing reduces observable IP risk score
                new_ip = clip_score(float(np.round(max(0.10, orig_ip * (1.0 - 0.55 * intensity)), 4)))
                mutated["IP_risk_score"] = new_ip

                record = {
                    "attack_type": attack_name,
                    "feature": "IP_risk_score",
                    "original_value": orig_ip,
                    "mutated_value": new_ip,
                    "reason": "Risk-Signal Suppression: Routing via clean residential ISP node",
                    "mutation_intensity": intensity,
                }
                tx_audit.append(record)
                self.audit_log.append(record)

        # Invariant Guarantees
        mutated["fraud_label"] = 1  # Strictly preserve ground-truth fraud label
        mutated["transaction_velocity_24h"] = max(mutated["transaction_velocity_24h"], mutated["transaction_velocity_1h"])
        mutated["transaction_amount"] = max(0.5, mutated["transaction_amount"])

        return mutated, tx_audit

    def mutate_dataframe(
        self,
        df_fraud: pd.DataFrame,
        attack_library: Optional[AttackLibrary] = None,
        target_attack_types: Optional[List[str]] = None,
        detector_weaknesses: Optional[List[str]] = None,
        mutation_intensity: float = 0.50,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Mutate all (or selected target) fraud transactions in a DataFrame.

        Args:
            df_fraud: DataFrame containing fraudulent transactions.
            attack_library: AttackLibrary instance for archetype metadata lookups.
            target_attack_types: Optional list of attack names to selectively mutate (if None, mutates all).
            detector_weaknesses: List of features to target for evasion.
            mutation_intensity: Strength of mutations (0.1 to 1.0).

        Returns:
            mutated_df: DataFrame of mutated transactions.
            audit_df: DataFrame containing all mutation log records.
        """
        library = attack_library or get_default_attack_library()
        mutated_records: List[Dict[str, Any]] = []
        batch_audit: List[Dict[str, Any]] = []

        for _, row in df_fraud.iterrows():
            tx_dict = row.to_dict()
            attack_name = tx_dict.get("attack_type", "")

            # If target attacks are specified and this is not in target, keep as is
            if target_attack_types is not None and attack_name not in target_attack_types:
                mutated_records.append(tx_dict)
                continue

            archetype = library.get_by_name(attack_name)
            mutated_tx, tx_audit = self.mutate_transaction(
                tx=tx_dict,
                archetype=archetype,
                detector_weaknesses=detector_weaknesses,
                mutation_intensity=mutation_intensity,
            )
            mutated_records.append(mutated_tx)
            batch_audit.extend(tx_audit)

        mutated_df = pd.DataFrame.from_records(mutated_records)
        audit_df = pd.DataFrame.from_records(batch_audit) if batch_audit else pd.DataFrame(
            columns=["attack_type", "feature", "original_value", "mutated_value", "reason", "mutation_intensity"]
        )

        return mutated_df, audit_df
