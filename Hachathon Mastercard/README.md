# FraudForge AI — Closed-Loop Red Team / Blue Team AI Defense Lab

> **Mastercard Innovation Challenge @ GFF 2026 — AI Defense Lab for Payment Security**  
> *"Attack the detector. Learn from the miss. Build a stronger defense."*

---

## 🛡️ Executive Summary

**FraudForge AI** is an adversarial payment security framework designed to detect, simulate, and defend against emerging **GenAI-powered payment fraud vectors**. Rather than relying on static rules or legacy point-in-time classifiers, FraudForge AI operates a closed-loop **IDENTIFY → GENERATE → DEFEND** cycle:

1. **IDENTIFY**: Curates a library of 28 realistic GenAI fraud archetypes (voice clone APP, deepfake video KYC, autonomous agent injection, behavioral mimicry, smurfing).
2. **GENERATE**: Synthesizes high-fidelity, statistically correlated transaction streams with behavioral telemetry.
3. **DEFEND**: Evaluates blue-team detection performance by individual attack vector, mutates missed attacks via red-team feedback, and hardens the model.

---

## 📁 Repository Structure

```
fraudforge-ai/
├── README.md                           # Project documentation & runbook
├── requirements.txt                    # Core Python dependencies
├── .gitignore                          # Standard Python & data gitignore
│
├── data/
│   ├── raw/                            # Raw data store (.gitkeep)
│   ├── generated/                      # Generated synthetic datasets
│   └── processed/                      # Transformed & normalized feature sets
│
├── src/
│   ├── __init__.py
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── attack_library.py           # 28 GenAI fraud archetypes catalog & query API
│   │   └── attack_mutator.py           # [Phase 3] Adversarial mutation engine
│   │
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── distributions.py            # Diurnal curves, category parameters, risk priors
│   │   └── transaction_generator.py    # Synthetic payment generator & validator
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py      # [Phase 2] Preprocessing & feature pipelines
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── train.py                    # [Phase 2] Baseline XGBoost / RF trainer
│   │   ├── predict.py                  # [Phase 2] Scoring inference engine
│   │   └── evaluate.py                 # [Phase 2] Attack-specific evaluation metrics
│   │
│   ├── adversarial/
│   │   ├── __init__.py
│   │   └── feedback_loop.py            # [Phase 3] Closed-loop retraining pipeline
│   │
│   └── utils/
│       ├── __init__.py
│       └── config.py                   # Global constants, schema definitions & paths
│
├── models/                             # Saved model artifacts (.gitkeep)
├── experiments/                        # Benchmark outputs & run metrics (.gitkeep)
├── notebooks/                          # Exploration & demonstration notebooks (.gitkeep)
└── tests/
    ├── __init__.py
    ├── test_attack_library.py          # Unit tests for attack intelligence catalog
    └── test_transaction_generator.py   # Unit tests for data generator & invariants
```

---

## 🧬 Attack Intelligence Library (28 Archetypes)

The library catalogs 28 distinct GenAI payment fraud vectors organized into 7 functional categories:

| Category | Count | Sample Archetypes |
| :--- | :---: | :--- |
| **AI Social Engineering & Impersonation** | 4 | Voice Clone Executive APP (`ATK-001`), Conversational Phishing Agent (`ATK-002`), Deepfake Family Emergency (`ATK-003`) |
| **Synthetic Identity & Deepfake Onboarding** | 4 | Deepfake Video KYC Bypass (`ATK-005`), Generative Identity Fabrication (`ATK-006`), Diffusion Statement Forgery (`ATK-007`) |
| **Automated ATO & Behavioral Mimicry** | 5 | Keystroke Cadence Mimicry (`ATK-009`), Adaptive Credential Stuffing (`ATK-010`), Stealth Biometric Injection (`ATK-012`) |
| **Evasive & Micro-Transaction Attacks** | 4 | Low-and-Slow Micro-Carding (`ATK-013`), AI Smurfing / Structuring (`ATK-014`), Velocity-Throttled Draining (`ATK-015`) |
| **AI Agent & API Payment Exploits** | 5 | Agent Prompt Hijack (`ATK-017`), Malicious MCP Tool Exploit (`ATK-018`), Risk Scoring Perturbation Evasion (`ATK-021`) |
| **Cross-Channel & Cross-Border Evasion** | 4 | AI Residential Proxy Swarm (`ATK-022`), Triangular Currency Arbitrage (`ATK-023`), POS-to-Web Fast Bypass (`ATK-024`) |
| **E-Commerce & Merchant Exploits** | 2 | AI-Generated RMA Return Fraud (`ATK-026`), Synthetic Subscription Layering (`ATK-027`) |

Each archetype defines:
* `attack_id`, `name`, `category`, `description`, `severity`
* `novelty_score` (0.0–1.0) and `detectability_score` (0.0–1.0)
* `behavioral_indicators` (telemetry and behavioral anomalies)
* `affected_payment_surface` (e-commerce, p2p, mobile_app, pos, api_gateway)
* `simulation_parameters` (precise numerical shifts for amount, velocity, timing, device, and risk priors)

---

## 📊 Dataset Schema & Behavioral Correlated Features

The generated dataset contains **22 features** with realistic statistical distributions and domain invariants:

### Numerical Features (15)
* `transaction_amount`: Transaction amount ($)
* `transaction_hour`: Hour of transaction (0–23, following diurnal curve)
* `account_age_days`: Age of account in days (1–1800)
* `device_age_days`: Age of active device in days (1–account_age_days)
* `device_change`: Binary indicator for newly registered device (0 or 1)
* `IP_risk_score`: IP reputation anomaly score (0.0–1.0)
* `merchant_risk_score`: Category and gateway risk score (0.0–1.0)
* `transaction_velocity_1h`: Transaction count in past 1 hour ($\ge 1$)
* `transaction_velocity_24h`: Transaction count in past 24 hours ($\ge \text{velocity\_1h}$)
* `average_customer_amount`: Historical customer average spending ($)
* `amount_deviation`: Ratio deviation: $\frac{\text{amount} - \text{avg}}{\text{avg}}$
* `geographic_deviation`: Binary cross-border anomaly flag (0 or 1)
* `behavioral_deviation`: Keystroke/navigation biometric anomaly score (0.0–1.0)
* `failed_authentication_count`: Recent failed 2FA/password attempts ($\ge 0$)
* `identity_risk_score`: Synthetic identity / KYC anomaly score (0.0–1.0)

### Categorical Features (5)
* `merchant_category`: `groceries`, `retail`, `dining`, `travel`, `digital_goods`, `gaming`, `luxury`, `crypto_exchange`, `money_transfer`, `utilities`, `electronics`, `marketplace`
* `payment_channel`: `pos_contactless`, `pos_chip`, `e-commerce`, `mobile_app`, `recurring_subscription`, `p2p_transfer`, `api_gateway`
* `authentication_method`: `biometric`, `3ds_v2`, `sms_otp`, `password`, `none`, `hardware_token`, `push_notification`
* `transaction_country`: ISO 2-letter country code
* `customer_country`: Customer primary registration country

### Target & Ground Truth (2)
* `attack_type`: `LEGITIMATE` or specific attack archetype name
* `fraud_label`: `0` (Legitimate) or `1` (Fraudulent)

---

## 🚀 Quick Start & Reproducibility

### 1. Installation
```bash
# Clone the repository and navigate into the root folder
cd fraudforge-ai

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset
```bash
# Generate 10,000 transactions with 15% fraud ratio (deterministic seed=42)
python -m src.simulation.transaction_generator --n_samples 10000 --fraud_ratio 0.15 --seed 42 --output data/generated/synthetic_transactions_v1.csv
```

### 3. Train Baseline XGBoost Detector
```bash
# Train baseline detector on synthetic dataset with stratified split
python -m src.detection.train --data data/generated/synthetic_transactions_v1.csv --model_type xgboost
```

### 4. Evaluate Detector & Attack Vulnerabilities
```bash
# Evaluate global classification metrics and per-attack detection rates on held-out test split
python -m src.detection.evaluate
```

### 5. Run Closed-Loop Adaptive Red-Team Experiment
```bash
# Run end-to-end 3-dataset closed loop: generate, mutate weak attacks, retrain, and evaluate on unseen attacks
python -m src.adversarial.feedback_loop --n_samples 10000 --fraud_ratio 0.15 --mutation_intensity 0.65
```

### 6. Run Robustness Benchmarks & Feature Ablation Experiments
```bash
# Execute model architecture comparisons (XGBoost vs Random Forest) and feature ablation studies
python -m src.detection.benchmarks
```

### 7. Run Automated Test Suite
```bash
pytest -v
```

---

## 🔬 Robustness & Model Benchmarking (Phase 4 / 4B)

### 1. Synthetic Payment Realism Upgrade (Phase 4B)

We calibrated the synthetic generator distributions to eliminate artificial separation cues:

| Feature | Pre-Realism Separation | Post-Realism Separation | Status |
| :--- | :---: | :---: | :---: |
| **`device_change`** | Legit: 3.75% \| Fraud: 75.75% ($d = 2.172, KS = 0.720$) | Legit: **15.51%** \| Fraud: **52.53%** ($d = 0.849, KS = 0.370$) | ✅ **Realistic Overlap** |
| **`merchant_risk_score`** | Legit: 0.1630 \| Fraud: 0.5166 ($d = 1.858, KS = 0.709$) | Legit: **0.1659** \| Fraud: **0.4431** ($d = 1.355, KS = 0.543$) | ✅ **Moderate Separation** |
| **`transaction_amount`** | Legit: $79.43 \| Fraud: $803.96 ($d = 1.257, KS = 0.596$) | Legit: **$92.47** \| Fraud: **$230.08** ($d = 0.730, KS = 0.307$) | ✅ **Realistic Overlap** |
| **`amount_deviation`** | Legit: 0.2360 \| Fraud: 14.65 ($d = 0.899, KS = 0.655$) | Legit: **0.4807** \| Fraud: **3.4327** ($d = 0.583, KS = 0.317$) | ✅ **Realistic Overlap** |

### 2. Model Architecture Comparison (Realistic Dataset)

Both models evaluated on identical stratified splits (Dataset A normal test & Dataset C unseen adversarial test):

| Model | Normal F1 | Normal Recall | Adversarial Recall (Dataset C) | Adversarial F1 | Adversarial Misses |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Full Features)** | **0.9554** | **96.33%** | 68.00% | 0.7846 | 96 missed |
| **Random Forest (Full Features)** | 0.9502 | 94.67% | **76.00%** | **0.8352** | **72 missed** |

### 3. Adaptive Red-Team Hardening on Realistic Data

| Metric | Baseline Detector | Hardened Detector | Improvement (Delta) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 94.40% | **97.05%** | **+2.65%** |
| **Precision** | **92.73%** | 90.85% | -1.88% |
| **Adversarial Recall** | 68.00% | **89.33%** | **+21.33%** |
| **F1 Score** | 0.7846 | **0.9008** | **+0.1162** |
| **ROC-AUC** | 0.9790 | **0.9885** | **+0.0095** |
| **False Positive Rate** | **0.94%** | 1.59% | +0.65% |
| **Missed Attacks (FN)** | 96 missed | **32 missed** | **-64 (-66.7% reduction)** |

### 4. Balanced Feature Importance (Zero Dominating Cue)

Top 10 features utilized by the detector after realism calibration:
1. `merchant_risk_score` (12.43%)
2. `transaction_velocity_1h` (7.42%)
3. `geographic_deviation` (6.79%)
4. `transaction_velocity_24h` (6.66%)
5. `behavioral_deviation` (5.28%)
6. `IP_risk_score` (4.87%)
7. `payment_channel_pos_chip` (4.82%)
8. `identity_risk_score` (4.47%)
9. `device_age_days` (4.44%)
10. `payment_channel_pos_contactless` (4.14%)

*(Note: `device_change` dropped from 59.17% down to < 4%, forcing the model to learn genuine multi-feature behavioral interactions).*

---

## 🗺️ Development Roadmap

- [x] **Phase 1: Repository Foundation & Synthetic Data Pipeline**
  - Attack intelligence library (28 archetypes)
  - Synthetic transaction generator with correlated features
  - Validation engine & automated unit test suite
- [x] **Phase 2: Baseline ML Detector & Attack-Specific Evaluation**
  - Preprocessing and feature engineering pipeline (`FraudFeaturePipeline`)
  - XGBoost baseline classifier trainer (`src/detection/train.py`)
  - Scoring & inference engine (`src/detection/predict.py`)
  - Granular per-attack detection rate & false negative analytics (`src/detection/evaluate.py`)
- [x] **Phase 3: Adaptive Red-Team Loop & Model Hardening**
  - False-negative detection analyzer & feature dependency extractor
  - Targeted, multi-strategy attack mutator (`AttackMutator`)
  - 3-Dataset closed-loop orchestration (`Dataset A`, `Dataset B`, `Dataset C`)
  - Retrained hardened model with 76.5% reduction in adversarial misses (`src/adversarial/feedback_loop.py`)
- [x] **Phase 4 & 4B: Robustness Benchmarks, Data Realism & Full Revalidation**
  - Calibrated synthetic distributions (device change, merchant risk, amount tails)
  - XGBoost vs. Random Forest architectural benchmark
  - Feature ablation studies and automated target leakage audit
  - Retrained and revalidated closed-loop feedback experiment (`experiments/realism_revalidation.json`)
