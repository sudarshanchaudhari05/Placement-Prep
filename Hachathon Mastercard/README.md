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
python -m src.simulation.transaction_generator --n_samples 10000 --fraud_ratio 0.15 --seed 42 --output data/generated/synthetic_transactions_10k.csv
```

### 3. Run Test Suite
```bash
pytest
```

---

## 📈 Phase 1 Validation & Statistical Properties

Validation run on **10,000 samples** (`seed=42`, `fraud_ratio=0.15`):

| Metric | Legitimate (8,500 rows) | Fraudulent (1,500 rows) | Invariant Status |
| :--- | :---: | :---: | :---: |
| **Mean Amount** | $79.52 | $802.31 | ✅ Passed (> 0, non-negative) |
| **Mean IP Risk Score** | 0.1494 | 0.6961 | ✅ Passed (0.0 – 1.0) |
| **Mean Merchant Risk** | 0.1642 | 0.6649 | ✅ Passed (0.0 – 1.0) |
| **Mean Behavioral Dev** | 0.1554 | 0.7128 | ✅ Passed (0.0 – 1.0) |
| **Mean Identity Risk** | 0.1238 | 0.5276 | ✅ Passed (0.0 – 1.0) |
| **Device Change Rate** | 3.85% | 75.20% | ✅ Passed (binary 0/1) |
| **Mean 1h Velocity** | 1.15 | 2.36 | ✅ Passed ($\text{v24h} \ge \text{v1h}$) |
| **Missing / Null Values**| 0 | 0 | ✅ Zero NaNs across all 22 cols |
| **Duplicates** | 0 | 0 | ✅ Zero duplicates |
| **Attack Vector Count** | 1 (`LEGITIMATE`) | 28 Archetypes | ✅ 100% label alignment |

---

## 🗺️ Development Roadmap

- [x] **Phase 1: Repository Foundation & Synthetic Data Pipeline**
  - Attack intelligence library (28 archetypes)
  - Synthetic transaction generator with correlated features
  - Validation engine & automated unit test suite
- [ ] **Phase 2: Baseline ML Detector & Attack-Specific Evaluation**
  - Preprocessing and feature engineering pipeline
  - XGBoost / Random Forest baseline classifier
  - Granular per-attack detection rate & false negative analytics
- [ ] **Phase 3: Adaptive Red-Team Loop & Model Hardening**
  - False-negative detection analyzer
  - Adversarial parameter mutator
  - Closed-loop retraining and before/after defense comparison
