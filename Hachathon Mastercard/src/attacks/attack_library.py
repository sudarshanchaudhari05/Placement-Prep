"""FraudForge AI: Attack Intelligence Library.

A structured catalog of 28 synthetic GenAI-enabled payment fraud archetypes.
All descriptions are strictly defensive and research-oriented.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
import pandas as pd


@dataclass
class AttackArchetype:
    """Represents a structured GenAI-enabled payment fraud archetype."""

    attack_id: str
    name: str
    category: str
    description: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    novelty_score: float  # 0.0 to 1.0 (higher = newer / more novel attack vector)
    detectability_score: float  # 0.0 to 1.0 (higher = easier for legacy baseline models to catch)
    behavioral_indicators: List[str]
    affected_payment_surface: str
    simulation_parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert archetype to standard dictionary representation."""
        return asdict(self)


ATTACK_CATALOG: List[AttackArchetype] = [
    # -------------------------------------------------------------------------
    # Category 1: AI Social Engineering & Impersonation (4 Archetypes)
    # -------------------------------------------------------------------------
    AttackArchetype(
        attack_id="ATK-001",
        name="Voice Clone Executive Impersonation (CEO Fraud / APP)",
        category="AI Social Engineering & Impersonation",
        description="Adversaries employ generative voice cloning of corporate executives to authorize high-value urgent push payments.",
        severity="CRITICAL",
        novelty_score=0.88,
        detectability_score=0.35,
        behavioral_indicators=[
            "Unusually high transaction amount compared to customer baseline",
            "Urgent timing during off-peak or after-hours windows",
            "Payment routed via authorized P2P or instant wire channel",
            "Customer passes standard 2FA because user was socially engineered into self-authorization",
        ],
        affected_payment_surface="p2p_transfer",
        simulation_parameters={
            "amount_multiplier": (3.5, 7.0),
            "hour_distribution": "off_hours",
            "auth_method_override": "sms_otp",
            "behavioral_dev_shift": 0.45,
            "ip_risk_shift": 0.15,
            "merchant_category": "money_transfer",
            "payment_channel": "p2p_transfer",
            "failed_auth_count": 0,
        },
    ),
    AttackArchetype(
        attack_id="ATK-002",
        name="Conversational Phishing Agent (AI Romance / Trust Scam)",
        category="AI Social Engineering & Impersonation",
        description="Autonomous LLM agents maintain multi-week relationship grooming with victims to extract recurring authorized transfers.",
        severity="HIGH",
        novelty_score=0.82,
        detectability_score=0.30,
        behavioral_indicators=[
            "Gradually escalating transaction amounts over multiple cycles",
            "Repeated P2P transfers to new beneficiary accounts",
            "Customer uses native trusted device with normal IP score",
            "High amount deviation relative to historical account average",
        ],
        affected_payment_surface="p2p_transfer",
        simulation_parameters={
            "amount_multiplier": (1.8, 3.8),
            "hour_distribution": "daytime",
            "auth_method_override": "biometric",
            "behavioral_dev_shift": 0.35,
            "ip_risk_shift": 0.05,
            "merchant_category": "money_transfer",
            "payment_channel": "p2p_transfer",
            "failed_auth_count": 0,
        },
    ),
    AttackArchetype(
        attack_id="ATK-003",
        name="Deepfake Family Emergency Push Payment",
        category="AI Social Engineering & Impersonation",
        description="Generative audio distress calls simulate urgent family emergencies to manipulate victims into immediate money transfers.",
        severity="HIGH",
        novelty_score=0.85,
        detectability_score=0.32,
        behavioral_indicators=[
            "Sudden high-velocity transfers within a 1-hour window",
            "High transaction amount deviation on mobile banking application",
            "Legitimate customer credentials and biometric authorization",
            "Beneficiary account registered recently in high-risk jurisdiction",
        ],
        affected_payment_surface="mobile_app",
        simulation_parameters={
            "amount_multiplier": (2.5, 5.5),
            "hour_distribution": "any",
            "auth_method_override": "biometric",
            "behavioral_dev_shift": 0.50,
            "velocity_1h_boost": 2,
            "merchant_category": "money_transfer",
            "payment_channel": "mobile_app",
            "failed_auth_count": 0,
        },
    ),
    AttackArchetype(
        attack_id="ATK-004",
        name="Automated AI Customer Support / Refund Phishing",
        category="AI Social Engineering & Impersonation",
        description="Automated conversational AI bots masquerade as merchant dispute agents to lure users into overpayment refund traps.",
        severity="MEDIUM",
        novelty_score=0.75,
        detectability_score=0.45,
        behavioral_indicators=[
            "Rapid succession of micro-refunds followed by large outbound reversal",
            "E-commerce or digital marketplace checkout redirection",
            "Elevated merchant risk rating due to newly onboarded gateway",
        ],
        affected_payment_surface="e-commerce",
        simulation_parameters={
            "amount_multiplier": (1.2, 2.5),
            "hour_distribution": "business_hours",
            "auth_method_override": "password",
            "behavioral_dev_shift": 0.40,
            "merchant_risk_shift": 0.45,
            "merchant_category": "marketplace",
            "payment_channel": "e-commerce",
            "failed_auth_count": 0,
        },
    ),

    # -------------------------------------------------------------------------
    # Category 2: Synthetic Identity & Deepfake Onboarding (4 Archetypes)
    # -------------------------------------------------------------------------
    AttackArchetype(
        attack_id="ATK-005",
        name="Deepfake Video KYC Onboarding Bypass",
        category="Synthetic Identity & Deepfake Onboarding",
        description="Diffusion video synthesis generates real-time liveness head movements to bypass automated KYC identity checks.",
        severity="CRITICAL",
        novelty_score=0.92,
        detectability_score=0.40,
        behavioral_indicators=[
            "Newly created account (account age < 14 days) conducting immediate high-tier transactions",
            "Elevated identity risk score despite passing onboarding liveness",
            "New device registration with zero historical telemetry",
            "Instant conversion to crypto or high-liquidity digital goods",
        ],
        affected_payment_surface="mobile_app",
        simulation_parameters={
            "account_age_max": 14,
            "device_age_max": 7,
            "device_change": 1,
            "identity_risk_shift": 0.65,
            "amount_multiplier": (2.0, 5.0),
            "merchant_category": "crypto_exchange",
            "payment_channel": "mobile_app",
            "failed_auth_count": 0,
        },
    ),
    AttackArchetype(
        attack_id="ATK-006",
        name="Generative Identity Fabrication (Franken-Identity)",
        category="Synthetic Identity & Deepfake Onboarding",
        description="LLMs weave real dormant SSNs with synthesized demographic profiles and fabricated credit histories.",
        severity="HIGH",
        novelty_score=0.86,
        detectability_score=0.48,
        behavioral_indicators=[
            "High identity risk score combined with clean legacy credit bureau pulls",
            "Device fingerprint incongruent with historical residential zip code",
            "Moderate initial purchases followed by rapid credit line exhaustion",
        ],
        affected_payment_surface="e-commerce",
        simulation_parameters={
            "account_age_max": 30,
            "device_age_max": 15,
            "device_change": 1,
            "identity_risk_shift": 0.55,
            "amount_multiplier": (1.5, 3.5),
            "merchant_category": "electronics",
            "payment_channel": "e-commerce",
            "failed_auth_count": 0,
        },
    ),
    AttackArchetype(
        attack_id="ATK-007",
        name="Diffusion Document Forgery (Statement Manipulation)",
        category="Synthetic Identity & Deepfake Onboarding",
        description="Pixel-perfect generative document editing modifies bank statements and utility bills to unlock elevated credit tiers.",
        severity="HIGH",
        novelty_score=0.80,
        detectability_score=0.42,
        behavioral_indicators=[
            "Sudden tier jump in transaction authorization limits",
            "High-value transactions initiated from unverified residential subnets",
            "Moderate account age with sudden velocity spike",
        ],
        affected_payment_surface="money_transfer",
        simulation_parameters={
            "account_age_range": (30, 90),
            "device_change": 1,
            "identity_risk_shift": 0.50,
            "amount_multiplier": (2.5, 6.0),
            "merchant_category": "money_transfer",
            "payment_channel": "api_gateway",
            "failed_auth_count": 0,
        },
    ),
    AttackArchetype(
        attack_id="ATK-008",
        name="Blended Minor Identity Sleeper Fraud",
        category="Synthetic Identity & Deepfake Onboarding",
        description="AI models monitor child SSNs to cultivate synthetic sleeper accounts over months before coordinated cash-out.",
        severity="MEDIUM",
        novelty_score=0.78,
        detectability_score=0.38,
        behavioral_indicators=[
            "Aged account with long quiet period suddenly showing high transaction velocity",
            "Device age mismatched with account creation date",
            "High transaction amount deviation against quiet baseline",
        ],
        affected_payment_surface="retail",
        simulation_parameters={
            "account_age_range": (180, 400),
            "device_age_max": 10,
            "device_change": 1,
            "identity_risk_shift": 0.40,
            "velocity_24h_boost": 4,
            "amount_multiplier": (2.0, 4.0),
            "merchant_category": "luxury",
            "payment_channel": "pos_chip",
            "failed_auth_count": 0,
        },
    ),

    # -------------------------------------------------------------------------
    # Category 3: Automated Account Takeover & Behavioral Mimicry (4 Archetypes)
    # -------------------------------------------------------------------------
    AttackArchetype(
        attack_id="ATK-009",
        name="Human Typing & Mouse Cadence Behavioral Mimicry",
        category="Automated Account Takeover & Behavioral Mimicry",
        description="Adversarial neural networks mimic legitimate user keystroke dynamics and pointer curvature to evade behavioral biometrics.",
        severity="CRITICAL",
        novelty_score=0.94,
        detectability_score=0.25,
        behavioral_indicators=[
            "Very low behavioral deviation metrics despite novel device fingerprint",
            "Suspicious IP address range or high IP risk score",
            "Device change flag triggered on high-value transaction checkout",
        ],
        affected_payment_surface="e-commerce",
        simulation_parameters={
            "device_change": 1,
            "device_age_max": 5,
            "behavioral_dev_shift": 0.12,  # Mimics human well, keeping deviation lower than usual attacks
            "ip_risk_shift": 0.70,
            "amount_multiplier": (2.0, 4.5),
            "merchant_category": "luxury",
            "payment_channel": "e-commerce",
            "failed_auth_count": 1,
        },
    ),
    AttackArchetype(
        attack_id="ATK-010",
        name="LLM-Orchestrated Adaptive Credential Stuffing",
        category="Automated Account Takeover & Behavioral Mimicry",
        description="LLMs dynamically solve CAPTCHAs, rotate user agents, and vary request payload tempos across proxy meshes.",
        severity="HIGH",
        novelty_score=0.81,
        detectability_score=0.55,
        behavioral_indicators=[
            "Multiple failed authentication attempts prior to successful checkout",
            "Elevated IP risk score with high transaction velocity across 24 hours",
            "Device change flag positive",
        ],
        affected_payment_surface="e-commerce",
        simulation_parameters={
            "device_change": 1,
            "failed_auth_count": 3,
            "ip_risk_shift": 0.60,
            "velocity_24h_boost": 5,
            "merchant_category": "retail",
            "payment_channel": "e-commerce",
        },
    ),
    AttackArchetype(
        attack_id="ATK-011",
        name="Autonomous Session Token Harvesting & Replay",
        category="Automated Account Takeover & Behavioral Mimicry",
        description="Infostealer malware exfiltrates active browser sessions and uses AI agents to replay checkout payloads seamlessly.",
        severity="HIGH",
        novelty_score=0.87,
        detectability_score=0.38,
        behavioral_indicators=[
            "Authenticated session utilized from discordant IP geolocation",
            "Geographic deviation between customer home location and transaction IP",
            "Fast checkout completion without typical browsing navigation time",
        ],
        affected_payment_surface="api_gateway",
        simulation_parameters={
            "geographic_deviation": 1,
            "ip_risk_shift": 0.55,
            "behavioral_dev_shift": 0.40,
            "auth_method_override": "none",
            "amount_multiplier": (1.8, 3.5),
            "merchant_category": "digital_goods",
            "payment_channel": "api_gateway",
        },
    ),
    AttackArchetype(
        attack_id="ATK-012",
        name="Stealth Biometric Hash Injection / Virtual Sensor Replay",
        category="Automated Account Takeover & Behavioral Mimicry",
        description="Adversaries inject pre-computed biometric cryptographic tokens into compromised mobile OS hardware abstraction layers.",
        severity="CRITICAL",
        novelty_score=0.95,
        detectability_score=0.28,
        behavioral_indicators=[
            "Biometric authentication passed instantly on rooted/tampered hardware runtime",
            "High device age with recent root exploit telemetry",
            "High transaction amount deviation with zero prior merchant interaction",
        ],
        affected_payment_surface="mobile_app",
        simulation_parameters={
            "auth_method_override": "biometric",
            "behavioral_dev_shift": 0.35,
            "amount_multiplier": (3.0, 6.0),
            "ip_risk_shift": 0.45,
            "merchant_category": "crypto_exchange",
            "payment_channel": "mobile_app",
        },
    ),

    # -------------------------------------------------------------------------
    # Category 4: Evasive & Micro-Transaction Attacks (4 Archetypes)
    # -------------------------------------------------------------------------
    AttackArchetype(
        attack_id="ATK-013",
        name="Low-and-Slow AI Micro-Carding Swarm",
        category="Evasive & Micro-Transaction Attacks",
        description="Distributed bot swarms perform sub-threshold $1-$5 card validation tests distributed across thousands of digital merchants.",
        severity="MEDIUM",
        novelty_score=0.76,
        detectability_score=0.60,
        behavioral_indicators=[
            "Unusually low transaction amounts below standard deviation threshold",
            "High 1-hour and 24-hour transaction velocity across distinct merchant IDs",
            "Low authentication friction channel (no 3DS challenge triggered)",
        ],
        affected_payment_surface="digital_goods",
        simulation_parameters={
            "fixed_amount_range": (1.0, 8.0),
            "auth_method_override": "none",
            "velocity_1h_boost": 4,
            "velocity_24h_boost": 12,
            "merchant_risk_shift": 0.30,
            "merchant_category": "digital_goods",
            "payment_channel": "e-commerce",
        },
    ),
    AttackArchetype(
        attack_id="ATK-014",
        name="AI Smurfing & Micro-Structuring Network",
        category="Evasive & Micro-Transaction Attacks",
        description="Reinforcement learning agents optimize transaction amounts just below regulatory AML and fraud detection trigger limits.",
        severity="HIGH",
        novelty_score=0.89,
        detectability_score=0.36,
        behavioral_indicators=[
            "Clustering of transaction amounts just below threshold tiers (e.g. $95-$99 or $950-$990)",
            "Synchronized multi-account fund aggregation",
            "Slightly elevated behavioral deviation with structured timing cadence",
        ],
        affected_payment_surface="money_transfer",
        simulation_parameters={
            "fixed_amount_range": (850.0, 995.0),
            "velocity_1h_boost": 2,
            "velocity_24h_boost": 6,
            "behavioral_dev_shift": 0.38,
            "merchant_category": "money_transfer",
            "payment_channel": "p2p_transfer",
        },
    ),
    AttackArchetype(
        attack_id="ATK-015",
        name="Velocity-Throttled Automated Account Draining",
        category="Evasive & Micro-Transaction Attacks",
        description="AI bots mathematically calculate rate limits to drain account balances at the exact maximum speed permitted by rules.",
        severity="HIGH",
        novelty_score=0.83,
        detectability_score=0.45,
        behavioral_indicators=[
            "Evenly spaced transaction intervals skirting 1-hour velocity tripwires",
            "Repeated moderate transaction amounts depleting available credit balance",
            "Elevated IP risk score masked by proxy rotation",
        ],
        affected_payment_surface="e-commerce",
        simulation_parameters={
            "velocity_1h_boost": 1,
            "velocity_24h_boost": 5,
            "amount_multiplier": (1.5, 2.8),
            "ip_risk_shift": 0.50,
            "merchant_category": "gaming",
            "payment_channel": "e-commerce",
        },
    ),
    AttackArchetype(
        attack_id="ATK-016",
        name="Dynamic Merchant Category Hopping Bot",
        category="Evasive & Micro-Transaction Attacks",
        description="Automated carding engines rapidly switch MCC codes between groceries, utilities, and gaming to avoid MCC-specific blocks.",
        severity="MEDIUM",
        novelty_score=0.74,
        detectability_score=0.52,
        behavioral_indicators=[
            "High variety of distinct merchant categories in short duration",
            "Elevated velocity across 24h window",
            "New device registration with moderate IP risk",
        ],
        affected_payment_surface="retail",
        simulation_parameters={
            "device_change": 1,
            "velocity_24h_boost": 7,
            "amount_multiplier": (1.0, 2.2),
            "merchant_risk_shift": 0.35,
            "payment_channel": "e-commerce",
        },
    ),

    # -------------------------------------------------------------------------
    # Category 5: AI Agent & API Payment Exploits (5 Archetypes)
    # -------------------------------------------------------------------------
    AttackArchetype(
        attack_id="ATK-017",
        name="Autonomous Shopping Agent Prompt Hijack",
        category="AI Agent & API Payment Exploits",
        description="Indirect prompt injection in merchant item descriptions forces user AI shopping agents to reroute payments to attacker wallets.",
        severity="CRITICAL",
        novelty_score=0.96,
        detectability_score=0.22,
        behavioral_indicators=[
            "API-initiated transaction with legitimate agent credentials",
            "Sudden rerouting to high-risk beneficiary or unexpected merchant category",
            "High amount deviation from shopping agent's typical budget bounds",
        ],
        affected_payment_surface="api_gateway",
        simulation_parameters={
            "auth_method_override": "none",
            "payment_channel": "api_gateway",
            "merchant_category": "crypto_exchange",
            "amount_multiplier": (2.5, 5.0),
            "behavioral_dev_shift": 0.55,
            "merchant_risk_shift": 0.60,
        },
    ),
    AttackArchetype(
        attack_id="ATK-018",
        name="Malicious MCP / Plugin Tool Unauthorized Transaction",
        category="AI Agent & API Payment Exploits",
        description="Compromised Model Context Protocol tools silently trigger background micro-payments during unrelated agent reasoning chains.",
        severity="HIGH",
        novelty_score=0.93,
        detectability_score=0.28,
        behavioral_indicators=[
            "Transaction initiated via headless API channel with zero user interaction telemetry",
            "Moderate transaction amount routed to affiliate digital service provider",
            "Account age mature but sudden uncharacteristic API usage",
        ],
        affected_payment_surface="api_gateway",
        simulation_parameters={
            "payment_channel": "api_gateway",
            "auth_method_override": "none",
            "amount_multiplier": (1.4, 3.2),
            "behavioral_dev_shift": 0.48,
            "merchant_risk_shift": 0.50,
            "merchant_category": "digital_goods",
        },
    ),
    AttackArchetype(
        attack_id="ATK-019",
        name="Agentic Webhook Signature Evasion & Race Exploit",
        category="AI Agent & API Payment Exploits",
        description="High-frequency LLM bot clusters exploit asynchronous race conditions between payment webhooks and order fulfillment state.",
        severity="HIGH",
        novelty_score=0.88,
        detectability_score=0.42,
        behavioral_indicators=[
            "Near-instantaneous parallel transaction attempts within milliseconds (high 1h velocity)",
            "High merchant risk rating and API payment channel",
            "Extreme speed of authorization confirmation requests",
        ],
        affected_payment_surface="api_gateway",
        simulation_parameters={
            "payment_channel": "api_gateway",
            "velocity_1h_boost": 6,
            "velocity_24h_boost": 10,
            "merchant_risk_shift": 0.65,
            "merchant_category": "digital_goods",
        },
    ),
    AttackArchetype(
        attack_id="ATK-020",
        name="Automated Cart-State Desynchronization & Discount Exploit",
        category="AI Agent & API Payment Exploits",
        description="Automated scripts manipulate client-side JSON cart states to settle high-value items at negative or discounted line items.",
        severity="MEDIUM",
        novelty_score=0.79,
        detectability_score=0.50,
        behavioral_indicators=[
            "High amount deviation (severely underpriced cart total compared to category norm)",
            "E-commerce checkout with high IP risk score",
            "Moderate failed authentication count due to gateway price integrity re-checks",
        ],
        affected_payment_surface="e-commerce",
        simulation_parameters={
            "amount_multiplier": (0.1, 0.4),  # Abnormally low for electronics/luxury
            "ip_risk_shift": 0.45,
            "merchant_category": "electronics",
            "payment_channel": "e-commerce",
            "failed_auth_count": 1,
        },
    ),
    AttackArchetype(
        attack_id="ATK-021",
        name="Adversarial Perturbation Evasion on Risk Scoring Engine",
        category="AI Agent & API Payment Exploits",
        description="Adversaries generate gradient-optimized transaction feature vectors that intentionally minimize ML model risk probabilities.",
        severity="CRITICAL",
        novelty_score=0.97,
        detectability_score=0.18,
        behavioral_indicators=[
            "Features mathematically engineered to mimic legitimate medians (e.g. rounded hour, common MCC)",
            "Device change masked by simulated browser fingerprints",
            "Very low superficial deviation hiding underlying fraudulent fund extraction",
        ],
        affected_payment_surface="e-commerce",
        simulation_parameters={
            "behavioral_dev_shift": 0.08,
            "ip_risk_shift": 0.15,
            "amount_multiplier": (1.1, 1.9),
            "device_change": 0,
            "failed_auth_count": 0,
            "merchant_category": "retail",
            "payment_channel": "e-commerce",
        },
    ),

    # -------------------------------------------------------------------------
    # Category 6: Cross-Channel & Cross-Border Evasion (4 Archetypes)
    # -------------------------------------------------------------------------
    AttackArchetype(
        attack_id="ATK-022",
        name="AI-Routed Residential Proxy Swarm Geolocation Spoofing",
        category="Cross-Channel & Cross-Border Evasion",
        description="Adversarial botnets route payment traffic through compromised local residential smart-home nodes matching customer zip codes.",
        severity="HIGH",
        novelty_score=0.85,
        detectability_score=0.34,
        behavioral_indicators=[
            "Residential IP looks benign superficially but possesses hidden network ASN anomalies",
            "High amount deviation and device change flag",
            "Low geographic deviation score concealing remote adversary control",
        ],
        affected_payment_surface="e-commerce",
        simulation_parameters={
            "geographic_deviation": 0,  # Geolocation spoofed to match customer
            "device_change": 1,
            "ip_risk_shift": 0.25,
            "behavioral_dev_shift": 0.42,
            "amount_multiplier": (2.2, 4.8),
            "merchant_category": "luxury",
            "payment_channel": "e-commerce",
        },
    ),
    AttackArchetype(
        attack_id="ATK-023",
        name="Multi-Currency Triangular Arbitrage Money Laundering",
        category="Cross-Channel & Cross-Border Evasion",
        description="Algorithmic fraud swarms execute rapid multi-currency currency conversions to obscure the audit trail of stolen funds.",
        severity="HIGH",
        novelty_score=0.87,
        detectability_score=0.40,
        behavioral_indicators=[
            "Cross-border transaction with mismatching transaction and customer country",
            "Geographic deviation flag triggered",
            "High transaction velocity in 1h window across international gateways",
        ],
        affected_payment_surface="money_transfer",
        simulation_parameters={
            "geographic_deviation": 1,
            "velocity_1h_boost": 3,
            "velocity_24h_boost": 8,
            "amount_multiplier": (2.0, 4.5),
            "merchant_category": "crypto_exchange",
            "payment_channel": "p2p_transfer",
        },
    ),
    AttackArchetype(
        attack_id="ATK-024",
        name="Omnichannel Fast Checkout Bypass (POS to Web Arbitrage)",
        category="Cross-Channel & Cross-Border Evasion",
        description="Simultaneous exploitation of contactless tokenization in physical POS and online web channels to bypass dual-presence rules.",
        severity="MEDIUM",
        novelty_score=0.77,
        detectability_score=0.46,
        behavioral_indicators=[
            "Physical POS and online checkout occurring in impossible chronological proximity",
            "High 1-hour velocity with disparate payment channels",
            "Device age fresh on secondary channel",
        ],
        affected_payment_surface="pos_contactless",
        simulation_parameters={
            "velocity_1h_boost": 3,
            "device_change": 1,
            "amount_multiplier": (1.5, 3.0),
            "merchant_category": "retail",
            "payment_channel": "pos_contactless",
        },
    ),
    AttackArchetype(
        attack_id="ATK-025",
        name="Deepfake Live Face Swap on NFC Contactless Terminal",
        category="Cross-Channel & Cross-Border Evasion",
        description="Wearable lens hardware displays dynamic GAN facial reconstructions to fool POS surveillance cameras and contactless verification.",
        severity="HIGH",
        novelty_score=0.91,
        detectability_score=0.31,
        behavioral_indicators=[
            "High-value contactless transaction at luxury physical retailer",
            "Device age low or newly provisioned digital wallet token",
            "High amount deviation from cardholder's historical baseline",
        ],
        affected_payment_surface="pos_contactless",
        simulation_parameters={
            "payment_channel": "pos_contactless",
            "auth_method_override": "biometric",
            "amount_multiplier": (2.5, 5.5),
            "behavioral_dev_shift": 0.35,
            "merchant_category": "luxury",
        },
    ),

    # -------------------------------------------------------------------------
    # Category 7: E-Commerce & Merchant Exploits (3 Archetypes)
    # -------------------------------------------------------------------------
    AttackArchetype(
        attack_id="ATK-026",
        name="AI-Generated RMA & Fake Tracking Return Fraud",
        category="E-Commerce & Merchant Exploits",
        description="Generative models forge digital carrier scan proofs and postal receipt timestamps to trigger instant merchant refund releases.",
        severity="MEDIUM",
        novelty_score=0.81,
        detectability_score=0.48,
        behavioral_indicators=[
            "High frequency of refund and dispute reversals on e-commerce accounts",
            "Merchant risk score elevated due to dispute ratio",
            "Moderate transaction amount with high velocity in 24h",
        ],
        affected_payment_surface="e-commerce",
        simulation_parameters={
            "merchant_risk_shift": 0.50,
            "amount_multiplier": (1.2, 2.5),
            "velocity_24h_boost": 4,
            "merchant_category": "marketplace",
            "payment_channel": "e-commerce",
        },
    ),
    AttackArchetype(
        attack_id="ATK-027",
        name="Synthetic Subscription Layering & Chargeback Cycling",
        category="E-Commerce & Merchant Exploits",
        description="Botnets cycle thousands of stolen BIN ranges across low-dollar recurring digital subscriptions to harvest payout cuts.",
        severity="MEDIUM",
        novelty_score=0.76,
        detectability_score=0.58,
        behavioral_indicators=[
            "Recurring subscription channel with zero authentication friction",
            "High velocity of recurring sign-ups across 24 hours",
            "Elevated IP risk score",
        ],
        affected_payment_surface="recurring_subscription",
        simulation_parameters={
            "payment_channel": "recurring_subscription",
            "auth_method_override": "none",
            "fixed_amount_range": (9.99, 49.99),
            "velocity_24h_boost": 8,
            "ip_risk_shift": 0.55,
            "merchant_category": "digital_goods",
        },
    ),
    AttackArchetype(
        attack_id="ATK-028",
        name="AI SIM-Swap + Automated OTP Exfiltration",
        category="Automated Account Takeover & Behavioral Mimicry",
        description="Social engineering LLMs compromise telecom customer care portals to execute SIM swaps, followed by automated OTP interception.",
        severity="CRITICAL",
        novelty_score=0.90,
        detectability_score=0.44,
        behavioral_indicators=[
            "Device change flag triggered right before high-value transaction",
            "SMS OTP passed successfully on brand new device ID",
            "High amount deviation and maximum 1h transaction velocity",
        ],
        affected_payment_surface="mobile_app",
        simulation_parameters={
            "device_change": 1,
            "device_age_max": 2,
            "auth_method_override": "sms_otp",
            "amount_multiplier": (3.0, 6.5),
            "velocity_1h_boost": 3,
            "ip_risk_shift": 0.55,
            "merchant_category": "electronics",
            "payment_channel": "mobile_app",
        },
    ),
]


class AttackLibrary:
    """Library service providing lookup, filtering, and catalog management."""

    def __init__(self, archetypes: Optional[List[AttackArchetype]] = None):
        self._archetypes: Dict[str, AttackArchetype] = {
            atk.attack_id: atk for atk in (archetypes or ATTACK_CATALOG)
        }

    def get_all(self) -> List[AttackArchetype]:
        """Return list of all registered attack archetypes."""
        return list(self._archetypes.values())

    def get_by_id(self, attack_id: str) -> Optional[AttackArchetype]:
        """Retrieve archetype by its unique ID (e.g. 'ATK-001')."""
        return self._archetypes.get(attack_id)

    def get_by_name(self, name: str) -> Optional[AttackArchetype]:
        """Retrieve archetype by its exact name."""
        for atk in self._archetypes.values():
            if atk.name.lower() == name.lower():
                return atk
        return None

    def filter_by_category(self, category: str) -> List[AttackArchetype]:
        """Filter archetypes by fraud category."""
        return [atk for atk in self._archetypes.values() if atk.category.lower() == category.lower()]

    def filter_by_severity(self, severity: str) -> List[AttackArchetype]:
        """Filter archetypes by severity rating (e.g. 'CRITICAL', 'HIGH')."""
        return [atk for atk in self._archetypes.values() if atk.severity.upper() == severity.upper()]

    def filter_by_surface(self, surface: str) -> List[AttackArchetype]:
        """Filter archetypes by affected payment channel or surface."""
        return [atk for atk in self._archetypes.values() if surface.lower() in atk.affected_payment_surface.lower()]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert entire attack library into a pandas DataFrame."""
        records = [atk.to_dict() for atk in self._archetypes.values()]
        return pd.DataFrame.from_records(records)

    def list_categories(self) -> List[str]:
        """Return sorted list of unique categories."""
        return sorted(list(set(atk.category for atk in self._archetypes.values())))

    def __len__(self) -> int:
        return len(self._archetypes)

    def __repr__(self) -> str:
        return f"<AttackLibrary: {len(self._archetypes)} archetypes across {len(self.list_categories())} categories>"


def get_default_attack_library() -> AttackLibrary:
    """Factory helper to obtain standard preloaded AttackLibrary instance."""
    return AttackLibrary()
