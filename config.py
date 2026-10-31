"""
Configuration file for Bank Loan Default Hybrid Model System.
Defines dataset schemas, model parameters, underwriting thresholds, and paths.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "first_time_borrowers_loan_data.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_data.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test_data.csv"

MODELS_DIR = BASE_DIR / "models_saved"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"
DECISION_TREE_PATH = MODELS_DIR / "decision_tree.joblib"
HYBRID_MODEL_PATH = MODELS_DIR / "hybrid_model.joblib"
STACKING_MODEL_PATH = MODELS_DIR / "stacking_hybrid.joblib"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"

REPORTS_DIR = BASE_DIR / "reports"

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Underwriting Risk Tiers & Thresholds (Default Probability)
RISK_THRESHOLDS = {
    "TIER_A_LOW": 0.12,          # < 12%: Tier A (Automatic Approval)
    "TIER_B_MODERATE": 0.22,     # 12% - 22%: Tier B (Standard Approval)
    "TIER_C_BORDERLINE": 0.35,   # 22% - 35%: Tier C (Manual Review / Conditional Approval)
    "TIER_D_HIGH": 0.50,         # 35% - 50%: Tier D (High Risk / Strict Conditions)
    "TIER_E_SEVERE": 1.00        # > 50%: Tier E (Automatic Decline)
}

# Decision Actions
DECISION_APPROVAL_THRESHOLD = 0.22   # Below 22% = Approved
DECISION_REVIEW_THRESHOLD = 0.35     # 22% - 35% = Manual Review / Conditional

# Credit Score Mapping Constants (FICO-like 300 to 850 range)
CREDIT_SCORE_BASE = 680
CREDIT_SCORE_PDO = 45   # Points to Double the Odds

# Features Definition
NUMERICAL_FEATURES = [
    "age",
    "employment_years",
    "annual_income",
    "existing_monthly_debt",
    "loan_amount",
    "loan_term_months",
    "interest_rate",
    "savings_balance",
    "rent_utility_payment_score",
    "num_recent_credit_inquiries",
    "num_bank_overdrafts_12m",
    "bank_tenure_months"
]

CATEGORICAL_FEATURES = [
    "education",
    "employment_type",
    "home_ownership",
    "loan_intent",
    "has_cosigner",
    "has_collateral"
]

ENGINEERED_FEATURES = [
    "debt_to_income_ratio",       # Total debt obligations / gross monthly income
    "loan_to_income_ratio",       # Requested loan / annual income
    "savings_to_loan_ratio",      # Liquid savings / loan amount
    "monthly_installment",        # Calculated amortized monthly installment
    "net_monthly_surplus"         # Monthly income - existing debt - new installment
]

TARGET_COLUMN = "loan_default"

# Cost Matrix for Banking Decisions
# Default cost is much higher than missed opportunity cost
COST_FALSE_NEGATIVE = 5.0   # Default accepted (bank loses principal & recovery costs)
COST_FALSE_POSITIVE = 1.0   # Good customer rejected (bank loses net interest margin)
