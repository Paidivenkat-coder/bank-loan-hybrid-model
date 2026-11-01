"""
Configuration file for Bank Loan Default Hybrid Model System.
Project: Design and Development of a Working Prototype for Bank loan default Using A Logistic Regression and Decision Tree hybrid model
Domain: Machine Learning & Predictive Analytics
Author / Lead Investigator: PAIDI VENKAT
Currency: Indian Rupee (INR / ₹)
"""
from pathlib import Path

# Project Metadata
PROJECT_TITLE = "Design and Development of a Working Prototype for Bank loan default Using A Logistic Regression and Decision Tree hybrid model"
DOMAIN = "Machine Learning & Predictive Analytics"
AUTHOR = "PAIDI VENKAT"
ABSTRACT = (
    "This project develops a Logistic Regression and Decision Tree hybrid model to assess "
    "creditworthiness of first-time borrowers, addressing the problem of bank loan default "
    "through a data-driven approach. The pipeline draws on anonymized loan applicant datasets, "
    "applying suitable pre-processing before training and testing the model on held-out data. "
    "The project culminates in a loan officer decision-support tool, bridging the gap between "
    "the underlying model and everyday use."
)
KEYWORDS = ["bank", "loan", "default", "logistic", "regression", "decision"]

# Currency Specification: Indian Rupee (INR / ₹)
CURRENCY_SYMBOL = "₹"
CURRENCY_CODE = "INR"

def format_inr(number: float, include_symbol: bool = True) -> str:
    """
    Formats a number into the Indian numbering system (Lakhs and Crores).
    E.g.: 125000 -> ₹1,25,000  |  12500000 -> ₹1.25 Cr
    """
    sym = "₹ " if include_symbol else ""
    if abs(number) >= 10000000:
        return f"{sym}{number / 10000000:.2f} Cr"
    elif abs(number) >= 100000:
        return f"{sym}{number / 100000:.2f} L"
    else:
        # Standard Indian comma placement for numbers < 1 Lakh
        s = f"{int(round(number)):,}"
        return f"{sym}{s}"

def format_inr_full(number: float, include_symbol: bool = True) -> str:
    """
    Formats a number into full Indian comma style (e.g., ₹12,34,567).
    """
    sym = "₹" if include_symbol else ""
    num_int = int(round(number))
    sign = "-" if num_int < 0 else ""
    num_str = str(abs(num_int))
    
    if len(num_str) <= 3:
        return f"{sym}{sign}{num_str}"
    
    last3 = num_str[-3:]
    remaining = num_str[:-3]
    
    # Insert commas every 2 digits from right to left
    parts = []
    while len(remaining) > 2:
        parts.insert(0, remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        parts.insert(0, remaining)
        
    formatted = ",".join(parts) + "," + last3
    return f"{sym}{sign}{formatted}"

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

# Underwriting Risk Tiers & Thresholds (Probability of Default)
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

# Indian Credit Bureau Score Mapping Constants (CIBIL-like 300 to 900 range)
CREDIT_SCORE_BASE = 720
CREDIT_SCORE_PDO = 50   # Points to Double the Odds

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
    "monthly_installment",        # Calculated amortized monthly installment (EMI) in ₹
    "net_monthly_surplus"         # Monthly income - existing debt - new installment in ₹
]

TARGET_COLUMN = "loan_default"

# Cost Matrix for Banking Decisions (RBI / Indian Commercial Bank standard)
# Default cost is significantly higher than missed interest margin
COST_FALSE_NEGATIVE = 5.0   # Default accepted (bank loses principal & recovery costs)
COST_FALSE_POSITIVE = 1.0   # Good customer rejected (bank loses net interest margin)
