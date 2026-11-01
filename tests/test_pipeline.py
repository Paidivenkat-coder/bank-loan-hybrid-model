"""
Unit and Integration Tests for Bank Loan Default Hybrid System.
Tests data generation in Indian Rupees (INR / ₹), preprocessing, model inference,
explainability, counterfactuals, and Indian currency formatting.
Author: PAIDI VENKAT
Project: Design and Development of a Working Prototype for Bank loan default Using A Logistic Regression and Decision Tree hybrid model
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    RAW_DATA_PATH, PREPROCESSOR_PATH, HYBRID_MODEL_PATH,
    TARGET_COLUMN, NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    format_inr, format_inr_full
)
from src.data_generator import generate_first_time_borrower_dataset, calculate_amortized_installment
from src.preprocessing import compute_financial_ratios, LoanDataPreprocessor
from src.models import TreeEmbeddingLogisticHybrid, build_baseline_logistic, build_baseline_decision_tree
from src.explainability import (
    probability_to_credit_score, get_risk_tier_and_decision,
    extract_decision_tree_path, generate_counterfactual_recommendations
)
import joblib


def test_amortized_installment_math():
    """Verifies that monthly installment (EMI) formula matches standard financial mathematics."""
    principal = np.array([100000.0])  # ₹1 Lakh
    annual_rate = np.array([12.0])    # 1% per month
    term_months = np.array([12])
    
    # Formula for 100000 at 12% annual (1% monthly) for 12 months = 100000 * (0.01 * 1.01^12) / (1.01^12 - 1) = ~8884.88
    installment = calculate_amortized_installment(principal, annual_rate, term_months)[0]
    assert np.isclose(installment, 8884.88, atol=1.0)


def test_indian_currency_formatting():
    """Verifies Indian numbering system formatting (Lakhs and Crores)."""
    assert "₹" in format_inr(50000)
    assert "L" in format_inr(650000)     # ₹ 6.50 L
    assert "Cr" in format_inr(15000000)  # ₹ 1.50 Cr
    assert format_inr_full(1234567) == "₹12,34,567"


def test_dataset_generation():
    """Tests synthetic dataset properties and statistical ranges in Indian Rupees."""
    df = generate_first_time_borrower_dataset(num_samples=200, random_seed=42)
    assert len(df) == 200
    assert TARGET_COLUMN in df.columns
    assert not df.isnull().any().any(), "Generated dataset should contain no missing values."
    for col in NUMERICAL_FEATURES:
        assert col in df.columns
    for col in CATEGORICAL_FEATURES:
        assert col in df.columns
    default_rate = df[TARGET_COLUMN].mean()
    assert 0.10 <= default_rate <= 0.35, f"Default rate {default_rate} outside expected range."
    # Verify INR range
    assert df["annual_income"].min() >= 200000
    assert df["loan_amount"].min() >= 40000


def test_preprocessing_and_ratios():
    """Verifies financial ratio computation and preprocessor transformation."""
    df = generate_first_time_borrower_dataset(num_samples=100, random_seed=42)
    df_calc = compute_financial_ratios(df)
    
    assert "monthly_installment" in df_calc.columns
    assert "debt_to_income_ratio" in df_calc.columns
    assert "loan_to_income_ratio" in df_calc.columns
    assert "savings_to_loan_ratio" in df_calc.columns
    assert "net_monthly_surplus" in df_calc.columns
    
    preprocessor = LoanDataPreprocessor()
    X = preprocessor.fit_transform(df)
    assert X.shape[0] == 100
    assert X.shape[1] == len(preprocessor.get_feature_names())
    assert not np.isnan(X).any()


def test_hybrid_model_fit_and_predict():
    """Tests TreeEmbeddingLogisticHybrid fit, predict, and predict_proba."""
    df = generate_first_time_borrower_dataset(num_samples=300, random_seed=42)
    preprocessor = LoanDataPreprocessor()
    X = preprocessor.fit_transform(df)
    y = df[TARGET_COLUMN].values
    
    hybrid = TreeEmbeddingLogisticHybrid(tree_max_depth=4, lr_c=0.5, random_state=42)
    hybrid.fit(X, y)
    
    probs = hybrid.predict_proba(X)
    assert probs.shape == (300, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.allclose(probs.sum(axis=1), 1.0)
    
    preds = hybrid.predict(X)
    assert set(preds).issubset({0, 1})


def test_cibil_credit_scoring_calibration():
    """Tests CIBIL credit score translation bounds (300 to 900) and monotonicity."""
    score_low_risk = probability_to_credit_score(0.05)
    score_mid_risk = probability_to_credit_score(0.20)
    score_high_risk = probability_to_credit_score(0.75)
    
    assert 300 <= score_low_risk <= 900
    assert 300 <= score_mid_risk <= 900
    assert 300 <= score_high_risk <= 900
    
    # Monotonicity: Lower default probability must yield a higher credit score
    assert score_low_risk > score_mid_risk > score_high_risk


def test_decision_and_counterfactuals():
    """Tests underwriting tier classification and counterfactual advisor in INR."""
    tier, dec, color, desc = get_risk_tier_and_decision(0.08)
    assert "Tier A" in tier
    assert "Approved" in dec
    
    tier_e, dec_e, _, _ = get_risk_tier_and_decision(0.65)
    assert "Tier E" in tier_e
    assert "Declined" in dec_e
    
    # Test counterfactual generator with an Indian profile
    preprocessor = LoanDataPreprocessor()
    df_sample = generate_first_time_borrower_dataset(num_samples=200, random_seed=42)
    preprocessor.fit(df_sample)
    
    hybrid_model = TreeEmbeddingLogisticHybrid(tree_max_depth=4, lr_c=0.5, random_state=42)
    hybrid_model.fit(preprocessor.transform(df_sample), df_sample[TARGET_COLUMN].values)
    
    sample_high_risk_inr = {
        "age": 24, "education": "Higher Secondary", "employment_type": "Gig / Freelance",
        "employment_years": 0.8, "annual_income": 300000, "existing_monthly_debt": 12000,
        "loan_amount": 450000, "loan_term_months": 36, "interest_rate": 18.5,
        "loan_intent": "Personal", "home_ownership": "Rented", "savings_balance": 15000,
        "rent_utility_payment_score": 52, "has_cosigner": 0, "has_collateral": "None",
        "num_recent_credit_inquiries": 4, "num_bank_overdrafts_12m": 2, "bank_tenure_months": 8
    }
    
    recs = generate_counterfactual_recommendations(sample_high_risk_inr, preprocessor, hybrid_model)
    assert len(recs) > 0
    assert any(r["reduction_pct"] > 0 for r in recs)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
