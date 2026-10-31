"""
Synthetic realistic anonymized loan applicant dataset generator for first-time borrowers.
Incorporates demographic, financial, alternative credit data, and non-linear risk dynamics.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RAW_DATA_PATH, RANDOM_SEED


def calculate_amortized_installment(principal: np.ndarray, annual_rate: np.ndarray, term_months: np.ndarray) -> np.ndarray:
    """Calculates monthly amortized installment using standard banking annuity formula."""
    monthly_rate = (annual_rate / 100.0) / 12.0
    # Guard against zero rate
    monthly_rate = np.where(monthly_rate == 0, 1e-6, monthly_rate)
    numerator = monthly_rate * ((1.0 + monthly_rate) ** term_months)
    denominator = ((1.0 + monthly_rate) ** term_months) - 1.0
    return principal * (numerator / denominator)


def generate_first_time_borrower_dataset(num_samples: int = 6000, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generates a realistic, anonymized dataset of first-time loan applicants.
    Combines linear risk factors with non-linear threshold interactions (e.g. DTI cliff, liquidity buffer).
    """
    np.random.seed(random_seed)
    
    # 1. Demographics & Employment
    # First-time borrowers skew towards young to mid-career applicants
    age = np.clip(np.random.normal(loc=29, scale=6.5, size=num_samples).astype(int), 21, 62)
    
    education_choices = ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
    education_probs = [0.22, 0.18, 0.44, 0.13, 0.03]
    education = np.random.choice(education_choices, size=num_samples, p=education_probs)
    
    employment_choices = ["Salaried", "Self-Employed", "Contract", "Freelance"]
    employment_probs = [0.65, 0.16, 0.12, 0.07]
    employment_type = np.random.choice(employment_choices, size=num_samples, p=employment_probs)
    
    # Employment length correlated with age
    max_possible_exp = np.maximum(0.2, (age - 20) * 0.75)
    employment_years = np.round(np.random.uniform(0.1, max_possible_exp, size=num_samples), 1)
    
    # 2. Financial Metrics (Income & Existing Debt)
    # Annual income follows log-normal distribution, adjusted by education
    base_income = np.random.lognormal(mean=10.8, sigma=0.45, size=num_samples)
    edu_multiplier = {
        "High School": 0.82,
        "Associate": 0.92,
        "Bachelor": 1.10,
        "Master": 1.28,
        "Doctorate": 1.45
    }
    multipliers = np.array([edu_multiplier[e] for e in education])
    annual_income = np.round(np.clip(base_income * multipliers, 18000, 160000), -2)
    
    # Existing monthly debt obligations (credit cards, student loans, auto)
    # Around 10% to 25% of gross monthly income
    monthly_gross = annual_income / 12.0
    existing_debt_ratio = np.random.beta(a=2.5, b=6.0, size=num_samples) * 0.45
    existing_monthly_debt = np.round(monthly_gross * existing_debt_ratio, -1)
    
    # 3. Loan Characteristics
    loan_intent_choices = [
        "Personal", "Education", "Debt Consolidation", 
        "Small Business", "Home Improvement", "Medical"
    ]
    loan_intent_probs = [0.28, 0.22, 0.20, 0.14, 0.10, 0.06]
    loan_intent = np.random.choice(loan_intent_choices, size=num_samples, p=loan_intent_probs)
    
    home_ownership_choices = ["Rent", "Own", "Mortgage", "With Parents"]
    home_ownership_probs = [0.52, 0.10, 0.18, 0.20]
    home_ownership = np.random.choice(home_ownership_choices, size=num_samples, p=home_ownership_probs)
    
    # Loan amount requested (proportional to income, but with variance)
    loan_multiplier = np.random.uniform(0.10, 0.45, size=num_samples)
    loan_amount = np.round(np.clip(annual_income * loan_multiplier, 2000, 45000), -2)
    
    loan_term_choices = [12, 24, 36, 48, 60]
    loan_term_probs = [0.10, 0.25, 0.40, 0.15, 0.10]
    loan_term_months = np.random.choice(loan_term_choices, size=num_samples, p=loan_term_probs)
    
    # Base interest rate: 8.5% to 24.0%
    interest_rate = np.round(np.random.normal(loc=13.5, scale=3.8, size=num_samples), 2)
    interest_rate = np.clip(interest_rate, 7.5, 25.5)
    
    # 4. Alternative & Relationship Banking Features (Thin-File Signals)
    # Liquid savings buffer ($)
    savings_ratio = np.random.exponential(scale=0.18, size=num_samples)
    savings_balance = np.round(np.clip(annual_income * savings_ratio, 200, 35000), -2)
    
    # Non-bureau utility and telecom payment score (30 - 100)
    rent_utility_payment_score = np.clip(
        np.random.normal(loc=74, scale=14, size=num_samples).astype(int), 30, 100
    )
    
    # Has co-signer (guarantor)
    has_cosigner_prob = np.where(age < 26, 0.48, 0.22)
    has_cosigner = (np.random.uniform(0, 1, size=num_samples) < has_cosigner_prob).astype(int)
    
    # Has collateral pledged
    collateral_choices = ["None", "Vehicle", "Fixed Deposit"]
    collateral_probs = [0.65, 0.25, 0.10]
    has_collateral = np.random.choice(collateral_choices, size=num_samples, p=collateral_probs)
    
    # Inquiries in past 6 months (Poisson)
    num_recent_credit_inquiries = np.clip(np.random.poisson(lam=1.4, size=num_samples), 0, 8)
    
    # Bank overdrafts in past 12 months (Poisson, highly skewed towards 0)
    num_bank_overdrafts_12m = np.clip(np.random.poisson(lam=0.6, size=num_samples), 0, 6)
    
    # Bank relationship tenure (months)
    bank_tenure_months = np.clip(
        np.random.gamma(shape=2.5, scale=12.0, size=num_samples).astype(int), 3, 140
    )
    
    # Calculate Installment and Debt-To-Income (DTI)
    monthly_installment = calculate_amortized_installment(loan_amount, interest_rate, loan_term_months)
    total_monthly_debt = existing_monthly_debt + monthly_installment
    dti_ratio = total_monthly_debt / monthly_gross
    loan_to_income = loan_amount / annual_income
    
    # 5. Latent Risk Model: Combining Linear Risk + Non-Linear Decision Tree Interactions
    # Base intercept targeting ~18% default rate for thin-file applicants
    z = -0.70
    
    # Linear Risk Contributions
    z += 2.8 * (dti_ratio - 0.35)                     # High DTI increases risk
    z += 1.8 * (loan_to_income - 0.25)                # High loan relative to income
    z += 0.08 * (interest_rate - 13.0)                # High interest rate burden
    z += 0.22 * num_recent_credit_inquiries           # Credit seeking behaviour
    z += 0.38 * num_bank_overdrafts_12m               # Cash-flow mismanagement
    z -= 0.025 * (rent_utility_payment_score - 70)    # Good utility history mitigates risk
    z -= 0.008 * (bank_tenure_months - 24)            # Bank relationship loyalty
    z -= 0.75 * has_cosigner                          # Co-signer significantly protects
    z -= 0.50 * (has_collateral != "None").astype(int)# Collateral protection
    z -= 0.60 * np.log1p(savings_balance / 1000.0)    # Liquid savings buffer
    
    # Non-Linear Interaction Spikes (True Non-Linearities)
    # Rule 1: DTI Cliff with Low Savings Buffer (Classic Underwriting Cliff)
    dti_cliff = (dti_ratio > 0.43) & (savings_balance < (2.0 * monthly_installment))
    z += 1.45 * dti_cliff.astype(float)
    
    # Rule 2: Inexperienced Speculative Small Business
    speculative_biz = (loan_intent == "Small Business") & (employment_years < 1.5) & (has_cosigner == 0)
    z += 1.25 * speculative_biz.astype(float)
    
    # Rule 3: Extreme Overdraft + Multi-Inquiry Cash Crunch
    liquidity_crunch = (num_bank_overdrafts_12m >= 2) & (num_recent_credit_inquiries >= 3)
    z += 1.35 * liquidity_crunch.astype(float)
    
    # Rule 4: Fortress Borrower (Virtually immune to default)
    fortress = (has_cosigner == 1) & (rent_utility_payment_score >= 85) & (savings_balance > loan_amount * 0.4)
    z -= 1.80 * fortress.astype(float)
    
    # Convert log-odds to default probability via sigmoid function
    default_prob = 1.0 / (1.0 + np.exp(-z))
    default_prob = np.clip(default_prob, 0.01, 0.98)
    
    # Bernoulli trial to generate actual ground truth default label
    loan_default = (np.random.uniform(0, 1, size=num_samples) < default_prob).astype(int)
    
    # Build DataFrame
    applicant_ids = [f"APP-{10000 + i}" for i in range(num_samples)]
    
    df = pd.DataFrame({
        "applicant_id": applicant_ids,
        "age": age,
        "education": education,
        "employment_type": employment_type,
        "employment_years": employment_years,
        "annual_income": annual_income,
        "existing_monthly_debt": existing_monthly_debt,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "interest_rate": interest_rate,
        "loan_intent": loan_intent,
        "home_ownership": home_ownership,
        "savings_balance": savings_balance,
        "rent_utility_payment_score": rent_utility_payment_score,
        "has_cosigner": has_cosigner,
        "has_collateral": has_collateral,
        "num_recent_credit_inquiries": num_recent_credit_inquiries,
        "num_bank_overdrafts_12m": num_bank_overdrafts_12m,
        "bank_tenure_months": bank_tenure_months,
        "loan_default": loan_default
    })
    
    return df


def generate_and_save_dataset(filepath: Path = RAW_DATA_PATH, num_samples: int = 6000) -> pd.DataFrame:
    """Generates dataset and saves it to the raw data path."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df = generate_first_time_borrower_dataset(num_samples=num_samples)
    df.to_csv(filepath, index=False)
    print(f"Dataset generated successfully at: {filepath}")
    print(f"Total applicants: {len(df)}")
    print(f"Default rate: {df['loan_default'].mean():.2%}")
    return df


if __name__ == "__main__":
    generate_and_save_dataset()
