"""
Synthetic realistic anonymized loan applicant dataset generator for first-time borrowers.
Tailored for Indian Retail Banking (INR / ₹ currency, Indian demographics, and RBI credit norms).
Author: PAIDI VENKAT
Project: Design and Development of a Working Prototype for Bank loan default Using A Logistic Regression and Decision Tree hybrid model
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RAW_DATA_PATH, RANDOM_SEED, format_inr, format_inr_full


def calculate_amortized_installment(principal: np.ndarray, annual_rate: np.ndarray, term_months: np.ndarray) -> np.ndarray:
    """
    Calculates monthly Equated Monthly Installment (EMI) in INR using standard banking annuity formula:
    EMI = P * r * (1 + r)^n / ((1 + r)^n - 1)
    """
    monthly_rate = (annual_rate / 100.0) / 12.0
    # Guard against zero rate
    monthly_rate = np.where(monthly_rate == 0, 1e-6, monthly_rate)
    numerator = monthly_rate * ((1.0 + monthly_rate) ** term_months)
    denominator = ((1.0 + monthly_rate) ** term_months) - 1.0
    return principal * (numerator / denominator)


def generate_first_time_borrower_dataset(num_samples: int = 6000, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generates a realistic, anonymized Indian banking dataset of first-time retail loan applicants.
    Monetary figures in Indian Rupees (INR / ₹).
    Combines linear risk factors with non-linear threshold interactions (DTI cliff, liquidity buffer, guarantor impact).
    """
    np.random.seed(random_seed)
    
    # 1. Demographics & Employment in Indian Retail Context
    # First-time applicants skew towards young to early-career professionals (age 21 to 58)
    age = np.clip(np.random.normal(loc=28.5, scale=6.0, size=num_samples).astype(int), 21, 58)
    
    education_choices = [
        "Higher Secondary", 
        "Diploma", 
        "Graduate (B.Tech/B.Sc/B.Com)", 
        "Post-Graduate (MBA/M.Tech)", 
        "Doctorate"
    ]
    education_probs = [0.18, 0.14, 0.48, 0.17, 0.03]
    education = np.random.choice(education_choices, size=num_samples, p=education_probs)
    
    employment_choices = [
        "Salaried (IT/Corporate)", 
        "Salaried (PSU/Govt)", 
        "Self-Employed Professional", 
        "Small Business / Proprietor", 
        "Gig / Freelance"
    ]
    employment_probs = [0.48, 0.18, 0.12, 0.14, 0.08]
    employment_type = np.random.choice(employment_choices, size=num_samples, p=employment_probs)
    
    # Work experience (years) correlated with age
    max_possible_exp = np.maximum(0.2, (age - 20) * 0.8)
    employment_years = np.round(np.random.uniform(0.2, max_possible_exp, size=num_samples), 1)
    
    # 2. Financial Metrics in Indian Rupees (INR / ₹)
    # Annual gross income follows log-normal distribution, adjusted by education & sector
    # Median ~ ₹6,50,000 (₹6.5 Lakhs), spanning ₹2,40,000 to ₹35,00,000
    base_income_inr = np.random.lognormal(mean=13.35, sigma=0.48, size=num_samples)
    edu_multiplier = {
        "Higher Secondary": 0.72,
        "Diploma": 0.85,
        "Graduate (B.Tech/B.Sc/B.Com)": 1.12,
        "Post-Graduate (MBA/M.Tech)": 1.38,
        "Doctorate": 1.55
    }
    multipliers = np.array([edu_multiplier[e] for e in education])
    annual_income = np.round(np.clip(base_income_inr * multipliers, 240000, 3500000), -3)
    
    # Existing monthly debt obligations (EMIs: student loan, consumer durables, two-wheeler) in ₹
    # Around 8% to 28% of gross monthly income
    monthly_gross = annual_income / 12.0
    existing_debt_ratio = np.random.beta(a=2.2, b=6.5, size=num_samples) * 0.42
    existing_monthly_debt = np.round(monthly_gross * existing_debt_ratio, -2)
    
    # 3. Loan Characteristics (INR / ₹)
    loan_intent_choices = [
        "Personal", 
        "Education / Higher Studies", 
        "Two-Wheeler / Vehicle", 
        "Debt Consolidation", 
        "MSME / Small Business", 
        "Home Improvement", 
        "Medical / Emergency"
    ]
    loan_intent_probs = [0.26, 0.20, 0.18, 0.14, 0.12, 0.06, 0.04]
    loan_intent = np.random.choice(loan_intent_choices, size=num_samples, p=loan_intent_probs)
    
    home_ownership_choices = ["Rented", "Self-Owned", "Ancestral / Family Owned", "Company Provided"]
    home_ownership_probs = [0.46, 0.18, 0.30, 0.06]
    home_ownership = np.random.choice(home_ownership_choices, size=num_samples, p=home_ownership_probs)
    
    # Requested loan amount in ₹ (proportional to income, from ₹50,000 to ₹18,00,000)
    loan_multiplier = np.random.uniform(0.12, 0.48, size=num_samples)
    loan_amount = np.round(np.clip(annual_income * loan_multiplier, 50000, 1800000), -3)
    
    loan_term_choices = [12, 24, 36, 48, 60, 72]
    loan_term_probs = [0.12, 0.28, 0.35, 0.13, 0.08, 0.04]
    loan_term_months = np.random.choice(loan_term_choices, size=num_samples, p=loan_term_probs)
    
    # Base retail interest rate: 10.5% to 22.0% per annum (standard Indian unsecured/retail rate)
    interest_rate = np.round(np.random.normal(loc=13.8, scale=2.8, size=num_samples), 2)
    interest_rate = np.clip(interest_rate, 9.5, 23.5)
    
    # 4. Alternative & Relationship Banking Features (Thin-File First-Timer Signals)
    # Liquid savings / bank deposit balance in ₹
    savings_ratio = np.random.exponential(scale=0.22, size=num_samples)
    savings_balance = np.round(np.clip(annual_income * savings_ratio, 15000, 900000), -3)
    
    # Alternative track score: Account Aggregator / UPI recurring payments & utility score (30 - 100)
    rent_utility_payment_score = np.clip(
        np.random.normal(loc=73, scale=14, size=num_samples).astype(int), 30, 100
    )
    
    # Has co-signer / parent guarantor (frequent in first-time Indian retail loans)
    has_cosigner_prob = np.where(age < 26, 0.52, 0.24)
    has_cosigner = (np.random.uniform(0, 1, size=num_samples) < has_cosigner_prob).astype(int)
    
    # Has collateral pledged (FD lien, vehicle title, gold deposit)
    collateral_choices = ["None", "Vehicle", "Fixed Deposit Lien", "Gold Deposit"]
    collateral_probs = [0.60, 0.20, 0.12, 0.08]
    has_collateral = np.random.choice(collateral_choices, size=num_samples, p=collateral_probs)
    
    # Recent credit inquiries in last 6 months (Poisson)
    num_recent_credit_inquiries = np.clip(np.random.poisson(lam=1.3, size=num_samples), 0, 8)
    
    # Bank overdrafts / ECS bounce events in last 12 months
    num_bank_overdrafts_12m = np.clip(np.random.poisson(lam=0.5, size=num_samples), 0, 6)
    
    # Relationship tenure with the bank (months)
    bank_tenure_months = np.clip(
        np.random.gamma(shape=2.6, scale=12.0, size=num_samples).astype(int), 3, 130
    )
    
    # Calculate Monthly EMI and Debt-To-Income (FOIR)
    monthly_installment = calculate_amortized_installment(loan_amount, interest_rate, loan_term_months)
    total_monthly_debt = existing_monthly_debt + monthly_installment
    dti_ratio = total_monthly_debt / monthly_gross
    loan_to_income = loan_amount / annual_income
    
    # 5. Latent Risk Model: Combining Linear Risk + Non-Linear Interactions
    # Base intercept targeting ~17.5% default rate for thin-file first-timers
    z = -0.75
    
    # Linear Risk Factors
    z += 2.9 * (dti_ratio - 0.36)                      # High FOIR / DTI is primary risk factor
    z += 1.7 * (loan_to_income - 0.26)                 # High loan sizing relative to annual capacity
    z += 0.07 * (interest_rate - 13.5)                 # Interest rate servicing friction
    z += 0.24 * num_recent_credit_inquiries            # Credit hunger signal
    z += 0.40 * num_bank_overdrafts_12m                # Cash flow mismanagement / NACH bounces
    z -= 0.024 * (rent_utility_payment_score - 72)     # Consistent bill/utility payment habit
    z -= 0.007 * (bank_tenure_months - 24)             # Depository banking loyalty
    z -= 0.72 * has_cosigner                           # Solvent guarantor / parent co-signer
    z -= 0.48 * (has_collateral != "None").astype(int) # Asset backing (FD lien, vehicle, gold)
    z -= 0.55 * np.log1p(savings_balance / 100000.0)   # Savings liquidity buffer (per Lakh INR)
    
    # Non-Linear Interaction Rules (Tree-exploitable patterns)
    # Rule 1: FOIR / DTI Cliff with Thin Liquid Buffer
    dti_cliff = (dti_ratio > 0.45) & (savings_balance < (2.0 * monthly_installment))
    z += 1.40 * dti_cliff.astype(float)
    
    # Rule 2: Inexperienced MSME / Small Business without guarantor
    speculative_msme = (loan_intent == "MSME / Small Business") & (employment_years < 1.5) & (has_cosigner == 0)
    z += 1.20 * speculative_msme.astype(float)
    
    # Rule 3: Liquidity Stress: Multiple NACH/ECS bounces + Multi-Inquiries
    liquidity_crunch = (num_bank_overdrafts_12m >= 2) & (num_recent_credit_inquiries >= 3)
    z += 1.30 * liquidity_crunch.astype(float)
    
    # Rule 4: Fortress Borrower (Virtually immune to default)
    fortress = (has_cosigner == 1) & (rent_utility_payment_score >= 82) & (savings_balance > loan_amount * 0.35)
    z -= 1.75 * fortress.astype(float)
    
    # Convert log-odds to default probability via logistic sigmoid function
    default_prob = 1.0 / (1.0 + np.exp(-z))
    default_prob = np.clip(default_prob, 0.01, 0.98)
    
    # Bernoulli trial to generate actual ground-truth binary default label
    loan_default = (np.random.uniform(0, 1, size=num_samples) < default_prob).astype(int)
    
    # Format Applicant IDs with Indian prefix
    applicant_ids = [f"IN-APP-{10000 + i}" for i in range(num_samples)]
    
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
    """Generates Indian retail loan applicant dataset and saves it to CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df = generate_first_time_borrower_dataset(num_samples=num_samples)
    df.to_csv(filepath, index=False)
    print(f"Dataset generated successfully at: {filepath}")
    print(f"Total applicants: {len(df)}")
    print(f"Default rate: {df['loan_default'].mean():.2%}")
    print(f"Mean Annual Income: {format_inr(df['annual_income'].mean())}")
    print(f"Mean Loan Requested: {format_inr(df['loan_amount'].mean())}")
    return df


if __name__ == "__main__":
    generate_and_save_dataset()
