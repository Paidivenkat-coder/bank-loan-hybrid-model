"""
Preprocessing and Feature Engineering pipeline for Bank Loan Default Hybrid Model.
Handles feature transformations, encoding, scaling, and train-test splitting.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import sys

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    RAW_DATA_PATH, PROCESSED_DATA_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH,
    PREPROCESSOR_PATH, RANDOM_SEED, NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    TARGET_COLUMN
)


def compute_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes banking credit metrics:
    - Monthly amortized installment
    - Debt-to-Income (DTI) ratio
    - Loan-to-Income (LTI) ratio
    - Savings-to-Loan ratio
    - Net monthly surplus cash flow
    """
    df = df.copy()
    
    # Monthly gross income
    monthly_gross = np.maximum(df["annual_income"] / 12.0, 1.0)
    
    # Monthly amortized loan installment
    monthly_rate = (df["interest_rate"] / 100.0) / 12.0
    monthly_rate = np.where(monthly_rate == 0, 1e-6, monthly_rate)
    terms = df["loan_term_months"]
    numerator = monthly_rate * ((1.0 + monthly_rate) ** terms)
    denominator = ((1.0 + monthly_rate) ** terms) - 1.0
    installment = df["loan_amount"] * (numerator / denominator)
    df["monthly_installment"] = np.round(installment, 2)
    
    # Total monthly debt obligations
    total_monthly_debt = df["existing_monthly_debt"] + df["monthly_installment"]
    df["debt_to_income_ratio"] = np.round(total_monthly_debt / monthly_gross, 4)
    df["loan_to_income_ratio"] = np.round(df["loan_amount"] / np.maximum(df["annual_income"], 1.0), 4)
    df["savings_to_loan_ratio"] = np.round(df["savings_balance"] / np.maximum(df["loan_amount"], 1.0), 4)
    df["net_monthly_surplus"] = np.round(monthly_gross - total_monthly_debt, 2)
    
    return df


class LoanDataPreprocessor:
    """
    Production-ready preprocessor encapsulating:
    1. Financial ratio engineering
    2. Numerical imputation and standardization
    3. Categorical imputation and one-hot encoding
    4. Exact feature name mapping for model explainability
    """
    def __init__(self):
        self.num_cols = list(NUMERICAL_FEATURES) + [
            "monthly_installment", "debt_to_income_ratio", 
            "loan_to_income_ratio", "savings_to_loan_ratio", 
            "net_monthly_surplus"
        ]
        self.cat_cols = list(CATEGORICAL_FEATURES)
        self.transformer: ColumnTransformer = None
        self.feature_names_: list[str] = []
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame):
        df_engineered = compute_financial_ratios(df)
        
        num_transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.num_cols)
            ]
        )
        
        # Build composite ColumnTransformer
        self.transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_cols)
            ]
        )
        
        self.transformer.fit(df_engineered)
        
        # Extract feature names
        cat_encoder = self.transformer.named_transformers_["cat"]
        encoded_cat_names = cat_encoder.get_feature_names_out(self.cat_cols)
        self.feature_names_ = list(self.num_cols) + list(encoded_cat_names)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before calling transform().")
        df_engineered = compute_financial_ratios(df)
        return self.transformer.transform(df_engineered)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def get_feature_names(self) -> list[str]:
        return self.feature_names_


def run_preprocessing_pipeline(
    raw_path: Path = RAW_DATA_PATH,
    test_size: float = 0.20,
    random_seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame, LoanDataPreprocessor]:
    """
    Loads raw applicant data, applies stratified split, fits preprocessor,
    and persists datasets and preprocessor.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if not raw_path.exists():
        from src.data_generator import generate_and_save_dataset
        generate_and_save_dataset(raw_path)
        
    df = pd.read_csv(raw_path)
    
    # Stratified Split on target
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_seed, stratify=df[TARGET_COLUMN]
    )
    
    # Save raw train & test splits
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    
    # Fit preprocessor on train only
    preprocessor = LoanDataPreprocessor()
    preprocessor.fit(train_df)
    
    # Save preprocessor artifact
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    
    print(f"Train split saved to: {TRAIN_DATA_PATH} ({len(train_df)} records)")
    print(f"Test split saved to: {TEST_DATA_PATH} ({len(test_df)} records)")
    print(f"Preprocessor saved to: {PREPROCESSOR_PATH} ({len(preprocessor.get_feature_names())} features)")
    
    return train_df, test_df, preprocessor


if __name__ == "__main__":
    run_preprocessing_pipeline()
