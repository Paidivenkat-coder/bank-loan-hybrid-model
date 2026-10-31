"""
Master Pipeline Orchestrator for Bank Loan Default Hybrid Model System.
Executes end-to-end workflow:
1. Synthetic / Anonymized Data Generation
2. Preprocessing & Stratified Train/Test Partitioning
3. Model Training (Logistic Regression, Decision Tree, Hybrid Model, Stacking)
4. Comprehensive Credit Risk Metric Evaluation
5. Report & Diagnostic Visualizations Generation
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    RAW_DATA_PATH, TARGET_COLUMN
)
from src.data_generator import generate_and_save_dataset
from src.preprocessing import run_preprocessing_pipeline
from src.models import train_and_save_all_models
from src.evaluate import run_full_evaluation


def run_complete_pipeline():
    print("=================================================================")
    print("  BANK LOAN DEFAULT HYBRID MODEL: END-TO-END WORKFLOW PIPELINE   ")
    print("=================================================================")
    
    # 1. Dataset Generation
    print("\n[Step 1/4] Checking / Generating Applicant Dataset...")
    if not RAW_DATA_PATH.exists():
        generate_and_save_dataset()
    else:
        print(f"Using existing dataset: {RAW_DATA_PATH}")
        
    # 2. Preprocessing & Partitioning
    print("\n[Step 2/4] Running Preprocessing & Feature Engineering...")
    train_df, test_df, preprocessor = run_preprocessing_pipeline()
    X_train = preprocessor.transform(train_df)
    y_train = train_df[TARGET_COLUMN].values
    
    # 3. Model Training
    print("\n[Step 3/4] Training Machine Learning Models...")
    models = train_and_save_all_models(X_train, y_train)
    
    # 4. Evaluation & Diagnostic Plots
    print("\n[Step 4/4] Evaluating Models & Rendering Visualizations...")
    metrics = run_full_evaluation()
    
    print("\n=================================================================")
    print("                       BENCHMARK SUMMARY                         ")
    print("=================================================================")
    print(f"{'Model':<35} | {'ROC-AUC':<8} | {'Gini':<8} | {'KS Stat':<8} | {'Recall':<8} | {'Loss Pts':<8}")
    print("-" * 85)
    for name, m in metrics.items():
        print(
            f"{name:<35} | {m['roc_auc']:<8.4f} | {m['gini_coefficient']:<8.4f} | "
            f"{m['ks_statistic']:<8.4f} | {m['recall_sensitivity']:<8.2%} | {m['financial_loss_cost']:<8.0f}"
        )
    print("=================================================================")
    print("Pipeline execution complete! Ready for Loan Officer Decision-Support Tool.")


if __name__ == "__main__":
    run_complete_pipeline()
