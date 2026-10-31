"""
Comprehensive Model Evaluation suite for Bank Loan Default Prediction.
Calculates regulatory credit risk metrics (ROC-AUC, Gini, KS-Statistic, Brier Score, Cost Matrix)
and produces publication-grade comparative visualizations.
"""
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

# Ensure matplotlib cache is in a clean local directory
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent.parent / ".mplconfig")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss
)
from scipy.stats import ks_2samp
import sys

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    BASE_DIR, TEST_DATA_PATH, PREPROCESSOR_PATH, LOGISTIC_MODEL_PATH,
    DECISION_TREE_PATH, HYBRID_MODEL_PATH, STACKING_MODEL_PATH,
    METRICS_PATH, REPORTS_DIR, TARGET_COLUMN, COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE, DECISION_APPROVAL_THRESHOLD
)
from src.preprocessing import LoanDataPreprocessor
from src.models import TreeEmbeddingLogisticHybrid


def compute_ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Computes the Kolmogorov-Smirnov (KS) statistic between default and non-default distributions."""
    defaults = y_proba[y_true == 1]
    non_defaults = y_proba[y_true == 0]
    ks_stat, _ = ks_2samp(defaults, non_defaults)
    return float(ks_stat)


def evaluate_single_model(
    name: str,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = DECISION_APPROVAL_THRESHOLD
) -> dict:
    """Evaluates a single model across credit risk metrics."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    
    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    
    roc_auc = roc_auc_score(y_test, proba)
    gini = 2.0 * roc_auc - 1.0
    ks = compute_ks_statistic(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    
    # Financial Cost (Banking Cost Function)
    financial_cost = (fn * COST_FALSE_NEGATIVE) + (fp * COST_FALSE_POSITIVE)
    
    metrics = {
        "model_name": name,
        "roc_auc": round(float(roc_auc), 4),
        "gini_coefficient": round(float(gini), 4),
        "ks_statistic": round(float(ks), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall_sensitivity": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "specificity": round(float(tn / (tn + fp)), 4),
        "f1_score": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "brier_score": round(float(brier), 4),
        "financial_loss_cost": round(float(financial_cost), 1),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }
    return metrics


def run_full_evaluation():
    """Evaluates all candidate models, saves metrics JSON, and renders diagnostic plots."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load test data and preprocessor
    test_df = pd.read_csv(TEST_DATA_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    X_test = preprocessor.transform(test_df)
    y_test = test_df[TARGET_COLUMN].values
    
    # Load models
    models = {
        "Logistic Regression": joblib.load(LOGISTIC_MODEL_PATH),
        "Decision Tree": joblib.load(DECISION_TREE_PATH),
        "Hybrid Model (Tree-Embedding LR)": joblib.load(HYBRID_MODEL_PATH),
        "Stacking Ensemble": joblib.load(STACKING_MODEL_PATH)
    }
    
    results = {}
    probas = {}
    
    for name, model in models.items():
        metrics = evaluate_single_model(name, model, X_test, y_test)
        results[name] = metrics
        probas[name] = model.predict_proba(X_test)[:, 1]
    
    # Save Metrics JSON
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Metrics JSON saved to: {METRICS_PATH}")
    
    # Generate Plots
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # 1. ROC Curves
    plt.figure(figsize=(9, 6.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    for (name, prob), color in zip(probas.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc_val = results[name]["roc_auc"]
        gini_val = results[name]["gini_coefficient"]
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f}, Gini={gini_val:.3f})", color=color, linewidth=2.2)
        
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC=0.500)")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.title("ROC Curves Comparison - First-Time Borrower Default Models", fontsize=14, weight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curves.png", dpi=300)
    plt.close()
    
    # 2. Precision-Recall Curves
    plt.figure(figsize=(9, 6.5))
    for (name, prob), color in zip(probas.items(), colors):
        prec, rec, _ = precision_recall_curve(y_test, prob)
        pr_auc = results[name]["pr_auc"]
        plt.plot(rec, prec, label=f"{name} (PR-AUC={pr_auc:.3f})", color=color, linewidth=2.2)
        
    plt.xlabel("Recall (Defaults Caught)", fontsize=12)
    plt.ylabel("Precision (Accuracy of Default Predictions)", fontsize=12)
    plt.title("Precision-Recall Curves - Bank Loan Underwriting", fontsize=14, weight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "precision_recall_curves.png", dpi=300)
    plt.close()
    
    # 3. Confusion Matrix Grid
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()
    
    for i, (name, res) in enumerate(results.items()):
        cm = np.array([
            [res["confusion_matrix"]["tn"], res["confusion_matrix"]["fp"]],
            [res["confusion_matrix"]["fn"], res["confusion_matrix"]["tp"]]
        ])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[i],
            xticklabels=["Repaid (0)", "Default (1)"],
            yticklabels=["Repaid (0)", "Default (1)"],
            annot_kws={"size": 13, "weight": "bold"}
        )
        axes[i].set_title(f"{name}\nCost: {res['financial_loss_cost']:.0f} pts", fontsize=12, weight="bold")
        axes[i].set_xlabel("Predicted Class")
        axes[i].set_ylabel("Actual Ground Truth")
        
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrices.png", dpi=300)
    plt.close()
    
    # 4. Metrics Bar Chart Comparison
    metrics_to_plot = ["roc_auc", "gini_coefficient", "ks_statistic", "recall_sensitivity", "f1_score"]
    df_metrics = pd.DataFrame([
        {
            "Model": name,
            "Metric": metric.replace("_", " ").title(),
            "Value": res[metric]
        }
        for name, res in results.items()
        for metric in metrics_to_plot
    ])
    
    plt.figure(figsize=(12, 6.5))
    ax = sns.barplot(data=df_metrics, x="Metric", y="Value", hue="Model", palette="tab10")
    plt.title("Comparative Performance across Credit Underwriting Metrics", fontsize=14, weight="bold")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "model_comparison_bar.png", dpi=300)
    plt.close()
    
    print("All diagnostic figures successfully rendered to:", REPORTS_DIR)
    return results


if __name__ == "__main__":
    results = run_full_evaluation()
    print("\n=== MODEL BENCHMARK RESULTS ===")
    for model_name, m in results.items():
        print(f"\n[{model_name}]")
        print(f"  ROC-AUC: {m['roc_auc']:.4f} | Gini: {m['gini_coefficient']:.4f} | KS: {m['ks_statistic']:.4f}")
        print(f"  Recall (Defaults Detected): {m['recall_sensitivity']:.2%} | Precision: {m['precision']:.2%}")
        print(f"  Financial Loss Cost: {m['financial_loss_cost']} pts")
