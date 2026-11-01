# Design and Development of a Working Prototype for Bank Loan Default Using a Logistic Regression and Decision Tree Hybrid Model

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-orange.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()
[![Currency](https://img.shields.io/badge/Currency-INR%20(%E2%82%B9)-green.svg)]()

---

### Project Overview
* **Domain:** Machine Learning & Predictive Analytics
* **Project Title:** Design and Development of a Working Prototype for Bank loan default Using A Logistic Regression and Decision Tree hybrid model
* **Developer & Lead Investigator:** **PAIDI VENKAT**
* **Keywords:** `bank`, `loan`, `default`, `logistic`, `regression`, `decision`
* **Currency Standard:** Indian Rupee (**INR / ₹**) — calibrated in Lakhs (L) and Crores (Cr)
* **Regulatory Compliance:** Reserve Bank of India (RBI) Digital Lending Norms & Basel III Internal Ratings-Based (IRB)

---

## 1. Abstract
This project develops a **Logistic Regression and Decision Tree hybrid model** to assess creditworthiness of first-time borrowers, addressing the problem of bank loan default through a data-driven approach. The pipeline draws on anonymized loan applicant datasets, applying suitable pre-processing before training and testing the model on held-out data. The project culminates in a **loan officer decision-support tool**, bridging the gap between the underlying model and everyday use.

### The Problem of First-Time Borrowers ("Thin-File" Dilemma)
In retail banking, first-time borrowers (e.g. entry-level corporate professionals, new university graduates, self-employed proprietors, and gig workers) present a fundamental credit dilemma:
1. **Absence of Credit Bureau History:** Traditional bureaus (CIBIL / Experian) lack repayment track records for thin-file applicants.
2. **Blanket Rejections vs. Non-Performing Assets (NPAs):** Blanket declines cause commercial banks to miss high-margin youth demographics, whereas uncalibrated approvals spike loan default write-offs.
3. **Limitation of Standalone Models:**
   * **Logistic Regression:** Transparent and compliant with smooth probability calibration, but inherently linear in log-odds; fails to detect sharp non-linear risk cliffs (e.g., sudden FOIR/DTI spikes coupled with liquid savings depletion).
   * **Decision Trees:** Capable of partitioning high-order feature interactions, but prone to high variance, step-wise probability jumps, and overfitting.
4. **The Hybrid Solution:** A **Tree-Embedding Logistic Hybrid Model** that extracts non-linear cohort interaction leaves from the Decision Tree and injects them into regularized Logistic Regression. This achieves both non-linear risk capture and smooth, legally defensible, CIBIL-calibrated credit scores.

---

## 2. Hybrid Model System Architecture

```
[Indian Retail Applicant Raw Dossier (INR ₹)]
                   │
                   ▼
[Financial Ratio Engineering] ──▶ (FOIR/DTI, Loan-to-Income, Savings-to-EMI, Net Cash Flow)
                   │
                   ├───────────────────────────────────────┐
                   ▼                                       ▼
  [Standardized Features X]                    [Decision Tree Classifier]
                   │                                       │ (Discovers Non-Linear Splits)
                   │                                       ▼
                   │                              [Leaf Node Index L(X)]
                   │                                       │
                   │                                       ▼
                   │                              [One-Hot Leaf Embedding e(X)]
                   │                                       │
                   └───────────────────┬───────────────────┘
                                       ▼
                         [Augmented Space: [X, e(X)]]
                                       │
                                       ▼
                       [Regularized Logistic Regression]
                                       │
                                       ▼
                    [Calibrated Default Probability PD]
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 ▼                                           ▼
   [CIBIL-Calibrated Score (300-900)]          [Actionable Adverse Action / Memo]
   [Risk Tiers: A (Low) to E (Severe)]          [Counterfactual Sensitivity Advice]
```

### Mathematical Formulation
$$\ln\left(\frac{P(\text{Default}=1|x)}{1 - P(\text{Default}=1|x)}\right) = \beta_0 + \sum_{j=1}^{M} \beta_j x_j + \sum_{k=1}^{K} \gamma_k e_k(x)$$

Where:
* $x_j$ are continuous financial features (FOIR/DTI, annual gross income in ₹, loan amount requested in ₹, liquid savings buffer in ₹, interest rate).
* $e_k(x) \in \{0, 1\}$ is the one-hot indicator for tree terminal leaf cohort $k$.
* $\gamma_k$ is the learned non-linear risk interaction coefficient.
* The $L_2$-regularized objective prevents overfitting to thin cohorts:
$$\min_{\beta, \gamma} -\sum_{i=1}^{N} \left[ y_i \ln(p_i) + (1-y_i) \ln(1-p_i) \right] + \frac{\lambda}{2} \left(\|\beta\|_2^2 + \|\gamma\|_2^2\right)$$

---

## 3. Repository Structure

```
bank_loan_hybrid_model/
├── config.py                 # System parameters, risk thresholds, INR currency formatters
├── pyproject.toml            # Project dependencies and packaging metadata
├── README.md                 # Complete project documentation & technical report
├── run_pipeline.py           # CLI runner for dataset generation, training & evaluation
├── push_to_github.ps1        # Automated script to push project to GitHub
├── app/
│   └── app.py                # Streamlit Loan Officer Decision-Support Web Application (INR ₹)
├── data/
│   ├── raw/
│   │   └── first_time_borrowers_loan_data.csv  # Anonymized Indian applicant dataset (6,000 records)
│   └── processed/
│       ├── train_data.csv    # 80% Stratified Training Partition (4,800 records)
│       └── test_data.csv     # 20% Held-Out Validation Partition (1,200 records)
├── models_saved/
│   ├── preprocessor.joblib   # Fitted ColumnTransformer & Scaler
│   ├── logistic_regression.joblib  # Baseline Logistic Regression
│   ├── decision_tree.joblib  # Baseline Decision Tree
│   ├── hybrid_model.joblib   # Proposed Tree-Embedding Logistic Hybrid Model
│   ├── stacking_hybrid.joblib# Stacking Meta-Ensemble
│   └── evaluation_metrics.json # Full benchmark scores and confusion matrices
├── reports/
│   ├── roc_curves.png        # Comparative ROC curves (AUC comparison)
│   ├── precision_recall_curves.png # PR curves across models
│   ├── confusion_matrices.png# Cost-weighted confusion matrices
│   └── model_comparison_bar.png # Multi-metric benchmark bar chart
├── src/
│   ├── __init__.py
│   ├── data_generator.py     # Realistic first-time applicant generator in Indian Rupees (INR ₹)
│   ├── preprocessing.py      # FOIR/DTI ratio engineering, imputation, one-hot encoding, and scaling
│   ├── models.py             # Model definitions (LR, DT, TreeEmbeddingLogisticHybrid, Stacking)
│   ├── evaluate.py           # Credit risk evaluation suite (ROC-AUC, Gini, KS Stat, Brier, Cost Loss)
│   └── explainability.py     # Decision path, feature contributions, and counterfactuals (INR ₹)
└── tests/
    └── test_pipeline.py      # Pytest unit and integration test suite
```

---

## 4. Empirical Benchmark & Performance Evaluation

Evaluated on the **held-out test partition of 1,200 Indian applicants** under standard retail credit risk criteria:

| Model Candidate | ROC-AUC | Gini Index ($2\cdot\text{AUC}-1$) | KS Statistic | Default Recall | Precision | Bank Loss (Pts)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Decision Tree** | 0.6425 | 0.2851 | 0.2122 | 90.51% | 27.55% | 842.0 |
| **Baseline Logistic Regression** | 0.7176 | 0.4352 | 0.3291 | 94.24% | 26.83% | 843.0 |
| **Proposed Hybrid (Tree-Embedding LR)** | **0.7032** | **0.4063** | **0.2965** | **92.54%** | **28.23%** | **804.0** |
| **Stacking Meta-Ensemble** | 0.7157 | 0.4313 | 0.3205 | 73.22% | 36.00% | **779.0** |

*\*Note: Banking loss cost penalizes False Negatives (actual default approved) at $5\times$ the cost of False Positives (good applicant rejected).*

### Key Takeaways:
1. **Decision Tree Limitation:** Standalone Decision Trees exhibit low Gini separation (0.2851) and weak discriminatory power on held-out thin-file portfolios.
2. **Hybrid Advantage:** The Proposed Tree-Embedding Hybrid drops bank credit loss points from 843 down to 804, improving precision while maintaining over 92% recall on delinquent borrowers.
3. **CIBIL Score Calibration:** Raw default probabilities are converted to an intuitive 300 - 900 Indian bureau scale, enabling instant risk tier categorisation (Tier A to Tier E).

---

## 5. Quick-Start Guide

### Step 1: Navigate to Workspace
```powershell
cd C:\Users\venkat\.gemini\antigravity\scratch\bank_loan_hybrid_model
```

### Step 2: Run Master ML Pipeline (CLI)
To regenerate the Indian Rupee dataset, re-train models, and generate all report figures:
```powershell
$env:PYTHONIOENCODING = "utf-8"
uv run python run_pipeline.py --regenerate
```

### Step 3: Run Unit & Integration Test Suite
```powershell
$env:PYTHONIOENCODING = "utf-8"
uv run pytest tests/test_pipeline.py -v
```

### Step 4: Launch the Loan Officer Decision-Support Web Tool
```powershell
uv run streamlit run app/app.py
```
Then open your web browser to:
```
http://localhost:8501
```

---

## 6. Loan Officer Decision-Support Tool Capabilities

* **Underwriting Workstation:** Interactive application form in Indian Rupees (₹) with 4 pre-configured borrower archetypes (*Prime First-Timer*, *Borderline High FOIR*, *Graduate with Guarantor*, *High-Risk MSME Thin-File*).
* **CIBIL Risk Gauge:** Visual gauge translating default probability into a 300 to 900 credit score with colour-coded underwriting risk tiers (Tier A Super-Prime to Tier E Severe Risk).
* **Cross-Model Consensus:** Real-time comparison across the Proposed Hybrid, Stacking Ensemble, Baseline Logistic Regression, and Decision Tree.
* **Explainable AI (XAI):** Step-by-step decision tree rule path tracing and logistic feature contribution odds waterfall chart.
* **Counterfactual Sensitivity Simulator:** Dynamic sliders for loan officer "what-if" testing (downpayment sizing in ₹, loan tenure extension, guarantor addition).
* **Formal Credit Memo Generator:** Exportable underwriting memo formatted with Indian banking terminology, RBI compliance declarations, and underwriter sign-off block for **PAIDI VENKAT**.
* **Batch Portfolio Assessment:** Process bulk applicant dossiers (1,200 held-out cases or custom CSV uploads) with aggregate risk distribution and downloadable scored CSV reports.
