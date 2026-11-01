"""
Explainability, Credit Scoring, and Counterfactual Sensitivity Engine
for the Bank Loan Default Hybrid Decision Support Tool.
Tailored for Indian Retail Banking (INR / ₹ currency, CIBIL 300-900 calibration).
Author: PAIDI VENKAT
Project: Design and Development of a Working Prototype for Bank loan default Using A Logistic Regression and Decision Tree hybrid model
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.tree import _tree
import sys

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    CREDIT_SCORE_BASE, CREDIT_SCORE_PDO, RISK_THRESHOLDS,
    DECISION_APPROVAL_THRESHOLD, DECISION_REVIEW_THRESHOLD,
    format_inr, format_inr_full
)


def probability_to_credit_score(default_prob: float) -> int:
    """
    Translates default probability into standard Indian Credit Bureau score (CIBIL-like 300 - 900)
    using logarithmic odds calibration:
    Score = Base - (PDO / ln(2)) * ln(p / (1 - p))
    Calibrated so that:
    - 5% default rate maps to ~795 (Super-Prime)
    - 15% default rate maps to ~730 (Prime)
    - 25% default rate maps to ~675 (Near-Prime)
    - 50% default rate maps to ~600 (Sub-Prime)
    - 80% default rate maps to ~480 (Severe Risk)
    """
    p = np.clip(default_prob, 0.005, 0.995)
    odds = p / (1.0 - p)
    factor = CREDIT_SCORE_PDO / np.log(2.0)
    score = CREDIT_SCORE_BASE - factor * np.log(odds)
    return int(np.clip(np.round(score), 300, 900))


def get_risk_tier_and_decision(default_prob: float) -> tuple[str, str, str, str]:
    """
    Maps probability of default to Tier, Underwriting Recommendation, Color, and Action Description.
    """
    if default_prob < RISK_THRESHOLDS["TIER_A_LOW"]:
        return (
            "Tier A (Super-Prime)",
            "Approved (Fast-Track STP)",
            "#10b981", # Green
            "Borrower demonstrates exceptional repayment capacity, low FOIR/DTI, and solid liquid reserves."
        )
    elif default_prob < RISK_THRESHOLDS["TIER_B_MODERATE"]:
        return (
            "Tier B (Standard Prime)",
            "Approved",
            "#059669", # Emerald
            "Applicant meets standard Indian retail credit criteria. Moderate to low default hazard."
        )
    elif default_prob < RISK_THRESHOLDS["TIER_C_BORDERLINE"]:
        return (
            "Tier C (Near-Prime / Borderline)",
            "Manual Review / Conditional Approval",
            "#f59e0b", # Amber
            "Applicant has elevated FOIR or thin liquidity cushion. Underwriter verification or mitigating conditions recommended."
        )
    elif default_prob < RISK_THRESHOLDS["TIER_D_HIGH"]:
        return (
            "Tier D (Sub-Prime / High Risk)",
            "Decline / Substantial Mitigant Required",
            "#ea580c", # Orange-Red
            "High default hazard. Approval requires significant collateral backing or a creditworthy guarantor/co-signer."
        )
    else:
        return (
            "Tier E (Severe Risk)",
            "Declined",
            "#dc2626", # Red
            "Default probability exceeds bank risk appetite under RBI prudential underwriting guidelines."
        )


def extract_decision_tree_path(tree_model, sample_feature_array: np.ndarray, feature_names: list[str]) -> list[dict]:
    """
    Traces the exact decision path of a single applicant down the Decision Tree.
    Returns step-by-step branching rules and threshold comparisons.
    """
    tree_ = tree_model.tree_
    feature = tree_.feature
    threshold = tree_.threshold
    
    node_indicator = tree_model.decision_path(sample_feature_array)
    leaf_id = tree_model.apply(sample_feature_array)[0]
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    
    rules = []
    for node_id in node_index:
        if node_id == leaf_id:
            val = tree_.value[node_id][0]
            total_samples = np.sum(val)
            default_prob_node = val[1] / total_samples if total_samples > 0 else 0.5
            rules.append({
                "type": "leaf",
                "node_id": int(node_id),
                "cohort_default_rate": round(float(default_prob_node), 4),
                "samples_in_cohort": int(total_samples),
                "text": f"Reached Terminal Risk Node #{node_id} (Cohort Historical Default Rate: {default_prob_node:.1%})"
            })
        else:
            feat_idx = feature[node_id]
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feat_{feat_idx}"
            thresh_val = threshold[node_id]
            actual_val = sample_feature_array[0, feat_idx]
            passed_left = actual_val <= thresh_val
            
            clean_feat = feat_name.replace("_", " ").title()
            op = "<=" if passed_left else ">"
            status = "Satisfies" if passed_left else "Exceeds"
            
            rules.append({
                "type": "split",
                "node_id": int(node_id),
                "feature": clean_feat,
                "raw_feature_name": feat_name,
                "actual_normalized_val": round(float(actual_val), 3),
                "threshold_val": round(float(thresh_val), 3),
                "comparison": f"{clean_feat} {op} {thresh_val:.2f}",
                "status": f"{status} threshold (Applicant normalized score: {actual_val:.2f})"
            })
    return rules


def compute_feature_contributions(
    logistic_model,
    sample_feature_array: np.ndarray,
    feature_names: list[str],
    top_n: int = 8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates log-odds impact for each feature on the applicant's default risk:
    Contribution = coefficient_j * value_j
    Positive contribution increases default risk; negative reduces default risk.
    """
    coefs = logistic_model.coef_[0][:len(feature_names)]
    vals = sample_feature_array[0][:len(feature_names)]
    contributions = coefs * vals
    
    df_contrib = pd.DataFrame({
        "Feature": [f.replace("_", " ").title() for f in feature_names],
        "Raw_Feature": feature_names,
        "Coefficient": coefs,
        "Value": vals,
        "Contribution": contributions,
        "Odds_Multiplier": np.exp(contributions)
    })
    
    # Sort into risk increasing and risk mitigating
    risk_increasing = df_contrib[df_contrib["Contribution"] > 0].sort_values("Contribution", ascending=False).head(top_n)
    risk_mitigating = df_contrib[df_contrib["Contribution"] < 0].sort_values("Contribution", ascending=True).head(top_n)
    
    return risk_increasing, risk_mitigating


def generate_counterfactual_recommendations(
    raw_applicant_dict: dict,
    preprocessor,
    hybrid_model,
    target_max_prob: float = DECISION_APPROVAL_THRESHOLD
) -> list[dict]:
    """
    Evaluates actionable parameter modifications (downpayment / loan reduction in INR,
    co-signer addition, term elongation) to help a declined or borderline applicant qualify.
    """
    current_df = pd.DataFrame([raw_applicant_dict])
    current_feat = preprocessor.transform(current_df)
    current_prob = float(hybrid_model.predict_proba(current_feat)[0, 1])
    
    if current_prob <= target_max_prob:
        return [{
            "action": "Already Qualifies",
            "detail": f"Applicant default probability is {current_prob:.1%}, which satisfies the approval threshold ({target_max_prob:.1%}).",
            "new_probability": current_prob,
            "reduction_pct": 0.0,
            "feasibility": "Ready for Underwriting"
        }]
    
    recommendations = []
    
    # Scenario 1: Adding a co-signer / parent guarantor
    if raw_applicant_dict.get("has_cosigner", 0) == 0:
        alt_dict = dict(raw_applicant_dict)
        alt_dict["has_cosigner"] = 1
        alt_df = pd.DataFrame([alt_dict])
        p_new = float(hybrid_model.predict_proba(preprocessor.transform(alt_df))[0, 1])
        recommendations.append({
            "action": "Add Solvent Co-Signer / Guarantor",
            "detail": "Adding an earning parent or creditworthy co-signer mitigates first-time thin-file risk.",
            "new_probability": p_new,
            "reduction_pct": round((current_prob - p_new) * 100, 1),
            "qualifies": p_new <= target_max_prob,
            "feasibility": "High"
        })
        
    # Scenario 2: Reducing requested loan amount (Borrower Downpayment) in INR
    curr_amount = float(raw_applicant_dict["loan_amount"])
    for discount in [0.15, 0.25, 0.35]:
        reduced_amt = np.round(curr_amount * (1.0 - discount), -3)
        alt_dict = dict(raw_applicant_dict)
        alt_dict["loan_amount"] = reduced_amt
        alt_df = pd.DataFrame([alt_dict])
        p_new = float(hybrid_model.predict_proba(preprocessor.transform(alt_df))[0, 1])
        if p_new < current_prob:
            diff_inr = curr_amount - reduced_amt
            recommendations.append({
                "action": f"Lower Loan Amount by {format_inr_full(diff_inr)} ({discount:.0%})",
                "detail": f"Reduce loan amount from {format_inr_full(curr_amount)} to {format_inr_full(reduced_amt)} (e.g., via borrower upfront margin money).",
                "new_probability": p_new,
                "reduction_pct": round((current_prob - p_new) * 100, 1),
                "qualifies": p_new <= target_max_prob,
                "feasibility": "Medium"
            })
            if p_new <= target_max_prob:
                break
                
    # Scenario 3: Extending loan term to lower monthly EMI and FOIR
    curr_term = int(raw_applicant_dict["loan_term_months"])
    if curr_term < 60:
        next_term = 48 if curr_term < 48 else 60
        alt_dict = dict(raw_applicant_dict)
        alt_dict["loan_term_months"] = next_term
        alt_df = pd.DataFrame([alt_dict])
        p_new = float(hybrid_model.predict_proba(preprocessor.transform(alt_df))[0, 1])
        recommendations.append({
            "action": f"Extend Repayment Tenure to {next_term} Months",
            "detail": f"Extending tenure from {curr_term} to {next_term} months reduces the monthly EMI burden and improves FOIR.",
            "new_probability": p_new,
            "reduction_pct": round((current_prob - p_new) * 100, 1),
            "qualifies": p_new <= target_max_prob,
            "feasibility": "High"
        })
        
    # Scenario 4: Pledging Collateral
    if raw_applicant_dict.get("has_collateral", "None") == "None":
        alt_dict = dict(raw_applicant_dict)
        alt_dict["has_collateral"] = "Fixed Deposit Lien"
        alt_df = pd.DataFrame([alt_dict])
        p_new = float(hybrid_model.predict_proba(preprocessor.transform(alt_df))[0, 1])
        recommendations.append({
            "action": "Pledge Asset Collateral (Fixed Deposit Lien or Vehicle)",
            "detail": "Securing the loan with a bank FD lien or hypothecation substantially enhances recovery rating.",
            "new_probability": p_new,
            "reduction_pct": round((current_prob - p_new) * 100, 1),
            "qualifies": p_new <= target_max_prob,
            "feasibility": "Medium"
        })
        
    # Sort recommendations by probability reduction
    recommendations.sort(key=lambda x: x.get("reduction_pct", 0), reverse=True)
    return recommendations
