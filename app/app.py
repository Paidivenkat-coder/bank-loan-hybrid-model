"""
Loan Officer Decision-Support Tool: Interactive Web Application
Built with Streamlit and Plotly for the Logistic Regression & Decision Tree Hybrid Model.
"""
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    PREPROCESSOR_PATH, LOGISTIC_MODEL_PATH, DECISION_TREE_PATH,
    HYBRID_MODEL_PATH, STACKING_MODEL_PATH, METRICS_PATH,
    REPORTS_DIR, TEST_DATA_PATH, DECISION_APPROVAL_THRESHOLD,
    DECISION_REVIEW_THRESHOLD
)
from src.preprocessing import LoanDataPreprocessor, compute_financial_ratios
from src.models import TreeEmbeddingLogisticHybrid
from src.explainability import (
    probability_to_credit_score, get_risk_tier_and_decision,
    extract_decision_tree_path, compute_feature_contributions,
    generate_counterfactual_recommendations
)

# Page Setup
st.set_page_config(
    page_title="Apex Horizon Bank | Credit Decision System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .decision-banner {
        padding: 18px 24px;
        border-radius: 8px;
        color: white;
        font-weight: 700;
        font-size: 1.4rem;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .rule-box {
        background-color: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 0 6px 6px 0;
        font-size: 0.92rem;
    }
    .badge-approved { background-color: #10b981; color: white; padding: 4px 10px; border-radius: 12px; }
    .badge-review { background-color: #f59e0b; color: white; padding: 4px 10px; border-radius: 12px; }
    .badge-declined { background-color: #ef4444; color: white; padding: 4px 10px; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system_artifacts():
    """Loads fitted preprocessor and trained models."""
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    lr_model = joblib.load(LOGISTIC_MODEL_PATH)
    dt_model = joblib.load(DECISION_TREE_PATH)
    hybrid_model = joblib.load(HYBRID_MODEL_PATH)
    stacking_model = joblib.load(STACKING_MODEL_PATH)
    
    metrics = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
            
    return preprocessor, lr_model, dt_model, hybrid_model, stacking_model, metrics


# Load Artifacts
try:
    preprocessor, lr_model, dt_model, hybrid_model, stacking_model, metrics = load_system_artifacts()
    artifacts_loaded = True
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please run `python run_pipeline.py` first.")
    artifacts_loaded = False


# Sidebar Configuration
st.sidebar.image("https://images.unsplash.com/photo-1541354329998-f4d9a9f9297f?w=400&auto=format&fit=crop&q=80", use_container_width=True)
st.sidebar.markdown("### 🏦 Apex Horizon Bank")
st.sidebar.markdown("**Retail Underwriting & Credit Risk**")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Navigation Mode",
    [
        "📋 Single Underwriting Workstation",
        "📂 Batch Portfolio Assessment",
        "📊 Model Benchmark & Analytics",
        "📖 Hybrid Architecture & Methodology"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**System Version:** v2.4 (Production Prototype)\n\n"
    "**Core Model:** Decision Tree & Logistic Regression Hybrid\n\n"
    "**Target Segment:** First-Time / Thin-File Borrowers\n\n"
    "**Compliance:** Basel III & Fair Lending Aligned"
)


# ==============================================================================
# TAB 1: SINGLE LOAN UNDERWRITING WORKSTATION
# ==============================================================================
if app_mode == "📋 Single Underwriting Workstation":
    st.markdown('<div class="main-title">Loan Officer Decision-Support Workstation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">AI-Assisted Credit Evaluation for First-Time Borrowers using the Logistic Regression & Decision Tree Hybrid Pipeline</div>',
        unsafe_allow_html=True
    )
    
    # Archetype Presets
    st.markdown("##### ⚡ Quick-Load Borrower Profiles:")
    col_arc1, col_arc2, col_arc3, col_arc4 = st.columns(4)
    
    archetypes = {
        "prime": {
            "name": "Sarah Chen", "age": 28, "education": "Bachelor", "employment_type": "Salaried",
            "employment_years": 4.5, "annual_income": 68000, "existing_monthly_debt": 350,
            "loan_amount": 12000, "loan_term_months": 36, "interest_rate": 10.5,
            "loan_intent": "Debt Consolidation", "home_ownership": "Rent", "savings_balance": 14000,
            "rent_utility_payment_score": 88, "has_cosigner": 0, "has_collateral": "None",
            "num_recent_credit_inquiries": 1, "num_bank_overdrafts_12m": 0, "bank_tenure_months": 36
        },
        "borderline": {
            "name": "David Miller", "age": 25, "education": "High School", "employment_type": "Contract",
            "employment_years": 1.8, "annual_income": 38000, "existing_monthly_debt": 520,
            "loan_amount": 16000, "loan_term_months": 36, "interest_rate": 15.8,
            "loan_intent": "Personal", "home_ownership": "Rent", "savings_balance": 1800,
            "rent_utility_payment_score": 68, "has_cosigner": 0, "has_collateral": "None",
            "num_recent_credit_inquiries": 3, "num_bank_overdrafts_12m": 1, "bank_tenure_months": 14
        },
        "grad": {
            "name": "Elena Rostova", "age": 23, "education": "Master", "employment_type": "Salaried",
            "employment_years": 0.8, "annual_income": 54000, "existing_monthly_debt": 280,
            "loan_amount": 10000, "loan_term_months": 36, "interest_rate": 11.2,
            "loan_intent": "Education", "home_ownership": "With Parents", "savings_balance": 5200,
            "rent_utility_payment_score": 82, "has_cosigner": 1, "has_collateral": "None",
            "num_recent_credit_inquiries": 1, "num_bank_overdrafts_12m": 0, "bank_tenure_months": 24
        },
        "highrisk": {
            "name": "Marcus Vance", "age": 31, "education": "Associate", "employment_type": "Freelance",
            "employment_years": 0.9, "annual_income": 32000, "existing_monthly_debt": 680,
            "loan_amount": 18000, "loan_term_months": 48, "interest_rate": 19.5,
            "loan_intent": "Small Business", "home_ownership": "Rent", "savings_balance": 750,
            "rent_utility_payment_score": 52, "has_cosigner": 0, "has_collateral": "None",
            "num_recent_credit_inquiries": 5, "num_bank_overdrafts_12m": 3, "bank_tenure_months": 8
        }
    }
    
    if "form_data" not in st.session_state:
        st.session_state["form_data"] = archetypes["prime"]
        
    with col_arc1:
        if st.button("🌟 Prime First-Timer", use_container_width=True):
            st.session_state["form_data"] = archetypes["prime"]
            st.rerun()
    with col_arc2:
        if st.button("⚖️ Borderline / High DTI", use_container_width=True):
            st.session_state["form_data"] = archetypes["borderline"]
            st.rerun()
    with col_arc3:
        if st.button("🎓 College Grad + Co-signer", use_container_width=True):
            st.session_state["form_data"] = archetypes["grad"]
            st.rerun()
    with col_arc4:
        if st.button("⚠️ High-Risk Thin File", use_container_width=True):
            st.session_state["form_data"] = archetypes["highrisk"]
            st.rerun()
            
    current = st.session_state["form_data"]
    
    st.markdown("---")
    
    # Input Form Layout
    with st.form("applicant_evaluation_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 Demographics & Stability")
            applicant_name = st.text_input("Applicant Full Name", value=current.get("name", "John Doe"))
            age = st.slider("Applicant Age", 21, 65, int(current.get("age", 28)))
            education = st.selectbox(
                "Highest Education Level",
                ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
                index=["High School", "Associate", "Bachelor", "Master", "Doctorate"].index(current.get("education", "Bachelor"))
            )
            employment_type = st.selectbox(
                "Employment Type",
                ["Salaried", "Self-Employed", "Contract", "Freelance"],
                index=["Salaried", "Self-Employed", "Contract", "Freelance"].index(current.get("employment_type", "Salaried"))
            )
            employment_years = st.slider(
                "Tenure in Current Employment (Years)", 0.0, 15.0, float(current.get("employment_years", 3.0)), step=0.5
            )
            home_ownership = st.selectbox(
                "Residential Status",
                ["Rent", "Own", "Mortgage", "With Parents"],
                index=["Rent", "Own", "Mortgage", "With Parents"].index(current.get("home_ownership", "Rent"))
            )
            
        with col2:
            st.markdown("#### 💰 Financial Profile")
            annual_income = st.number_input("Annual Gross Income ($)", min_value=12000, max_value=250000, value=int(current.get("annual_income", 55000)), step=1000)
            existing_monthly_debt = st.number_input("Existing Monthly Debt Payments ($)", min_value=0, max_value=15000, value=int(current.get("existing_monthly_debt", 400)), step=50)
            savings_balance = st.number_input("Liquid Savings / Emergency Buffer ($)", min_value=0, max_value=100000, value=int(current.get("savings_balance", 8000)), step=500)
            rent_utility_payment_score = st.slider(
                "Utility / Rent Alternative Credit Score (30-100)", 30, 100, int(current.get("rent_utility_payment_score", 75)),
                help="Derived from telecom and utility recurring payment reliability."
            )
            bank_tenure_months = st.slider(
                "Account Relationship Tenure (Months)", 1, 140, int(current.get("bank_tenure_months", 24))
            )
            
        with col3:
            st.markdown("#### 📑 Requested Loan Details")
            loan_amount = st.number_input("Loan Amount Requested ($)", min_value=1000, max_value=60000, value=int(current.get("loan_amount", 12000)), step=500)
            loan_term_months = st.selectbox(
                "Loan Term (Months)", [12, 24, 36, 48, 60],
                index=[12, 24, 36, 48, 60].index(int(current.get("loan_term_months", 36)))
            )
            interest_rate = st.slider(
                "Offered Annual Interest Rate (%)", 6.0, 26.0, float(current.get("interest_rate", 12.5)), step=0.25
            )
            loan_intent = st.selectbox(
                "Loan Intent / Purpose",
                ["Personal", "Education", "Debt Consolidation", "Small Business", "Home Improvement", "Medical"],
                index=["Personal", "Education", "Debt Consolidation", "Small Business", "Home Improvement", "Medical"].index(current.get("loan_intent", "Personal"))
            )
            col_mit1, col_mit2 = st.columns(2)
            with col_mit1:
                has_cosigner = st.selectbox("Has Co-Signer?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=int(current.get("has_cosigner", 0)))
            with col_mit2:
                has_collateral = st.selectbox("Collateral Pledged", ["None", "Vehicle", "Fixed Deposit"], index=["None", "Vehicle", "Fixed Deposit"].index(current.get("has_collateral", "None")))
                
            num_recent_credit_inquiries = st.slider("Recent Inquiries (6 Mo)", 0, 8, int(current.get("num_recent_credit_inquiries", 1)))
            num_bank_overdrafts_12m = st.slider("Bank Overdrafts (12 Mo)", 0, 6, int(current.get("num_bank_overdrafts_12m", 0)))
            
        submit_btn = st.form_submit_button("🔍 Run Hybrid Credit Underwriting Assessment", use_container_width=True)
        
    # Applicant Dictionary Construction
    applicant_dict = {
        "age": age, "education": education, "employment_type": employment_type,
        "employment_years": employment_years, "annual_income": annual_income,
        "existing_monthly_debt": existing_monthly_debt, "loan_amount": loan_amount,
        "loan_term_months": loan_term_months, "interest_rate": interest_rate,
        "loan_intent": loan_intent, "home_ownership": home_ownership,
        "savings_balance": savings_balance, "rent_utility_payment_score": rent_utility_payment_score,
        "has_cosigner": has_cosigner, "has_collateral": has_collateral,
        "num_recent_credit_inquiries": num_recent_credit_inquiries,
        "num_bank_overdrafts_12m": num_bank_overdrafts_12m,
        "bank_tenure_months": bank_tenure_months
    }
    
    # Process Applicant through Pipeline
    applicant_df = pd.DataFrame([applicant_dict])
    applicant_feats = preprocessor.transform(applicant_df)
    
    # Model Predictions
    prob_hybrid = float(hybrid_model.predict_proba(applicant_feats)[0, 1])
    prob_lr = float(lr_model.predict_proba(applicant_feats)[0, 1])
    prob_dt = float(dt_model.predict_proba(applicant_feats)[0, 1])
    prob_stack = float(stacking_model.predict_proba(applicant_feats)[0, 1])
    
    # Underwriting Metrics
    credit_score = probability_to_credit_score(prob_hybrid)
    tier_name, decision_text, decision_color, decision_desc = get_risk_tier_and_decision(prob_hybrid)
    
    # Financial Ratios
    df_calc = compute_financial_ratios(applicant_df).iloc[0]
    
    st.markdown("---")
    
    # UNDERWRITING DECISION BANNER
    st.markdown(
        f'<div class="decision-banner" style="background-color: {decision_color};">'
        f'{decision_text.upper()} &nbsp;|&nbsp; {tier_name} &nbsp;|&nbsp; Estimated Default Risk: {prob_hybrid:.1%}'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # Primary KPI Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Credit Bureau Score", f"{credit_score}", f"{'Prime' if credit_score >= 680 else 'Sub-Prime'}")
    with m2:
        st.metric("Hybrid Default Probability", f"{prob_hybrid:.1%}", f"{round((prob_hybrid - 0.18)*100, 1):+0.1f}% vs Portfolio")
    with m3:
        st.metric("Debt-to-Income (DTI)", f"{df_calc['debt_to_income_ratio']:.1%}", "Acceptable" if df_calc['debt_to_income_ratio'] <= 0.40 else "High Leverage", delta_color="inverse")
    with m4:
        st.metric("Monthly Installment", f"${df_calc['monthly_installment']:,.0f}", f"Term: {loan_term_months}m")
    with m5:
        st.metric("Net Monthly Cash Flow", f"${df_calc['net_monthly_surplus']:,.0f}", "Healthy" if df_calc['net_monthly_surplus'] > 500 else "Tight Buffer")
        
    st.markdown("---")
    
    # Two Columns: Gauges & Model Comparison vs Detailed Explainability
    left_col, right_col = st.columns([1, 1.2])
    
    with left_col:
        st.markdown("#### 🎯 Credit Risk Gauge")
        
        # Plotly Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=credit_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Calibrated Credit Score: {credit_score} / 850", 'font': {'size': 18, 'color': '#1e293b'}},
            gauge={
                'axis': {'range': [300, 850], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': decision_color, 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#cbd5e1",
                'steps': [
                    {'range': [300, 550], 'color': 'rgba(239, 68, 68, 0.25)'},
                    {'range': [550, 650], 'color': 'rgba(245, 158, 11, 0.25)'},
                    {'range': [650, 750], 'color': 'rgba(16, 185, 129, 0.25)'},
                    {'range': [750, 850], 'color': 'rgba(5, 150, 105, 0.40)'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': credit_score
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown("##### 🔍 Cross-Model Consensus Comparison")
        df_models = pd.DataFrame({
            "Underwriting Model": [
                "Proposed Hybrid (Tree-Embedding LR)",
                "Stacking Meta-Ensemble",
                "Baseline Logistic Regression",
                "Baseline Decision Tree"
            ],
            "Default Risk (%)": [prob_hybrid, prob_stack, prob_lr, prob_dt],
            "Derived Score": [
                credit_score,
                probability_to_credit_score(prob_stack),
                probability_to_credit_score(prob_lr),
                probability_to_credit_score(prob_dt)
            ],
            "Assessment": [
                decision_text,
                "Approved" if prob_stack <= 0.22 else ("Review" if prob_stack <= 0.35 else "Decline"),
                "Approved" if prob_lr <= 0.22 else ("Review" if prob_lr <= 0.35 else "Decline"),
                "Approved" if prob_dt <= 0.22 else ("Review" if prob_dt <= 0.35 else "Decline")
            ]
        })
        
        fig_bar = px.bar(
            df_models, x="Underwriting Model", y="Default Risk (%)",
            color="Default Risk (%)",
            color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
            range_y=[0, 1.0],
            text=df_models["Default Risk (%)"].apply(lambda v: f"{v:.1%}")
        )
        fig_bar.update_layout(height=240, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.caption(
            "💡 *The Tree-Embedding Hybrid synthesizes non-linear interactions discovered by the Decision Tree "
            "with calibrated log-odds from Logistic Regression for compliant underwriting.*"
        )
        
    with right_col:
        st.markdown("#### 🧠 Explainable AI: Underwriting Drivers")
        
        feature_names = preprocessor.get_feature_names()
        risk_inc, risk_mit = compute_feature_contributions(lr_model, applicant_feats, feature_names, top_n=5)
        
        # Risk Contributing Factors Plot
        plot_df = pd.concat([
            risk_inc.assign(Direction="Risk Increasing"),
            risk_mit.assign(Direction="Risk Mitigating")
        ]).sort_values("Contribution")
        
        fig_waterfall = px.bar(
            plot_df, x="Contribution", y="Feature", orientation="h",
            color="Direction",
            color_discrete_map={"Risk Increasing": "#ef4444", "Risk Mitigating": "#10b981"},
            title="Logistic Impact Breakdown (Odds Impact Multiplier)"
        )
        fig_waterfall.update_layout(height=260, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        # Decision Tree Logic Path
        st.markdown("##### 🌳 Decision Tree Branching Path Activated:")
        tree_rules = extract_decision_tree_path(dt_model, applicant_feats, feature_names)
        
        for rule in tree_rules:
            if rule["type"] == "split":
                st.markdown(
                    f'<div class="rule-box">'
                    f'<strong>Node #{rule["node_id"]}:</strong> {rule["comparison"]} '
                    f'<em>({rule["status"]})</em>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="rule-box" style="border-left-color: #8b5cf6; background-color: #faf5ff;">'
                    f'<strong>{rule["text"]}</strong>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
    st.markdown("---")
    
    # Counterfactual "What-If" Sensitivity & Actionable Advice
    st.markdown("### 🎛️ Counterfactual Sensitivity & Underwriting Optimization")
    st.markdown("Explore what actionable adjustments would qualify this borrower or lower their risk profile.")
    
    sens_col1, sens_col2 = st.columns([1, 1.2])
    
    with sens_col1:
        st.markdown("##### 🔄 Interactive What-If Scenario Simulator")
        sim_loan = st.slider("Simulate Adjusted Loan Amount ($)", 1000, 50000, int(loan_amount), step=1000)
        sim_term = st.selectbox("Simulate Adjusted Term (Months)", [12, 24, 36, 48, 60], index=[12, 24, 36, 48, 60].index(int(loan_term_months)))
        sim_cosigner = st.radio("Simulate Adding Co-Signer / Guarantor", [0, 1], index=int(has_cosigner), horizontal=True, format_func=lambda x: "No Co-Signer" if x==0 else "Yes (Guarantor)")
        
        sim_dict = dict(applicant_dict)
        sim_dict["loan_amount"] = sim_loan
        sim_dict["loan_term_months"] = sim_term
        sim_dict["has_cosigner"] = sim_cosigner
        
        sim_feats = preprocessor.transform(pd.DataFrame([sim_dict]))
        sim_prob = float(hybrid_model.predict_proba(sim_feats)[0, 1])
        sim_score = probability_to_credit_score(sim_prob)
        sim_tier, sim_dec, sim_col, _ = get_risk_tier_and_decision(sim_prob)
        
        st.markdown(
            f'<div style="background-color: {sim_col}; color: white; padding: 12px; border-radius: 6px; text-align: center; font-weight: bold;">'
            f'SIMULATED RESULT: {sim_dec.upper()}<br>'
            f'Score: {sim_score} | Default Risk: {sim_prob:.1%} (Δ {(sim_prob - prob_hybrid)*100:+.1f}%)'
            f'</div>',
            unsafe_allow_html=True
        )
        
    with sens_col2:
        st.markdown("##### 📋 Actionable Underwriter Path to Approval:")
        recommendations = generate_counterfactual_recommendations(applicant_dict, preprocessor, hybrid_model)
        
        for rec in recommendations:
            status_icon = "✅" if rec.get("qualifies", False) else "ℹ️"
            st.markdown(f"**{status_icon} {rec['action']}**")
            st.markdown(f"> {rec['detail']}")
            if rec.get("reduction_pct", 0) > 0:
                st.caption(f"📉 **Risk Reduction:** -{rec['reduction_pct']}% | **New Default Hazard:** {rec['new_probability']:.1%}")
            st.markdown("---")
            
    # Printable Credit Memo
    with st.expander("📄 Generate Formal Loan Underwriting Credit Memo"):
        memo_text = f"""================================================================================
APEX HORIZON BANK - RETAIL UNDERWRITING MEMORANDUM
================================================================================
APPLICANT DETAILS:
Applicant Name      : {applicant_name}
Age / Education     : {age} Years | {education}
Employment Status   : {employment_type} ({employment_years} yrs tenure)
Annual Gross Income : ${annual_income:,.2f}
Liquid Savings      : ${savings_balance:,.2f}

CREDIT & LEVERAGE PROFILE:
Loan Request        : ${loan_amount:,.2f} over {loan_term_months} Months @ {interest_rate:.2f}%
Monthly Installment : ${df_calc['monthly_installment']:,.2f}
Debt-to-Income (DTI): {df_calc['debt_to_income_ratio']:.2%}
Loan-to-Income (LTI): {df_calc['loan_to_income_ratio']:.2%}
Utility Score (Alt) : {rent_utility_payment_score} / 100
Guarantor / Cosigner: {'Yes' if has_cosigner == 1 else 'None'}
Collateral Pledged  : {has_collateral}

HYBRID MODEL DECISION ENGINE:
Credit Risk Tier    : {tier_name}
Credit Bureau Score : {credit_score} / 850
Default Probability : {prob_hybrid:.2%}
Underwriting Action : {decision_text.upper()}

COMMITTEE SUMMARY:
{decision_desc}
================================================================================"""
        st.text_area("Credit Memo Output", memo_text, height=280)
        st.download_button("📥 Download Credit Memo (.txt)", memo_text, file_name=f"Credit_Memo_{applicant_name.replace(' ', '_')}.txt", mime="text/plain")


# ==============================================================================
# TAB 2: BATCH PORTFOLIO ASSESSMENT
# ==============================================================================
elif app_mode == "📂 Batch Portfolio Assessment":
    st.markdown('<div class="main-title">Batch Application Processing & Portfolio Risk</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Underwrite multiple first-time applicant dossiers simultaneously with aggregate portfolio risk metrics</div>', unsafe_allow_html=True)
    
    st.markdown("##### 📥 Applicant Data Source")
    data_source = st.radio("Choose Input Mode:", ["Use Held-Out Validation Test Set (1,200 Applicants)", "Upload Custom Applicants CSV"], horizontal=True)
    
    if data_source == "Use Held-Out Validation Test Set (1,200 Applicants)":
        df_batch = pd.read_csv(TEST_DATA_PATH)
    else:
        uploaded_file = st.file_uploader("Upload CSV formatted applicant dataset", type=["csv"])
        if uploaded_file is not None:
            df_batch = pd.read_csv(uploaded_file)
        else:
            st.info("Awaiting CSV file upload. Defaulting to sample test data.")
            df_batch = pd.read_csv(TEST_DATA_PATH)
            
    # Sample Selector
    sample_size = st.slider("Select Portfolio Sample Size to Evaluate", 50, len(df_batch), min(250, len(df_batch)), step=50)
    df_eval = df_batch.head(sample_size).copy()
    
    with st.spinner("Executing Hybrid Model batch underwriting..."):
        batch_feats = preprocessor.transform(df_eval)
        df_eval["default_prob"] = np.round(hybrid_model.predict_proba(batch_feats)[:, 1], 4)
        df_eval["credit_score"] = [probability_to_credit_score(p) for p in df_eval["default_prob"]]
        df_eval["underwriting_decision"] = [
            "Approved" if p < DECISION_APPROVAL_THRESHOLD else ("Manual Review" if p < DECISION_REVIEW_THRESHOLD else "Declined")
            for p in df_eval["default_prob"]
        ]
        df_eval["risk_tier"] = [
            "Tier A" if p < 0.12 else ("Tier B" if p < 0.22 else ("Tier C" if p < 0.35 else ("Tier D" if p < 0.50 else "Tier E")))
            for p in df_eval["default_prob"]
        ]
        
    # Portfolio Summary KPIs
    total_apps = len(df_eval)
    approved_count = int(np.sum(df_eval["underwriting_decision"] == "Approved"))
    review_count = int(np.sum(df_eval["underwriting_decision"] == "Manual Review"))
    declined_count = int(np.sum(df_eval["underwriting_decision"] == "Declined"))
    
    total_volume = df_eval["loan_amount"].sum()
    approved_volume = df_eval[df_eval["underwriting_decision"] == "Approved"]["loan_amount"].sum()
    avg_portfolio_risk = df_eval["default_prob"].mean()
    
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric("Total Applications", f"{total_apps:,}", f"${total_volume:,.0f} Volume")
    with p2:
        st.metric("Approval Rate", f"{approved_count / total_apps:.1%}", f"${approved_volume:,.0f} Approved")
    with p3:
        st.metric("Conditional Review Rate", f"{review_count / total_apps:.1%}", f"{review_count} Cases")
    with p4:
        st.metric("Declined Rate", f"{declined_count / total_apps:.1%}", f"Avg Hazard: {avg_portfolio_risk:.1%}")
        
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("##### 📊 Portfolio Risk Tier Distribution")
        fig_tier = px.pie(
            df_eval, names="risk_tier",
            color="risk_tier",
            color_discrete_map={
                "Tier A": "#10b981",
                "Tier B": "#059669",
                "Tier C": "#f59e0b",
                "Tier D": "#ea580c",
                "Tier E": "#dc2626"
            },
            hole=0.45
        )
        fig_tier.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_tier, use_container_width=True)
        
    with col_chart2:
        st.markdown("##### 📈 Default Probability vs Credit Score Dispersion")
        fig_scatter = px.scatter(
            df_eval, x="credit_score", y="default_prob", color="underwriting_decision",
            size="loan_amount", hover_data=["applicant_id", "annual_income", "loan_intent"],
            color_discrete_map={"Approved": "#10b981", "Manual Review": "#f59e0b", "Declined": "#ef4444"},
            labels={"credit_score": "Calibrated Credit Score", "default_prob": "Default Probability"}
        )
        fig_scatter.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    st.markdown("##### 📋 Underwriting Roster Table")
    filter_choice = st.multiselect("Filter Decision Status:", ["Approved", "Manual Review", "Declined"], default=["Approved", "Manual Review", "Declined"])
    display_df = df_eval[df_eval["underwriting_decision"].isin(filter_choice)][[
        "applicant_id", "age", "education", "annual_income", "loan_amount",
        "loan_intent", "savings_balance", "credit_score", "default_prob", "underwriting_decision", "risk_tier"
    ]]
    st.dataframe(display_df, use_container_width=True, height=320)
    
    csv_data = df_eval.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Scored Portfolio (CSV)",
        csv_data,
        "underwritten_loan_portfolio.csv",
        "text/csv",
        use_container_width=True
    )


# ==============================================================================
# TAB 3: MODEL BENCHMARK & ANALYTICS
# ==============================================================================
elif app_mode == "📊 Model Benchmark & Analytics":
    st.markdown('<div class="main-title">Model Benchmark & Regulatory Diagnostics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Empirical comparison between Baseline Logistic Regression, Decision Tree, and the Proposed Hybrid Model across Basel III metrics</div>',
        unsafe_allow_html=True
    )
    
    if metrics:
        st.markdown("##### 🏆 Comparative Scorecard Table")
        table_rows = []
        for name, m in metrics.items():
            table_rows.append({
                "Model Candidate": name,
                "ROC-AUC": m["roc_auc"],
                "Gini Index": m["gini_coefficient"],
                "KS Statistic": m["ks_statistic"],
                "Default Recall": f"{m['recall_sensitivity']:.2%}",
                "Precision": f"{m['precision']:.2%}",
                "Brier Score": m["brier_score"],
                "Cost Loss Pts": m["financial_loss_cost"]
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
        
    st.markdown("---")
    
    col_img1, col_img2 = st.columns(2)
    roc_img = REPORTS_DIR / "roc_curves.png"
    pr_img = REPORTS_DIR / "precision_recall_curves.png"
    cm_img = REPORTS_DIR / "confusion_matrices.png"
    bar_img = REPORTS_DIR / "model_comparison_bar.png"
    
    with col_img1:
        st.markdown("##### 📈 ROC Curves (Receiver Operating Characteristic)")
        if roc_img.exists():
            st.image(str(roc_img), use_container_width=True)
        else:
            st.info("Run evaluation pipeline to generate ROC plot.")
            
    with col_img2:
        st.markdown("##### 🎯 Precision-Recall Curves")
        if pr_img.exists():
            st.image(str(pr_img), use_container_width=True)
        else:
            st.info("Run evaluation pipeline to generate PR plot.")
            
    col_img3, col_img4 = st.columns(2)
    with col_img3:
        st.markdown("##### 🔲 Confusion Matrices (Cost-Weighted)")
        if cm_img.exists():
            st.image(str(cm_img), use_container_width=True)
        else:
            st.info("Run evaluation pipeline to generate confusion matrices.")
            
    with col_img4:
        st.markdown("##### 📊 Comparative Bar Chart Across Underwriting Dimensions")
        if bar_img.exists():
            st.image(str(bar_img), use_container_width=True)
        else:
            st.info("Run evaluation pipeline to generate comparison bar chart.")


# ==============================================================================
# TAB 4: HYBRID ARCHITECTURE & METHODOLOGY
# ==============================================================================
elif app_mode == "📖 Hybrid Architecture & Methodology":
    st.markdown('<div class="main-title">Hybrid Architecture & Technical Underwriting Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Design rationale, mathematical formulation, and Basel III regulatory compliance</div>', unsafe_allow_html=True)
    
    st.markdown("""
### 1. Executive Problem Statement
First-time borrowers represent the **"credit invisible" / thin-file dilemma**:
* Traditional credit bureaus (FICO / CIBIL) have virtually zero repayment history for young professionals and new business founders.
* Banks cannot afford blanket rejections (lost market share), nor uncalibrated approvals (excessive default write-offs).
* Standalone **Logistic Regression** satisfies regulatory interpretability, but is fundamentally linear in log-odds; it fails to automatically discover high-order risk interactions (such as the DTI cliff and liquidity crunch).
* Standalone **Decision Trees** discover non-linear threshold rules easily, but yield coarse step-wise probabilities, overfit readily, and lack smooth credit score calibration.

---

### 2. The Hybrid Model Architecture
Our architecture fuses both paradigms into an end-to-end mathematical pipeline:

```
[Raw Applicant Dossier]
        │
        ▼
[Feature Engineering & Ratios] (DTI, Loan-to-Income, Savings-to-Loan, Installment)
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
[Standardized Numerical & Categorical X]  [Decision Tree Classifier]
        │                                      │ (Discovers Non-Linear Splits)
        │                                      ▼
        │                              [Leaf Node Index L(X)]
        │                                      │
        │                                      ▼
        │                              [One-Hot Leaf Embedding e(X)]
        │                                      │
        └──────────────────┬───────────────────┘
                           ▼
                 [Augmented Space: [X, e(X)]]
                           │
                           ▼
            [L2-Regularized Logistic Regression]
                           │
                           ▼
           [Calibrated Default Probability P(Default|X)]
                           │
                           ▼
     [Logarithmic Credit Score (300-850) & Risk Tier]
```

#### Mathematical Formulation:
$$\ln\left(\frac{P(\text{Default}=1|x)}{1 - P(\text{Default}=1|x)}\right) = \beta_0 + \sum_{j=1}^{M} \beta_j x_j + \sum_{k=1}^{K} \gamma_k e_k(x)$$

Where:
* $x_j$ represents the normalized linear financial ratios (DTI, income, savings, interest rate).
* $e_k(x) \in \{0, 1\}$ represents the indicator variable for whether the applicant falls into leaf node cohort $k$ of the Decision Tree.
* $\gamma_k$ captures the non-linear risk premium / penalty associated with that specific structural cohort.
* The regularized loss objective prevents overfitting to thin sub-cohorts:

$$\min_{\beta, \gamma} -\sum_{i=1}^{N} \left[ y_i \ln(p_i) + (1-y_i) \ln(1-p_i) \right] + \frac{\lambda}{2} \left(\|\beta\|_2^2 + \|\gamma\|_2^2\right)$$

---

### 3. Banking & Regulatory Advantages
1. **Basel III Regulatory Compliance**: Calibrated probability outputs directly map to Internal Ratings-Based (IRB) Expected Loss:
   $$\text{Expected Loss (EL)} = \text{PD} \times \text{LGD} \times \text{EAD}$$
2. **Transparent Adverse Action Reporting (Equal Credit Opportunity Act / ECOA)**:
   Whenever an applicant is rejected or placed in manual review, the decision tree path and logistic odds multipliers provide instantaneous, legally defensible adverse action reasons.
3. **Counterfactual Underwriting Advisor**:
   Instead of a binary "No", loan officers can offer constructive conditions (e.g. "Add a co-signer or reduce loan amount by $3,000 to qualify").
""")
