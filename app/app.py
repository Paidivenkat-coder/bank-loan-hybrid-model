"""
Loan Officer Decision-Support Tool: Interactive Web Application
Built with Streamlit and Plotly for the Logistic Regression & Decision Tree Hybrid Model.
Domain: Machine Learning & Predictive Analytics
Project: Design and Development of a Working Prototype for Bank loan default Using A Logistic Regression and Decision Tree hybrid model
Developer / Lead Investigator: PAIDI VENKAT
Currency: Indian Rupee (INR / ₹)
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
    PROJECT_TITLE, DOMAIN, AUTHOR, ABSTRACT, KEYWORDS,
    CURRENCY_SYMBOL, CURRENCY_CODE, format_inr, format_inr_full,
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
    page_title="Bharat Credit Decision Support System | Bank Loan Default Hybrid Model",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0f2d59;
        margin-bottom: 0.15rem;
        letter-spacing: -0.5px;
    }
    .author-badge {
        display: inline-block;
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.88rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.02rem;
        color: #475569;
        margin-bottom: 1.2rem;
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
        font-size: 1.35rem;
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
st.sidebar.markdown(f"### 🏦 BHARAT CREDIT INTELLIGENCE")
st.sidebar.markdown("**Retail Loan Underwriting System**")
st.sidebar.markdown(f"**Developer / Lead:** `{AUTHOR}`")
st.sidebar.markdown(f"**Currency:** `Indian Rupee ({CURRENCY_SYMBOL} / INR)`")
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
    f"**Project:** {PROJECT_TITLE}\n\n"
    f"**Domain:** {DOMAIN}\n\n"
    f"**Investigator:** {AUTHOR}\n\n"
    f"**Standard:** RBI Digital Lending & Basel III Aligned\n\n"
    f"**Currency Standard:** Indian Rupees (₹ / Lakhs)"
)


# ==============================================================================
# TAB 1: SINGLE LOAN UNDERWRITING WORKSTATION
# ==============================================================================
if app_mode == "📋 Single Underwriting Workstation":
    st.markdown('<div class="author-badge">Project Developer: PAIDI VENKAT | Currency: Indian Rupee (₹ INR)</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Loan Officer Decision-Support Workstation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">AI-Assisted Credit Evaluation for First-Time Indian Borrowers using the Logistic Regression & Decision Tree Hybrid Pipeline</div>',
        unsafe_allow_html=True
    )
    
    # Archetype Presets in Indian Context & Rupees
    st.markdown("##### ⚡ Quick-Load Indian Borrower Profiles:")
    col_arc1, col_arc2, col_arc3, col_arc4 = st.columns(4)
    
    archetypes = {
        "prime": {
            "name": "Arjun Sharma", "age": 28, "education": "Graduate (B.Tech/B.Sc/B.Com)", "employment_type": "Salaried (IT/Corporate)",
            "employment_years": 4.5, "annual_income": 1150000, "existing_monthly_debt": 12000,
            "loan_amount": 350000, "loan_term_months": 36, "interest_rate": 11.5,
            "loan_intent": "Debt Consolidation", "home_ownership": "Rented", "savings_balance": 280000,
            "rent_utility_payment_score": 88, "has_cosigner": 0, "has_collateral": "None",
            "num_recent_credit_inquiries": 1, "num_bank_overdrafts_12m": 0, "bank_tenure_months": 36
        },
        "borderline": {
            "name": "Priya Patel", "age": 26, "education": "Diploma", "employment_type": "Salaried (IT/Corporate)",
            "employment_years": 2.0, "annual_income": 480000, "existing_monthly_debt": 14000,
            "loan_amount": 320000, "loan_term_months": 36, "interest_rate": 15.5,
            "loan_intent": "Personal", "home_ownership": "Rented", "savings_balance": 35000,
            "rent_utility_payment_score": 68, "has_cosigner": 0, "has_collateral": "None",
            "num_recent_credit_inquiries": 3, "num_bank_overdrafts_12m": 1, "bank_tenure_months": 15
        },
        "grad": {
            "name": "Rohit Varma", "age": 23, "education": "Post-Graduate (MBA/M.Tech)", "employment_type": "Salaried (IT/Corporate)",
            "employment_years": 0.8, "annual_income": 550000, "existing_monthly_debt": 4000,
            "loan_amount": 250000, "loan_term_months": 36, "interest_rate": 12.0,
            "loan_intent": "Education / Higher Studies", "home_ownership": "Ancestral / Family Owned", "savings_balance": 75000,
            "rent_utility_payment_score": 82, "has_cosigner": 1, "has_collateral": "None",
            "num_recent_credit_inquiries": 1, "num_bank_overdrafts_12m": 0, "bank_tenure_months": 24
        },
        "highrisk": {
            "name": "Vikram Singh", "age": 32, "education": "Higher Secondary", "employment_type": "Small Business / Proprietor",
            "employment_years": 1.0, "annual_income": 420000, "existing_monthly_debt": 18000,
            "loan_amount": 650000, "loan_term_months": 48, "interest_rate": 18.5,
            "loan_intent": "MSME / Small Business", "home_ownership": "Rented", "savings_balance": 22000,
            "rent_utility_payment_score": 50, "has_cosigner": 0, "has_collateral": "None",
            "num_recent_credit_inquiries": 5, "num_bank_overdrafts_12m": 3, "bank_tenure_months": 9
        }
    }
    
    if "form_data" not in st.session_state:
        st.session_state["form_data"] = archetypes["prime"]
        
    with col_arc1:
        if st.button("🌟 Prime First-Timer (₹11.5L Income)", use_container_width=True):
            st.session_state["form_data"] = archetypes["prime"]
            st.rerun()
    with col_arc2:
        if st.button("⚖️ Borderline / High FOIR (₹4.8L)", use_container_width=True):
            st.session_state["form_data"] = archetypes["borderline"]
            st.rerun()
    with col_arc3:
        if st.button("🎓 Graduate + Guarantor (₹5.5L)", use_container_width=True):
            st.session_state["form_data"] = archetypes["grad"]
            st.rerun()
    with col_arc4:
        if st.button("⚠️ High-Risk Thin File (₹4.2L)", use_container_width=True):
            st.session_state["form_data"] = archetypes["highrisk"]
            st.rerun()
            
    current = st.session_state["form_data"]
    
    st.markdown("---")
    
    # Input Form Layout in Indian Rupees
    with st.form("applicant_evaluation_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 Demographics & Stability")
            applicant_name = st.text_input("Applicant Full Name", value=current.get("name", "Arjun Sharma"))
            age = st.slider("Applicant Age", 21, 65, int(current.get("age", 28)))
            
            edu_options = [
                "Higher Secondary", 
                "Diploma", 
                "Graduate (B.Tech/B.Sc/B.Com)", 
                "Post-Graduate (MBA/M.Tech)", 
                "Doctorate"
            ]
            cur_edu = current.get("education", "Graduate (B.Tech/B.Sc/B.Com)")
            edu_idx = edu_options.index(cur_edu) if cur_edu in edu_options else 2
            education = st.selectbox("Highest Education Level", edu_options, index=edu_idx)
            
            emp_options = [
                "Salaried (IT/Corporate)", 
                "Salaried (PSU/Govt)", 
                "Self-Employed Professional", 
                "Small Business / Proprietor", 
                "Gig / Freelance"
            ]
            cur_emp = current.get("employment_type", "Salaried (IT/Corporate)")
            emp_idx = emp_options.index(cur_emp) if cur_emp in emp_options else 0
            employment_type = st.selectbox("Employment Sector", emp_options, index=emp_idx)
            
            employment_years = st.slider(
                "Tenure in Current Employment (Years)", 0.0, 20.0, float(current.get("employment_years", 3.5)), step=0.5
            )
            
            home_options = ["Rented", "Self-Owned", "Ancestral / Family Owned", "Company Provided"]
            cur_home = current.get("home_ownership", "Rented")
            home_idx = home_options.index(cur_home) if cur_home in home_options else 0
            home_ownership = st.selectbox("Residential Status", home_options, index=home_idx)
            
        with col2:
            st.markdown("#### 💰 Financial Profile (Indian Rupees ₹)")
            annual_income = st.number_input(
                "Annual Gross Income (₹ INR)", 
                min_value=150000, 
                max_value=10000000, 
                value=int(current.get("annual_income", 950000)), 
                step=25000,
                help=f"E.g., ₹6,50,000 = 6.5 Lakhs/yr. Current: {format_inr(current.get('annual_income', 950000))}"
            )
            st.caption(f"ℹ️ Equivalent to **{format_inr(annual_income)}/yr** (~{format_inr(annual_income/12.0)}/month)")
            
            existing_monthly_debt = st.number_input(
                "Existing Monthly EMIs / Debt (₹ INR)", 
                min_value=0, 
                max_value=250000, 
                value=int(current.get("existing_monthly_debt", 8000)), 
                step=1000,
                help="Sum of ongoing EMIs (credit cards, bike loans, consumer loans)."
            )
            
            savings_balance = st.number_input(
                "Liquid Bank Savings / Fixed Deposits (₹ INR)", 
                min_value=0, 
                max_value=5000000, 
                value=int(current.get("savings_balance", 120000)), 
                step=10000,
                help=f"Emergency buffer available in bank savings or liquid deposits. Current: {format_inr(current.get('savings_balance', 120000))}"
            )
            
            rent_utility_payment_score = st.slider(
                "Alternative Track Score / Account Aggregator (30-100)", 30, 100, int(current.get("rent_utility_payment_score", 75)),
                help="Derived from UPI payment track, utility bills, and bank statement cash-flow regularity."
            )
            bank_tenure_months = st.slider(
                "Relationship Tenure with Bank (Months)", 1, 140, int(current.get("bank_tenure_months", 24))
            )
            
        with col3:
            st.markdown("#### 📑 Requested Loan Details (₹)")
            loan_amount = st.number_input(
                "Loan Amount Requested (₹ INR)", 
                min_value=25000, 
                max_value=3000000, 
                value=int(current.get("loan_amount", 300000)), 
                step=10000,
                help=f"Requested borrowing. Current: {format_inr(current.get('loan_amount', 300000))}"
            )
            st.caption(f"ℹ️ Requested Principal: **{format_inr(loan_amount)}**")
            
            loan_term_months = st.selectbox(
                "Loan Tenure (Months)", [12, 24, 36, 48, 60, 72],
                index=[12, 24, 36, 48, 60, 72].index(int(current.get("loan_term_months", 36)))
            )
            interest_rate = st.slider(
                "Annual Interest Rate (%)", 8.5, 26.0, float(current.get("interest_rate", 12.5)), step=0.25
            )
            
            intent_options = [
                "Personal", 
                "Education / Higher Studies", 
                "Two-Wheeler / Vehicle", 
                "Debt Consolidation", 
                "MSME / Small Business", 
                "Home Improvement", 
                "Medical / Emergency"
            ]
            cur_intent = current.get("loan_intent", "Personal")
            intent_idx = intent_options.index(cur_intent) if cur_intent in intent_options else 0
            loan_intent = st.selectbox("Loan Purpose", intent_options, index=intent_idx)
            
            col_mit1, col_mit2 = st.columns(2)
            with col_mit1:
                has_cosigner = st.selectbox(
                    "Co-Signer / Guarantor?", [0, 1], 
                    format_func=lambda x: "Yes (Guarantor)" if x == 1 else "No", 
                    index=int(current.get("has_cosigner", 0))
                )
            with col_mit2:
                collat_options = ["None", "Vehicle", "Fixed Deposit Lien", "Gold Deposit"]
                cur_collat = current.get("has_collateral", "None")
                collat_idx = collat_options.index(cur_collat) if cur_collat in collat_options else 0
                has_collateral = st.selectbox("Collateral Pledged", collat_options, index=collat_idx)
                
            num_recent_credit_inquiries = st.slider("Credit Inquiries (Past 6 Mo)", 0, 8, int(current.get("num_recent_credit_inquiries", 1)))
            num_bank_overdrafts_12m = st.slider("NACH / ECS Bounces (Past 12 Mo)", 0, 6, int(current.get("num_bank_overdrafts_12m", 0)))
            
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
    
    # Primary KPI Metrics Row in Indian Rupees
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(
            "CIBIL-Calibrated Score", 
            f"{credit_score}", 
            f"{'Super-Prime' if credit_score >= 750 else ('Prime' if credit_score >= 700 else 'Sub-Prime')}"
        )
    with m2:
        st.metric(
            "Hybrid Default Probability", 
            f"{prob_hybrid:.1%}", 
            f"{round((prob_hybrid - 0.175)*100, 1):+0.1f}% vs Portfolio"
        )
    with m3:
        st.metric(
            "FOIR / Debt-to-Income", 
            f"{df_calc['debt_to_income_ratio']:.1%}", 
            "Acceptable (≤ 45%)" if df_calc['debt_to_income_ratio'] <= 0.45 else "High FOIR Cliff", 
            delta_color="inverse"
        )
    with m4:
        st.metric(
            "Monthly EMI", 
            f"{format_inr_full(df_calc['monthly_installment'])}", 
            f"Tenure: {loan_term_months} Months"
        )
    with m5:
        st.metric(
            "Net Monthly Cash Surplus", 
            f"{format_inr_full(df_calc['net_monthly_surplus'])}", 
            "Comfortable Buffer" if df_calc['net_monthly_surplus'] > 20000 else "Tight Liquidity"
        )
        
    st.markdown("---")
    
    # Two Columns: Gauges & Model Comparison vs Detailed Explainability
    left_col, right_col = st.columns([1, 1.2])
    
    with left_col:
        st.markdown("#### 🎯 Indian Credit Risk Gauge (300 - 900 CIBIL Scale)")
        
        # Plotly Gauge Chart calibrated for Indian CIBIL scale
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=credit_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Calibrated Credit Bureau Score: {credit_score} / 900", 'font': {'size': 17, 'color': '#0f2d59'}},
            gauge={
                'axis': {'range': [300, 900], 'tickwidth': 1, 'tickcolor': "#334155"},
                'bar': {'color': decision_color, 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#cbd5e1",
                'steps': [
                    {'range': [300, 550], 'color': 'rgba(239, 68, 68, 0.25)'},
                    {'range': [550, 675], 'color': 'rgba(245, 158, 11, 0.25)'},
                    {'range': [675, 750], 'color': 'rgba(16, 185, 129, 0.25)'},
                    {'range': [750, 900], 'color': 'rgba(5, 150, 105, 0.40)'}
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
            "💡 *The Tree-Embedding Hybrid Model synthesizes non-linear interactions discovered by the Decision Tree "
            "with smoothly calibrated log-odds from regularized Logistic Regression.*"
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
            title="Logistic Impact Breakdown (Odds Multiplier)"
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
    
    # Counterfactual "What-If" Sensitivity in Indian Rupees
    st.markdown("### 🎛️ Counterfactual Sensitivity & Underwriting Optimization (INR ₹)")
    st.markdown("Explore actionable adjustments to help this applicant qualify or reduce their default hazard.")
    
    sens_col1, sens_col2 = st.columns([1, 1.2])
    
    with sens_col1:
        st.markdown("##### 🔄 Interactive What-If Scenario Simulator")
        sim_loan = st.slider(
            "Simulate Adjusted Loan Amount (₹)", 
            25000, 
            int(max(50000, loan_amount * 1.5)), 
            int(loan_amount), 
            step=25000,
            help="Simulate the impact of reducing loan sizing via applicant margin money."
        )
        st.caption(f"Simulated Loan: **{format_inr(sim_loan)}**")
        
        sim_term = st.selectbox(
            "Simulate Adjusted Tenure (Months)", 
            [12, 24, 36, 48, 60, 72], 
            index=[12, 24, 36, 48, 60, 72].index(int(loan_term_months))
        )
        sim_cosigner = st.radio(
            "Simulate Adding Co-Signer / Guarantor", 
            [0, 1], 
            index=int(has_cosigner), 
            horizontal=True, 
            format_func=lambda x: "No Guarantor" if x==0 else "Yes (Parent/Spouse Guarantor)"
        )
        
        sim_dict = dict(applicant_dict)
        sim_dict["loan_amount"] = sim_loan
        sim_dict["loan_term_months"] = sim_term
        sim_dict["has_cosigner"] = sim_cosigner
        
        sim_feats = preprocessor.transform(pd.DataFrame([sim_dict]))
        sim_prob = float(hybrid_model.predict_proba(sim_feats)[0, 1])
        sim_score = probability_to_credit_score(sim_prob)
        sim_tier, sim_dec, sim_col, _ = get_risk_tier_and_decision(sim_prob)
        
        st.markdown(
            f'<div style="background-color: {sim_col}; color: white; padding: 14px; border-radius: 6px; text-align: center; font-weight: bold; margin-top: 15px;">'
            f'SIMULATED RESULT: {sim_dec.upper()}<br>'
            f'Score: {sim_score} / 900 | Default Risk: {sim_prob:.1%} (Δ {(sim_prob - prob_hybrid)*100:+.1f}%)'
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
                st.caption(f"📉 **Default Hazard Reduction:** -{rec['reduction_pct']}% | **New Expected Risk:** {rec['new_probability']:.1%}")
            st.markdown("---")
            
    # Printable Indian Credit Memo
    with st.expander("📄 Generate Formal Indian Banking Credit Memo"):
        memo_text = f"""================================================================================
BHARAT CREDIT INTELLIGENCE SYSTEM - RETAIL UNDERWRITING MEMORANDUM
Project Lead / Developer: {AUTHOR}
Domain: {DOMAIN}
================================================================================
APPLICANT DETAILS:
Applicant Name          : {applicant_name}
Age / Education         : {age} Years | {education}
Employment Sector       : {employment_type} ({employment_years} yrs tenure)
Annual Gross Income     : {format_inr_full(annual_income)} ({format_inr(annual_income)})
Liquid Savings / FD     : {format_inr_full(savings_balance)} ({format_inr(savings_balance)})

CREDIT & LEVERAGE PROFILE (INR / ₹):
Loan Amount Requested   : {format_inr_full(loan_amount)} ({format_inr(loan_amount)})
Tenure & Interest Rate  : {loan_term_months} Months @ {interest_rate:.2f}% p.a.
Monthly EMI             : {format_inr_full(df_calc['monthly_installment'])}
Fixed Obligation Ratio  : {df_calc['debt_to_income_ratio']:.2%} (FOIR / DTI)
Loan-to-Income (LTI)    : {df_calc['loan_to_income_ratio']:.2%}
Alternative Score Track : {rent_utility_payment_score} / 100 (UPI / Account Aggregator)
Guarantor / Co-Signer   : {'Yes (Solvent Guarantor)' if has_cosigner == 1 else 'None'}
Collateral Backing      : {has_collateral}

HYBRID MODEL DECISION ENGINE:
Credit Risk Tier        : {tier_name}
CIBIL-Calibrated Score  : {credit_score} / 900
Default Probability (PD): {prob_hybrid:.2%}
Underwriting Action     : {decision_text.upper()}

CREDIT COMMITTEE ACTION SUMMARY:
{decision_desc}

UNDERWRITING SIGN-OFF:
Lead Investigator       : {AUTHOR}
Model Architecture      : Logistic Regression & Decision Tree Hybrid Model
Regulatory Compliance   : RBI Digital Lending Norms & Basel III IRB
Date of Evaluation      : 2026-11-01
================================================================================"""
        st.text_area("Official Credit Memo", memo_text, height=300)
        st.download_button(
            "📥 Download Formal Credit Memo (.txt)", 
            memo_text, 
            file_name=f"Credit_Memo_{applicant_name.replace(' ', '_')}_INR.txt", 
            mime="text/plain"
        )


# ==============================================================================
# TAB 2: BATCH PORTFOLIO ASSESSMENT
# ==============================================================================
elif app_mode == "📂 Batch Portfolio Assessment":
    st.markdown('<div class="author-badge">Lead Investigator: PAIDI VENKAT | Currency: Indian Rupee (₹ INR)</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Batch Application Processing & Portfolio Risk (INR ₹)</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Underwrite multiple first-time applicant dossiers simultaneously with aggregate Indian retail portfolio risk metrics</div>', unsafe_allow_html=True)
    
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
    
    with st.spinner("Executing Hybrid Model batch underwriting in Indian Rupees..."):
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
        
    # Portfolio Summary KPIs in Indian Rupees (Lakhs & Crores)
    total_apps = len(df_eval)
    approved_count = int(np.sum(df_eval["underwriting_decision"] == "Approved"))
    review_count = int(np.sum(df_eval["underwriting_decision"] == "Manual Review"))
    declined_count = int(np.sum(df_eval["underwriting_decision"] == "Declined"))
    
    total_volume = df_eval["loan_amount"].sum()
    approved_volume = df_eval[df_eval["underwriting_decision"] == "Approved"]["loan_amount"].sum()
    avg_portfolio_risk = df_eval["default_prob"].mean()
    
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric("Total Applications", f"{total_apps:,}", f"{format_inr(total_volume)} Volume")
    with p2:
        st.metric("Approval Rate", f"{approved_count / total_apps:.1%}", f"{format_inr(approved_volume)} Approved")
    with p3:
        st.metric("Conditional Review Rate", f"{review_count / total_apps:.1%}", f"{review_count} Dossiers")
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
        st.markdown("##### 📈 Default Probability vs CIBIL Score Dispersion")
        fig_scatter = px.scatter(
            df_eval, x="credit_score", y="default_prob", color="underwriting_decision",
            size="loan_amount", hover_data=["applicant_id", "annual_income", "loan_intent"],
            color_discrete_map={"Approved": "#10b981", "Manual Review": "#f59e0b", "Declined": "#ef4444"},
            labels={"credit_score": "CIBIL Bureau Score (300-900)", "default_prob": "Probability of Default"}
        )
        fig_scatter.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    st.markdown("##### 📋 Underwriting Roster Table (Indian Rupees)")
    filter_choice = st.multiselect("Filter Decision Status:", ["Approved", "Manual Review", "Declined"], default=["Approved", "Manual Review", "Declined"])
    
    display_df = df_eval[df_eval["underwriting_decision"].isin(filter_choice)].copy()
    display_df["annual_income_fmt"] = display_df["annual_income"].apply(format_inr_full)
    display_df["loan_amount_fmt"] = display_df["loan_amount"].apply(format_inr_full)
    display_df["savings_balance_fmt"] = display_df["savings_balance"].apply(format_inr_full)
    
    table_view = display_df[[
        "applicant_id", "age", "education", "annual_income_fmt", "loan_amount_fmt",
        "loan_intent", "savings_balance_fmt", "credit_score", "default_prob", "underwriting_decision", "risk_tier"
    ]].rename(columns={
        "annual_income_fmt": "Annual Income (₹)",
        "loan_amount_fmt": "Loan Amount (₹)",
        "savings_balance_fmt": "Savings (₹)"
    })
    
    st.dataframe(table_view, use_container_width=True, height=320)
    
    csv_data = df_eval.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Scored Indian Loan Portfolio (CSV)",
        csv_data,
        "indian_underwritten_loan_portfolio.csv",
        "text/csv",
        use_container_width=True
    )


# ==============================================================================
# TAB 3: MODEL BENCHMARK & ANALYTICS
# ==============================================================================
elif app_mode == "📊 Model Benchmark & Analytics":
    st.markdown('<div class="author-badge">Lead Investigator: PAIDI VENKAT | Currency: Indian Rupee (₹ INR)</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Model Benchmark & Regulatory Diagnostics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Empirical comparison between Baseline Logistic Regression, Decision Tree, and Proposed Hybrid Model across Basel III and RBI metrics</div>',
        unsafe_allow_html=True
    )
    
    if metrics:
        st.markdown("##### 🏆 Comparative Scorecard Table")
        table_rows = []
        for name, m in metrics.items():
            table_rows.append({
                "Model Candidate": name,
                "ROC-AUC": m["roc_auc"],
                "Gini Index (2·AUC - 1)": m["gini_coefficient"],
                "KS Statistic": m["ks_statistic"],
                "Default Recall (Sensitivity)": f"{m['recall_sensitivity']:.2%}",
                "Precision": f"{m['precision']:.2%}",
                "Brier Score": m["brier_score"],
                "Financial Loss Cost (Pts)": m["financial_loss_cost"]
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
        st.markdown("##### 🔲 Confusion Matrices (Cost-Weighted Banking Loss)")
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
    st.markdown(f'<div class="author-badge">Project Developer: {AUTHOR} | Domain: {DOMAIN}</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Hybrid Architecture & Technical Underwriting Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Design rationale, mathematical formulation, and RBI Digital Lending / Basel III regulatory compliance</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
### 1. Executive Abstract & Motivation
**Investigator:** {AUTHOR}  
**Domain:** {DOMAIN}  
**Project:** {PROJECT_TITLE}  
**Keywords:** {", ".join(KEYWORDS)}  

> **Abstract:**  
> {ABSTRACT}

First-time borrowers in India (e.g. fresh graduates, entry-level professionals, gig workers, and micro-entrepreneurs) represent the **"thin-file / credit invisible" dilemma**:
* Traditional credit bureaus (such as CIBIL / Experian) possess no prior repayment history for this cohort.
* Blanket rejection deprives the bank of high-margin retail expansion, while uncalibrated approval triggers severe Non-Performing Asset (NPA) spikes.
* Standalone **Logistic Regression** provides Basel III interpretability and monotonic log-odds, but fails to automatically model sharp non-linear risk cliffs (such as sudden FOIR/DTI spikes coupled with liquid savings depletion).
* Standalone **Decision Trees** easily partition non-linear risk interactions, but produce stepped, uncalibrated risk probabilities that fluctuate wildly on small sample shifts.

---

### 2. Proposed Tree-Embedding Logistic Hybrid Architecture

```
[Indian Retail Applicant Dossier (INR ₹)]
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

#### Mathematical Formulation:
$$\\ln\\left(\\frac{{P(\\text{{Default}}=1|x)}}{{1 - P(\\text{{Default}}=1|x)}}\\right) = \\beta_0 + \\sum_{{j=1}}^{{M}} \\beta_j x_j + \\sum_{{k=1}}^{{K}} \\gamma_k e_k(x)$$

Where:
* $x_j$ represents continuous financial features (FOIR/DTI, annual income in ₹, loan amount in ₹, savings balance in ₹, interest rate).
* $e_k(x) \\in \\{{0, 1\\}}$ is the one-hot indicator for tree terminal leaf cohort $k$.
* $\\gamma_k$ is the learned non-linear risk interaction coefficient.
* Regularized loss objective guarantees smooth calibration:

$$\\min_{{\\beta, \\gamma}} -\\sum_{{i=1}}^{{N}} \\left[ y_i \\ln(p_i) + (1-y_i) \\ln(1-p_i) \\right] + \\frac{{\\lambda}}{{2}} \\left(\\|\\beta\\|_2^2 + \\|\\gamma\\|_2^2\\right)$$

---

### 3. Banking & Regulatory Compliance (RBI & Basel III)
1. **RBI Digital Lending Guidelines Compliance**:
   All decision paths are fully transparent, auditable, and explainable with zero black-box opacity.
2. **Fair Lending & Non-Discriminatory Scoring**:
   No proxy discrimination variables; model relies strictly on financial capacity, debt serviceability, and alternative utility tracks.
3. **Counterfactual Underwriting Advisor**:
   Instead of a binary rejection, the system delivers precise, actionable stipulations (e.g., adding a solvent guarantor, reducing loan sizing via margin money, or tenure extension).
""")
