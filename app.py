"""
Home Loan Approval Predictor — Streamlit app
Run locally:   streamlit run app.py
"""
import joblib
import streamlit as st
from loan_utils import predict_and_explain

st.set_page_config(page_title="Home Loan Predictor", page_icon="🏦", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("model/loan_model.joblib")


model = load_model()

st.title("🏦 Home Loan Approval Predictor")
st.caption("Built live in class · Data → Train → Evaluate → Deploy → Use · "
           "Demo model on synthetic data — not real financial advice.")

with st.form("applicant"):
    st.subheader("Applicant")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 21, 65, 30)
    employment_type = c2.selectbox("Employment", ["Salaried", "Self-Employed", "Business"])
    city_tier = c3.selectbox("City", ["Tier 1", "Tier 2", "Tier 3"])

    c1, c2, c3 = st.columns(3)
    monthly_income = c1.number_input("Monthly income (₹)", 10_000, 2_000_000, 90_000, step=5_000)
    coapplicant_income = c2.number_input("Co-applicant income (₹)", 0, 2_000_000, 0, step=5_000)
    dependents = c3.selectbox("Dependents", ["0", "1", "2", "3+"])

    c1, c2 = st.columns(2)
    credit_score = c1.slider("Credit score (CIBIL)", 300, 900, 740)
    existing_emis = c2.number_input("Existing EMIs per month (₹)", 0, 1_000_000, 5_000, step=1_000)

    st.subheader("Loan")
    c1, c2, c3 = st.columns(3)
    property_value = c1.number_input("Property value (₹)", 500_000, 200_000_000, 7_500_000, step=100_000)
    loan_amount = c2.number_input("Loan amount (₹)", 100_000, 200_000_000, 5_500_000, step=100_000)
    years = c3.selectbox("Tenure (years)", [10, 15, 20, 25, 30], index=2)

    threshold = st.slider("Bank strictness — approve if probability is at least", 0.3, 0.9, 0.5, 0.05,
                          help="Higher = fewer risky loans, but more good customers turned away.")
    submitted = st.form_submit_button("Check approval", type="primary", use_container_width=True)

if submitted:
    result = predict_and_explain(model, {
        "age": age, "employment_type": employment_type, "city_tier": city_tier,
        "dependents": dependents, "monthly_income": monthly_income,
        "coapplicant_income": coapplicant_income, "credit_score": credit_score,
        "existing_emis": existing_emis, "property_value": property_value,
        "loan_amount": loan_amount, "loan_term_months": years * 12,
    }, threshold)

    p = result["approval_probability"]
    if result["decision"] == "APPROVE":
        st.success(f"✅ Likely APPROVED — approval probability {p:.0%}")
    else:
        st.error(f"❌ Likely REJECTED — approval probability {p:.0%}")
    st.progress(p)

    c1, c2, c3 = st.columns(3)
    c1.metric("New EMI / month", f"₹{result['monthly_emi_rupees']:,}")
    c2.metric("EMI burden (FOIR)", f"{result['foir_percent']}%",
              help="Share of total income going to EMIs. Banks prefer under ~50–60%.")
    c3.metric("Loan-to-value", f"{result['ltv_percent']}%",
              help="Loan as a share of the property's value.")

    st.caption("Reasons compare this applicant with an average applicant in the training data.")
    c1, c2 = st.columns(2)
    c1.markdown("**👍 Helping this application**\n" +
                "".join(f"\n- {x}" for x in result["helping"]) if result["helping"] else "")
    c2.markdown("**👎 Hurting this application**\n" +
                "".join(f"\n- {x}" for x in result["hurting"]) if result["hurting"] else "")
