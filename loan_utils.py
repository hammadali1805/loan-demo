"""
Shared cleaning + feature code.
The notebook (training) and the app (prediction) BOTH use this file,
so the model always sees data prepared in exactly the same way.
"""
import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "age", "monthly_income", "coapplicant_income", "credit_score",
    "existing_emis", "loan_amount", "loan_term_months",
    "new_emi", "foir", "ltv", "years_past_60",
]
CATEGORICAL_FEATURES = ["employment_type", "city_tier", "dependents"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "loan_approved"


def monthly_emi(loan_amount, term_months, yearly_rate=0.09):
    """Standard EMI formula used by Indian banks."""
    r = yearly_rate / 12
    return loan_amount * r * (1 + r) ** term_months / ((1 + r) ** term_months - 1)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the messy text and obvious data-entry mistakes."""
    df = df.copy()

    # 1. Same thing written in different ways -> one standard label
    emp = df["employment_type"].astype("string").str.strip().str.lower()
    df["employment_type"] = emp.map({
        "salaried": "Salaried",
        "self employed": "Self-Employed", "self-employed": "Self-Employed",
        "business": "Business",
    })
    df["city_tier"] = (df["city_tier"].astype("string").str.strip()
                       .str.replace(r"^T(\d)$", r"Tier \1", regex=True))
    df["dependents"] = df["dependents"].astype("string").str.strip()
    # Use plain Python objects with np.nan for "missing" (what scikit-learn expects)
    for col in ["employment_type", "city_tier", "dependents"]:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)

    # 2. Impossible values -> treat as missing (the model pipeline fills them in)
    df.loc[df["monthly_income"] <= 0, "monthly_income"] = np.nan
    # Income typed with an extra zero (more than 50 lakh a month) -> missing
    df.loc[df["monthly_income"] > 5_000_000, "monthly_income"] = np.nan
    df["coapplicant_income"] = df["coapplicant_income"].fillna(0)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ratios a real loan officer looks at."""
    df = df.copy()
    total_income = df["monthly_income"] + df["coapplicant_income"]
    df["new_emi"] = monthly_emi(df["loan_amount"], df["loan_term_months"])
    # FOIR: what share of income goes to EMIs (banks like it under ~50-60%)
    df["foir"] = (df["existing_emis"] + df["new_emi"]) / total_income
    # LTV: loan as a share of property value (RBI caps it at 75-90%)
    df["ltv"] = df["loan_amount"] / df["property_value"]
    # Years the loan runs past age 60 (retirement risk)
    df["years_past_60"] = np.maximum(df["age"] + df["loan_term_months"] / 12 - 60, 0)
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    return add_features(clean(df))


# ---------------------------------------------------------------------------
# Used by the app and the agent: predict + explain in plain words
# ---------------------------------------------------------------------------
FRIENDLY_NAMES = {
    "foir": "EMI burden (share of income going to EMIs)",
    "credit_score": "Credit score (CIBIL)",
    "employment_type": "Type of employment",
    "new_emi": "Size of the new EMI",
    "loan_amount": "Loan amount",
    "city_tier": "City tier",
    "dependents": "Number of dependents",
    "ltv": "Loan-to-property-value ratio",
    "years_past_60": "Loan running past age 60",
    "monthly_income": "Monthly income",
    "coapplicant_income": "Co-applicant income",
    "existing_emis": "Existing EMIs",
    "loan_term_months": "Loan tenure",
    "age": "Age",
}


def predict_and_explain(model, applicant: dict, threshold: float = 0.5) -> dict:
    """Return probability, decision, key ratios and the top reasons."""
    row = prepare(pd.DataFrame([applicant]))
    X = row[FEATURES]
    prob = float(model.predict_proba(X)[0, 1])

    # For logistic regression: contribution of each feature = coefficient x (scaled) value
    prep, lr = model.named_steps["prep"], model.named_steps["model"]
    values = prep.transform(X)
    values = values.toarray()[0] if hasattr(values, "toarray") else np.asarray(values)[0]
    contrib = pd.Series(values * lr.coef_[0], index=prep.get_feature_names_out())
    contrib.index = [n.split("__", 1)[1] for n in contrib.index]
    # one-hot columns like "city_tier_Tier 1" -> back to "city_tier"
    contrib.index = [next((c for c in CATEGORICAL_FEATURES if n.startswith(c + "_")), n)
                     for n in contrib.index]
    contrib = contrib.groupby(level=0).sum()
    contrib = contrib[contrib.abs() > 0.05].sort_values()

    return {
        "approval_probability": round(prob, 3),
        "decision": "APPROVE" if prob >= threshold else "REJECT",
        "threshold": threshold,
        "monthly_emi_rupees": round(float(row["new_emi"].iloc[0])),
        "foir_percent": round(float(row["foir"].iloc[0]) * 100, 1),
        "ltv_percent": round(float(row["ltv"].iloc[0]) * 100, 1),
        "helping": [FRIENDLY_NAMES.get(k, k) for k in contrib[contrib > 0].index[::-1][:3]],
        "hurting": [FRIENDLY_NAMES.get(k, k) for k in contrib[contrib < 0].index[:3]],
    }
