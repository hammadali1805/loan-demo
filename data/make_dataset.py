"""
Creates a realistic (synthetic) Indian home-loan dataset: data/loans.csv

Why synthetic? Real bank data is private. This file follows real lending rules
(credit score, EMI-to-income ratio, loan-to-value) and is deliberately MESSY
(missing values, typos, inconsistent text, duplicates) so students can see
what real-world cleaning looks like.

To use a real dataset instead, swap in any public "Loan Prediction" CSV from
Kaggle and update the column names in the notebook.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 5000

age = rng.integers(21, 61, N)
employment = rng.choice(["Salaried", "Self-Employed", "Business"], N, p=[0.62, 0.23, 0.15])
city_tier = rng.choice(["Tier 1", "Tier 2", "Tier 3"], N, p=[0.45, 0.35, 0.20])
dependents = rng.choice(["0", "1", "2", "3+"], N, p=[0.30, 0.28, 0.27, 0.15])

# Monthly income in rupees (log-normal, higher in Tier 1 cities)
tier_boost = np.select([city_tier == "Tier 1", city_tier == "Tier 2"], [1.25, 1.0], 0.8)
monthly_income = (rng.lognormal(mean=11.2, sigma=0.45, size=N) * tier_boost).round(-2)
coapplicant_income = np.where(rng.random(N) < 0.45,
                              (rng.lognormal(10.8, 0.5, N)).round(-2), 0)

credit_score = np.clip(rng.normal(715, 70, N), 300, 900).round()
existing_emis = np.where(rng.random(N) < 0.55,
                         (monthly_income * rng.uniform(0.05, 0.35, N)).round(-2), 0)

property_value = (monthly_income * rng.uniform(40, 110, N)).round(-4)
loan_amount = (property_value * rng.uniform(0.55, 0.95, N)).round(-4)
loan_term_months = rng.choice([120, 180, 240, 300, 360], N, p=[0.08, 0.17, 0.35, 0.15, 0.25])

# New EMI at ~9% yearly interest
r = 0.09 / 12
new_emi = loan_amount * r * (1 + r) ** loan_term_months / ((1 + r) ** loan_term_months - 1)
total_income = monthly_income + coapplicant_income
foir = (existing_emis + new_emi) / total_income          # Fixed Obligation to Income Ratio
ltv = loan_amount / property_value                        # Loan to Value

# Bank-style decision score (hidden "true" rule + noise)
score = (
    0.022 * (credit_score - 700)
    - 6.0 * (foir - 0.50)
    - 3.0 * (ltv - 0.80)
    + np.select([employment == "Salaried", employment == "Business"], [0.4, -0.1], -0.3)
    - 0.03 * np.maximum(age + loan_term_months / 12 - 65, 0)   # loan running past retirement
    + rng.normal(0, 0.8, N)
    + 0.9
)
approved = np.where(1 / (1 + np.exp(-score)) > rng.random(N), "Y", "N")

df = pd.DataFrame({
    "applicant_id": [f"APP{100000 + i}" for i in range(N)],
    "age": age,
    "employment_type": employment,
    "city_tier": city_tier,
    "dependents": dependents,
    "monthly_income": monthly_income,
    "coapplicant_income": coapplicant_income,
    "credit_score": credit_score,
    "existing_emis": existing_emis,
    "property_value": property_value,
    "loan_amount": loan_amount,
    "loan_term_months": loan_term_months,
    "loan_approved": approved,
})

# ---------- Make it messy, like real data ----------
def blank(col, frac):
    idx = rng.choice(N, int(N * frac), replace=False)
    df.loc[idx, col] = np.nan

blank("credit_score", 0.06)
blank("employment_type", 0.03)
blank("dependents", 0.02)
blank("coapplicant_income", 0.04)

# Inconsistent text typed by different branch staff
idx = rng.choice(N, 300, replace=False)
df.loc[idx, "employment_type"] = df.loc[idx, "employment_type"].map(
    {"Salaried": "salaried ", "Self-Employed": "self employed", "Business": "BUSINESS"})
idx = rng.choice(N, 150, replace=False)
df.loc[idx, "city_tier"] = df.loc[idx, "city_tier"].str.replace("Tier ", "T")

# Data-entry mistakes: income typed with an extra zero, or as 0
idx = rng.choice(N, 15, replace=False)
df.loc[idx, "monthly_income"] *= 10
idx = rng.choice(N, 10, replace=False)
df.loc[idx, "monthly_income"] = 0

# Duplicate rows (same form submitted twice)
df = pd.concat([df, df.sample(60, random_state=1)]).sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("data/loans.csv", index=False)
print(f"Saved data/loans.csv with {len(df)} rows")
print(df["loan_approved"].value_counts(normalize=True).round(3))
