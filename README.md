# 🏦 Home Loan Approval Predictor — live class demo

A complete machine-learning project, start to finish:
**Data → Clean → Features → Train → Evaluate → Save → Deploy → Use**

| File | What it does | When to show it |
|---|---|---|
| `data/make_dataset.py` | Creates the messy, realistic loan dataset (`data/loans.csv`) | Step 1: Data |
| `loan_model.ipynb` | The full walkthrough: explore, clean, train, evaluate, save | Steps 1–4 (already run — a ready backup) |
| `loan_utils.py` | Cleaning + feature rules shared by training AND the app | Step 2: "same rules everywhere" |
| `model/loan_model.joblib` | The trained model (≈5 KB) | Step 4: Save |
| `app.py` | Streamlit web app that people can use | Steps 5–6: Deploy & Use |
| `agent_demo.py` | Bonus: an AI agent that calls the model as a tool | Bridge to LLMs / harnesses |

## Run it on your laptop
```bash
pip install -r requirements.txt
python data/make_dataset.py          # optional: data/loans.csv is already included
jupyter notebook loan_model.ipynb    # or open it in Google Colab / VS Code
streamlit run app.py
```

## Run in Google Colab
Upload `loan_model.ipynb`, `loan_utils.py` and the `data/` folder, then Runtime → Run all.

## Deploy free (do this BEFORE the class)
**Streamlit Community Cloud**
1. Push this folder to a public GitHub repo (include `model/loan_model.joblib`).
2. Go to share.streamlit.io → New app → pick the repo → main file `app.py` → Deploy.
3. Turn the app URL into a QR code for the slides.

**Hugging Face Spaces (alternative)**
New Space → SDK: Streamlit → upload all files → it builds automatically.

## Bonus agent
```bash
export ANTHROPIC_API_KEY=your_key
python agent_demo.py
```
Try: *"I'm 29, salaried in Pune, earn 1.1 lakh a month, CIBIL 760, no EMIs. I want a 65 lakh loan on an 85 lakh flat for 20 years."*

## Results you will see (test set, 1,000 applicants)
| Model | Accuracy | Precision (approved) | Recall (good customers) | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.819 | 0.835 | 0.921 | 0.867 |
| Random Forest | 0.804 | 0.821 | 0.916 | 0.851 |

The simple model wins — a great teaching moment.

⚠️ The data is synthetic and the model is for teaching only — not for real lending decisions.
