"""
Bonus: turn the ML model into a TOOL that an AI agent can use.

The LLM (Claude) talks to the customer in plain language.
When it needs a decision, it calls our trained model — it does NOT guess.
This "LLM + tools + rules around it" setup is what we call a harness.

Setup:
    pip install anthropic
    export ANTHROPIC_API_KEY=your_key_here
    python agent_demo.py
"""
import json
import joblib
import anthropic
from loan_utils import predict_and_explain

MODEL_NAME = "claude-sonnet-5-5"   # change to any current Claude model you have access to
loan_model = joblib.load("model/loan_model.joblib")
client = anthropic.Anthropic()

TOOLS = [{
    "name": "check_loan_eligibility",
    "description": ("Runs the bank's trained ML model on a home-loan application and returns "
                    "the approval probability, EMI, FOIR, LTV and the main reasons."),
    "input_schema": {
        "type": "object",
        "properties": {
            "age": {"type": "integer"},
            "employment_type": {"type": "string", "enum": ["Salaried", "Self-Employed", "Business"]},
            "city_tier": {"type": "string", "enum": ["Tier 1", "Tier 2", "Tier 3"]},
            "dependents": {"type": "string", "enum": ["0", "1", "2", "3+"]},
            "monthly_income": {"type": "number", "description": "Rupees per month"},
            "coapplicant_income": {"type": "number", "description": "Rupees per month, 0 if none"},
            "credit_score": {"type": "number", "description": "CIBIL score, 300-900"},
            "existing_emis": {"type": "number", "description": "Rupees per month, 0 if none"},
            "property_value": {"type": "number", "description": "Rupees"},
            "loan_amount": {"type": "number", "description": "Rupees"},
            "loan_term_months": {"type": "integer"},
        },
        "required": ["age", "employment_type", "city_tier", "dependents", "monthly_income",
                     "coapplicant_income", "credit_score", "existing_emis",
                     "property_value", "loan_amount", "loan_term_months"],
    },
}]

SYSTEM = """You are a friendly home-loan assistant for an Indian bank.
Rules (this is the 'harness'):
- Never decide approval yourself. Always call check_loan_eligibility.
- If any required detail is missing, ask the customer for it (one short question at a time).
- Explain the result in simple words. If the result is weak, suggest practical fixes
  (bigger down payment, longer tenure, adding a co-applicant, improving the credit score).
- Say clearly this is an estimate, not a final bank decision."""


def chat():
    messages = []
    print("🏦 Loan assistant ready. Describe the loan you want (type 'quit' to exit).\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit"}:
            break
        messages.append({"role": "user", "content": user})

        # Agent loop: keep going while the model wants to use tools
        while True:
            response = client.messages.create(
                model=MODEL_NAME, max_tokens=1000, system=SYSTEM,
                tools=TOOLS, messages=messages)
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                text = "".join(b.text for b in response.content if b.type == "text")
                print(f"\nAssistant: {text}\n")
                break

            results = []
            for block in response.content:
                if block.type == "tool_use":
                    output = predict_and_explain(loan_model, block.input)
                    print(f"   🔧 [model called] {json.dumps(output)}")
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": json.dumps(output)})
            messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    chat()
