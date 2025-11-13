# fraud_service/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
from sklearn.ensemble import IsolationForest
import os
import openai
import uvicorn

openai.api_key = os.getenv("OPENAI_API_KEY")  # GPT-4o-mini key

app = FastAPI(title="CrediLens Fraud Service")

# ---------- ✅ ADD THIS HEALTH CHECK ----------
@app.get("/")
def health_check():
    return {"status": "ok", "service": "CrediLens Fraud API is running"}

# ---------- Request/Response models ----------
class Transaction(BaseModel):
    date: str
    description: str
    amount: float
    type: str  # 'credit' or 'debit'
    balance: float | None = None

class FraudRequest(BaseModel):
    transactions: List[Transaction]
    metadata: Dict[str, Any] = {}

class FraudResponse(BaseModel):
    fraud_score: int
    fraud_flags: List[str]
    confidence: float
    merchant_classifications: Dict[str, str] = {}

# ---------- Helper: call GPT-4o-mini to classify merchants ----------
def classify_merchants_with_gpt(descriptions: List[str]) -> Dict[str, str]:
    text_prompt = (
        "You are an assistant that maps a short bank transaction description to one of: "
        "salary, informal_income, grocery, transport, utilities, loan_repayment, gambling, "
        "entertainment, savings, bank_fee, other. Return JSON mapping description->category. "
        "If uncertain, use 'other'.\n\n"
        "DESCRIPTIONS:\n" + "\n".join(f"- {d}" for d in descriptions) +
        "\n\nReturn JSON object like {\"desc\":\"category\", ...}."
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": text_prompt}],
            max_tokens=512,
            temperature=0.0
        )
        reply = resp["choices"][0]["message"]["content"]
        import json, re
        json_text = re.search(r'\{.*\}', reply, flags=re.S)
        if not json_text:
            return {}
        return json.loads(json_text.group(0))
    except Exception:
        return {}

# ---------- Fraud rules + ML ----------
def run_rule_checks(transactions):
    flags = set()
    balances = [t.balance for t in transactions if t.balance is not None]
    if balances:
        opening = balances[0]
        closing = balances[-1]
        flow = sum([t.amount if t.type == "credit" else -t.amount for t in transactions])
        if abs(opening + flow - closing) > max(1, abs(flow) * 0.02):
            flags.add("balance_mismatch")

    seen = {}
    for t in transactions:
        key = (round(t.amount, 2), t.description.lower()[:30])
        if key in seen:
            flags.add("duplicate_transaction")
        else:
            seen[key] = True

    dates = [t.date for t in transactions]
    if len(set(dates)) < max(10, len(transactions)//3):
        flags.add("incomplete_period")

    neg_count = sum(1 for b in balances if b is not None and b < 0)
    if balances and (neg_count / len(balances)) > 0.3:
        flags.add("chronic_overdraft")

    return list(flags)

def compute_ml_anomaly_score(transactions):
    amounts = np.array([abs(t.amount) for t in transactions]).reshape(-1, 1)
    if len(amounts) < 2:
        return 0.0
    try:
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(amounts)
        preds = model.predict(amounts)
        anomaly = int((-1 in preds))
        score = int(min(100, anomaly * 60 + np.std(amounts)))
        return score
    except Exception:
        return 0

@app.post("/fraud", response_model=FraudResponse)
async def detect_fraud(req: FraudRequest):
    txs = req.transactions
    if not txs:
        raise HTTPException(status_code=400, detail="No transactions provided")

    descriptions = [t.description for t in txs][:50]
    merchant_map = classify_merchants_with_gpt(descriptions)
    for d in descriptions:
        if d not in merchant_map:
            low = d.lower()
            if any(x in low for x in ["sassa","grant","gov"]):
                merchant_map[d] = "informal_income"
            elif any(x in low for x in ["shoprite","checkers","pick n pay","picknpay"]):
                merchant_map[d] = "grocery"
            elif any(x in low for x in ["bet","betway","casino","gamenet"]):
                merchant_map[d] = "gambling"
            else:
                merchant_map[d] = "other"

    rule_flags = run_rule_checks(txs)
    ml_score = compute_ml_anomaly_score(txs)

    fraud_flags = list(set(rule_flags))
    gambling_count = sum(1 for d, cat in merchant_map.items() if cat == "gambling")
    if gambling_count > 0:
        fraud_flags.append("gambling_pattern")

    fraud_score = min(100, int(ml_score + len(fraud_flags) * 15))
    confidence = 0.7 + (0.25 if ml_score > 10 or fraud_flags else 0.0)

    return FraudResponse(
        fraud_score=fraud_score,
        fraud_flags=fraud_flags,
        confidence=round(confidence, 2),
        merchant_classifications=merchant_map
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
