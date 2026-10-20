from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import time

app = FastAPI(title="Fraud Detection API")

model    = joblib.load("fraud_xgb_model.pkl")
FEATURES = joblib.load("fraud_features.pkl")


class Transaction(BaseModel):
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    is_transfer: int  # 1 = TRANSFER, 0 = CASH_OUT


def build_features(t: Transaction) -> np.ndarray:
    balance_diff_orig    = t.oldbalanceOrg - t.newbalanceOrig
    balance_diff_dest    = t.newbalanceDest - t.oldbalanceDest
    amount_to_orig_ratio = t.amount / (t.oldbalanceOrg + 1)
    zero_orig_after_txn  = int(t.newbalanceOrig == 0)
    zero_dest_before_txn = int(t.oldbalanceDest == 0)
    balance_mismatch     = abs(balance_diff_orig - t.amount)
    return np.array([[
        t.amount, t.oldbalanceOrg, t.newbalanceOrig,
        t.oldbalanceDest, t.newbalanceDest,
        balance_diff_orig, balance_diff_dest,
        amount_to_orig_ratio, zero_orig_after_txn,
        zero_dest_before_txn, balance_mismatch, t.is_transfer
    ]])


@app.post("/predict")
def predict(transaction: Transaction):
    start = time.perf_counter()
    X     = build_features(transaction)
    prob  = float(model.predict_proba(X)[0][1])
    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "fraud_probability": round(prob, 4),
        "is_fraud":  prob >= 0.5,
        "risk_level": "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW",
        "latency_ms": round(latency_ms, 2)
    }


@app.get("/health")
def health():
    return {"status": "ok"}
