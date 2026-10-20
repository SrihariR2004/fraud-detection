# Real-Time Fraud Detection System

XGBoost-based fraud detection on 6.3M financial transactions with SHAP explainability and a FastAPI inference server.

## Results

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression (baseline) | ~0.95 | ~0.95 |
| XGBoost (engineered features) | ~0.999 | ~0.999 |

Confusion matrix (XGBoost, test set of ~1,643 fraud cases): **3 false negatives**.

## Project Structure

```
fraud-detection/
├── fraud_detection.py   # Full pipeline: EDA → features → training → SHAP → ONNX export
├── app.py               # FastAPI inference server
├── shap_importance.png  # Generated after training
├── shap_beeswarm.png    # Generated after training
├── shap_waterfall_fraud.png  # Generated after training
├── fraud_xgb_model.pkl  # Saved model (generated after training)
└── fraud_model.onnx     # ONNX export (generated after training)
```

## Dataset

[Fraud Detection Dataset — Aman Ali Siddiqui](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset/data)  
6.3M transactions · 8,213 fraud cases (0.13% fraud rate)

## Feature Engineering

Raw features alone are weak. These engineered features drove 94%+ of XGBoost importance (SHAP):

| Feature | Description |
|---|---|
| `balance_diff_orig` | Origin balance before − after transaction |
| `balance_diff_dest` | Destination balance after − before transaction |
| `amount_to_orig_ratio` | Transaction amount / origin balance |
| `zero_orig_after_txn` | Flag: origin account drained to 0 |
| `zero_dest_before_txn` | Flag: destination account was empty before |
| `balance_mismatch` | Difference between expected and actual balance change |
| `is_transfer` | 1 = TRANSFER, 0 = CASH_OUT |

Key insight: fraudulent TRANSFER and CASH_OUT transactions almost always drain the origin account to exactly zero — `zero_orig_after_txn` is the strongest single predictor.

## Setup

```bash
pip install xgboost shap scikit-learn fastapi uvicorn joblib matplotlib
# Optional — for ONNX export:
pip install onnxmltools skl2onnx
```

## Run Training

```bash
# Place "AIML Dataset.csv" in the same directory
python fraud_detection.py
```

Outputs:
- `fraud_xgb_model.pkl` — trained model
- `fraud_features.pkl` — feature list
- `shap_importance.png` — global feature importance
- `shap_beeswarm.png` — SHAP beeswarm plot
- `shap_waterfall_fraud.png` — single prediction explanation
- `fraud_model.onnx` — ONNX export (if onnxmltools installed)

## Run API Server

```bash
uvicorn app:app --reload
```

### Predict endpoint

`POST /predict`

```json
{
  "amount": 200000.0,
  "oldbalanceOrg": 200000.0,
  "newbalanceOrig": 0.0,
  "oldbalanceDest": 0.0,
  "newbalanceDest": 200000.0,
  "is_transfer": 1
}
```

Response:

```json
{
  "fraud_probability": 0.9972,
  "is_fraud": true,
  "risk_level": "HIGH",
  "latency_ms": 2.4
}
```

`GET /health` — liveness check

## Why SHAP?

Banking regulators (RBI, Basel III model risk guidelines) require explainability for credit and fraud models. SHAP provides per-prediction feature attributions that satisfy model governance requirements — not just global importance but transaction-level reasoning.

## Tech Stack

- **Modeling:** XGBoost, scikit-learn
- **Explainability:** SHAP
- **Serving:** FastAPI, Uvicorn
- **Export:** ONNX Runtime compatible
- **Language:** Python
