import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("AIML Dataset.csv")
print(df.head())
print("\nFraud distribution:")
print(df["isFraud"].value_counts())

# ─────────────────────────────────────────────
# 2. EDA — fraud rate by transaction type
# ─────────────────────────────────────────────
print("\nFraud rate by type (%):")
print(df["isFraud"].groupby(df['type']).mean().mul(100).sort_values(ascending=False))
# TRANSFER ~0.77%  CASH_OUT ~0.18%  — all others 0

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
df_model = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()

# Fraudsters drain accounts to exactly 0 — capture that signal
df_model['balance_diff_orig']    = df_model['oldbalanceOrg'] - df_model['newbalanceOrig']
df_model['balance_diff_dest']    = df_model['newbalanceDest'] - df_model['oldbalanceDest']
df_model['amount_to_orig_ratio'] = df_model['amount'] / (df_model['oldbalanceOrg'] + 1)
df_model['zero_orig_after_txn']  = (df_model['newbalanceOrig'] == 0).astype(int)
df_model['zero_dest_before_txn'] = (df_model['oldbalanceDest'] == 0).astype(int)
df_model['balance_mismatch']     = np.abs(
    (df_model['oldbalanceOrg'] - df_model['newbalanceOrig']) - df_model['amount']
)
df_model['is_transfer'] = (df_model['type'] == 'TRANSFER').astype(int)

FEATURES = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'balance_diff_orig', 'balance_diff_dest',
    'amount_to_orig_ratio', 'zero_orig_after_txn',
    'zero_dest_before_txn', 'balance_mismatch', 'is_transfer'
]

# ─────────────────────────────────────────────
# 4. CLASS BALANCING (undersample majority)
# ─────────────────────────────────────────────
fraud     = df_model[df_model['isFraud'] == 1]
non_fraud = df_model[df_model['isFraud'] == 0]
non_fraud_sample = non_fraud.sample(n=len(fraud), random_state=42)

df_balanced = (
    pd.concat([fraud, non_fraud_sample])
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)
print("\nBalanced class distribution:")
print(df_balanced['isFraud'].value_counts())

# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
from sklearn.model_selection import train_test_split

X = df_balanced[FEATURES]
y = df_balanced['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")

# ─────────────────────────────────────────────
# 6. BASELINE — LOGISTIC REGRESSION
# ─────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

print("\n── Logistic Regression ──")
print(classification_report(y_test, y_pred_lr))
print(f"ROC-AUC : {roc_auc_score(y_test, y_prob_lr):.4f}")
print(f"PR-AUC  : {average_precision_score(y_test, y_prob_lr):.4f}")

# ─────────────────────────────────────────────
# 7. XGBOOST MODEL
# ─────────────────────────────────────────────
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='aucpr',
    random_state=42
)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("\n── XGBoost ──")
print(classification_report(y_test, y_pred_xgb))
print(f"ROC-AUC : {roc_auc_score(y_test, y_prob_xgb):.4f}")
print(f"PR-AUC  : {average_precision_score(y_test, y_prob_xgb):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

# ─────────────────────────────────────────────
# 8. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────
import shap
import matplotlib.pyplot as plt

explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Global feature importance bar chart
shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig("shap_importance.png", dpi=150)
plt.close()
print("Saved shap_importance.png")

# Beeswarm — direction of impact per feature
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=150)
plt.close()
print("Saved shap_beeswarm.png")

# Waterfall — explain one fraud prediction
fraud_idx  = y_test[y_test == 1].index[0]
sample_pos = X_test.index.get_loc(fraud_idx)
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[sample_pos],
        base_values=explainer.expected_value,
        data=X_test.iloc[sample_pos],
        feature_names=FEATURES
    ),
    show=False
)
plt.tight_layout()
plt.savefig("shap_waterfall_fraud.png", dpi=150)
plt.close()
print("Saved shap_waterfall_fraud.png")

# ─────────────────────────────────────────────
# 9. SAVE MODEL
# ─────────────────────────────────────────────
import joblib

joblib.dump(xgb_model, 'fraud_xgb_model.pkl')
joblib.dump(FEATURES,  'fraud_features.pkl')
print("\nModel saved → fraud_xgb_model.pkl")

# ─────────────────────────────────────────────
# 10. EXPORT TO ONNX  (pip install onnxmltools skl2onnx)
# ─────────────────────────────────────────────
try:
    from skl2onnx.common.data_types import FloatTensorType
    from onnxmltools import convert_xgboost

    initial_type = [('float_input', FloatTensorType([None, len(FEATURES)]))]
    onnx_model   = convert_xgboost(xgb_model, initial_types=initial_type)
    with open('fraud_model.onnx', 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print("ONNX model saved → fraud_model.onnx")
except ImportError:
    print("Skipping ONNX export — install onnxmltools & skl2onnx to enable")
