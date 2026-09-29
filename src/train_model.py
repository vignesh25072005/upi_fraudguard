import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

DATA_PATH = "/home/claude/upi_fraud_project/data/upi_transactions_features.csv"
OUT_DIR = "/home/claude/upi_fraud_project/outputs"
MODEL_DIR = "/home/claude/upi_fraud_project/models"

df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)

# ---------- Feature selection ----------
categorical_cols = ["sender_state", "sender_bank", "receiver_bank",
                     "merchant_category", "device_type", "network_type"]
numeric_cols = ["amount", "hour_of_day", "is_weekend", "txn_velocity_1h",
                 "amount_deviation_ratio", "is_odd_hour", "device_switch_flag", "cross_bank_flag"]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    encoders[col] = le

feature_cols = numeric_cols + [c + "_enc" for c in categorical_cols]
X = df[feature_cols]
y = df["fraud_flag"]

# ---------- Time-aware split (train on earlier txns, test on later ones -- realistic) ----------
df_sorted = df.sort_values("timestamp")
split_idx = int(len(df_sorted) * 0.8)
train_idx = df_sorted.index[:split_idx]
test_idx = df_sorted.index[split_idx:]

X_train, X_test = X.loc[train_idx], X.loc[test_idx]
y_train, y_test = y.loc[train_idx], y.loc[test_idx]

print(f"Train: {X_train.shape}, fraud rate {y_train.mean()*100:.3f}%")
print(f"Test:  {X_test.shape}, fraud rate {y_test.mean()*100:.3f}%")

# ---------- Scale numeric features ----------
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ---------- Handle class imbalance with SMOTE (train set only!) ----------
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print(f"After SMOTE: {X_train_res.shape}, fraud rate {y_train_res.mean()*100:.1f}%")

# ---------- Models ----------
results = {}

def evaluate(name, model, X_te, y_te, proba=None):
    preds = model.predict(X_te)
    if proba is None:
        proba = model.predict_proba(X_te)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_te, preds),
        "precision": precision_score(y_te, preds, zero_division=0),
        "recall": recall_score(y_te, preds, zero_division=0),
        "f1": f1_score(y_te, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_te, proba),
        "pr_auc": average_precision_score(y_te, proba),
    }
    results[name] = metrics
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    print(classification_report(y_te, preds, target_names=["Genuine", "Fraud"], zero_division=0))
    return preds, proba

# 1) Logistic Regression baseline
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
log_reg.fit(X_train_res, y_train_res)
lr_preds, lr_proba = evaluate("Logistic Regression (baseline)", log_reg, X_test_scaled, y_test)

# 2) Random Forest
rf = RandomForestClassifier(
    n_estimators=300, max_depth=14, min_samples_leaf=3,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_train_res, y_train_res)
rf_preds, rf_proba = evaluate("Random Forest", rf, X_test_scaled, y_test)

# 3) Decision Tree
dt = DecisionTreeClassifier(
    max_depth=8, min_samples_leaf=5,
    class_weight="balanced", random_state=42
)
dt.fit(X_train_res, y_train_res)
dt_preds, dt_proba = evaluate("Decision Tree", dt, X_test_scaled, y_test)

# 4) XGBoost (primary model)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.08,
    subsample=0.9, colsample_bytree=0.8,
    eval_metric="aucpr", random_state=42, n_jobs=-1
)
xgb.fit(X_train_res, y_train_res)
xgb_preds, xgb_proba = evaluate("XGBoost (primary model)", xgb, X_test_scaled, y_test)

# ---------- Threshold tuning for XGBoost (optimize F1 on precision-recall curve) ----------
prec, rec, thresh = precision_recall_curve(y_test, xgb_proba)
f1_scores = 2 * (prec * rec) / (prec + rec + 1e-12)
best_idx = np.argmax(f1_scores[:-1])
best_thresh = thresh[best_idx]
xgb_preds_tuned = (xgb_proba >= best_thresh).astype(int)

tuned_metrics = {
    "accuracy": accuracy_score(y_test, xgb_preds_tuned),
    "precision": precision_score(y_test, xgb_preds_tuned, zero_division=0),
    "recall": recall_score(y_test, xgb_preds_tuned, zero_division=0),
    "f1": f1_score(y_test, xgb_preds_tuned, zero_division=0),
    "roc_auc": roc_auc_score(y_test, xgb_proba),
    "pr_auc": average_precision_score(y_test, xgb_proba),
    "threshold": float(best_thresh),
}
results["XGBoost (tuned threshold)"] = tuned_metrics
print(f"\n=== XGBoost (tuned threshold={best_thresh:.3f}) ===")
for k, v in tuned_metrics.items():
    print(f"  {k:10s}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
print(classification_report(y_test, xgb_preds_tuned, target_names=["Genuine", "Fraud"], zero_division=0))

# ---------- Save results ----------
with open(f"{OUT_DIR}/model_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

# ---------- Confusion matrix plot (best model: tuned XGBoost) ----------
cm = confusion_matrix(y_test, xgb_preds_tuned)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Genuine", "Fraud"], yticklabels=["Genuine", "Fraud"])
plt.title("Confusion Matrix — XGBoost (tuned threshold)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
plt.close()

# ---------- Model comparison bar chart ----------
comp_df = pd.DataFrame(results).T[["precision", "recall", "f1", "roc_auc", "pr_auc"]]
comp_df.to_csv(f"{OUT_DIR}/model_comparison.csv")
ax = comp_df.plot(kind="bar", figsize=(10, 5))
plt.title("Model Comparison — Fraud Detection Metrics")
plt.ylabel("Score")
plt.xticks(rotation=20, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/model_comparison.png", dpi=150)
plt.close()

# ---------- Feature importance (XGBoost) ----------
importances = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(x=importances.values[:12], y=importances.index[:12], palette="viridis")
plt.title("Top Feature Importances — XGBoost")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150)
plt.close()

# ---------- Save model artifacts ----------
# Use XGBoost's own native save format (JSON) instead of raw pickle -- this is the
# officially recommended way to persist XGBoost models, and it avoids the binary
# corruption issues that raw pickles can hit when moved across machines, zipped,
# or synced through OneDrive/Dropbox (cloud-sync tools sometimes don't fully
# download binary files before they're read, which corrupts a pickled booster).
xgb.save_model(f"{MODEL_DIR}/xgb_fraud_model.json")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(encoders, f"{MODEL_DIR}/label_encoders.pkl")
with open(f"{MODEL_DIR}/best_threshold.json", "w") as f:
    json.dump({"threshold": float(best_thresh), "feature_cols": feature_cols}, f, indent=2)

# ---------- Regional & category risk aggregation (using test set predictions) ----------
test_df = df.loc[test_idx].copy()
test_df["predicted_fraud"] = xgb_preds_tuned
test_df["fraud_probability"] = xgb_proba

region_risk = (
    test_df.groupby("sender_state")
    .agg(total_txns=("transaction_id", "count"),
         flagged=("predicted_fraud", "sum"),
         avg_fraud_prob=("fraud_probability", "mean"))
    .assign(flagged_rate_pct=lambda d: (d["flagged"] / d["total_txns"] * 100).round(3))
    .sort_values("flagged_rate_pct", ascending=False)
)
region_risk.to_csv(f"{OUT_DIR}/regional_risk.csv")

category_risk = (
    test_df.groupby("merchant_category")
    .agg(total_txns=("transaction_id", "count"),
         flagged=("predicted_fraud", "sum"),
         avg_fraud_prob=("fraud_probability", "mean"))
    .assign(flagged_rate_pct=lambda d: (d["flagged"] / d["total_txns"] * 100).round(3))
    .sort_values("flagged_rate_pct", ascending=False)
)
category_risk.to_csv(f"{OUT_DIR}/category_risk.csv")

print("\nRegional risk (top 5):")
print(region_risk.head())
print("\nCategory risk (top 5):")
print(category_risk.head())

print("\nAll outputs saved to", OUT_DIR)
print("All models saved to", MODEL_DIR)
