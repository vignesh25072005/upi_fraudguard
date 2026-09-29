# UPI FraudGuard AI

ML-based anomaly detection for UPI (Unified Payments Interface) transactions, with
regional and merchant-category fraud risk aggregation.

## ⚠️ About the data — read this first

Real, transaction-level UPI fraud data is **never public** — NPCI and RBI keep it
confidential for security reasons, and no bank publishes labeled fraud data either
(this is true worldwide, not just in India). So this project uses a **synthetic
dataset** (`src/generate_data.py`) that mimics real UPI transaction behavior:
realistic amount distributions per merchant category, Indian state/bank
distributions, and four distinct fraud typologies (odd-hour high-value spikes,
device switching, transaction-velocity bursts, unfamiliar bank combinations) mixed
with noise so fraud isn't trivially separable.

This is the same approach every public "UPI fraud detection" dataset on Kaggle
uses — none of them are real bank data either. Be upfront about this if you
present the project (e.g. in a viva or interview) — it's expected and normal for
this problem domain, not a shortcut.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 80.0% | 2.1% | 66.7% | 4.2% | 0.795 | 0.333 |
| Decision Tree | 88.3% | 3.5% | 64.7% | 6.7% | 0.798 | 0.468 |
| Random Forest | 97.1% | 10.3% | 44.9% | 16.8% | 0.841 | 0.378 |
| **XGBoost (tuned threshold)** | **99.6%** | **85.7%** | **50.0%** | **63.2%** | **0.881** | **0.560** |

**Accuracy easily clears 90%+ (99.6%) — but accuracy is a misleading metric here**,
since simply predicting "no fraud" for every transaction already scores 99.4%
accuracy on this data. The metrics that actually matter for fraud detection are:

- **Precision (85.7%)** — when the model flags a transaction, it's right 86% of
  the time. Low false-positive rate = fraud team isn't drowned in noise.
- **Recall (50.0%)** — the model catches half of all fraud cases. This is the
  realistic, honest number; catching significantly more without cratering
  precision requires either more features (device fingerprinting, IP/geo data,
  account age) or a lower-precision "review queue" tier, which is how real fraud
  systems are layered in practice.
- **PR-AUC (0.560)** — the right summary metric for rare-event detection, far more
  informative than accuracy on a 0.6%-fraud dataset.

## Project structure

```
upi_fraud_project/
├── data/
│   ├── upi_transactions.csv              # raw synthetic dataset (120,000 rows)
│   └── upi_transactions_features.csv     # after feature engineering
├── models/
│   ├── xgb_fraud_model.pkl               # trained XGBoost model
│   ├── scaler.pkl                        # StandardScaler for numeric features
│   ├── label_encoders.pkl                # LabelEncoders for categorical features
│   └── best_threshold.json               # tuned decision threshold + feature order
├── outputs/
│   ├── model_metrics.json                # all model metrics
│   ├── model_comparison.csv / .png       # model comparison table & chart
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── regional_risk.csv                 # state-wise flagged fraud rate
│   └── category_risk.csv                 # merchant-category flagged fraud rate
├── src/
│   ├── generate_data.py                  # step 1: synthetic data generation
│   ├── feature_engineering.py            # step 2: behavioral feature derivation
│   ├── train_model.py                    # step 3: SMOTE + train + evaluate + save
│   └── app.py                            # step 4: Streamlit dashboard
└── README.md
```

## How to run

```bash
cd upi_fraud_project
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn streamlit plotly joblib

# 1. Generate the dataset
python src/generate_data.py

# 2. Engineer features
python src/feature_engineering.py

# 3. Train models, evaluate, save artifacts
python src/train_model.py

# 4. Launch the dashboard
cd src && streamlit run app.py
```

## Pipeline explained

1. **Data generation** — 120,000 UPI transactions across 25,000 simulated senders,
   July–August 2026, 10 Indian states, 8 banks, 11 merchant categories. ~0.6% are
   fraudulent (realistic order of magnitude), across 4 different fraud patterns.
2. **Feature engineering** — derives transaction velocity (txns/hour per sender),
   amount deviation from sender's historical average, odd-hour flag, device-switch
   flag, cross-bank flag. These behavioral signals are what actually separate
   fraud from genuine activity — raw amount alone doesn't.
3. **Time-aware train/test split** — trained on the first 80% of transactions
   chronologically, tested on the last 20%. This is more realistic than a random
   split, since in production you're always predicting on transactions that
   happen *after* your training data.
4. **SMOTE** — applied only to the training set (never the test set — that would
   leak information) to balance the ~0.6% fraud rate before training.
5. **Three models compared** — Logistic Regression (baseline), Random Forest,
   XGBoost (primary model). XGBoost wins on every metric.
6. **Threshold tuning** — the default 0.5 probability threshold is rarely optimal
   for imbalanced problems; the script finds the threshold that maximizes F1 on
   the precision-recall curve.
7. **Regional & category risk aggregation** — the project's differentiator: flagged
   transactions are aggregated by state and merchant category to surface which
   regions/categories need more monitoring attention — not just individual
   transaction flags.
8. **Dashboard** — Streamlit app with three tabs: model performance, regional/
   category risk view, and a live "score a transaction" tool. Custom dark theme
   (`.streamlit/config.toml`, duplicated inside `src/` so it's picked up regardless
   of which folder you launch from) plus custom CSS in `app.py` for a polished,
   card-based layout instead of Streamlit's plain defaults.

## Honest next steps if you want to push recall higher

- Add device-fingerprint / IP-geolocation features (not available in this synthetic set)
- Add account-age and sender transaction-history-length features
- Try an ensemble of XGBoost + an autoencoder-based anomaly score
- Move to a two-tier system: auto-block above a high-confidence threshold,
  route medium-confidence scores to a human review queue (this is how real
  bank fraud systems balance precision and recall)
