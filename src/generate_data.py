"""
Generates a synthetic but behaviorally realistic UPI transaction dataset for India,
with a small, deliberately hard-to-spot fraction of fraudulent transactions.

Real, labeled fraud data is never public (RBI/NPCI keep it confidential), so every
UPI fraud-detection project -- including the Kaggle ones -- uses simulated data like this.
The point is to learn the MODELING PIPELINE (imbalance handling, feature engineering,
evaluation) on data that mimics real transaction behavior.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N_TRANSACTIONS = 120_000
FRAUD_RATE = 0.006  # ~0.6% fraud -- realistic order of magnitude for payment fraud

states = ["Maharashtra", "Karnataka", "Delhi", "Tamil Nadu", "West Bengal",
          "Gujarat", "Rajasthan", "Uttar Pradesh", "Andhra Pradesh", "Telangana"]
state_weights = [0.16, 0.11, 0.09, 0.10, 0.08, 0.08, 0.07, 0.13, 0.09, 0.09]

banks = ["SBI", "HDFC", "ICICI", "Axis", "PNB", "Kotak", "IndusInd", "Yes Bank"]
bank_weights = [0.24, 0.18, 0.16, 0.11, 0.09, 0.08, 0.08, 0.06]

merchant_categories = ["Food", "Grocery", "Fuel", "Entertainment", "Shopping",
                        "Healthcare", "Education", "Transport", "Utilities", "P2P Transfer", "Other"]
merchant_weights = [0.14, 0.13, 0.08, 0.07, 0.13, 0.06, 0.04, 0.09, 0.09, 0.13, 0.04]

devices = ["Android", "iOS", "Web"]
device_weights = [0.78, 0.18, 0.04]

networks = ["4G", "5G", "WiFi", "3G"]
network_weights = [0.55, 0.25, 0.18, 0.02]

# category-specific typical amount ranges (INR)
category_amount_range = {
    "Food": (100, 900), "Grocery": (150, 2500), "Fuel": (200, 3000),
    "Entertainment": (150, 1500), "Shopping": (300, 8000), "Healthcare": (200, 5000),
    "Education": (500, 20000), "Transport": (30, 800), "Utilities": (200, 6000),
    "P2P Transfer": (100, 25000), "Other": (100, 3000),
}

n_senders = 25_000
sender_ids = [f"USR{100000+i}" for i in range(n_senders)]
sender_state = np.random.choice(states, size=n_senders, p=state_weights)
sender_bank = np.random.choice(banks, size=n_senders, p=bank_weights)
sender_home_device = np.random.choice(devices, size=n_senders, p=device_weights)

start_date = datetime(2026, 7, 1)
end_date = datetime(2026, 8, 31)
total_seconds = int((end_date - start_date).total_seconds())

rows = []
n_fraud_target = int(N_TRANSACTIONS * FRAUD_RATE)
fraud_indices = set(np.random.choice(N_TRANSACTIONS, size=n_fraud_target, replace=False))

for i in range(N_TRANSACTIONS):
    sidx = np.random.randint(0, n_senders)
    sender = sender_ids[sidx]
    s_state = sender_state[sidx]
    s_bank = sender_bank[sidx]
    home_device = sender_home_device[sidx]

    merchant_cat = np.random.choice(merchant_categories, p=merchant_weights)
    lo, hi = category_amount_range[merchant_cat]
    amount = round(np.random.uniform(lo, hi), 2)

    ts = start_date + timedelta(seconds=int(np.random.randint(0, total_seconds)))
    hour = ts.hour
    is_weekend = 1 if ts.weekday() >= 5 else 0

    receiver_bank = np.random.choice(banks, p=bank_weights)
    device_type = home_device
    network_type = np.random.choice(networks, p=network_weights)

    is_fraud = 0

    if i in fraud_indices:
        is_fraud = 1
        # Fraud patterns: mix of behaviors so the model has to learn several signals,
        # not just one giveaway column (this mirrors real fraud typologies)
        pattern = np.random.choice(["odd_hour_high_amt", "device_switch", "velocity_burst", "new_bank_combo"])

        if pattern == "odd_hour_high_amt":
            hour = np.random.choice([1, 2, 3, 4, 23])
            amount = round(amount * np.random.uniform(3, 8), 2)
        elif pattern == "device_switch":
            other_devices = [d for d in devices if d != home_device]
            device_type = np.random.choice(other_devices)
            amount = round(amount * np.random.uniform(2, 5), 2)
        elif pattern == "velocity_burst":
            amount = round(np.random.uniform(50, 500), 2)  # many small transactions
            hour = np.random.choice(range(0, 24))
        elif pattern == "new_bank_combo":
            receiver_bank = np.random.choice([b for b in banks if b != s_bank])
            amount = round(amount * np.random.uniform(4, 10), 2)
            hour = np.random.choice([0, 1, 2, 22, 23])

        # fraud still needs SOME noise so it's not perfectly separable (realistic)
        if np.random.random() < 0.15:
            amount = round(np.random.uniform(lo, hi), 2)

    rows.append((
        f"TXN{i+1:07d}", sender, ts, amount, s_state, s_bank, receiver_bank,
        merchant_cat, device_type, network_type, hour, is_weekend, is_fraud
    ))

df = pd.DataFrame(rows, columns=[
    "transaction_id", "sender_id", "timestamp", "amount", "sender_state", "sender_bank",
    "receiver_bank", "merchant_category", "device_type", "network_type", "hour_of_day",
    "is_weekend", "fraud_flag"
])

df = df.sort_values("timestamp").reset_index(drop=True)
df.to_csv("/home/claude/upi_fraud_project/data/upi_transactions.csv", index=False)

print(f"Generated {len(df):,} transactions")
print(f"Fraud cases: {df['fraud_flag'].sum():,} ({df['fraud_flag'].mean()*100:.3f}%)")
print(df.head())
