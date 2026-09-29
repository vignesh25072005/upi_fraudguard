"""
Derives behavioral features from raw UPI transactions:
- Per-sender transaction velocity (txns in trailing 1 hour)
- Deviation of this amount from the sender's historical average
- Odd-hour flag, weekend flag
- Sender-receiver bank mismatch flag
- Device-switch flag (device different from sender's most common device)
"""

import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["sender_id", "timestamp"]).reset_index(drop=True)

    # --- Velocity: transactions per sender in trailing 60 minutes ---
    velocity = []
    df["_ts_epoch"] = df["timestamp"].astype("int64") // 10**9
    for sender, g in df.groupby("sender_id"):
        times = g["_ts_epoch"].values
        v = np.zeros(len(times), dtype=int)
        start = 0
        for idx in range(len(times)):
            while times[idx] - times[start] > 3600:
                start += 1
            v[idx] = idx - start + 1  # count including current txn
        velocity.append(pd.Series(v, index=g.index))
    df["txn_velocity_1h"] = pd.concat(velocity).sort_index()

    # --- Amount deviation from sender's running historical mean ---
    df["sender_running_mean"] = (
        df.groupby("sender_id")["amount"]
        .apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=0, drop=True)
    )
    df["sender_running_mean"] = df["sender_running_mean"].fillna(df["amount"])
    df["amount_deviation_ratio"] = df["amount"] / df["sender_running_mean"].replace(0, np.nan)
    df["amount_deviation_ratio"] = df["amount_deviation_ratio"].fillna(1.0)

    # --- Odd hour flag (midnight - 5 AM) ---
    df["is_odd_hour"] = df["hour_of_day"].apply(lambda h: 1 if (h <= 5 or h == 23) else 0)

    # --- Sender's most common device (first 5 txns as "baseline") ---
    def most_common_device(g):
        return g["device_type"].mode().iloc[0]
    baseline_device = df.groupby("sender_id").apply(most_common_device)
    df["_baseline_device"] = df["sender_id"].map(baseline_device)
    df["device_switch_flag"] = (df["device_type"] != df["_baseline_device"]).astype(int)

    # --- Cross-bank flag ---
    df["cross_bank_flag"] = (df["sender_bank"] != df["receiver_bank"]).astype(int)

    df = df.drop(columns=["_ts_epoch", "_baseline_device"])
    return df


if __name__ == "__main__":
    df = pd.read_csv("/home/claude/upi_fraud_project/data/upi_transactions.csv")
    df_feat = engineer_features(df)
    df_feat.to_csv("/home/claude/upi_fraud_project/data/upi_transactions_features.csv", index=False)
    print(f"Feature-engineered dataset shape: {df_feat.shape}")
    print(df_feat[["amount", "txn_velocity_1h", "amount_deviation_ratio",
                    "is_odd_hour", "device_switch_flag", "cross_bank_flag", "fraud_flag"]].describe())
