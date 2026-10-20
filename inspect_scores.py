
from pathlib import Path
import pandas as pd

path = Path("data/processed/threshold_test_results.parquet")
df = pd.read_parquet(path)

df["Label"] = df["Label"].astype(str).str.strip()
df["Anomaly_Score"] = pd.to_numeric(
    df["Anomaly_Score"], errors="coerce"
)

print("\n=== Attack labels in the saved test set ===")
print(df["Label"].value_counts().to_string())

print("\n=== Anomaly-score distribution by label ===")
summary = df.groupby("Label")["Anomaly_Score"].agg(
    ["count", "min", "median", "mean", "max"]
)
print(summary.round(5).to_string())

print("\n=== Detection rate by label ===")
for label, group in df.groupby("Label"):
    rate = 100 * group["Prediction"].eq("Suspect").mean()
    print(f"{label}: {rate:.2f}% flagged ({len(group)} flows)")
