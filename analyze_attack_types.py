
from pathlib import Path
import pandas as pd

path = Path("data/processed/threshold_test_results.parquet")

if not path.exists():
    raise FileNotFoundError(
        "Run python evaluate_thresholds.py first."
    )

df = pd.read_parquet(path)

df["Label"] = df["Label"].astype(str).str.strip()
df["is_suspect"] = df["Prediction"].eq("Suspect")
df["is_attack"] = df["Label"].ne("Benign")

summary = (
    df.groupby("Label")
    .agg(
        total_flows=("Prediction", "size"),
        flagged_flows=("is_suspect", "sum"),
        attacks=("is_attack", "sum"),
    )
)

summary["flagged_rate_pct"] = (
    100 * summary["flagged_flows"] / summary["total_flows"]
).round(2)

summary["detection_rate_pct"] = summary.apply(
    lambda row: (
        round(100 * row["flagged_flows"] / row["total_flows"], 2)
        if row.name != "Benign"
        else float("nan")
    ),
    axis=1,
)

print("\nDetection by traffic label:")
print(summary.to_string())

benign = df[df["Label"].eq("Benign")]
if len(benign):
    fpr = benign["is_suspect"].mean() * 100
    print(f"\nBenign false-positive rate: {fpr:.2f}%")

attacks = df[df["is_attack"]]
if len(attacks):
    recall = attacks["is_suspect"].mean() * 100
    print(f"Overall attack detection rate: {recall:.2f}%")
