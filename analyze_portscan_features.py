
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DATA = Path("data/raw")
BENIGN_FILE = DATA / "Benign-Monday-no-metadata.parquet"
PORTSCAN_FILE = DATA / "Portscan-Friday-no-metadata.parquet"

print("Loading datasets...")
benign = pd.read_parquet(BENIGN_FILE)
portscan = pd.read_parquet(PORTSCAN_FILE)

# Exclude the label from the input features.
label_col = "Label"
feature_cols = [
    c for c in benign.columns
    if c != label_col
    and pd.api.types.is_numeric_dtype(benign[c])
]

# Keep only benign Monday traffic for the baseline.
benign[label_col] = benign[label_col].astype(str).str.strip()
portscan[label_col] = portscan[label_col].astype(str).str.strip()

normal = benign[
    benign[label_col].str.lower().isin(["benign", "normal"])
][feature_cols].copy()

# Remove invalid values and use the same feature columns in both datasets.
normal = normal.replace([float("inf"), float("-inf")], float("nan"))
normal = normal.dropna()

portscan_features = portscan[feature_cols].replace(
    [float("inf"), float("-inf")], float("nan")
)

# Use a manageable sample for the first comparison.
normal_sample = normal.sample(
    n=min(50000, len(normal)), random_state=42
)

print(f"Numeric features: {len(feature_cols)}")
print(f"Benign training sample: {len(normal_sample)}")

# Compare medians and interquartile ranges.
attack_mask = portscan[label_col].str.lower().str.contains(
    "portscan|port scan", regex=True, na=False
)

benign_friday = portscan.loc[
    portscan[label_col].str.lower().isin(["benign", "normal"]),
    feature_cols
]
attack_friday = portscan.loc[attack_mask, feature_cols]

print(f"Friday benign flows: {len(benign_friday)}")
print(f"PortScan flows: {len(attack_friday)}")

if attack_friday.empty:
    print("\nCould not identify PortScan labels.")
    print("Available labels:")
    print(portscan[label_col].value_counts().to_string())
    raise SystemExit(1)

# Rank features by the absolute difference in medians,
# normalized by the benign interquartile range.
rows = []

for col in feature_cols:
    b = benign_friday[col].replace(
        [float("inf"), float("-inf")], float("nan")
    ).dropna()
    a = attack_friday[col].replace(
        [float("inf"), float("-inf")], float("nan")
    ).dropna()

    if len(b) < 10 or len(a) < 10:
        continue

    q25, q75 = b.quantile([0.25, 0.75])
    iqr = q75 - q25
    difference = abs(a.median() - b.median())

    if iqr > 0:
        normalized_difference = difference / iqr
    else:
        normalized_difference = 0.0 if difference == 0 else float("inf")

    rows.append({
        "feature": col,
        "benign_median": b.median(),
        "portscan_median": a.median(),
        "median_difference": difference,
        "normalized_difference": normalized_difference,
    })

ranking = pd.DataFrame(rows).sort_values(
    "normalized_difference", ascending=False
)

out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)
output_file = out / "portscan_feature_comparison.csv"
ranking.to_csv(output_file, index=False)

print("\nTop 15 features distinguishing PortScan from Friday benign traffic:")
print(ranking.head(15).to_string(index=False))
print(f"\nSaved: {output_file}")
