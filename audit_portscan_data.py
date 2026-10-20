
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_FILE = Path("data/raw/Portscan-Friday-no-metadata.parquet")

print("=== CYBERTRACE AI: PORTSCAN DATA AUDIT ===")

# 1. Load the original dataset
df = pd.read_parquet(DATA_FILE)
df["Label"] = df["Label"].astype(str).str.strip()

labels = df["Label"].str.lower()
mask = (
    labels.isin(["benign", "normal"])
    | labels.str.contains("portscan|port scan", regex=True, na=False)
)
df = df.loc[mask].copy()

y = (
    df["Label"].str.lower().str.contains(
        "portscan|port scan", regex=True, na=False
    )
).astype("int8")

X = df.drop(columns=["Label"]).select_dtypes(include="number")
X = X.replace([float("inf"), float("-inf")], float("nan"))

print(f"\nRows: {len(df):,}")
print(f"Numeric features: {X.shape[1]}")
print(f"Benign: {(y == 0).sum():,}")
print(f"PortScan: {(y == 1).sum():,}")

# 2. Check exact duplicate feature rows in the full dataset
# Missing values are normalized before hashing.
X_hashable = X.fillna("<MISSING>")
row_hashes = pd.util.hash_pandas_object(
    X_hashable, index=False
)

hash_frame = pd.DataFrame({
    "hash": row_hashes,
    "label": y.to_numpy(),
})

duplicate_mask = hash_frame.duplicated("hash", keep=False)
duplicate_rows = hash_frame.loc[duplicate_mask]

print("\n=== Exact duplicate feature rows ===")
print(f"Rows involved in duplicates: {duplicate_mask.sum():,}")
print(f"Duplicate groups: {duplicate_rows['hash'].nunique():,}")

# Check whether identical feature vectors have conflicting labels.
labels_per_hash = hash_frame.groupby("hash")["label"].nunique()
conflicting_hashes = labels_per_hash[labels_per_hash > 1]

print(f"Feature vectors with conflicting labels: {len(conflicting_hashes):,}")

# 3. Reproduce the same random split as the benchmark
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

train_hashes = set(
    pd.util.hash_pandas_object(
        X_train.fillna("<MISSING>"), index=False
    )
)
test_hashes = set(
    pd.util.hash_pandas_object(
        X_test.fillna("<MISSING>"), index=False
    )
)

overlap = train_hashes.intersection(test_hashes)

print("\n=== Train/test overlap ===")
print(f"Unique feature rows in train: {len(train_hashes):,}")
print(f"Unique feature rows in test: {len(test_hashes):,}")
print(f"Feature-row hashes shared across splits: {len(overlap):,}")

# 4. Find constant and nearly constant features
print("\n=== Constant features ===")
constant_cols = [
    col for col in X.columns
    if X[col].nunique(dropna=False) <= 1
]
print(constant_cols if constant_cols else "None")

# 5. Compare feature missingness by class
print("\n=== Missing values by class ===")
missing_report = pd.DataFrame({
    "Benign_missing_pct": X.loc[y == 0].isna().mean() * 100,
    "PortScan_missing_pct": X.loc[y == 1].isna().mean() * 100,
})
missing_report["difference_pp"] = (
    missing_report["PortScan_missing_pct"]
    - missing_report["Benign_missing_pct"]
)
print(
    missing_report.reindex(
        missing_report["difference_pp"].abs()
        .sort_values(ascending=False).index
    ).head(15).round(3).to_string()
)

# Save a concise audit report
out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)

report = {
    "rows": int(len(df)),
    "numeric_features": int(X.shape[1]),
    "benign_rows": int((y == 0).sum()),
    "portscan_rows": int((y == 1).sum()),
    "rows_in_duplicate_groups": int(duplicate_mask.sum()),
    "duplicate_feature_groups": int(duplicate_rows["hash"].nunique()),
    "conflicting_label_feature_groups": int(len(conflicting_hashes)),
    "train_test_shared_feature_hashes": int(len(overlap)),
    "constant_features": constant_cols,
}

report_file = out / "portscan_audit_report.json"
pd.Series(report).to_json(report_file, indent=2)

print(f"\nAudit report saved to: {report_file}")
print("\nAudit completed.")
