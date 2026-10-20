
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

DATA_DIR = Path("data/raw")
RANDOM_STATE = 42
TRAIN_SIZE = 50_000
THRESHOLD = 0.012066

# 1. Train on benign Monday traffic
print("Training model on benign Monday traffic...")

monday = pd.read_parquet(
    DATA_DIR / "Benign-Monday-no-metadata.parquet"
)
monday = monday[
    monday["Label"].astype(str).str.strip() == "Benign"
]

train = monday.sample(
    n=min(TRAIN_SIZE, len(monday)),
    random_state=RANDOM_STATE,
)

# Keep a separate benign Monday evaluation sample.
monday_remaining = monday.drop(index=train.index)
monday_eval = monday_remaining.sample(
    n=min(10_000, len(monday_remaining)),
    random_state=RANDOM_STATE + 1,
)

X_train = train.drop(columns=["Label"])
X_train = X_train.select_dtypes(include="number")
X_train = X_train.replace([np.inf, -np.inf], np.nan)

features = [
    c for c in X_train.columns
    if X_train[c].notna().any()
    and X_train[c].nunique(dropna=True) > 1
]
X_train = X_train[features]

imputer = SimpleImputer(strategy="median")
X_train_clean = imputer.fit_transform(X_train)

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model.fit(X_train_clean)

# 2. Load benign Friday and PortScan Friday
print("Loading Friday data...")

friday = pd.read_parquet(
    DATA_DIR / "Portscan-Friday-no-metadata.parquet"
)
friday["Label"] = friday["Label"].astype(str).str.strip()

friday_benign = friday[friday["Label"] == "Benign"]
friday_portscan = friday[friday["Label"] == "PortScan"]

# Use all Friday benign records and all PortScan records.
datasets = {
    "Benign Monday (held out)": monday_eval,
    "Benign Friday": friday_benign,
    "PortScan Friday": friday_portscan,
}

# 3. Compare score distributions and detection rates
for name, df in datasets.items():
    X = df.reindex(columns=features)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    scores = model.decision_function(imputer.transform(X))
    flagged = scores < THRESHOLD

    print(f"\n=== {name} ===")
    print(f"Flows: {len(df):,}")
    print(f"Score median: {np.median(scores):.6f}")
    print(f"Score 10th percentile: {np.quantile(scores, 0.10):.6f}")
    print(f"Score 90th percentile: {np.quantile(scores, 0.90):.6f}")
    print(f"Flagged: {flagged.sum():,} ({100 * flagged.mean():.2f}%)")
