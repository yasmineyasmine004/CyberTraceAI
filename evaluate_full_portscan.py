
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

RANDOM_STATE = 42
TRAIN_SIZE = 50_000

# Threshold selected by our previous validation experiment.
THRESHOLD = 0.012066

# 1. Train on benign traffic
print("Loading benign training data...")

normal = pd.read_parquet(
    DATA_DIR / "Benign-Monday-no-metadata.parquet"
)
normal = normal[
    normal["Label"].astype(str).str.strip() == "Benign"
]

train = normal.sample(
    n=min(TRAIN_SIZE, len(normal)),
    random_state=RANDOM_STATE,
)

X_train = train.drop(columns=["Label"])
X_train = X_train.select_dtypes(include="number")
X_train = X_train.replace([np.inf, -np.inf], np.nan)

features = [
    col for col in X_train.columns
    if X_train[col].notna().any()
    and X_train[col].nunique(dropna=True) > 1
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

print("Model trained.")

# 2. Load ALL rows from the PortScan Friday file
print("\nLoading the full PortScan Friday file...")

df = pd.read_parquet(
    DATA_DIR / "Portscan-Friday-no-metadata.parquet"
)
df["Label"] = df["Label"].astype(str).str.strip()

print("Total rows:", len(df))
print("\nActual labels:")
print(df["Label"].value_counts().to_string())

y_true = df["Label"].ne("Benign").astype(int).to_numpy()

X_test = df.reindex(columns=features)
X_test = X_test.apply(pd.to_numeric, errors="coerce")
X_test = X_test.replace([np.inf, -np.inf], np.nan)

scores = model.decision_function(imputer.transform(X_test))

# 3. Compare the default model threshold and our tuned threshold
thresholds = {
    "Default Isolation Forest": 0.0,
    "Previously selected threshold": THRESHOLD,
}

for name, threshold in thresholds.items():
    pred = (scores < threshold).astype(int)

    print(f"\n=== {name} (threshold={threshold:.6f}) ===")
    print("Confusion matrix [Benign, Attack]:")
    print(confusion_matrix(y_true, pred, labels=[0, 1]))

    print(f"Precision: {precision_score(y_true, pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_true, pred, zero_division=0):.4f}")

    benign_mask = y_true == 0
    attack_mask = y_true == 1

    if benign_mask.any():
        fpr = (pred[benign_mask] == 1).mean()
        print(f"Benign false-positive rate: {fpr:.4f}")

    if attack_mask.any():
        attack_recall = (pred[attack_mask] == 1).mean()
        print(f"PortScan detection rate:    {attack_recall:.4f}")

# 4. Save diagnostic results
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df["Anomaly_Score"] = scores
df["Prediction"] = np.where(
    scores < THRESHOLD, "Suspect", "Normal"
)

output_path = OUTPUT_DIR / "full_portscan_results.parquet"
df.to_parquet(output_path, index=False)

print(f"\nSaved full evaluation to: {output_path}")
