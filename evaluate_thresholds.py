
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
RANDOM_STATE = 42

# 1. Train using benign traffic only
print("Loading benign training data...")
normal_df = pd.read_parquet(
    DATA_DIR / "Benign-Monday-no-metadata.parquet"
)
normal_df = normal_df[
    normal_df["Label"].astype(str).str.strip() == "Benign"
]

train_df = normal_df.sample(
    n=min(50_000, len(normal_df)),
    random_state=RANDOM_STATE,
)

remaining_benign = normal_df.drop(index=train_df.index)
benign_eval = remaining_benign.sample(
    n=min(10_000, len(remaining_benign)),
    random_state=RANDOM_STATE,
)

X_train = train_df.drop(columns=["Label"])
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

# 2. Prepare benign and attack evaluation data
benign_eval = benign_eval.copy()
benign_eval["_target"] = 0

attack_parts = []
for filename in [
    "DDoS-Friday-no-metadata.parquet",
    "Portscan-Friday-no-metadata.parquet",
]:
    df = pd.read_parquet(DATA_DIR / filename)
    df = df.sample(
        n=min(10_000, len(df)),
        random_state=RANDOM_STATE,
    )
    df["_target"] = (
        df["Label"].astype(str).str.strip() != "Benign"
    ).astype(int)
    attack_parts.append(df)

attacks = pd.concat(attack_parts, ignore_index=True)

# Split benign and attack records separately to keep both
# classes represented in validation and test sets.
rng = np.random.RandomState(RANDOM_STATE)
benign_eval = benign_eval.sample(frac=1, random_state=RANDOM_STATE)
attacks = attacks.sample(frac=1, random_state=RANDOM_STATE)

validation = pd.concat([
    benign_eval.iloc[:len(benign_eval) // 2],
    attacks.groupby("_target", group_keys=False).apply(
        lambda g: g.iloc[:len(g) // 2],
        include_groups=True,
    ).reset_index(drop=True),
], ignore_index=True)

# Use the remaining records for final evaluation.
test = pd.concat([
    benign_eval.iloc[len(benign_eval) // 2:],
    attacks.groupby("_target", group_keys=False).apply(
        lambda g: g.iloc[len(g) // 2:],
        include_groups=True,
    ).reset_index(drop=True),
], ignore_index=True)

def get_scores(frame):
    X = frame.reindex(columns=features)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")
    return model.decision_function(imputer.transform(X))

y_val = validation["_target"].astype(int).to_numpy()
val_scores = get_scores(validation)

# 3. Compare candidate thresholds on validation data.
# A flow is flagged when its score is below the threshold.
thresholds = np.unique(
    np.quantile(val_scores, np.linspace(0.01, 0.50, 100))
)

rows = []
for threshold in thresholds:
    pred = (val_scores < threshold).astype(int)
    rows.append({
        "threshold": threshold,
        "precision": precision_score(y_val, pred, zero_division=0),
        "recall": recall_score(y_val, pred, zero_division=0),
        "f1": f1_score(y_val, pred, zero_division=0),
        "false_positive_rate": (
            ((pred == 1) & (y_val == 0)).sum()
            / max((y_val == 0).sum(), 1)
        ),
    })

comparison = pd.DataFrame(rows).sort_values("f1", ascending=False)
best_threshold = float(comparison.iloc[0]["threshold"])

print("\nTop 10 thresholds on validation data:")
print(comparison.head(10).to_string(index=False))
print(f"\nSelected threshold: {best_threshold:.6f}")

# 4. Evaluate the selected threshold on unseen test records.
y_test = test["_target"].astype(int).to_numpy()
test_scores = get_scores(test)
test_pred = (test_scores < best_threshold).astype(int)

print("\nFinal test results:")
print(f"Accuracy:  {accuracy_score(y_test, test_pred):.3f}")
print(f"Precision: {precision_score(y_test, test_pred, zero_division=0):.3f}")
print(f"Recall:    {recall_score(y_test, test_pred, zero_division=0):.3f}")
print(f"F1-score:  {f1_score(y_test, test_pred, zero_division=0):.3f}")
print("Confusion matrix [Benign, Attack]:")
print(confusion_matrix(y_test, test_pred, labels=[0, 1]))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
comparison.to_csv(
    OUTPUT_DIR / "threshold_comparison.csv",
    index=False,
)

test_results = test.drop(columns=["_target"]).copy()
test_results["Anomaly_Score"] = test_scores
test_results["Prediction"] = np.where(
    test_pred == 1, "Suspect", "Normal"
)
test_results.to_parquet(
    OUTPUT_DIR / "threshold_test_results.parquet",
    index=False,
)

print("\nSaved threshold comparison and test results.")
