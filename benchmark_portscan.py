
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
)

DATA_FILE = Path("data/raw/Portscan-Friday-no-metadata.parquet")

print("Loading PortScan Friday dataset...")
df = pd.read_parquet(DATA_FILE)
df["Label"] = df["Label"].astype(str).str.strip()

# Binary classification: PortScan vs benign.
labels = df["Label"].str.lower()
mask = labels.isin(["benign", "normal"]) | labels.str.contains(
    "portscan|port scan", regex=True, na=False
)

df = df.loc[mask].copy()
y = (
    df["Label"].str.lower().str.contains(
        "portscan|port scan", regex=True, na=False
    )
).astype(int)

X = df.drop(columns=["Label"])
X = X.select_dtypes(include="number")
X = X.replace([float("inf"), float("-inf")], float("nan"))

# Replace missing values using training-set medians only.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

medians = X_train.median().fillna(0)
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

print(f"Training flows: {len(X_train)}")
print(f"Test flows: {len(X_test)}")
print(f"PortScan in training: {int(y_train.sum())}")
print(f"PortScan in test: {int(y_test.sum())}")

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\n=== Classification report ===")
print(classification_report(
    y_test,
    predictions,
    target_names=["Benign", "PortScan"],
    zero_division=0,
))

print("=== Confusion matrix ===")
print(confusion_matrix(y_test, predictions))

print(
    "PortScan average precision:",
    round(average_precision_score(y_test, probabilities), 4),
)

importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns,
).sort_values(ascending=False)

print("\n=== Top 15 model features ===")
print(importance.head(15).to_string())

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)
importance.to_csv(output_dir / "portscan_feature_importance.csv")

print("\nSaved feature importances to:")
print("data/processed/portscan_feature_importance.csv")
