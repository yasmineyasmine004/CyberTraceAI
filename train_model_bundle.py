from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from sklearn.model_selection import train_test_split

DATA = Path("data/raw")
OUT = Path("data/processed")
RANDOM_STATE = 42
OUT.mkdir(parents=True, exist_ok=True)


def numeric_features(frame):
    X = frame.select_dtypes(include="number").copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    # Remove all-empty and constant columns using this training frame.
    X = X.dropna(axis=1, how="all")
    constant = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    return X.drop(columns=constant)


# ------------------------------------------------------------
# 1) General anomaly detector: Isolation Forest trained on benign Monday flows.
# ------------------------------------------------------------
print("[1/2] Training Isolation Forest...")
benign_monday = pd.read_parquet(DATA / "Benign-Monday-no-metadata.parquet")
benign_monday["Label"] = benign_monday["Label"].astype(str).str.strip()
benign_monday = benign_monday[benign_monday["Label"].str.lower() == "benign"]
train_normal = benign_monday.sample(n=min(50_000, len(benign_monday)), random_state=RANDOM_STATE)
X_if = numeric_features(train_normal.drop(columns=["Label"], errors="ignore"))
if_features = X_if.columns.tolist()
if_imputer = SimpleImputer(strategy="median")
X_if_clean = if_imputer.fit_transform(X_if)
if_model = IsolationForest(
    n_estimators=100, contamination=0.05, random_state=RANDOM_STATE, n_jobs=-1
)
if_model.fit(X_if_clean)
joblib.dump(
    {"model": if_model, "imputer": if_imputer, "feature_cols": if_features,
     "model_type": "isolation_forest", "description": "General anomaly detector trained on benign Monday flows"},
    OUT / "model_bundle.joblib",
)
print(f"Saved {OUT / 'model_bundle.joblib'} ({len(if_features)} features)")

# ------------------------------------------------------------
# 2) Specialized supervised classifier: benign vs PortScan.
#    A held-out split is reported, then a final deployment model is fit on all
#    labeled Friday rows. This is an internal same-file validation, not an
#    independent external benchmark.
# ------------------------------------------------------------
print("\n[2/2] Training PortScan classifier...")
friday = pd.read_parquet(DATA / "Portscan-Friday-no-metadata.parquet")
friday["Label"] = friday["Label"].astype(str).str.strip()
labels = friday["Label"].str.lower()
mask = labels.isin(["benign", "normal"]) | labels.str.contains("portscan|port scan", regex=True, na=False)
friday = friday.loc[mask].copy().reset_index(drop=True)
y = friday["Label"].str.lower().str.contains("portscan|port scan", regex=True, na=False).astype(int)
X_raw = friday.drop(columns=["Label"], errors="ignore")
X = numeric_features(X_raw)
portscan_features = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
validation_imputer = SimpleImputer(strategy="median")
X_train_clean = validation_imputer.fit_transform(X_train)
X_test_clean = validation_imputer.transform(X_test)
validation_model = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
)
validation_model.fit(X_train_clean, y_train)
pred = validation_model.predict(X_test_clean)
proba = validation_model.predict_proba(X_test_clean)[:, 1]
print("\nInternal random-split validation (same source file; not independent):")
print(classification_report(y_test, pred, labels=[0, 1], target_names=["Benign", "PortScan"], zero_division=0))
print("Confusion matrix [Benign, PortScan]:")
print(confusion_matrix(y_test, pred, labels=[0, 1]))
print("PortScan average precision:", round(average_precision_score(y_test, proba), 4))

# Fit final deployment model on all available labeled rows.
final_imputer = SimpleImputer(strategy="median")
X_final = final_imputer.fit_transform(X)
final_model = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
)
final_model.fit(X_final, y)
joblib.dump(
    {"model": final_model, "imputer": final_imputer, "feature_cols": portscan_features,
     "model_type": "random_forest_portscan", "positive_class": "PortScan",
     "description": "Binary classifier trained on CICIDS2017 Friday benign/PortScan labels"},
    OUT / "portscan_model_bundle.joblib",
)
importance = pd.Series(final_model.feature_importances_, index=portscan_features).sort_values(ascending=False)
importance.to_csv(OUT / "portscan_feature_importance.csv")
print(f"Saved {OUT / 'portscan_model_bundle.joblib'} ({len(portscan_features)} features)")
print("Saved feature importances to", OUT / "portscan_feature_importance.csv")
print("\nDone. Keep in mind: the PortScan validation split comes from the same file,")
print("so external generalization is not yet established.")
