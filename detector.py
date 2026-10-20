
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_DIR = Path("data/raw")
RANDOM_STATE = 42

TRAIN_SIZE = 50_000
TEST_SIZE = 10_000

# --------------------------------------------------
# 2. Chargement du trafic normal
# --------------------------------------------------

train_file = DATA_DIR / "Benign-Monday-no-metadata.parquet"

print("Chargement du trafic normal...")
normal_df = pd.read_parquet(train_file)

print("Dimensions initiales :", normal_df.shape)
print("Labels d'entraînement :", normal_df["Label"].value_counts())

# On utilise uniquement les flux Benign pour l'entraînement.
normal_df = normal_df[normal_df["Label"] == "Benign"]

# Échantillonnage pour limiter le temps d'entraînement.
train_df = normal_df.sample(
    n=min(TRAIN_SIZE, len(normal_df)),
    random_state=RANDOM_STATE,
)

X_train = train_df.drop(columns=["Label"])

# Remplacer les valeurs infinies par des valeurs manquantes.
X_train = X_train.replace([np.inf, -np.inf], np.nan)

# Supprimer les colonnes entièrement vides.
X_train = X_train.dropna(axis=1, how="all")

# Supprimer les colonnes constantes.
constant_cols = [
    col for col in X_train.columns
    if X_train[col].nunique(dropna=True) <= 1
]
X_train = X_train.drop(columns=constant_cols)

features = X_train.columns.tolist()

# Remplacer les valeurs manquantes restantes par la médiane.
imputer = SimpleImputer(strategy="median")
X_train_clean = imputer.fit_transform(X_train)

print("\nNombre de flux normaux utilisés :", len(X_train))
print("Nombre de caractéristiques retenues :", len(features))

# --------------------------------------------------
# 3. Entraînement du modèle non supervisé
# --------------------------------------------------

print("\nEntraînement d'Isolation Forest...")

model = IsolationForest(
    n_estimators=100,
    contamination=0.05, # Proportion d'anomalies dans le jeu de données
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(X_train_clean)

print("Entraînement terminé.")

# --------------------------------------------------
# 4. Tester sur des flux DDoS et PortScan
# --------------------------------------------------

test_files = [
    "DDoS-Friday-no-metadata.parquet",
    "Portscan-Friday-no-metadata.parquet",
]

test_parts = []

for filename in test_files:
    path = DATA_DIR / filename
    df = pd.read_parquet(path)

    # Échantillonner sans remplacer les lignes.
    df = df.sample(
        n=min(TEST_SIZE, len(df)),
        random_state=RANDOM_STATE,
    )

    test_parts.append(df)
    print(f"{filename} : {len(df)} flux sélectionnés")

test_df = pd.concat(test_parts, ignore_index=True)

# Le label est conservé uniquement pour l'évaluation.
y_true = (
    test_df["Label"]
    .astype(str)
    .str.strip()
    .ne("Benign")
    .astype(int)
)

X_test = test_df.reindex(columns=features)
X_test = X_test.replace([np.inf, -np.inf], np.nan)
X_test_clean = imputer.transform(X_test)

# Isolation Forest retourne -1 pour une anomalie et 1 sinon.
predictions = model.predict(X_test_clean)
y_pred = (predictions == -1).astype(int)

# Score : plus il est faible, plus le flux est inhabituel.
scores = model.decision_function(X_test_clean)

# --------------------------------------------------
# 5. Résultats
# --------------------------------------------------

test_df["Anomaly_Score"] = scores
test_df["Prediction"] = np.where(
    y_pred == 1,
    "Suspect",
    "Normal",
)

print("\nRépartition des prédictions :")
print(test_df["Prediction"].value_counts())

print("\nMatrice de confusion :")
print(
    confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )
)

print("\nRapport d'évaluation :")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Attack"],
        zero_division=0,
    )
)

# --------------------------------------------------
# 6. Export des résultats
# --------------------------------------------------

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

test_df.to_parquet(
    output_dir / "detection_results.parquet",
    index=False,
)

# Exporter les flux suspects pour faciliter leur analyse.
suspects = test_df[test_df["Prediction"] == "Suspect"]

suspects.to_csv(
    output_dir / "suspects.csv",
    index=False,
)

print("\nRésultats enregistrés dans data/processed/")
print("Nombre de flux suspects :", len(suspects))
print("Fichiers : detection_results.parquet et suspects.csv")
