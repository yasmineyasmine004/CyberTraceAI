
from pathlib import Path
import pandas as pd
import numpy as np

DOSSIER = Path("data/raw")

for fichier in sorted(DOSSIER.glob("*.parquet")):
    print("\n" + "=" * 65)
    print(f"FICHIER : {fichier.name}")

    df = pd.read_parquet(fichier)

    print("Dimensions :", df.shape)
    print("Nombre de colonnes :", len(df.columns))

    print("\nColonnes :")
    print(df.columns.tolist())

    print("\nTypes de données :")
    print(df.dtypes.value_counts())

    print("\nValeurs manquantes :", int(df.isna().sum().sum()))

    # Recherche de la colonne qui contient les labels
    colonnes_label = [
        c for c in df.columns
        if c.strip().lower() in ["label", "class", "attack"]
    ]

    for colonne in colonnes_label:
        print(f"\nRépartition de {colonne} :")
        print(df[colonne].value_counts(dropna=False).head(20))

    print("\nExemple de 3 lignes :")
    print(df.head(3).to_string(index=False))

    del df

print("\nInspection terminée.")
