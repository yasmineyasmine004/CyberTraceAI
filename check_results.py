
import pandas as pd

path = "data/processed/detection_results.parquet"
df = pd.read_parquet(path)

print("Dimensions :", df.shape)
print("\nColonnes :")
print(df.columns.tolist())

print("\nExemples :")
print(df.head(3).to_string())

print("\nRépartition des prédictions :")
print(df["Prediction"].value_counts(dropna=False))

print("\nValeurs manquantes :")
print(df.isna().sum().sort_values(ascending=False).head(10))
