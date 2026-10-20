
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

files = [
    "Benign-Monday-no-metadata.parquet",
    "DDoS-Friday-no-metadata.parquet",
    "Portscan-Friday-no-metadata.parquet",
]

for filename in files:
    path = DATA_DIR / filename

    print(f"\n{'=' * 55}")
    print(f"File: {filename}")

    df = pd.read_parquet(path, columns=["Label"])
    labels = df["Label"].astype(str).str.strip()

    print(f"Total flows: {len(df):,}")
    print("\nLabel distribution:")
    print(labels.value_counts().to_string())

    print("\nLabel percentages:")
    print(
        (100 * labels.value_counts(normalize=True))
        .round(2)
        .astype(str)
        .add("%")
        .to_string()
    )
