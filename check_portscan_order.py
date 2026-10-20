
from pathlib import Path
import pandas as pd

path = Path("data/raw/Portscan-Friday-no-metadata.parquet")

df = pd.read_parquet(path)
labels = df["Label"].astype(str).str.strip().str.lower()

is_portscan = labels.str.contains(
    "portscan|port scan", regex=True, na=False
)
is_benign = labels.isin(["benign", "normal"])

df = df.loc[is_portscan | is_benign].copy()
df["is_portscan"] = is_portscan.loc[df.index].astype(int)

# Keep the original row order and divide it into five consecutive chunks.
df["chunk"] = pd.qcut(
    range(len(df)),
    q=5,
    labels=["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4", "Chunk 5"],
)

summary = df.groupby("chunk", observed=False).agg(
    flows=("is_portscan", "size"),
    portscan_count=("is_portscan", "sum"),
)

summary["benign_count"] = summary["flows"] - summary["portscan_count"]
summary["portscan_rate_pct"] = (
    100 * summary["portscan_count"] / summary["flows"]
).round(3)

print("=== PortScan distribution in original row order ===")
print(summary.to_string())

print("\nInterpretation:")
print("- Similar rates across chunks suggest a more uniform class distribution.")
print("- Concentration in specific chunks suggests possible temporal or ordering effects.")
print("- Row order alone does not prove that the chunks correspond to time.")
