"""
data_normalization.py

Normalises the Fondium 15-min and 60-min datasets for publication.
Normalisation: each numeric column is divided by its column maximum (ignoring NaN).
The timestamp column is kept unchanged.
Output files are written to the norm_data/ subfolder.
"""

import os
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILES = [
    "fondium_15_min_data_2023.csv",
    "fondium_60_min_data_2023.csv",
]
OUTPUT_DIR  = "norm_data"
TIMESTAMP_COL = "new_time"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Normalisation ─────────────────────────────────────────────────────────────

for filename in INPUT_FILES:
    df = pd.read_csv(filename)

    numeric_cols = [c for c in df.columns if c != TIMESTAMP_COL]

    col_max = df[numeric_cols].max()          # NaN-ignoring by default in pandas
    zero_max = col_max[col_max == 0].index.tolist()
    if zero_max:
        print(f"  Warning: columns with max=0 (skipped): {zero_max}")
        col_max = col_max.replace(0, float("nan"))  # avoid division by zero

    df_norm = df.copy()
    df_norm[numeric_cols] = df[numeric_cols] / col_max

    out_path = os.path.join(OUTPUT_DIR, f"norm_{filename}")
    df_norm.to_csv(out_path, index=False)

    print(f"{filename}")
    print(f"  Rows:    {len(df)}")
    print(f"  Columns: {len(numeric_cols)} numeric + timestamp")
    print(f"  Saved:   {out_path}")
    print()

print("Done.")
