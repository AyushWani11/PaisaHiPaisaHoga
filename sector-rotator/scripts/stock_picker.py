#!/usr/bin/env python3
"""
stock_picker.py
---------------
Algorithmic stock selection per sector using:
- Momentum, volatility, valuation, technical strength
- Optional clustering-based filtering (KMeans)
- Z-score based scoring with safe fallbacks
Outputs:
  → metadata/selected_current.yaml
"""

import os
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import zscore
from config import META_DIR, UNIVERSE_FILES

FACT_FILE = "data/factors/factor_snapshot.csv"
OUT_FILE  = f"{META_DIR}/selected_current.yaml"

# Modular factor weights (total = 1.0)
WEIGHTS = {
    "mom3":      0.2,
    "mom6":      0.2,
    "rsi14":     0.1,
    "breakout":  0.1,
    "pe_inv":    0.2,
    "atr_inv":   0.1,
    "vol30_inv": 0.1,
}

# Read data
df = pd.read_csv(FACT_FILE)

# Map sector from universe files
sym2sector = {}
for sector, csv_path in UNIVERSE_FILES.items():
    tickers = pd.read_csv(csv_path)["symbol"].tolist()
    for sym in tickers:
        sym2sector[sym] = sector

df["sector"] = df["symbol"].map(sym2sector)
df = df.dropna(subset=["sector"])

# Invert "lower is better" metrics
df["pe_inv"]     = -df["pe"]
df["atr_inv"]    = -df["atr_pct"]
df["vol30_inv"]  = -df["vol30"]

score_cols = list(WEIGHTS.keys())
df = df.dropna(subset=score_cols)

selected = {}

print("Selected stocks")
print("-" * 24)

for sector in df["sector"].unique():
    sector_df = df[df["sector"] == sector].copy()

    # Clustering (optional ML layer)
    cluster_features = ["mom3", "mom6", "vol30", "pe"]
    cluster_data = sector_df[cluster_features].dropna()

    if len(cluster_data) >= 3:
        scaled = StandardScaler().fit_transform(cluster_data)
        kmeans = KMeans(n_clusters=3, random_state=42).fit(scaled)
        sector_df.loc[cluster_data.index, "cluster"] = kmeans.labels_

        # Pick best momentum cluster
        cluster_scores = (
            sector_df.groupby("cluster")[["mom3", "mom6"]].mean().sum(axis=1)
        )
        best_cluster = cluster_scores.idxmax()
        sector_df = sector_df[sector_df["cluster"] == best_cluster].copy()
    else:
        print(f"Skipping clustering for {sector} — not enough data")

    # Defensive: Check again
    sector_df = sector_df.dropna(subset=score_cols)
    if len(sector_df) < 2:
        print(f"Skipping {sector} — not enough stocks after filtering.")
        continue

    # Z-score normalization
   # Remove any score column with no variance (e.g., all 0s)
    valid_cols = [col for col in score_cols if sector_df[col].nunique() > 1]

    if not valid_cols:
        print(f"No valid scoring columns in {sector}, skipping.")
        continue

    sector_df[valid_cols] = sector_df[valid_cols].apply(zscore)
    sector_df["score"] = sum(sector_df[col] * WEIGHTS[col] for col in valid_cols)

    # Final score
    sector_df["score"] = sum(sector_df[col] * w for col, w in WEIGHTS.items())

    top = sector_df.sort_values("score", ascending=False).iloc[0]
    selected[sector] = top["symbol"]
    print(f"{sector:<6}: {top['symbol']:<15} score={top['score']:.2f}")

# Save YAML
os.makedirs(META_DIR, exist_ok=True)
with open(OUT_FILE, "w") as f:
    yaml.dump(selected, f)

print(f"\n Saved → {OUT_FILE}")
