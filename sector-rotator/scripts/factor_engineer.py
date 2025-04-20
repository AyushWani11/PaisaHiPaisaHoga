#!/usr/bin/env python3
"""
factor_engineer.py
------------------
Creates factor snapshot for each stock:
  • 3mo & 6mo momentum
  • ATR% (volatility)
  • 30-day log return stdev
  • RSI-14
  • 52-week breakout flag
  • trailing PE from pe_ratios.csv

Outputs:
  → data/factors/factor_snapshot.csv
"""

import os
import glob
import numpy as np
import pandas as pd
import pandas_ta as ta
from config import RAW_DIR, META_DIR

SNAP_DIR = "data/factors"
SNAP_FILE = f"{SNAP_DIR}/factor_snapshot.csv"
PE_FILE = f"{META_DIR}/pe_ratios.csv"

os.makedirs(SNAP_DIR, exist_ok=True)

# Read PE data
pe_df = pd.read_csv(PE_FILE).set_index("symbol")

rows = []
for fpath in glob.glob(f"{RAW_DIR}/*.csv"):
    sym = os.path.basename(fpath).replace("_", ".").replace(".csv", "")
    df = pd.read_csv(fpath)

    if "date" not in df.columns or "close" not in df.columns:
        print(f"⚠️  Skipping {sym} (missing columns)")
        continue

    df = df.sort_values("date")

    # --- Ensure numeric dtypes ---
    for col in ["open", "high", "low", "close", "adj close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close", "high", "low"])
    if len(df) < 260:  # ~1 year of trading days
        print(f"⚠️  Skipping {sym} (not enough data)")
        continue

    df["ret"] = df["close"].pct_change()
    df["logret"] = np.log(df["close"]).diff()

    try:
        mom3 = df["close"].pct_change(63).iloc[-1]
        mom6 = df["close"].pct_change(126).iloc[-1]
        atr_pct = (
            ta.atr(df["high"], df["low"], df["close"], 20).iloc[-1]
            / df["close"].iloc[-1]
        )
        vol30 = df["logret"].rolling(30).std().iloc[-1]
        rsi14 = ta.rsi(df["close"], 14).iloc[-1]
        breakout = int(df["close"].iloc[-1] > df["close"].rolling(252).max().iloc[-2])
    except Exception as e:
        print(f"❌ {sym} failed factor calc: {e}")
        continue

    pe = pe_df.at[sym, "pe"] if sym in pe_df.index else np.nan

    rows.append(
        {
            "symbol": sym,
            "mom3": mom3,
            "mom6": mom6,
            "atr_pct": atr_pct,
            "vol30": vol30,
            "rsi14": rsi14,
            "breakout": breakout,
            "pe": pe,
        }
    )

# Save results
pd.DataFrame(rows).to_csv(SNAP_FILE, index=False)
print(f"\n✅  factor snapshot → {SNAP_FILE}")
