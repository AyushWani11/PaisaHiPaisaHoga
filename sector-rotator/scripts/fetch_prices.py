#!/usr/bin/env python3
"""
fetch_prices.py
---------------
Download daily OHLCV for every symbol listed in metadata/*_universe.csv
and store one CSV per symbol under data/raw/.
"""

import os
import pandas as pd
import yfinance as yf

# ---- import project‑wide settings -----------------------------------------
from config import START_DATE, RAW_DIR, UNIVERSE_FILES
# (If you kept the file name as config.py instead of settings.py,
#  just change the import above back to `from config import …`)

# ---------------------------------------------------------------------------
os.makedirs(RAW_DIR, exist_ok=True)

# 1) build the master symbol list from the three universe files
universe = []
for csv_file in UNIVERSE_FILES.values():
    df = pd.read_csv(csv_file)
    universe.extend(df["symbol"].tolist())

# 2) download each symbol (skip if CSV already exists)
for sym in sorted(set(universe)):
    out_path = f"{RAW_DIR}/{sym.replace('.', '_')}.csv"
    if os.path.exists(out_path):
        print(f"✔  {sym} already downloaded")
        continue

    print(f"↓  {sym}")
    df = yf.download(sym, start=START_DATE, auto_adjust=True, progress=False)
    if df.empty:
        print(f"⚠  No data for {sym}, skipping.")
        continue

    # yfinance returns a DatetimeIndex; move it to a column called 'date'
    df = df.rename(columns=str.lower)               # open, high, low, close…
    df = (
        df.reset_index()                            # 'Date' -> column
          .rename(columns={"index": "date", "Date": "date"})
    )

    df.to_csv(out_path, index=False)
    print(f"   saved {len(df):,} rows  →  {out_path}")

print("\n✅  All downloads finished.")
