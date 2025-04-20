#!/usr/bin/env python3
"""
pull_fund_data.py
-----------------
Fetch trailing P/E for every symbol in the three universe CSVs.
Uses yahoo_fin; falls back to NSE's nsetools if Yahoo is missing the metric.
Outputs: metadata/pe_ratios.csv
"""

import os, csv, json, time
import pandas as pd
from yahoo_fin import stock_info as yfi
from nsetools import Nse
from config import META_DIR, UNIVERSE_FILES
import yfinance as yf

out_path = f"{META_DIR}/pe_ratios.csv"
os.makedirs(META_DIR, exist_ok=True)
nse = Nse()

# ── build symbol list ─────────────────────────────────────────────
symbols = []
for csv_file in UNIVERSE_FILES.values():
    symbols.extend(pd.read_csv(csv_file)["symbol"])

records = []
for sym in sorted(set(symbols)):
    pe = None
    try:                                            # 1️⃣ Yahoo Finance
       pe = yf.Ticker(sym).info.get("trailingPE")
    except Exception:
        pass

    if pe in (None, "N/A"):
        try:                                        # 2️⃣ NSE fallback
            pe = float(nse.get_quote(sym.split(".")[0])["pe"])
        except Exception:
            pe = None

    records.append({"symbol": sym, "pe": pe})
    print(f"{sym:<12}  PE = {pe}")
    time.sleep(0.3)        # polite delay (Yahoo free tier)

pd.DataFrame(records).to_csv(out_path, index=False)
print(f"\n✅  saved PE ratios → {out_path}")
