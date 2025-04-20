# scripts/generate_flags.py

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd, yaml, os
from importlib import import_module

SECTOR_MODULES = {
    "TECH": "signals.tech_rubberband",
    "FMCG": "signals.fmcg_turnofmonth",
    "BANK": "signals.bank_momentum",
}

import yfinance as yf

# Download index data (adjust start as needed)
index_data = {
    "TECH": yf.download("^CNXIT", start="2017-01-01")["Close"],
    "FMCG": yf.download("^CNXFMCG", start="2017-01-01")["Close"],
    "BANK": yf.download("^NSEBANK", start="2017-01-01")["Close"]
}

# Save to disk if needed
for sector, df in index_data.items():
    df.to_csv(f"data/indices/{sector}_index.csv", index_label="date")

def compute_macro_filter(index_series: pd.Series, window: int = 200) -> pd.Series:
    ma = index_series.rolling(window).mean()
    return (index_series > ma).astype(int)  # 1 = strong trend, 0 = weak


def main():
    with open("metadata/selected_current.yaml") as f:
        selected = yaml.safe_load(f)

    os.makedirs("data/signals", exist_ok=True)

    for sector, symbol in selected.items():
        print(f"Sector: {sector}  | Symbol: {symbol}")
        df = pd.read_csv(f"data/raw/{symbol.replace('.', '_')}.csv")
        mod = import_module(SECTOR_MODULES[sector])
        signal = mod.generate_signal(df)
        signal.name = "flag"
        outpath = f"data/signals/{sector}_flag.csv"
        signal.to_csv(outpath, index=False)
        print(f"Saved → {outpath}")

if __name__ == "__main__":
    main()
