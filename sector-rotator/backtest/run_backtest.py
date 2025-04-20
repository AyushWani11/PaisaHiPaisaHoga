#!/usr/bin/env python3
"""
run_backtest.py
---------------
Simulates portfolio equity curve using:
- weights from:       data/weights/allocations.csv
- tickers from:       metadata/selected_current.yaml
- prices from:        data/raw/*.csv
- outputs:            data/backtest/portfolio_value.csv
"""

import os
import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt

def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# Paths
META_DIR   = "metadata"
RAW_DIR    = "data/raw"
WEIGHT_CSV = "data/weights/allocations.csv"
OUT_DIR    = "data/backtest"
os.makedirs(OUT_DIR, exist_ok=True)

# Load weights
weights = pd.read_csv(WEIGHT_CSV)
flat_days = weights.abs().sum(axis=1) == 0
flat_pct = flat_days.sum() / len(weights) * 100
print(f"Flat exposure days: {flat_days.sum()} ({flat_pct:.2f}% of total)")

# Load tickers from YAML
with open(f"{META_DIR}/selected_current.yaml") as f:
    selected = yaml.safe_load(f)

# Load sector flags
flag_dir = "data/signals"
flag_files = [f for f in os.listdir(flag_dir) if f.endswith("_flag.csv")]
signal_flags = [pd.read_csv(os.path.join(flag_dir, fname))["flag"].values for fname in flag_files]

min_signal_len = min(map(len, signal_flags))
signal_flags = [x[-min_signal_len:] for x in signal_flags]
combined_flag = pd.DataFrame(signal_flags).max(axis=0).reset_index(drop=True)

# Collect returns
rets = {}
min_len = float("inf")
for sector, symbol in selected.items():
    fpath = f"{RAW_DIR}/{symbol.replace('.', '_')}.csv"
    if not os.path.exists(fpath):
        print(f"⚠ Missing data for {symbol}, skipping.")
        continue
    df = pd.read_csv(fpath, parse_dates=["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    df["ret"] = df["close"].pct_change().fillna(0)
    rets[sector] = df["ret"].values
    if 'sample_dates' not in locals():
        sample_dates = df["date"].iloc[-min_signal_len:].reset_index(drop=True)
    min_len = min(min_len, len(df))

for k in rets:
    rets[k] = rets[k][-min_signal_len:]

rets_df = pd.DataFrame(rets).reset_index(drop=True)

# Align to signal length
weights = weights.iloc[-min_signal_len:].reset_index(drop=True)
rets_df = rets_df.iloc[-min_signal_len:].reset_index(drop=True)

# Portfolio returns and equity
portfolio_returns = (weights * rets_df).sum(axis=1)
equity_curve = (1 + portfolio_returns).cumprod()
equity_curve.name = "PortfolioValue"
equity_curve.index = sample_dates
equity_curve.index.name = "Date"
portfolio_returns.index = sample_dates

# Risk overlays
equity_rsi = rsi(equity_curve, 14).fillna(0)
rolling_vol = portfolio_returns.rolling(30).std().fillna(0) * np.sqrt(252)
k = 0.125
adaptive_dd_limit = -k * rolling_vol

rolling_max = equity_curve.cummax()
drawdown = equity_curve / rolling_max - 1

# Trailing stop logic
active = pd.Series(1, index=equity_curve.index)
in_cash = False
for i in range(1, len(drawdown)):
    if in_cash:
        if equity_curve.iloc[i] >= rolling_max.iloc[i]:
            in_cash = False
            active.iloc[i] = 1
        else:
            active.iloc[i] = 0
    elif drawdown.iloc[i] < adaptive_dd_limit.iloc[i]:
        in_cash = True
        active.iloc[i] = 0

# Apply stops
portfolio_returns = portfolio_returns * active
equity_curve = (1 + portfolio_returns).cumprod()
initial_capital = 1_000_000
equity_curve = equity_curve * initial_capital

# Save final equity curve
out_path = f"{OUT_DIR}/portfolio_value.csv"
equity_curve.to_frame(name="PortfolioValue").to_csv(out_path)

# Rolling 30-day returns
rolling_30d_return = equity_curve.pct_change(periods=30, fill_method=None) * 100
rolling_30d_return = rolling_30d_return.dropna()
rolling_30d_return.name = "Rolling30dReturn"
rolling_30d_return.to_csv(f"{OUT_DIR}/rolling_30d_return.csv")

# Average rolling returns
intervals = {
    "1 Month": 21,
    "1 Year": 252,
    "3 Years": 756,
    "5 Years": 1260
}

def avg_rolling_return_over(days: int):
    if len(rolling_30d_return) < days:
        return None
    return rolling_30d_return[-days:].mean()

print("\nAverage 30-Day Rolling Returns Over Intervals:")
for label, days in intervals.items():
    avg_ret = avg_rolling_return_over(days)
    if avg_ret is not None:
        print(f"   {label:<8}: {avg_ret:.2f}%")
    else:
        print(f"   {label:<8}: Not enough data")

avg_full = rolling_30d_return.mean()

# Plot daily returns
plt.figure(figsize=(10, 4))
portfolio_returns.plot(color="green", title="Daily Portfolio Returns")
plt.axhline(0, linestyle="--", color="gray")
plt.grid(True)
plt.tight_layout()
os.makedirs("report", exist_ok=True)
plt.savefig("report/daily_returns.png")
plt.close()

# Plot rolling 30-day return
plt.figure(figsize=(10, 4))
plt.plot(rolling_30d_return, color='orange', label="30-Day Rolling Return (%)")
plt.axhline(0, linestyle='--', color='gray')
plt.axhline(avg_full, linestyle='--', color='blue', label=f"Avg: {avg_full:.2f}%")
plt.title("Rolling 30-Day Portfolio Return")
plt.xlabel("Days")
plt.ylabel("Return (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("report/rolling_30d_return.png")
plt.close()

print(f"\nBacktest complete → {out_path}")