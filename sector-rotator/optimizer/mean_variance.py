import pandas as pd
import numpy as np
import cvxpy as cp
import os
import yaml

# Input paths
WEIGHT_CSV = "data/weights/allocations.csv"
SIGNAL_DIR = "data/signals"
META_FILE = "metadata/selected_current.yaml"
RET_DIR = "data/raw"
OUT_FILE = "data/weights/allocations.csv"

# Load selected tickers (from YAML)
with open(META_FILE) as f:
    selected = yaml.safe_load(f)

# Load signal flags (only sectors with flag=1 are active)
signals = {}
for sector, symbol in selected.items():
    fname = f"{SIGNAL_DIR}/{sector.lower()}_flag.csv"
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        signals[sector] = df["flag"].values

min_len = min(map(len, signals.values()))
for k in signals:
    signals[k] = signals[k][-min_len:]

# Load return data
rets = {}
for sector, symbol in selected.items():
    fpath = f"{RET_DIR}/{symbol.replace('.', '_')}.csv"
    if not os.path.exists(fpath):
        continue
    df = pd.read_csv(fpath)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    df["ret"] = df["close"].pct_change().fillna(0)
    rets[sector] = df["ret"].values

min_ret_len = min(map(len, rets.values()))
for k in rets:
    rets[k] = rets[k][-min_ret_len:]

rets_df = pd.DataFrame(rets).reset_index(drop=True)
rets_df = rets_df.iloc[-min_len:].reset_index(drop=True)

# Allocate weights dynamically with mean-variance optimization
weights_all = []
for i in range(len(rets_df)):
    # Step 1: Active sectors at time i
    active_sectors = [s for s in signals if signals[s][i] == 1 and s in rets_df.columns]
    if len(active_sectors) == 0:
        weights_all.append([0] * len(rets_df.columns))
        continue

    mu = rets_df[active_sectors].iloc[:i+1].mean().values
    sigma = rets_df[active_sectors].iloc[:i+1].cov().values

    x = cp.Variable(len(mu))
    objective = cp.Maximize(mu @ x - 0.5 * cp.quad_form(x, sigma))
    constraints = [
        x >= 0,
        cp.sum(x) == 1
    ]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()
        w = x.value
    except:
        w = np.ones(len(mu)) / len(mu)  # fallback equal-weight
    # Step 2: Convert to full-sector weight
    full_weights = []
    for col in rets_df.columns:
        if col in active_sectors:
            idx = active_sectors.index(col)
            full_weights.append(w[idx])
        else:
            full_weights.append(0.0)
    weights_all.append(full_weights)

# Final output
weights_df = pd.DataFrame(weights_all, columns=rets_df.columns)
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
weights_df.to_csv(OUT_FILE, index=False)

print(f"✅ Saved mean-variance weights → {OUT_FILE}")
