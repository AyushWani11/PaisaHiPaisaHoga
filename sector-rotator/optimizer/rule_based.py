import pandas as pd
import numpy as np
import os, glob
from scipy.optimize import minimize
import yaml

def generate_allocations(signal_dir="data/signals/", return_dir="data/raw/", lookback=30) -> pd.DataFrame:
    # Load selected stock per sector
    with open("metadata/selected_current.yaml") as f:
        selected = yaml.safe_load(f)  # {TECH: INFY.NS, ...}

    # Load signals
    signals, prices = {}, {}
    lengths = []

    for fpath in glob.glob(os.path.join(signal_dir, "*_flag.csv")):
        sector = os.path.basename(fpath).split("_")[0].upper()
        df = pd.read_csv(fpath)
        signals[sector] = df["flag"].values
        lengths.append(len(df))

        symbol = selected.get(sector)
        if symbol:
            price_path = os.path.join(return_dir, f"{symbol.replace('.', '_')}.csv")
            if os.path.exists(price_path):
                p = pd.read_csv(price_path)
                p["close"] = pd.to_numeric(p["close"], errors="coerce")
                p = p.dropna(subset=["close"])
                prices[sector] = p["close"].pct_change().dropna()

    # Determine min length across all
    min_len = min([len(x) for x in signals.values()] + [len(p) for p in prices.values()])
    for k in signals:
        signals[k] = signals[k][-min_len:]
    for k in prices:
        prices[k] = prices[k][-min_len:]

    signal_df = pd.DataFrame(signals).reset_index(drop=True)
    return_df = pd.DataFrame(prices).reset_index(drop=True)

    # MVO optimizer for one day
    def mvo_alloc(signal_row, t):
        active = signal_row[signal_row != 0].index.tolist()
        if t < lookback:
            return pd.Series(0, index=signal_row.index)

        # If only 1 sector is active, add second best sector
        if len(active) == 1:
            rest = [s for s in return_df.columns if s not in active]
            if rest:
                rets_subset = return_df[rest].iloc[t - lookback:t]
                # Rank by Sharpe-like score: mean / std
                sharpe_scores = rets_subset.mean() / rets_subset.std()
                second = sharpe_scores.idxmax()
                active.append(second)

        valid = [s for s in active if s in return_df.columns]
        if not valid:
            return pd.Series(0, index=signal_row.index)

        rets = return_df[valid].iloc[t - lookback:t]
        mu = rets.mean().values
        cov = rets.cov().values

        def objective(w):
            return -np.dot(w, mu) + 0.5 * np.dot(w.T, np.dot(cov, w))

        bounds = [(-0.5, 0.5)] * len(valid)
        cons = [{"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1}]
        x0 = np.array([1 / len(valid)] * len(valid))

        res = minimize(objective, x0, bounds=bounds, constraints=cons)
        w_opt = res.x if res.success else x0

        alloc = pd.Series(0, index=signal_row.index, dtype=float)
        for i, sector in enumerate(valid):
            alloc[sector] = w_opt[i]
        return alloc

    # Run optimizer across all days
    weights = pd.DataFrame([mvo_alloc(signal_df.iloc[i], i) for i in range(min_len)])
    weights.index.name = "Date"
    weights.columns = signal_df.columns
    return weights
