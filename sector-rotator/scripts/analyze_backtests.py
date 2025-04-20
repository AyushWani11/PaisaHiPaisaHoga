#!/usr/bin/env python3
"""
analyze_backtests.py
---------------------
Evaluates portfolio performance metrics:
- CAGR
- Volatility
- Sharpe Ratio
- Max Drawdown
- Value at Risk (VaR)
- Conditional VaR (CVaR)
- Win/Loss Ratio
- Alpha/Beta vs Nifty 50
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load equity curve with actual date index
equity = pd.read_csv("data/backtest/portfolio_value.csv", parse_dates=["date"])
equity = equity.set_index("date")
# BACKTEST_START = "2018-01-01"
# BACKTEST_END   = "2025-12-31"
# equity = equity.loc[BACKTEST_START:BACKTEST_END]

# Initial capital
initial_capital = 1_000_000

# Compute daily returns
returns = equity["PortfolioValue"].pct_change().dropna()

# Final capital
final_value = equity["PortfolioValue"].iloc[-1]
total_return = final_value - initial_capital

# CAGR
n_years = len(returns) / 252
cagr = (final_value / initial_capital)**(1 / n_years) - 1

# Volatility (annualized)
volatility = returns.std() * np.sqrt(252)

# Sharpe Ratio (assumes 0% risk-free rate)
sharpe = cagr / volatility

# Max Drawdown
roll_max = equity["PortfolioValue"].cummax()
drawdown = equity["PortfolioValue"] / roll_max - 1
max_dd = drawdown.min()

# Value at Risk (VaR 95%)
var_95 = np.percentile(returns, 5)

# Conditional VaR (CVaR 95%)
cvar_95 = returns[returns <= var_95].mean()

# Win/Loss Ratio
wins = (returns > 0).sum()
losses = (returns < 0).sum()
win_loss = wins / losses if losses != 0 else np.nan

# Print results
print("\nPerformance Summary")
print(f"Initial Capital:  ₹{initial_capital:,.2f}")
print(f"Final Capital:    ₹{final_value:,.2f}")
print(f"Total Return:     ₹{total_return:,.2f}")
print(f"CAGR:             {cagr * 100:.2f}%")
print(f"Volatility:       {volatility * 100:.2f}%")
print(f"Sharpe Ratio:     {sharpe:.2f}")
print(f"Max Drawdown:     {max_dd * 100:.2f}%")
print(f"VaR (95%):        {var_95 * 100:.2f}%")
print(f"CVaR (95%):       {cvar_95 * 100:.2f}%")
print(f"Win/Loss:         {win_loss:.2f} ({wins}W / {losses}L)")

# Benchmark Comparison
nifty = yf.download("^NSEI", start="2018-01-01", end="2025-04-01")
nifty = nifty[["Close"]].dropna()
nifty.columns = ["Nifty50"]
nifty.index = pd.to_datetime(nifty.index)

# Align portfolio to actual Nifty dates
portfolio = equity.copy()
portfolio = portfolio.loc[portfolio.index.intersection(nifty.index)]
nifty = nifty.loc[portfolio.index]

# Returns
portfolio_returns = portfolio["PortfolioValue"].pct_change().dropna()
nifty_returns = nifty["Nifty50"].pct_change().dropna()

# Regime-Based Analysis
regimes = {
    "IL&FS Bear": ("2018-09-01", "2018-11-30"),
    "Pre-COVID Bull": ("2019-01-01", "2020-01-31"),
    "COVID Crash": ("2020-02-01", "2020-04-30"),
    "Post-COVID Bull": ("2020-05-01", "2021-12-31"),
    "Rate Hike Bear": ("2022-01-01", "2022-06-30"),
    "Recent Bull": ("2023-01-01", "2025-01-01")
}

print("\n📅 Regime-Based Analysis")
for name, (start, end) in regimes.items():
    try:
        segment = equity.loc[start:end]
        if len(segment) < 30:
            print(f"{name}: Too few data points.")
            continue
        start_val = segment["PortfolioValue"].iloc[0]
        end_val = segment["PortfolioValue"].iloc[-1]
        years = len(segment) / 252
        cagr = (end_val / start_val) ** (1 / years) - 1
        print(f"{name:<18}: {cagr*100:>6.2f}% CAGR over {len(segment)} days")
    except Exception as e:
        print(f"{name}: Error → {e}")

# Alpha/Beta via linear regression
common_returns = pd.concat([portfolio_returns, nifty_returns], axis=1).dropna()
common_returns.columns = ["Portfolio", "Nifty"]

if not common_returns.empty:
    X = common_returns["Nifty"].values.reshape(-1, 1)
    y = common_returns["Portfolio"].values

    model = LinearRegression().fit(X, y)
    alpha = model.intercept_
    beta = model.coef_[0]
    max_loss = portfolio_returns.min()

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot((1 + portfolio_returns).cumprod(), label="Portfolio")
    plt.plot((1 + nifty_returns).cumprod(), label="Nifty 50")
    plt.title("Portfolio vs Nifty 50")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("report", exist_ok=True)
    plt.savefig("report/benchmark_vs_portfolio.png")
    plt.close()

    # Print metrics
    print(f"\nBenchmark Comparison")
    print(f"Alpha:           {alpha:.4f}")
    print(f"Beta:            {beta:.4f}")
    print(f"Max Daily Loss:  {max_loss:.4f}")
else:
    print("\nNot enough data for alpha/beta regression. Skipping benchmark comparison.")