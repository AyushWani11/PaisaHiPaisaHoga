import pandas as pd

def compute_rsi(series: pd.Series, period=2) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def generate_signal(df: pd.DataFrame) -> pd.Series:
    df.loc[:, "close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    rsi2 = compute_rsi(df["close"], 2)

    signal = pd.Series(0, index=rsi2.index)
    signal[rsi2 < 30] = 1   # Long
    signal[rsi2 > 70] = -1  # Short
    return signal
