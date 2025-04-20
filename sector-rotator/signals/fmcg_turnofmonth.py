import pandas as pd

def atr(df: pd.DataFrame, window=5) -> pd.Series:
    for col in ["high", "low", "close"]:
        df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["high", "low", "close"])

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window).mean()

def generate_signal(df: pd.DataFrame) -> pd.Series:
    for col in ["high", "low", "close"]:
        df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["high", "low", "close"])

    hi5 = df["high"].rolling(5).max()
    lo5 = df["low"].rolling(5).min()
    atr5 = atr(df, 5)

    lower_band = hi5 - 2.5 * atr5
    upper_band = lo5 + 2.5 * atr5

    signal = pd.Series(0, index=df.index)
    signal[df["close"] < lower_band] = 1    # Long entry
    signal[df["close"] > upper_band] = -1   # Short entry
    return signal
