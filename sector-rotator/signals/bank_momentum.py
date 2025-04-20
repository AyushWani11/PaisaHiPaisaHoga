import pandas as pd

def generate_signal(df: pd.DataFrame) -> pd.Series:
    df.loc[:, "close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])

    sma20 = df["close"].rolling(20).mean()
    sma63 = df["close"].rolling(63).mean()

    signal = pd.Series(0, index=df.index)
    signal[sma20 > sma63] = 1   # Long signal
    signal[sma20 < sma63] = -1  # Short signal

    return signal
