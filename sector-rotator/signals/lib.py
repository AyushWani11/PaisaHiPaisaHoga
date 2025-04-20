# signals/lib.py
import pandas as pd
import numpy as np

def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def mfi(df, window=3):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]
    pos = mf.where(tp.diff() > 0, 0.0)
    neg = mf.where(tp.diff() < 0, 0.0).abs()
    pmf = pos.rolling(window).sum()
    nmf = neg.rolling(window).sum()
    return 100 - (100 / (1 + pmf / nmf))

def atr(df, window=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def macd_histogram(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd - macd_signal

def breakout(df, window=252):
    return (df["close"] > df["close"].rolling(window).max().shift(1)).astype(int)
