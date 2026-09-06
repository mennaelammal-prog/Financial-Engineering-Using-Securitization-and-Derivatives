"""
Pure, vectorized technical indicators over a price/OHLCV DataFrame.

Every function takes and returns pandas Series/DataFrames aligned on the
input index, so they compose directly with the ``charts`` and ``signals``
modules. No network calls here — this module is pure math and is unit
tested without any market-data dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int = 20) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int = 20) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def bollinger_bands(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands: middle/upper/lower bands around an SMA."""
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    return pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": mid + num_std * std,
            "bb_lower": mid - num_std * std,
        }
    )


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    # avg_loss == 0: a pure uptrend (RSI = 100) unless there's also no gain
    # at all (a flat series, RSI = 50 by convention) — the division above
    # turns both of those into NaN, so recover the correct value explicitly.
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return out.fillna(50)  # neutral where still undefined (insufficient data)


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram}
    )


def atr(ohlc: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range. Expects columns High, Low, Close.
    """
    high, low, close = ohlc["High"], ohlc["Low"], ohlc["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def add_all_indicators(ohlc: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience: attach the full standard indicator set to a copy of an
    OHLCV DataFrame (columns Open/High/Low/Close/Volume).
    """
    df = ohlc.copy()
    close = df["Close"]
    df["sma_20"] = sma(close, 20)
    df["sma_50"] = sma(close, 50)
    df["sma_200"] = sma(close, 200)
    df["ema_20"] = ema(close, 20)
    df = df.join(bollinger_bands(close))
    df["rsi_14"] = rsi(close, 14)
    df = df.join(macd(close))
    if {"High", "Low"}.issubset(df.columns):
        df["atr_14"] = atr(df, 14)
    return df
