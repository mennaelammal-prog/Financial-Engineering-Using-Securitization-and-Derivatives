"""
Market data retrieval layer.

Wraps yfinance with:
- multi-ticker OHLCV download normalized to a tidy shape
- an on-disk parquet/CSV cache so repeated runs (dashboard reloads, CLI
  re-runs) don't re-hit the network every time
- a friendly error when a ticker returns no data (bad symbol, delisted,
  market holiday range, etc.)
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "market_data"
CACHE_TTL_SECONDS = 15 * 60  # 15 minutes


class DataFetchError(RuntimeError):
    """Raised when market data can't be retrieved for one or more tickers."""


def _cache_key(tickers: list[str], period: str, interval: str) -> Path:
    raw = f"{sorted(tickers)}|{period}|{interval}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:20]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{digest}.parquet"


def fetch_ohlcv(
    tickers: str | list[str],
    period: str = "2y",
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV data for one or more tickers.

    Returns a DataFrame with a MultiIndex of columns ``(field, ticker)`` where
    field is one of Open/High/Low/Close/Volume — the same shape yfinance
    gives you for multiple tickers, so a single ticker is just a one-wide
    slice of it.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = list(dict.fromkeys(tickers))  # de-dupe, keep order

    cache_path = _cache_key(tickers, period, interval)
    if use_cache and cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            return pd.read_parquet(cache_path)

    raw = yf.download(
        tickers,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
        group_by="column",
    )

    if raw is None or raw.empty:
        raise DataFetchError(
            f"No market data returned for {tickers} (period={period}, "
            f"interval={interval}). Check the ticker symbols and try again."
        )

    # yfinance returns flat columns for a single ticker; normalize to the
    # same MultiIndex shape as the multi-ticker case for downstream code.
    if len(tickers) == 1 and not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product([raw.columns, tickers])

    raw = raw.dropna(how="all")
    if use_cache:
        try:
            raw.to_parquet(cache_path)
        except Exception:
            pass  # cache is a convenience, never fatal

    return raw


def close_prices(
    tickers: str | list[str], period: str = "2y", interval: str = "1d"
) -> pd.DataFrame:
    """Convenience: just the (adjusted) Close columns, one per ticker."""
    ohlcv = fetch_ohlcv(tickers, period=period, interval=interval)
    closes = ohlcv["Close"]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame()
    return closes.dropna(how="all")


def single_ticker_ohlcv(
    ticker: str, period: str = "1y", interval: str = "1d"
) -> pd.DataFrame:
    """OHLCV for one ticker with flat columns (Open/High/Low/Close/Volume)."""
    ohlcv = fetch_ohlcv([ticker], period=period, interval=interval)
    flat = ohlcv.xs(ticker, axis=1, level=1)
    return flat.dropna(how="all")
