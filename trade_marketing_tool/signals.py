"""
Rule-based trade signals and a lightweight backtester.

Combines the indicator set (SMA/EMA crossovers, RSI, MACD, Bollinger Bands)
into a simple scored signal per bar, then evaluates a naive
long/flat strategy against buy-and-hold so a signal's usefulness can be
sanity-checked before anyone trades on it.

This is intentionally simple — a research/education aid, not an execution
system. See the README's "Disclaimer" section.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import add_all_indicators

TRADING_DAYS_PER_YEAR = 252


def generate_signals(ohlc: pd.DataFrame) -> pd.DataFrame:
    """
    Attach indicator columns plus a per-bar signal score in [-3, +3]:
    +1 for each bullish condition triggered, -1 for each bearish one:
      - SMA50 crossing above/below SMA200 (golden/death cross)
      - RSI leaving oversold (<30) upward / overbought (>70) downward
      - MACD line crossing above/below its signal line
    A ``position`` column of {0, 1} (flat/long) is derived by holding long
    while the score is > 0, flat otherwise — the simplest possible rule so
    the backtest below is transparent, not tuned.
    """
    df = add_all_indicators(ohlc)

    score = pd.Series(0, index=df.index, dtype=float)

    golden_cross = (df["sma_50"] > df["sma_200"]) & (
        df["sma_50"].shift(1) <= df["sma_200"].shift(1)
    )
    death_cross = (df["sma_50"] < df["sma_200"]) & (
        df["sma_50"].shift(1) >= df["sma_200"].shift(1)
    )
    score += golden_cross.astype(int) - death_cross.astype(int)

    rsi_bull = (df["rsi_14"] > 30) & (df["rsi_14"].shift(1) <= 30)
    rsi_bear = (df["rsi_14"] < 70) & (df["rsi_14"].shift(1) >= 70)
    score += rsi_bull.astype(int) - rsi_bear.astype(int)

    macd_bull = (df["macd"] > df["macd_signal"]) & (
        df["macd"].shift(1) <= df["macd_signal"].shift(1)
    )
    macd_bear = (df["macd"] < df["macd_signal"]) & (
        df["macd"].shift(1) >= df["macd_signal"].shift(1)
    )
    score += macd_bull.astype(int) - macd_bear.astype(int)

    df["signal_score"] = score
    trend_state = np.where(
        df["sma_50"] > df["sma_200"], 1, np.where(df["sma_50"] < df["sma_200"], -1, 0)
    )
    df["position"] = (pd.Series(trend_state, index=df.index) > 0).astype(int)
    return df


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame  # strategy vs buy-and-hold cumulative returns
    total_return: float
    buy_hold_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float


def backtest(
    df_with_signals: pd.DataFrame,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> BacktestResult:
    """
    Backtest the ``position`` column from ``generate_signals`` against
    buy-and-hold, using next-bar returns (position is applied with a one-bar
    lag to avoid look-ahead bias).
    """
    close = df_with_signals["Close"]
    daily_returns = close.pct_change().fillna(0)
    position = df_with_signals["position"].shift(1).fillna(0)

    strategy_returns = daily_returns * position
    equity_curve = pd.DataFrame(
        {
            "strategy": (1 + strategy_returns).cumprod(),
            "buy_and_hold": (1 + daily_returns).cumprod(),
        }
    )

    total_return = float(equity_curve["strategy"].iloc[-1] - 1)
    buy_hold_return = float(equity_curve["buy_and_hold"].iloc[-1] - 1)

    n_days = len(strategy_returns)
    years = max(n_days / trading_days_per_year, 1e-9)
    annualized_return = float((1 + total_return) ** (1 / years) - 1)
    annualized_volatility = float(
        strategy_returns.std() * np.sqrt(trading_days_per_year)
    )

    excess = strategy_returns - risk_free_rate / trading_days_per_year
    sharpe = (
        float(excess.mean() / excess.std() * np.sqrt(trading_days_per_year))
        if excess.std() > 0
        else 0.0
    )

    running_max = equity_curve["strategy"].cummax()
    drawdown = equity_curve["strategy"] / running_max - 1
    max_drawdown = float(drawdown.min())

    winning_days = strategy_returns[strategy_returns > 0]
    active_days = strategy_returns[position > 0]
    win_rate = (
        float(len(winning_days) / len(active_days)) if len(active_days) > 0 else 0.0
    )

    return BacktestResult(
        equity_curve=equity_curve,
        total_return=total_return,
        buy_hold_return=buy_hold_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
    )
