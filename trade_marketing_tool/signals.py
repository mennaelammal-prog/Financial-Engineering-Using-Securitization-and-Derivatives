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


def generate_signals(ohlc: pd.DataFrame, require_confirmation: bool = True) -> pd.DataFrame:
    """
    Attach indicator columns plus a per-bar signal score in [-3, +3]:
    +1 for each bullish condition triggered, -1 for each bearish one:
      - SMA50 crossing above/below SMA200 (golden/death cross)
      - RSI leaving oversold (<30) upward / overbought (>70) downward
      - MACD line crossing above/below its signal line

    A ``position`` column of {0, 1} (flat/long) is derived from the SMA50/
    SMA200 trend regime. When ``require_confirmation`` is True (default),
    entering long additionally requires ``signal_score > 0`` on that bar —
    a golden cross, RSI leaving oversold, or a bullish MACD cross — instead
    of buying the instant the regime turns up; this filters out weak
    trend-starts that reverse immediately, at the cost of taking some
    trades late or missing choppy regimes. The position always still exits
    the moment SMA50 crosses back below SMA200, same as with confirmation
    off. Set ``require_confirmation=False`` to recover the plain SMA50/200
    regime filter (no RSI/MACD gating).
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
    trend_up = (df["sma_50"] > df["sma_200"]).to_numpy()

    if not require_confirmation:
        df["position"] = trend_up.astype(int)
        return df

    bullish_confirmation = (score > 0).to_numpy()
    position = np.zeros(len(df), dtype=int)
    in_position = False
    for i in range(len(df)):
        if in_position:
            in_position = trend_up[i]
        elif trend_up[i] and bullish_confirmation[i]:
            in_position = True
        position[i] = 1 if in_position else 0
    df["position"] = position
    return df


def _apply_exit_overlay(
    close: pd.Series,
    raw_position: pd.Series,
    stop_loss_pct: float | None,
    take_profit_pct: float | None,
) -> pd.Series:
    """
    Overlay an optional trailing stop and/or take-profit target on a raw
    long/flat position series. Once long, exit early — before the raw
    signal itself would exit on trend-reversal — if price falls more than
    ``stop_loss_pct`` below the highest close seen since entry (a trailing
    stop), or rises more than ``take_profit_pct`` above the entry price (a
    fixed target that locks in a gain instead of riding out every
    pullback along the way to trend-reversal). Either can be disabled by
    passing ``None``. Stays flat for the remainder of that raw signal's
    run — until the raw position drops back to 0, i.e. a fresh entry
    decision — rather than immediately re-buying into the same exit.
    """
    position = np.zeros(len(close), dtype=int)
    in_position = False
    closed_out = False
    entry_price = None
    trailing_high = None
    raw = raw_position.astype(int).to_numpy()
    prices = close.to_numpy()

    for i in range(len(close)):
        if raw[i] == 0:
            in_position = False
            closed_out = False
            entry_price = None
            trailing_high = None
        elif not closed_out:
            if entry_price is None:
                entry_price = prices[i]
                trailing_high = prices[i]
            else:
                trailing_high = max(trailing_high, prices[i])

            hit_stop = stop_loss_pct is not None and prices[i] <= trailing_high * (
                1 - stop_loss_pct
            )
            hit_target = take_profit_pct is not None and prices[i] >= entry_price * (
                1 + take_profit_pct
            )

            if hit_stop or hit_target:
                closed_out = True
                in_position = False
            else:
                in_position = True
        position[i] = 1 if in_position else 0

    return pd.Series(position, index=close.index, dtype=int)


def _trade_returns(strategy_returns: list[float], position: list[float]) -> list[float]:
    """Compound consecutive in-position bars into per-trade returns."""
    trades: list[list[float]] = []
    current: list[float] = []
    for ret, pos in zip(strategy_returns, position):
        if pos > 0:
            current.append(ret)
        elif current:
            trades.append(current)
            current = []
    if current:
        trades.append(current)
    return [float(np.prod([1 + r for r in trade]) - 1) for trade in trades]


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame  # strategy vs buy-and-hold cumulative returns
    total_return: float
    buy_hold_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float           # fraction of individual in-position DAYS that closed positive
    n_trades: int              # number of separate long trades taken
    trade_win_rate: float      # fraction of complete TRADES (entry-to-exit) that were profitable


def backtest(
    df_with_signals: pd.DataFrame,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    stop_loss_pct: float | None = 0.08,
    take_profit_pct: float | None = None,
) -> BacktestResult:
    """
    Backtest the ``position`` column from ``generate_signals`` against
    buy-and-hold, using next-bar returns (position is applied with a one-bar
    lag to avoid look-ahead bias).

    ``stop_loss_pct`` (default 8%) overlays a trailing stop on top of the
    raw signal to cut losing trades short instead of riding them out to the
    next trend-reversal exit. ``take_profit_pct`` (default: disabled)
    optionally overlays a fixed target that locks in a gain once price has
    risen that far above entry, instead of giving it all back on the
    pullback that eventually triggers the trend-reversal exit — the
    intended lever for raising the per-day win rate specifically, since it
    converts "still winning overall but red today" days into closed-out
    green ones. In practice, on synthetic backtests across many random
    seeds, it moved the per-day win rate by well under a percentage point
    while cutting average total return by roughly 10 points — it gives up
    the rare large trend run for almost no win-rate benefit. It's kept
    here as an opt-in experiment, not a default, precisely because it
    doesn't hold up: pushing per-day win rate up this way is fighting the
    strategy's own trend-following nature rather than improving it. See
    ``_apply_exit_overlay``. Pass either as ``None`` to disable it and fall
    back to the raw signal's position unmodified for that exit rule.

    Two win-rate figures are returned: ``win_rate`` is the per-day figure
    (naturally low for trend-following, since a single profitable trade
    still contains plenty of red days along the way); ``trade_win_rate`` is
    the fraction of complete entry-to-exit trades that were net
    profitable, usually a more honest read on whether the signal "works".
    """
    close = df_with_signals["Close"]
    daily_returns = close.pct_change().fillna(0)
    raw_position = df_with_signals["position"]

    if stop_loss_pct is not None or take_profit_pct is not None:
        effective_position = _apply_exit_overlay(
            close, raw_position, stop_loss_pct, take_profit_pct
        )
    else:
        effective_position = raw_position

    position = effective_position.shift(1).fillna(0)

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

    trade_rets = _trade_returns(strategy_returns.tolist(), position.tolist())
    n_trades = len(trade_rets)
    trade_win_rate = (
        float(sum(1 for r in trade_rets if r > 0) / n_trades) if n_trades > 0 else 0.0
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
        n_trades=n_trades,
        trade_win_rate=trade_win_rate,
    )
