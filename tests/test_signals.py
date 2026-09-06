import numpy as np
import pandas as pd

from trade_marketing_tool.signals import generate_signals, backtest


def _trending_ohlc(n=300, seed=11):
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.001, scale=0.01, size=n)
    close = 100 * np.cumprod(1 + steps)
    idx = pd.bdate_range("2023-01-02", periods=n)
    high = close * 1.01
    low = close * 0.99
    open_ = close * (1 + rng.normal(0, 0.001, n))
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def test_generate_signals_adds_expected_columns():
    ohlc = _trending_ohlc()
    df = generate_signals(ohlc)
    for col in ["sma_50", "sma_200", "rsi_14", "macd", "signal_score", "position"]:
        assert col in df.columns
    assert set(df["position"].dropna().unique()).issubset({0, 1})


def test_backtest_runs_and_returns_sane_metrics():
    ohlc = _trending_ohlc()
    signaled = generate_signals(ohlc)
    result = backtest(signaled)
    assert result.equity_curve["strategy"].iloc[0] > 0
    assert -1.0 <= result.max_drawdown <= 0.0
    assert 0.0 <= result.win_rate <= 1.0


def test_flat_position_produces_zero_strategy_return():
    ohlc = _trending_ohlc()
    signaled = generate_signals(ohlc)
    signaled["position"] = 0
    result = backtest(signaled)
    assert result.total_return == 0.0
    assert result.buy_hold_return != 0.0
