import numpy as np
import pandas as pd

from trade_marketing_tool.signals import _apply_exit_overlay, backtest, generate_signals


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


def test_confirmation_off_matches_plain_sma_trend_regime():
    ohlc = _trending_ohlc()
    df = generate_signals(ohlc, require_confirmation=False)
    expected = (df["sma_50"] > df["sma_200"]).astype(int)
    assert (df["position"] == expected).all()


def test_confirmation_on_never_holds_long_against_the_trend():
    ohlc = _trending_ohlc()
    df = generate_signals(ohlc, require_confirmation=True)
    long_bars = df[df["position"] == 1]
    assert (long_bars["sma_50"] > long_bars["sma_200"]).all()


def test_backtest_runs_and_returns_sane_metrics():
    ohlc = _trending_ohlc()
    signaled = generate_signals(ohlc)
    result = backtest(signaled)
    assert result.equity_curve["strategy"].iloc[0] > 0
    assert -1.0 <= result.max_drawdown <= 0.0
    assert 0.0 <= result.win_rate <= 1.0


def test_backtest_reports_trade_level_win_rate():
    ohlc = _trending_ohlc()
    signaled = generate_signals(ohlc)
    result = backtest(signaled)
    assert result.n_trades >= 0
    assert 0.0 <= result.trade_win_rate <= 1.0
    # A strongly uptrending synthetic series should take at least one trade.
    assert result.n_trades > 0


def test_backtest_stop_loss_can_be_disabled():
    ohlc = _trending_ohlc()
    signaled = generate_signals(ohlc)
    with_stop = backtest(signaled, stop_loss_pct=0.08, take_profit_pct=None)
    without_stop = backtest(signaled, stop_loss_pct=None, take_profit_pct=None)
    # Disabling the stop should never take MORE trades than having it on
    # (the stop can only split/shorten runs, never lengthen them) — and a
    # tighter stop should never produce a deeper max drawdown than none at all.
    assert without_stop.max_drawdown <= with_stop.max_drawdown + 1e-9


def test_take_profit_exits_once_target_hit():
    close = pd.Series([100, 105, 112, 116, 108, 95, 90], dtype=float)
    raw_position = pd.Series([1, 1, 1, 1, 1, 1, 1])
    position = _apply_exit_overlay(close, raw_position, stop_loss_pct=None, take_profit_pct=0.10)
    # 10% target = 110, first reached at index 2 (112) -> flat from that bar on.
    assert list(position) == [1, 1, 0, 0, 0, 0, 0]


def test_trailing_stop_exits_on_drawdown_from_running_high():
    close = pd.Series([100, 110, 100, 95, 90], dtype=float)
    raw_position = pd.Series([1, 1, 1, 1, 1])
    position = _apply_exit_overlay(close, raw_position, stop_loss_pct=0.10, take_profit_pct=None)
    # Trailing high reaches 110 at index 1; 10% stop = 99, first breached at index 3 (95).
    assert list(position) == [1, 1, 1, 0, 0]


def test_exit_overlay_rearms_after_raw_signal_resets():
    close = pd.Series([100, 112, 90, 100, 130], dtype=float)
    raw_position = pd.Series([1, 1, 0, 1, 1])
    position = _apply_exit_overlay(close, raw_position, stop_loss_pct=None, take_profit_pct=0.10)
    # First run hits its 10% target at index 1 (112) and exits; the raw
    # signal dropping to 0 at index 2 rearms a fresh entry (price=100) at
    # index 3, which hits its own target at index 4 (130).
    assert list(position) == [1, 0, 0, 1, 0]


def test_backtest_take_profit_can_be_disabled():
    ohlc = _trending_ohlc()
    signaled = generate_signals(ohlc)
    with_tp = backtest(signaled, stop_loss_pct=None, take_profit_pct=0.15)
    without_tp = backtest(signaled, stop_loss_pct=None, take_profit_pct=None)
    assert isinstance(with_tp.win_rate, float)
    assert isinstance(without_tp.win_rate, float)


def test_flat_position_produces_zero_strategy_return():
    ohlc = _trending_ohlc()
    signaled = generate_signals(ohlc)
    signaled["position"] = 0
    result = backtest(signaled)
    assert result.total_return == 0.0
    assert result.buy_hold_return != 0.0
    assert result.n_trades == 0
    assert result.trade_win_rate == 0.0
