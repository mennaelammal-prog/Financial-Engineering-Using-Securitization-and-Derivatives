import numpy as np
import pandas as pd

from trade_marketing_tool.indicators import sma, ema, rsi, macd, bollinger_bands, atr


def _price_series(n=300, seed=7):
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0005, scale=0.01, size=n)
    prices = 100 * np.cumprod(1 + steps)
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.Series(prices, index=idx)


def test_sma_matches_manual_rolling_mean():
    s = _price_series()
    result = sma(s, window=10)
    expected = s.rolling(10).mean()
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_ema_with_shorter_span_reacts_faster_to_a_shock():
    s = _price_series()
    shocked = s.copy()
    shocked.iloc[-1] *= 1.5  # sudden jump
    fast = ema(shocked, span=5)
    slow = ema(shocked, span=50)
    # A shorter span weights the new observation more heavily, so it should
    # move further in response to the same shock than a longer span does.
    assert abs(fast.iloc[-1] - fast.iloc[-2]) > abs(slow.iloc[-1] - slow.iloc[-2])


def test_rsi_is_bounded_0_100():
    s = _price_series()
    r = rsi(s)
    assert r.dropna().between(0, 100).all()


def test_rsi_pegs_high_for_monotonic_uptrend():
    idx = pd.bdate_range("2023-01-02", periods=30)
    s = pd.Series(np.linspace(100, 130, 30), index=idx)
    r = rsi(s, window=14)
    assert r.iloc[-1] > 90


def test_macd_histogram_is_difference_of_macd_and_signal():
    s = _price_series()
    df = macd(s)
    diff = df["macd"] - df["macd_signal"]
    pd.testing.assert_series_equal(df["macd_hist"], diff, check_names=False)


def test_bollinger_bands_bracket_the_middle_band():
    s = _price_series()
    bb = bollinger_bands(s)
    valid = bb.dropna()
    assert (valid["bb_upper"] >= valid["bb_mid"]).all()
    assert (valid["bb_lower"] <= valid["bb_mid"]).all()


def test_atr_is_non_negative():
    idx = pd.bdate_range("2023-01-02", periods=50)
    high = pd.Series(np.linspace(101, 120, 50), index=idx)
    low = pd.Series(np.linspace(99, 118, 50), index=idx)
    close = (high + low) / 2
    ohlc = pd.DataFrame({"High": high, "Low": low, "Close": close})
    a = atr(ohlc, window=14)
    assert (a.dropna() >= 0).all()
