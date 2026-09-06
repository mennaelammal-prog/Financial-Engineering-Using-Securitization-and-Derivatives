import numpy as np
import pandas as pd
import pytest

import trade_marketing_tool.sensitivity as sensitivity_module
from trade_marketing_tool.sensitivity import analyze_portfolio_sensitivity


def _synthetic_prices(n=300, seed=3):
    """
    Build a synthetic index and two stocks with known betas so the
    regression output can be checked against ground truth, without any
    network access.
    """
    rng = np.random.default_rng(seed)
    idx_returns = rng.normal(0.0004, 0.01, n)
    idx_prices = 100 * np.cumprod(1 + idx_returns)

    # Stock A: beta ~2.0, no alpha, no idiosyncratic noise (deterministic beta)
    a_returns = 2.0 * idx_returns
    a_prices = 50 * np.cumprod(1 + a_returns)

    # Stock B: beta ~0.5 plus a bit of idiosyncratic noise
    noise = rng.normal(0, 0.002, n)
    b_returns = 0.5 * idx_returns + noise
    b_prices = 80 * np.cumprod(1 + b_returns)

    date_idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame(
        {"STOCKA": a_prices, "STOCKB": b_prices, "^IDX": idx_prices}, index=date_idx
    )


@pytest.fixture
def patched_close_prices(monkeypatch):
    prices = _synthetic_prices()

    def fake_close_prices(tickers, period="2y", interval="1d"):
        return prices[tickers]

    monkeypatch.setattr(sensitivity_module, "close_prices", fake_close_prices)
    return prices


def test_beta_recovers_known_value_for_noiseless_asset(patched_close_prices):
    result = analyze_portfolio_sensitivity(
        {"STOCKA": 1.0}, index_ticker="^IDX", period="2y"
    )
    beta = result.per_asset.loc[result.per_asset["Ticker"] == "STOCKA", "Beta"].item()
    assert beta == pytest.approx(2.0, abs=0.01)


def test_weights_are_normalized_when_not_summing_to_one(patched_close_prices):
    result = analyze_portfolio_sensitivity(
        {"STOCKA": 3.0, "STOCKB": 1.0}, index_ticker="^IDX", period="2y"
    )
    assert result.per_asset["Weight"].sum() == pytest.approx(1.0)
    assert result.per_asset.loc[
        result.per_asset["Ticker"] == "STOCKA", "Weight"
    ].item() == pytest.approx(0.75)


def test_portfolio_beta_is_weighted_average_of_asset_betas(patched_close_prices):
    result = analyze_portfolio_sensitivity(
        {"STOCKA": 0.5, "STOCKB": 0.5}, index_ticker="^IDX", period="2y"
    )
    expected = (
        result.per_asset.set_index("Ticker")["Weight"]
        * result.per_asset.set_index("Ticker")["Beta"]
    ).sum()
    assert result.portfolio_beta == pytest.approx(expected)


def test_scenario_table_uses_beta_implied_moves(patched_close_prices):
    result = analyze_portfolio_sensitivity(
        {"STOCKA": 1.0}, index_ticker="^IDX", period="2y", shocks=(-0.10, 0.10)
    )
    assert list(result.scenario_table["Market Move"]) == ["-10%", "+10%"]


def test_empty_portfolio_raises():
    with pytest.raises(ValueError):
        analyze_portfolio_sensitivity({})


def test_zero_total_weight_raises():
    with pytest.raises(ValueError):
        analyze_portfolio_sensitivity({"STOCKA": 0.0, "STOCKB": 0.0})
