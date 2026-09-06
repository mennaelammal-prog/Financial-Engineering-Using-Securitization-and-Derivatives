"""
Share vs. Index Sensitivity Module.

Generalizes the multi-stock portfolio Beta/Alpha/Volatility/R-squared script
from the project notes into a reusable, testable function: quantifying each
holding's systematic risk against a benchmark index (the book's risk-
taxonomy applied to equities), plus weighted portfolio-level metrics and a
simple market-shock scenario table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .data import close_prices

TRADING_DAYS_PER_YEAR = 252


@dataclass
class PortfolioSensitivityResult:
    per_asset: pd.DataFrame          # Ticker, Weight, Beta, Alpha, Volatility, R-Squared
    portfolio_beta: float
    portfolio_alpha: float           # annualized
    portfolio_volatility: float      # annualized
    index_volatility: float          # annualized
    index_ticker: str
    scenario_table: pd.DataFrame     # market shock -> estimated portfolio move


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Portfolio weights must sum to a positive number")
    if not np.isclose(total, 1.0):
        weights = {k: v / total for k, v in weights.items()}
    return weights


def analyze_portfolio_sensitivity(
    portfolio: dict[str, float],
    index_ticker: str = "^GSPC",
    period: str = "2y",
    shocks: tuple[float, ...] = (-0.10, -0.05, 0.05, 0.10),
) -> PortfolioSensitivityResult:
    """
    Compute per-asset and portfolio-level Beta/Alpha/Volatility/R-squared
    against a benchmark index, plus a scenario table estimating portfolio
    impact from hypothetical market moves (using each asset's Beta).

    ``portfolio`` maps ticker -> weight (weights need not sum to 1; they are
    normalized automatically).
    """
    if not portfolio:
        raise ValueError("portfolio must contain at least one ticker")

    weights = _normalize_weights(portfolio)
    tickers = list(weights.keys())

    prices = close_prices(tickers + [index_ticker], period=period)
    missing = [t for t in tickers + [index_ticker] if t not in prices.columns]
    if missing:
        raise ValueError(f"No price data returned for: {missing}")

    prices = prices[tickers + [index_ticker]].dropna()
    returns = prices.pct_change().dropna()
    index_returns = returns[index_ticker]
    X = sm.add_constant(index_returns)

    rows = []
    for ticker in tickers:
        stock_returns = returns[ticker]
        model = sm.OLS(stock_returns, X).fit()

        beta = model.params[index_ticker]
        alpha_annual = model.params["const"] * TRADING_DAYS_PER_YEAR
        volatility = stock_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        r_squared = model.rsquared

        rows.append(
            {
                "Ticker": ticker,
                "Weight": weights[ticker],
                "Beta": beta,
                "Alpha": alpha_annual,
                "Volatility": volatility,
                "R-Squared": r_squared,
            }
        )

    per_asset = pd.DataFrame(rows)

    portfolio_beta = float((per_asset["Weight"] * per_asset["Beta"]).sum())
    portfolio_alpha = float((per_asset["Weight"] * per_asset["Alpha"]).sum())

    weight_vector = per_asset.set_index("Ticker")["Weight"]
    portfolio_daily_returns = (returns[tickers] * weight_vector).sum(axis=1)
    portfolio_volatility = float(
        portfolio_daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    index_volatility = float(index_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

    scenario_table = pd.DataFrame(
        {
            "Market Move": [f"{s:+.0%}" for s in shocks],
            "Estimated Portfolio Move": [
                f"{portfolio_beta * s:+.2%}" for s in shocks
            ],
        }
    )

    return PortfolioSensitivityResult(
        per_asset=per_asset,
        portfolio_beta=portfolio_beta,
        portfolio_alpha=portfolio_alpha,
        portfolio_volatility=portfolio_volatility,
        index_volatility=index_volatility,
        index_ticker=index_ticker,
        scenario_table=scenario_table,
    )


def format_report(result: PortfolioSensitivityResult) -> str:
    """Render a console-friendly text report, same shape as the original script."""
    lines = []
    lines.append("=" * 60)
    lines.append("           INDIVIDUAL ASSET SENSITIVITY")
    lines.append("=" * 60)
    lines.append(
        result.per_asset.to_string(
            index=False,
            formatters={
                "Weight": "{:.1%}".format,
                "Beta": "{:.4f}".format,
                "Alpha": "{:.2%}".format,
                "Volatility": "{:.2%}".format,
                "R-Squared": "{:.4f}".format,
            },
        )
    )
    lines.append("")
    lines.append("=" * 60)
    lines.append("           PORTFOLIO LEVEL SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Benchmark Index:              {result.index_ticker}")
    lines.append(f"Weighted Portfolio Beta (b):   {result.portfolio_beta:.4f}")
    lines.append(f"Weighted Portfolio Alpha (a):  {result.portfolio_alpha:.2%}")
    lines.append(f"Portfolio Annual Volatility:   {result.portfolio_volatility:.2%}")
    lines.append(f"Index Annual Volatility:       {result.index_volatility:.2%}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Scenario Analysis (Beta-implied portfolio move):")
    lines.append(result.scenario_table.to_string(index=False))
    return "\n".join(lines)
