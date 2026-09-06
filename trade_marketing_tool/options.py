"""
Black-Scholes European option pricing and Greeks.

This is the derivatives-pricing counterpart to the book's chapters on
forwards/futures/options/swaps used for hedging exposure — applied here to
individual equities or index ETFs (e.g. an SPY/QQQ option chain).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt

from scipy.stats import norm
from scipy.optimize import brentq


@dataclass
class OptionInputs:
    spot: float          # S: current underlying price
    strike: float        # K: strike price
    time_to_expiry: float  # T: in years (e.g. 30/365)
    rate: float          # r: annualized risk-free rate (decimal, e.g. 0.05)
    volatility: float    # sigma: annualized volatility (decimal, e.g. 0.25)
    dividend_yield: float = 0.0  # q: continuous dividend yield


@dataclass
class OptionResult:
    price: float
    delta: float
    gamma: float
    vega: float   # per 1.00 (100%) change in volatility
    theta: float  # per year; divide by 365 for per-day decay
    rho: float    # per 1.00 (100%) change in rate


def _d1_d2(inp: OptionInputs) -> tuple[float, float]:
    if inp.time_to_expiry <= 0 or inp.volatility <= 0:
        raise ValueError("time_to_expiry and volatility must both be > 0")
    S, K, T, r, sigma, q = (
        inp.spot,
        inp.strike,
        inp.time_to_expiry,
        inp.rate,
        inp.volatility,
        inp.dividend_yield,
    )
    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return d1, d2


def price_option(inp: OptionInputs, option_type: str = "call") -> OptionResult:
    """
    Price a European call or put and return price + the standard Greeks.
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    S, K, T, r, sigma, q = (
        inp.spot,
        inp.strike,
        inp.time_to_expiry,
        inp.rate,
        inp.volatility,
        inp.dividend_yield,
    )
    d1, d2 = _d1_d2(inp)
    disc_q = exp(-q * T)
    disc_r = exp(-r * T)

    if option_type == "call":
        price = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
        delta = disc_q * norm.cdf(d1)
        rho = K * T * disc_r * norm.cdf(d2) / 100
        theta = (
            -S * disc_q * norm.pdf(d1) * sigma / (2 * sqrt(T))
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        )
    else:  # put
        price = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
        delta = -disc_q * norm.cdf(-d1)
        rho = -K * T * disc_r * norm.cdf(-d2) / 100
        theta = (
            -S * disc_q * norm.pdf(d1) * sigma / (2 * sqrt(T))
            + r * K * disc_r * norm.cdf(-d2)
            - q * S * disc_q * norm.cdf(-d1)
        )

    gamma = disc_q * norm.pdf(d1) / (S * sigma * sqrt(T))
    vega = S * disc_q * norm.pdf(d1) * sqrt(T) / 100

    return OptionResult(
        price=price, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho
    )


def implied_volatility(
    market_price: float,
    inp: OptionInputs,
    option_type: str = "call",
    lo: float = 1e-4,
    hi: float = 5.0,
) -> float:
    """
    Solve for the volatility that reproduces ``market_price`` under
    Black-Scholes, via bisection (Brent's method). Raises ValueError if no
    solution exists in [lo, hi] (e.g. price outside no-arbitrage bounds).
    """

    def objective(sigma: float) -> float:
        candidate = OptionInputs(
            spot=inp.spot,
            strike=inp.strike,
            time_to_expiry=inp.time_to_expiry,
            rate=inp.rate,
            volatility=sigma,
            dividend_yield=inp.dividend_yield,
        )
        return price_option(candidate, option_type).price - market_price

    try:
        return brentq(objective, lo, hi, xtol=1e-6)
    except ValueError as exc:
        raise ValueError(
            "Could not solve for implied volatility in the given range — "
            "the market price may be outside no-arbitrage bounds."
        ) from exc
