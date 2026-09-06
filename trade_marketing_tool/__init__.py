"""
Trade Marketing & Share/Index Analytics Tool
=============================================

Grounded in the financial-engineering framework of Dr. Munir Ibrahim Hindi's
*Modern Thought in Risk Management: Financial Engineering Using
Securitization and Derivatives* — the systematic-risk taxonomy, bond/price
sensitivity mechanics, and derivatives-pricing chapters translate directly
into the market-analytics modules below. The retail-trade-marketing modules
(``promotion_impact``, ``shelf_sensitivity``, ``outlet_segmentation``) apply
the equivalent applied-statistics toolkit — mean-difference tests,
regression/correlation, and cluster/factor analysis — described in
*Analytical Statistics with SPSS Applications*.

- ``data``               : live market data retrieval (yfinance) with local caching
- ``indicators``         : SMA/EMA/RSI/MACD/Bollinger/ATR technical indicators
- ``charts``             : interactive TradingView-style candlestick charts (Plotly)
- ``sensitivity``        : portfolio Beta / Alpha / R-squared vs. a benchmark index
                           (the book's systematic-risk taxonomy, quantified)
- ``options``            : Black-Scholes option pricing + Greeks (the book's
                           derivatives-for-hedging chapter)
- ``signals``            : rule-based trade signals + a lightweight backtester
- ``promotion_impact``   : t-tests / one-way ANOVA to measure campaign uplift
                           and compare promotion variants
- ``shelf_sensitivity``  : Pearson correlation + linear regression of sales
                           against shelf space, price, or any trade lever
- ``outlet_segmentation``: PCA (factor analysis) + K-Means clustering to
                           segment outlets/distributors by value
- ``dashboard``          : a Streamlit app tying every module into one workbench
- ``cli``                : command-line entry points for quick, scriptable use
"""

__version__ = "0.1.0"
