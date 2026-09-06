"""
Trade Marketing & Share/Index Analytics Tool
=============================================

Grounded in the financial-engineering framework of Dr. Munir Ibrahim Hindi's
*Modern Thought in Risk Management: Financial Engineering Using
Securitization and Derivatives* — the systematic-risk taxonomy, bond/price
sensitivity mechanics, and derivatives-pricing chapters translate directly
into the modules below:

- ``data``        : live market data retrieval (yfinance) with local caching
- ``indicators``  : SMA/EMA/RSI/MACD/Bollinger/ATR technical indicators
- ``charts``      : interactive TradingView-style candlestick charts (Plotly)
- ``sensitivity`` : portfolio Beta / Alpha / R-squared vs. a benchmark index
                    (the book's systematic-risk taxonomy, quantified)
- ``options``     : Black-Scholes option pricing + Greeks (the book's
                    derivatives-for-hedging chapter)
- ``signals``      : rule-based trade signals + a lightweight backtester
- ``dashboard``   : a Streamlit app tying every module into one workbench
- ``cli``         : command-line entry points for quick, scriptable use
"""

__version__ = "0.1.0"
