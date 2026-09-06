# Share & Index Analytics Workbench

A Python tool for **trade marketing / share & index analysis**: TradingView-style
interactive charts, portfolio Beta/Alpha sensitivity vs. a benchmark index, a
Black-Scholes options pricer, and a rule-based trade-signal backtester — all
in one Streamlit dashboard or a scriptable CLI.

## Why this exists

This project operationalizes the "Share & Index Analytics Engine" blueprint
sketched out while reading Dr. Munir Ibrahim Hindi's *Modern Thought in Risk
Management: Financial Engineering Using Securitization and Derivatives*:

| Book concept | Tool module |
|---|---|
| Systematic risk taxonomy, benchmark-relative risk | `sensitivity.py` — portfolio Beta/Alpha/R² vs. an index |
| Bond/price sensitivity mechanics | same regression machinery, applied to equities |
| Derivatives for hedging (forwards/options) | `options.py` — Black-Scholes pricing + Greeks |
| Securitization turning cash flows into tradeable instruments | see **Roadmap** below (Sukuk/MBS simulator) |

It grew out of a single-portfolio PowerShell/`yfinance` script into a proper
package: reusable functions, tests with no network dependency, an
interactive TradingView-style chart layer, and a dashboard tying it together.

## Install

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Requires internet access to Yahoo Finance (via `yfinance`) for live data —
the options pricer and indicator math work fully offline.

## Quick start

### Dashboard (recommended)

```bash
streamlit run trade_marketing_tool/dashboard.py
```

Opens a browser tab with four tabs:
1. **TradingView Chart** — candlesticks + SMA/EMA/Bollinger overlays, volume, RSI, MACD, zoom/pan/crosshair.
2. **Portfolio Sensitivity** — enter holdings as `TICKER:WEIGHT`, get weighted portfolio Beta/Alpha/volatility vs. an index (e.g. `^GSPC`, `^AXJO`), plus a Beta-implied market-shock scenario table.
3. **Options Pricer** — Black-Scholes price + Delta/Gamma/Vega/Theta/Rho for a call or put.
4. **Trade Signals** — SMA50/200 trend filter confirmed by RSI/MACD crossovers, backtested against buy-and-hold with Sharpe ratio, max drawdown, win rate.

### CLI

```bash
# Portfolio sensitivity (weights auto-normalize)
python -m trade_marketing_tool.cli sensitivity AAPL:0.30 MSFT:0.25 NVDA:0.25 JNJ:0.20 --index ^GSPC --period 2y

# Australian ASX 200 example
python -m trade_marketing_tool.cli sensitivity BHP.AX:0.40 CBA.AX:0.35 WES.AX:0.25 --index ^AXJO

# Save an interactive TradingView-style chart to HTML
python -m trade_marketing_tool.cli chart AAPL --period 1y --out aapl_chart.html

# Price an option (Black-Scholes)
python -m trade_marketing_tool.cli option --spot 190 --strike 195 --days 30 --rate 0.05 --vol 0.28 --type call

# Backtest the built-in trend/RSI/MACD signal vs. buy & hold
python -m trade_marketing_tool.cli backtest AAPL --period 2y
```

### From Python

```python
from trade_marketing_tool.sensitivity import analyze_portfolio_sensitivity, format_report

result = analyze_portfolio_sensitivity(
    {"AAPL": 0.30, "MSFT": 0.25, "NVDA": 0.25, "JNJ": 0.20},
    index_ticker="^GSPC",
    period="2y",
)
print(format_report(result))
```

## Project layout

```
trade_marketing_tool/
  data.py          live OHLCV retrieval (yfinance) + local cache
  indicators.py    SMA/EMA/RSI/MACD/Bollinger/ATR (pure, unit-tested math)
  charts.py        TradingView-style interactive Plotly candlestick charts
  sensitivity.py   portfolio Beta/Alpha/Volatility/R-squared vs. an index
  options.py       Black-Scholes pricing + Greeks + implied volatility
  signals.py       rule-based trade signals + a simple backtester
  dashboard.py     Streamlit app tying every module into one workbench
  cli.py           scriptable command-line entry points
tests/             pytest suite (indicators/options/sensitivity/signals — no network needed)
```

## Testing

```bash
pytest
```

`tests/test_sensitivity.py` uses a synthetic index + two assets with known
betas (monkeypatching the data layer) so the regression math is checked
against ground truth without hitting the network. Every other test module
is pure math and needs no network access either.

## Roadmap

- **Structured Equity/Sukuk Simulator** — cash-flow waterfall modeling for
  securitized/Sharia-compliant instruments (the book's Part 4/5 material).
- **Duration & Convexity module** — bond price-sensitivity calculator,
  extending the same regression-style sensitivity approach to fixed income.
- **Pair-trading / cointegration engine** — mean-reversion signal between a
  stock (or basket) and its benchmark index.

## Disclaimer

Educational/research tool. The trade-signal backtester is intentionally
simple and not tuned or validated for live trading — it exists to make a
signal's assumptions and performance visible, not to recommend trades.
Nothing here is investment advice.
