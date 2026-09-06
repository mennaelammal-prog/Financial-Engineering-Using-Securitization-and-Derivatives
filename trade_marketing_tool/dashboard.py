"""
Streamlit dashboard — the "one tool" that ties every module together:
a TradingView-style chart, portfolio sensitivity (Beta/Alpha/R-squared),
a Black-Scholes options pricer, and a rule-based trade-signal backtest.

Run with:
    streamlit run trade_marketing_tool/dashboard.py
"""

from __future__ import annotations

import streamlit as st

from .charts import build_candlestick_chart
from .data import DataFetchError, single_ticker_ohlcv
from .options import OptionInputs, price_option
from .sensitivity import analyze_portfolio_sensitivity
from .signals import backtest, generate_signals

st.set_page_config(
    page_title="Share & Index Analytics Workbench", layout="wide", page_icon="📈"
)

st.title("📈 Share & Index Analytics Workbench")
st.caption(
    "Grounded in the risk-taxonomy, price-sensitivity, and derivatives "
    "chapters of Dr. Munir Ibrahim Hindi's *Financial Engineering Using "
    "Securitization and Derivatives*."
)

tab_chart, tab_sensitivity, tab_options, tab_signals = st.tabs(
    ["TradingView Chart", "Portfolio Sensitivity", "Options Pricer", "Trade Signals"]
)

# ----------------------------------------------------------------------
# Tab 1: TradingView-style chart
# ----------------------------------------------------------------------
with tab_chart:
    col1, col2, col3 = st.columns([2, 1, 1])
    ticker = col1.text_input("Ticker", value="AAPL", key="chart_ticker").upper()
    period = col2.selectbox(
        "Period", ["3mo", "6mo", "1y", "2y", "5y", "max"], index=2, key="chart_period"
    )
    interval = col3.selectbox("Interval", ["1d", "1wk", "1mo"], index=0, key="chart_interval")

    show_volume = st.checkbox("Show volume", value=True)
    show_rsi = st.checkbox("Show RSI", value=True)
    show_macd = st.checkbox("Show MACD", value=True)

    if st.button("Load chart", key="load_chart"):
        try:
            with st.spinner(f"Fetching {ticker}..."):
                ohlc = single_ticker_ohlcv(ticker, period=period, interval=interval)
            fig = build_candlestick_chart(
                ohlc, ticker, show_volume=show_volume, show_rsi=show_rsi, show_macd=show_macd
            )
            st.plotly_chart(fig, use_container_width=True)
        except DataFetchError as exc:
            st.error(str(exc))

# ----------------------------------------------------------------------
# Tab 2: Portfolio sensitivity (Beta/Alpha/R-squared)
# ----------------------------------------------------------------------
with tab_sensitivity:
    st.write("Enter holdings as `TICKER:WEIGHT`, one per line (weights auto-normalize).")
    default_portfolio = "AAPL:0.30\nMSFT:0.25\nNVDA:0.25\nJNJ:0.20"
    holdings_text = st.text_area("Portfolio", value=default_portfolio, height=140)
    index_ticker = st.text_input("Benchmark index", value="^GSPC")
    sens_period = st.selectbox("History window", ["1y", "2y", "5y"], index=1)

    if st.button("Analyze sensitivity"):
        try:
            portfolio = {}
            for line in holdings_text.strip().splitlines():
                if not line.strip():
                    continue
                t, w = line.split(":")
                portfolio[t.strip().upper()] = float(w)

            with st.spinner("Computing Beta/Alpha/R-squared..."):
                result = analyze_portfolio_sensitivity(
                    portfolio, index_ticker=index_ticker, period=sens_period
                )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Portfolio Beta (β)", f"{result.portfolio_beta:.3f}")
            c2.metric("Portfolio Alpha (α, ann.)", f"{result.portfolio_alpha:.2%}")
            c3.metric("Portfolio Volatility", f"{result.portfolio_volatility:.2%}")
            c4.metric(f"{index_ticker} Volatility", f"{result.index_volatility:.2%}")

            st.subheader("Per-asset sensitivity")
            st.dataframe(
                result.per_asset.style.format(
                    {
                        "Weight": "{:.1%}",
                        "Beta": "{:.4f}",
                        "Alpha": "{:.2%}",
                        "Volatility": "{:.2%}",
                        "R-Squared": "{:.4f}",
                    }
                ),
                use_container_width=True,
            )

            st.subheader("Market-shock scenario (Beta-implied)")
            st.table(result.scenario_table)
        except (ValueError, DataFetchError) as exc:
            st.error(str(exc))

# ----------------------------------------------------------------------
# Tab 3: Options pricer
# ----------------------------------------------------------------------
with tab_options:
    c1, c2, c3 = st.columns(3)
    spot = c1.number_input("Spot price (S)", value=190.0)
    strike = c2.number_input("Strike price (K)", value=195.0)
    days = c3.number_input("Days to expiry", value=30, min_value=1)

    c4, c5, c6 = st.columns(3)
    rate = c4.number_input("Risk-free rate", value=0.05, format="%.4f")
    vol = c5.number_input("Volatility (annualized)", value=0.25, format="%.4f")
    dividend_yield = c6.number_input("Dividend yield", value=0.0, format="%.4f")

    option_type = st.radio("Option type", ["call", "put"], horizontal=True)

    if st.button("Price option"):
        try:
            inp = OptionInputs(
                spot=spot,
                strike=strike,
                time_to_expiry=days / 365.0,
                rate=rate,
                volatility=vol,
                dividend_yield=dividend_yield,
            )
            result = price_option(inp, option_type=option_type)
            c1, c2, c3 = st.columns(3)
            c1.metric("Price", f"{result.price:.4f}")
            c2.metric("Delta", f"{result.delta:.4f}")
            c3.metric("Gamma", f"{result.gamma:.4f}")
            c4, c5 = st.columns(2)
            c4.metric("Vega (per 1% vol)", f"{result.vega/100:.4f}")
            c5.metric("Theta (per day)", f"{result.theta/365:.4f}")
        except ValueError as exc:
            st.error(str(exc))

# ----------------------------------------------------------------------
# Tab 4: Trade signals + backtest
# ----------------------------------------------------------------------
with tab_signals:
    sig_ticker = st.text_input("Ticker", value="AAPL", key="sig_ticker").upper()
    sig_period = st.selectbox("History window", ["1y", "2y", "5y"], index=1, key="sig_period")

    if st.button("Run signal backtest"):
        try:
            with st.spinner(f"Fetching {sig_ticker} and running backtest..."):
                ohlc = single_ticker_ohlcv(sig_ticker, period=sig_period)
                signaled = generate_signals(ohlc)
                result = backtest(signaled)

            c1, c2, c3 = st.columns(3)
            c1.metric("Strategy return", f"{result.total_return:.2%}")
            c2.metric("Buy & hold return", f"{result.buy_hold_return:.2%}")
            c3.metric("Sharpe ratio", f"{result.sharpe_ratio:.2f}")
            c4, c5 = st.columns(2)
            c4.metric("Max drawdown", f"{result.max_drawdown:.2%}")
            c5.metric("Win rate (active days)", f"{result.win_rate:.2%}")

            st.line_chart(result.equity_curve)
            st.caption(
                "Rule-based signal: SMA50/SMA200 trend filter, confirmed by RSI "
                "and MACD crossovers. Educational/research use — not investment advice."
            )
        except DataFetchError as exc:
            st.error(str(exc))
