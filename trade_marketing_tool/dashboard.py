"""
Streamlit dashboard — the "one tool" that ties every module together:
a TradingView-style chart, portfolio sensitivity (Beta/Alpha/R-squared),
a Black-Scholes options pricer, and a rule-based trade-signal backtest.

Run with:
    streamlit run trade_marketing_tool/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Streamlit executes this file as a standalone script rather than as part of
# the `trade_marketing_tool` package, so relative imports (`from .charts import
# ...`) fail with "attempted relative import with no known parent package".
# Put the project root on sys.path and import with the full package path
# instead — this works whether or not the package has been pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trade_marketing_tool.charts import build_candlestick_chart
from trade_marketing_tool.data import DataFetchError, single_ticker_ohlcv
from trade_marketing_tool.options import OptionInputs, price_option
from trade_marketing_tool.outlet_segmentation import segment_outlets
from trade_marketing_tool.promotion_impact import compare_before_after, compare_campaign_vs_control
from trade_marketing_tool.sensitivity import analyze_portfolio_sensitivity
from trade_marketing_tool.shelf_sensitivity import analyze_sensitivity, predict as predict_shelf
from trade_marketing_tool.signals import backtest, generate_signals

st.set_page_config(
    page_title="Share & Index Analytics Workbench", layout="wide", page_icon="📈"
)

st.title("📈 Share & Index Analytics Workbench")
st.caption(
    "Grounded in the risk-taxonomy, price-sensitivity, and derivatives "
    "chapters of Dr. Munir Ibrahim Hindi's *Financial Engineering Using "
    "Securitization and Derivatives*."
)

(
    tab_chart,
    tab_sensitivity,
    tab_options,
    tab_signals,
    tab_promo,
    tab_shelf,
    tab_segment,
) = st.tabs(
    [
        "TradingView Chart",
        "Portfolio Sensitivity",
        "Options Pricer",
        "Trade Signals",
        "Promotion Impact",
        "Shelf/Price Sensitivity",
        "Outlet Segmentation",
    ]
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

    c1, c2 = st.columns(2)
    require_confirmation = c1.checkbox(
        "Require RSI/MACD confirmation to enter", value=True, key="sig_confirm"
    )
    use_stop_loss = c2.checkbox("Use trailing stop", value=True, key="sig_use_stop")
    stop_loss_pct = (
        st.slider("Trailing stop %", min_value=2, max_value=20, value=8, key="sig_stop_pct") / 100
        if use_stop_loss
        else None
    )

    if st.button("Run signal backtest"):
        try:
            with st.spinner(f"Fetching {sig_ticker} and running backtest..."):
                ohlc = single_ticker_ohlcv(sig_ticker, period=sig_period)
                signaled = generate_signals(ohlc, require_confirmation=require_confirmation)
                result = backtest(signaled, stop_loss_pct=stop_loss_pct)

            c1, c2, c3 = st.columns(3)
            c1.metric("Strategy return", f"{result.total_return:.2%}")
            c2.metric("Buy & hold return", f"{result.buy_hold_return:.2%}")
            c3.metric("Sharpe ratio", f"{result.sharpe_ratio:.2f}")
            c4, c5, c6 = st.columns(3)
            c4.metric("Max drawdown", f"{result.max_drawdown:.2%}")
            c5.metric("Win rate (per day)", f"{result.win_rate:.2%}")
            c6.metric("Win rate (per trade)", f"{result.trade_win_rate:.2%}", help=f"{result.n_trades} trades taken")

            st.line_chart(result.equity_curve)
            st.caption(
                "Rule-based signal: SMA50/SMA200 trend filter" +
                (", confirmed by RSI/MACD crossovers" if require_confirmation else "") +
                (f", with an {stop_loss_pct:.0%} trailing stop" if stop_loss_pct else "") +
                ". Educational/research use — not investment advice. "
                "Per-day win rate is naturally low for trend-following even when "
                "profitable; per-trade win rate is usually the more honest read."
            )
        except DataFetchError as exc:
            st.error(str(exc))

# ----------------------------------------------------------------------
# Tab 5: Promotion impact (t-test)
# ----------------------------------------------------------------------
with tab_promo:
    st.caption(
        "Applied-statistics chapters (mean-difference tests) of *Analytical "
        "Statistics with SPSS Applications*, applied to trade-promotion sales."
    )
    default_promo = pd.DataFrame(
        {
            "outlet": [f"P{i}" for i in range(1, 9)],
            "before": [12, 15, 8, 22, 18, 10, 14, 20],
            "after": [15, 19, 9, 27, 23, 11, 17, 24],
        }
    )
    promo_df = st.data_editor(default_promo, num_rows="dynamic", key="promo_editor")
    mode = st.radio(
        "Comparison type", ["Paired (before vs. after)", "Independent (campaign vs. control)"],
        horizontal=True,
    )

    if st.button("Run test", key="run_promo"):
        try:
            if mode.startswith("Paired"):
                result = compare_before_after(promo_df["before"], promo_df["after"])
            else:
                result = compare_campaign_vs_control(promo_df["after"], promo_df["before"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Mean before/control", f"{result.mean_before:.2f}")
            c2.metric("Mean after/campaign", f"{result.mean_after:.2f}")
            c3.metric("Uplift", f"{result.uplift:+.2f} ({result.uplift_pct:+.1%})")

            c4, c5 = st.columns(2)
            c4.metric("t-statistic", f"{result.t_statistic:.3f}")
            c5.metric("p-value", f"{result.p_value:.4f}")

            if result.significant:
                st.success(f"Statistically significant at alpha={result.alpha} — the promotion moved sales.")
            else:
                st.warning(f"Not statistically significant at alpha={result.alpha}.")

            chart_df = pd.DataFrame(
                {"Group": [mode.split(" ")[0], "After/Campaign"], "Mean": [result.mean_before, result.mean_after]}
            )
            st.plotly_chart(px.bar(chart_df, x="Group", y="Mean"), use_container_width=True)
        except ValueError as exc:
            st.error(str(exc))

# ----------------------------------------------------------------------
# Tab 6: Shelf/price sensitivity (correlation + regression)
# ----------------------------------------------------------------------
with tab_shelf:
    st.caption(
        "Correlation/regression chapters of *Analytical Statistics with SPSS "
        "Applications*: how strongly does a trade lever (shelf space, price, "
        "display type) move sales, and by how much per unit?"
    )
    default_shelf = pd.DataFrame(
        {
            "outlet": [f"P{i}" for i in range(1, 11)],
            "shelf_space": [2.5, 3.0, 1.5, 4.0, 3.5, 2.0, 3.2, 1.8, 4.5, 2.8],
            "sales": [12, 15, 8, 22, 18, 10, 16, 9, 25, 13],
        }
    )
    shelf_df = st.data_editor(default_shelf, num_rows="dynamic", key="shelf_editor")
    x_col = st.selectbox("X (explanatory)", [c for c in shelf_df.columns if c != "outlet"], index=0)
    y_col = st.selectbox("Y (response)", [c for c in shelf_df.columns if c != "outlet"], index=1)

    if st.button("Fit regression", key="run_shelf"):
        try:
            result = analyze_sensitivity(shelf_df[x_col], shelf_df[y_col], x_label=x_col, y_label=y_col)
            c1, c2, c3 = st.columns(3)
            c1.metric("Pearson r", f"{result.r:.4f}")
            c2.metric("R-squared", f"{result.r_squared:.4f}")
            c3.metric("Slope p-value", f"{result.p_value:.4f}")
            st.write(f"**Equation:** {y_col} = {result.intercept:.3f} + {result.slope:.3f} × {x_col}")

            fig = px.scatter(shelf_df, x=x_col, y=y_col, trendline="ols")
            st.plotly_chart(fig, use_container_width=True)

            predict_x = st.number_input(f"Predict {y_col} at {x_col} =", value=float(shelf_df[x_col].mean()))
            st.metric(f"Predicted {y_col}", f"{predict_shelf(result, predict_x):.2f}")
        except ValueError as exc:
            st.error(str(exc))

# ----------------------------------------------------------------------
# Tab 7: Outlet segmentation (PCA + K-Means)
# ----------------------------------------------------------------------
with tab_segment:
    st.caption(
        "Cluster/factor-analysis chapter of *Analytical Statistics with SPSS "
        "Applications*: group outlets/distributors by behavior to target the "
        "most profitable segment."
    )
    default_seg = pd.DataFrame(
        {
            "outlet": [f"P{i}" for i in range(1, 11)],
            "sales_volume": [12, 15, 8, 22, 18, 10, 16, 9, 25, 13],
            "basket_size": [45, 52, 30, 65, 58, 38, 50, 33, 70, 47],
            "footfall": [300, 340, 210, 420, 380, 260, 330, 220, 460, 310],
        }
    )
    seg_df = st.data_editor(default_seg, num_rows="dynamic", key="seg_editor")
    feature_cols = st.multiselect(
        "Features to cluster on",
        [c for c in seg_df.columns if c != "outlet"],
        default=[c for c in seg_df.columns if c != "outlet"],
    )
    n_clusters = st.slider("Number of segments", min_value=2, max_value=5, value=3)

    if st.button("Segment outlets", key="run_segment"):
        try:
            result = segment_outlets(
                seg_df, features=feature_cols, n_clusters=n_clusters, id_column="outlet"
            )
            st.subheader("Cluster profile")
            st.dataframe(result.cluster_profile, use_container_width=True)

            st.subheader("Outlet assignments")
            st.dataframe(result.assignments, use_container_width=True)

            if len(feature_cols) >= 2:
                fig = px.scatter(
                    result.assignments,
                    x=feature_cols[0],
                    y=feature_cols[1],
                    color="Segment",
                    hover_data=["outlet"] if "outlet" in result.assignments.columns else None,
                )
                st.plotly_chart(fig, use_container_width=True)
        except ValueError as exc:
            st.error(str(exc))
