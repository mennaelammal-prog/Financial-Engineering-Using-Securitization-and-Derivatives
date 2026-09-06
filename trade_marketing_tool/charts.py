"""
Interactive, TradingView-style candlestick charts (Plotly).

Produces a dark-themed, multi-panel chart — price candles with moving-
average/Bollinger overlays, a volume panel, and RSI/MACD panels — that
behaves like TradingView in the browser: zoom, pan, crosshair, and a
range slider. Each chart is a self-contained Plotly Figure you can show
inline (Streamlit/Jupyter) or export to a standalone HTML file.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .indicators import add_all_indicators

# TradingView's familiar dark theme palette
BG_COLOR = "#131722"
GRID_COLOR = "#2a2e39"
TEXT_COLOR = "#d1d4dc"
UP_COLOR = "#26a69a"
DOWN_COLOR = "#ef5350"


def build_candlestick_chart(
    ohlc: pd.DataFrame,
    ticker: str,
    show_volume: bool = True,
    show_rsi: bool = True,
    show_macd: bool = True,
    show_bollinger: bool = True,
) -> go.Figure:
    """
    Build a TradingView-style interactive chart from an OHLCV DataFrame
    (columns Open/High/Low/Close/[Volume]).
    """
    df = add_all_indicators(ohlc)

    panels = ["price"]
    if show_volume and "Volume" in df.columns:
        panels.append("volume")
    if show_rsi:
        panels.append("rsi")
    if show_macd:
        panels.append("macd")

    row_heights = {"price": 0.55, "volume": 0.15, "rsi": 0.15, "macd": 0.15}
    heights = [row_heights[p] for p in panels]
    row_of = {p: i + 1 for i, p in enumerate(panels)}

    fig = make_subplots(
        rows=len(panels),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=heights,
    )

    # --- Price panel: candlesticks + moving averages + Bollinger Bands ---
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=ticker,
            increasing_line_color=UP_COLOR,
            decreasing_line_color=DOWN_COLOR,
            increasing_fillcolor=UP_COLOR,
            decreasing_fillcolor=DOWN_COLOR,
        ),
        row=row_of["price"],
        col=1,
    )
    for col, name, color, width in [
        ("sma_20", "SMA 20", "#f5a623", 1.2),
        ("sma_50", "SMA 50", "#2196f3", 1.2),
        ("sma_200", "SMA 200", "#e040fb", 1.4),
    ]:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                name=name,
                line=dict(color=color, width=width),
                mode="lines",
            ),
            row=row_of["price"],
            col=1,
        )
    if show_bollinger:
        for col, name, dash in [
            ("bb_upper", "BB Upper", "dot"),
            ("bb_lower", "BB Lower", "dot"),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    name=name,
                    line=dict(color="#787b86", width=1, dash=dash),
                    mode="lines",
                ),
                row=row_of["price"],
                col=1,
            )

    # --- Volume panel ---
    if "volume" in panels:
        volume_colors = [
            UP_COLOR if c >= o else DOWN_COLOR
            for o, c in zip(df["Open"], df["Close"])
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker_color=volume_colors,
                showlegend=False,
            ),
            row=row_of["volume"],
            col=1,
        )

    # --- RSI panel ---
    if "rsi" in panels:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["rsi_14"],
                name="RSI 14",
                line=dict(color="#ab47bc", width=1.3),
            ),
            row=row_of["rsi"],
            col=1,
        )
        for level, color in [(70, DOWN_COLOR), (30, UP_COLOR)]:
            fig.add_hline(
                y=level,
                line=dict(color=color, width=1, dash="dash"),
                row=row_of["rsi"],
                col=1,
            )

    # --- MACD panel ---
    if "macd" in panels:
        hist_colors = [UP_COLOR if v >= 0 else DOWN_COLOR for v in df["macd_hist"]]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["macd_hist"],
                name="MACD Hist",
                marker_color=hist_colors,
                showlegend=False,
            ),
            row=row_of["macd"],
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["macd"],
                name="MACD",
                line=dict(color="#2196f3", width=1.2),
            ),
            row=row_of["macd"],
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["macd_signal"],
                name="Signal",
                line=dict(color="#f5a623", width=1.2),
            ),
            row=row_of["macd"],
            col=1,
        )

    fig.update_layout(
        title=f"{ticker} — Price, Volume & Technicals",
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR),
        xaxis_rangeslider_visible=False,
        height=250 * len(panels) + 200,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        margin=dict(l=40, r=20, t=60, b=20),
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        rangeslider_visible=(panels[-1] == "price"),
        rangebreaks=[dict(bounds=["sat", "mon"])],  # skip weekends, no gaps
    )
    fig.update_yaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(range=[0, 100], row=row_of.get("rsi", 1), col=1)

    return fig


def save_chart_html(fig: go.Figure, path: str) -> str:
    """Export a chart to a standalone, interactive HTML file."""
    fig.write_html(path, include_plotlyjs="cdn")
    return path
