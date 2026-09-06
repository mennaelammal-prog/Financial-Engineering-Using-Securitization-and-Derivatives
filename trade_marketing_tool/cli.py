"""
Command-line entry points — for quick, scriptable use without launching the
Streamlit dashboard (the PowerShell one-liner style from the project notes).

Examples
--------
    python -m trade_marketing_tool.cli sensitivity AAPL:0.3 MSFT:0.25 NVDA:0.25 JNJ:0.2 --index ^GSPC
    python -m trade_marketing_tool.cli chart AAPL --period 1y --out aapl_chart.html
    python -m trade_marketing_tool.cli option --spot 190 --strike 195 --days 30 --rate 0.05 --vol 0.28 --type call
    python -m trade_marketing_tool.cli backtest AAPL --period 2y
"""

from __future__ import annotations

import argparse
import sys

from .charts import build_candlestick_chart, save_chart_html
from .data import DataFetchError, single_ticker_ohlcv
from .options import OptionInputs, price_option
from .sensitivity import analyze_portfolio_sensitivity, format_report
from .signals import backtest, generate_signals


def _parse_portfolio(pairs: list[str]) -> dict[str, float]:
    portfolio: dict[str, float] = {}
    for pair in pairs:
        if ":" not in pair:
            raise SystemExit(
                f"Invalid holding '{pair}': expected TICKER:WEIGHT (e.g. AAPL:0.3)"
            )
        ticker, weight = pair.split(":", 1)
        portfolio[ticker.upper()] = float(weight)
    return portfolio


def cmd_sensitivity(args: argparse.Namespace) -> None:
    portfolio = _parse_portfolio(args.holdings)
    result = analyze_portfolio_sensitivity(
        portfolio, index_ticker=args.index, period=args.period
    )
    print(format_report(result))


def cmd_chart(args: argparse.Namespace) -> None:
    ohlc = single_ticker_ohlcv(args.ticker, period=args.period, interval=args.interval)
    fig = build_candlestick_chart(ohlc, args.ticker)
    out = args.out or f"{args.ticker}_chart.html"
    save_chart_html(fig, out)
    print(f"Chart saved to {out} — open it in a browser for the interactive view.")


def cmd_option(args: argparse.Namespace) -> None:
    inp = OptionInputs(
        spot=args.spot,
        strike=args.strike,
        time_to_expiry=args.days / 365.0,
        rate=args.rate,
        volatility=args.vol,
        dividend_yield=args.dividend_yield,
    )
    result = price_option(inp, option_type=args.type)
    print(f"{args.type.upper()} price: {result.price:.4f}")
    print(f"  Delta: {result.delta:.4f}")
    print(f"  Gamma: {result.gamma:.4f}")
    print(f"  Vega:  {result.vega:.4f}  (per 1% vol change: {result.vega/100:.4f})")
    print(f"  Theta: {result.theta:.4f}/yr  ({result.theta/365:.4f}/day)")
    print(f"  Rho:   {result.rho:.4f}")


def cmd_backtest(args: argparse.Namespace) -> None:
    ohlc = single_ticker_ohlcv(args.ticker, period=args.period, interval="1d")
    signaled = generate_signals(ohlc)
    result = backtest(signaled)
    print(f"Backtest: {args.ticker} ({args.period})")
    print(f"  Strategy total return:    {result.total_return:.2%}")
    print(f"  Buy & hold total return:  {result.buy_hold_return:.2%}")
    print(f"  Annualized return:        {result.annualized_return:.2%}")
    print(f"  Annualized volatility:    {result.annualized_volatility:.2%}")
    print(f"  Sharpe ratio:             {result.sharpe_ratio:.2f}")
    print(f"  Max drawdown:             {result.max_drawdown:.2%}")
    print(f"  Win rate (active days):   {result.win_rate:.2%}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trade_marketing_tool",
        description="Share & Index Analytics: sensitivity, charts, options, signals.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sens = sub.add_parser("sensitivity", help="Portfolio Beta/Alpha/Volatility vs an index")
    p_sens.add_argument("holdings", nargs="+", help="TICKER:WEIGHT pairs, e.g. AAPL:0.3 MSFT:0.25")
    p_sens.add_argument("--index", default="^GSPC", help="Benchmark index ticker (default ^GSPC)")
    p_sens.add_argument("--period", default="2y", help="History window (default 2y)")
    p_sens.set_defaults(func=cmd_sensitivity)

    p_chart = sub.add_parser("chart", help="Save a TradingView-style interactive chart to HTML")
    p_chart.add_argument("ticker")
    p_chart.add_argument("--period", default="1y")
    p_chart.add_argument("--interval", default="1d")
    p_chart.add_argument("--out", default=None, help="Output HTML path")
    p_chart.set_defaults(func=cmd_chart)

    p_opt = sub.add_parser("option", help="Black-Scholes option price + Greeks")
    p_opt.add_argument("--spot", type=float, required=True)
    p_opt.add_argument("--strike", type=float, required=True)
    p_opt.add_argument("--days", type=float, required=True, help="Days to expiry")
    p_opt.add_argument("--rate", type=float, required=True, help="Risk-free rate, e.g. 0.05")
    p_opt.add_argument("--vol", type=float, required=True, help="Annualized volatility, e.g. 0.25")
    p_opt.add_argument("--dividend-yield", type=float, default=0.0, dest="dividend_yield")
    p_opt.add_argument("--type", choices=["call", "put"], default="call")
    p_opt.set_defaults(func=cmd_option)

    p_bt = sub.add_parser("backtest", help="Backtest the built-in rule-based signal vs buy & hold")
    p_bt.add_argument("ticker")
    p_bt.add_argument("--period", default="2y")
    p_bt.set_defaults(func=cmd_backtest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except DataFetchError as exc:
        print(f"Data error: {exc}", file=sys.stderr)
        return 1
    except (ValueError, SystemExit) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
