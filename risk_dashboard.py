#!/usr/bin/env python3
"""
risk_dashboard.py

Portfolio risk read for a set of positions (ticker + share count), not just
prices: concentration, annualized volatility, market beta, and a simple
1-day Value-at-Risk figure — all computed from real trailing daily-return
history via yfinance, not just today's snapshot.

Concentration is a plain weight check (position value / total value).
Volatility, beta, and VaR come from a single aligned daily-return table
built across every position plus the benchmark (pandas' pct_change() on a
date-intersected price frame) — the portfolio return series is the
weighted sum of the individual return series at *today's* weights, so its
volatility already reflects real correlation between holdings, not a naive
average of the individual volatilities. Beta per ticker (and for the
portfolio as a whole) is the slope of a linregress() of ticker return vs.
benchmark return — same tool earnings_reaction.py already uses for its
surprise/reaction regressions. VaR is the parametric (normal-distribution)
1-day figure at the chosen confidence level; real returns have fatter
tails than normal, so treat it as a rough floor, not a hard limit.

Mixing an every-day-of-the-week asset (e.g. BTC-USD) into positions.csv
alongside stocks will shrink the aligned-date sample, since only dates
where every position AND the benchmark all traded survive the alignment.

USAGE
-----
    python3 risk_dashboard.py                         # reads positions.csv
    python3 risk_dashboard.py --file other.csv
    python3 risk_dashboard.py --benchmark QQQ
    python3 risk_dashboard.py --var-confidence 0.99
"""

import argparse
import csv
import sys

import pandas as pd
import yfinance as yf
from scipy.stats import linregress

DEFAULT_POSITIONS_FILE = "positions.csv"
DEFAULT_BENCHMARK = "SPY"
HISTORY_PERIOD = "1y"
TRADING_DAYS = 252
CONCENTRATION_WARN = 0.25  # flag any position over 25% of the portfolio
VAR_Z = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}  # one-tailed normal z-scores


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_positions(path: str) -> list[dict]:
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            positions = []
            for row in reader:
                ticker = (row.get("ticker") or "").strip().upper()
                shares_raw = (row.get("shares") or "").strip()
                if not ticker or not shares_raw:
                    continue
                positions.append({"ticker": ticker, "shares": float(shares_raw)})
    except FileNotFoundError:
        raise FileNotFoundError(f"Positions file '{path}' not found.")

    if not positions:
        raise ValueError(f"No positions found in '{path}' — expects a 'ticker,shares' header and at least one row.")
    return positions


def fetch_close_series(ticker: str, period: str = HISTORY_PERIOD):
    try:
        hist = yf.Ticker(ticker).history(period=period)
    except Exception:
        return None
    if hist is None or hist.empty:
        return None
    closes = hist["Close"]
    if closes.index.tz is not None:
        closes.index = closes.index.tz_localize(None)
    return closes


# --------------------------------------------------------------------------
# Risk computation
# --------------------------------------------------------------------------

def compute_risk(positions_file: str = DEFAULT_POSITIONS_FILE, benchmark: str = DEFAULT_BENCHMARK,
                  var_confidence: float = 0.95) -> dict:
    """Fetches + computes everything for the risk report and returns plain
    data — same decoupling as dcf.py/outcomes.py/valuation_model.py, so
    dashboard.py can reuse it without scraping printed text."""
    benchmark = benchmark.upper()
    positions = load_positions(positions_file)

    rows, close_series = [], {}
    for pos in positions:
        ticker = pos["ticker"]
        closes = fetch_close_series(ticker)
        if closes is None or closes.empty:
            rows.append({**pos, "price": None, "market_value": None, "weight": None,
                         "ann_volatility": None, "beta": None, "error": "no price data available"})
            continue
        price = float(closes.iloc[-1])
        rows.append({**pos, "price": price, "market_value": pos["shares"] * price,
                     "weight": None, "ann_volatility": None, "beta": None, "error": None})
        close_series[ticker] = closes

    usable = [r for r in rows if r["error"] is None]
    if not usable:
        return {"positions": rows, "error": "None of the tickers in this file returned price data."}

    total_value = sum(r["market_value"] for r in usable)
    for r in usable:
        r["weight"] = (r["market_value"] / total_value) if total_value > 0 else None

    concentrated = [{"ticker": r["ticker"], "weight": r["weight"]}
                     for r in usable if r["weight"] and r["weight"] > CONCENTRATION_WARN]

    bench_closes = fetch_close_series(benchmark)
    price_frame = pd.DataFrame(close_series)
    if bench_closes is not None:
        price_frame[benchmark] = bench_closes
    returns = price_frame.dropna().pct_change().dropna()

    for r in usable:
        t = r["ticker"]
        if t in returns.columns and len(returns[t]) > 1:
            r["ann_volatility"] = float(returns[t].std() * (TRADING_DAYS ** 0.5))
        if benchmark in returns.columns and t in returns.columns and t != benchmark and len(returns) > 1:
            r["beta"] = float(linregress(returns[benchmark], returns[t]).slope)

    portfolio_volatility = portfolio_beta = daily_var_pct = daily_var_dollar = None
    if total_value > 0:
        weighted_cols = [t for t in usable_tickers(usable) if t in returns.columns]
        if weighted_cols and len(returns) > 1:
            weights = {r["ticker"]: r["weight"] for r in usable}
            port_returns = sum(returns[t] * weights[t] for t in weighted_cols)
            daily_vol = float(port_returns.std())
            portfolio_volatility = daily_vol * (TRADING_DAYS ** 0.5)
            if benchmark in returns.columns:
                portfolio_beta = float(linregress(returns[benchmark], port_returns).slope)
            z = VAR_Z.get(var_confidence)
            if z is not None:
                daily_var_pct = z * daily_vol
                daily_var_dollar = daily_var_pct * total_value

    return {
        "positions": rows, "benchmark": benchmark, "total_value": total_value,
        "concentrated": concentrated, "concentration_threshold": CONCENTRATION_WARN,
        "portfolio_volatility": portfolio_volatility, "portfolio_beta": portfolio_beta,
        "var_confidence": var_confidence, "daily_var_pct": daily_var_pct,
        "daily_var_dollar": daily_var_dollar, "history_period": HISTORY_PERIOD, "error": None,
    }


def usable_tickers(usable_rows: list[dict]) -> list[str]:
    return [r["ticker"] for r in usable_rows]


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------

def money(x) -> str:
    if x is None:
        return "N/A"
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"


def fmt_pct_abs(x) -> str:
    return "N/A" if x is None else f"{x * 100:.1f}%"


def print_report(data: dict):
    if data.get("error"):
        print(data["error"])
        sys.exit(1)

    print("=" * 88)
    print("PORTFOLIO RISK DASHBOARD".center(88))
    print("=" * 88)
    print(f"Total value: {money(data['total_value'])}   "
          f"({data['history_period']} of daily returns vs {data['benchmark']})")

    print()
    print("POSITIONS")
    print("-" * 88)
    print(f"{'Ticker':<8}{'Shares':>10}{'Price':>12}{'Value':>14}{'Weight':>9}{'Ann. Vol':>11}{'Beta':>8}")
    for r in data["positions"]:
        if r["error"]:
            print(f"{r['ticker']:<8}{r['shares']:>10,.2f}{'N/A':>12}{'N/A':>14}"
                  f"{'—':>9}{'—':>11}{'—':>8}   ({r['error']})")
            continue
        flag = " ⚠" if r["weight"] and r["weight"] > data["concentration_threshold"] else ""
        beta_txt = f"{r['beta']:.2f}" if r["beta"] is not None else "N/A"
        print(f"{r['ticker']:<8}{r['shares']:>10,.2f}{money(r['price']):>12}{money(r['market_value']):>14}"
              f"{fmt_pct_abs(r['weight']):>9}{fmt_pct_abs(r['ann_volatility']):>11}{beta_txt:>8}{flag}")

    if data["concentrated"]:
        print()
        print(f"⚠ CONCENTRATION — {len(data['concentrated'])} position(s) over "
              f"{data['concentration_threshold'] * 100:.0f}% of the portfolio:")
        for c in data["concentrated"]:
            print(f"   {c['ticker']}: {fmt_pct_abs(c['weight'])}")

    print()
    print("PORTFOLIO-LEVEL RISK")
    print("-" * 88)
    beta_txt = f"{data['portfolio_beta']:.2f}" if data["portfolio_beta"] is not None else "N/A"
    conf_pct = int(data["var_confidence"] * 100)
    print(f"  Annualized volatility:  {fmt_pct_abs(data['portfolio_volatility'])}")
    print(f"  Beta vs {data['benchmark']:<8}      {beta_txt}")
    print(f"  1-day VaR ({conf_pct}%):       {fmt_pct_abs(data['daily_var_pct'])}  ({money(data['daily_var_dollar'])})")

    print()
    print("=" * 88)
    print("Volatility/beta/VaR come from realized daily returns over the trailing period,")
    print("aligned across every position + the benchmark. VaR assumes a normal return")
    print("distribution — a simplification, since real returns have fatter tails. Not")
    print("investment advice.")
    print("=" * 88)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Portfolio risk dashboard: concentration, volatility, beta, and VaR from a positions CSV.")
    parser.add_argument("--file", default=DEFAULT_POSITIONS_FILE,
                         help=f"Path to positions CSV (ticker,shares columns). Default: {DEFAULT_POSITIONS_FILE}")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK,
                         help=f"Benchmark ticker for beta. Default: {DEFAULT_BENCHMARK}")
    parser.add_argument("--var-confidence", type=float, default=0.95, choices=sorted(VAR_Z),
                         help="Confidence level for the 1-day VaR figure. Default: 0.95")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        data = compute_risk(args.file, args.benchmark, args.var_confidence)
    except (FileNotFoundError, ValueError) as e:
        print(str(e))
        sys.exit(1)
    print_report(data)


if __name__ == "__main__":
    main()
