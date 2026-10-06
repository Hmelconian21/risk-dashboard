"""
app.py

Streamlit front end for risk_dashboard.py: edit a portfolio's tickers and
weights in the sidebar and see its risk metrics, cumulative return vs. a
benchmark, and the correlation between holdings.

Reuses risk_dashboard.py's data loading and constants rather than
duplicating them. The math is the same as compute_risk() -- an aligned
daily-return table and a portfolio return series that is the weighted sum
of the holdings' returns -- but driven by weights instead of share counts.

Run: streamlit run app.py
"""

import logging

import altair as alt
import pandas as pd
import streamlit as st
from scipy.stats import linregress

import risk_dashboard as rd

logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # bad tickers get a friendly message instead

RISK_FREE_RATE = 0.045      # annual, for the Sharpe ratio (~ 10yr Treasury)
VAR_CONFIDENCE = 0.95

# Chart colors (reference data-viz palette). Light and dark are separate
# steps of the same hues, picked to suit each background.
COLORS = {
    "light": {"portfolio": "#2a78d6", "benchmark": "#eb6834", "muted": "#898781", "ink": "#0b0b0b",
              "neg": "#e34948", "mid": "#f0efec", "pos": "#2a78d6"},
    "dark":  {"portfolio": "#3987e5", "benchmark": "#d95926", "muted": "#898781", "ink": "#ffffff",
              "neg": "#e66767", "mid": "#383835", "pos": "#3987e5"},
}


# --------------------------------------------------------------------------
# Data (cached)
# --------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_closes(ticker: str) -> pd.Series:
    """A year of daily closes. Raises rather than returning None so a bad
    ticker or a network blip isn't cached for an hour."""
    closes = rd.fetch_close_series(ticker)
    if closes is None or closes.empty:
        raise ValueError(f"no price data for {ticker}")
    return closes


@st.cache_data(ttl=3600, show_spinner=False)
def sample_portfolio() -> pd.DataFrame:
    """positions.csv as market-value weights at the latest close, falling
    back to equal weights for any ticker whose price can't be fetched."""
    positions = rd.load_positions(rd.DEFAULT_POSITIONS_FILE)
    values = {}
    for p in positions:
        try:
            values[p["ticker"]] = p["shares"] * float(get_closes(p["ticker"]).iloc[-1])
        except ValueError:
            values[p["ticker"]] = None
    if any(v is None for v in values.values()):
        weights = {t: 100 / len(values) for t in values}
    else:
        total = sum(values.values())
        weights = {t: v / total * 100 for t, v in values.items()}
    return pd.DataFrame({"Ticker": list(weights), "Weight (%)": [round(w, 1) for w in weights.values()]})


# --------------------------------------------------------------------------
# Risk math
# --------------------------------------------------------------------------

def compute_metrics(returns: pd.DataFrame, weights: dict, benchmark: str) -> dict:
    holdings = list(weights)
    port = sum(returns[t] * weights[t] for t in holdings)
    bench = returns[benchmark]

    daily_vol = port.std()
    ann_vol = daily_vol * rd.TRADING_DAYS ** 0.5
    ann_return = (1 + port).prod() ** (rd.TRADING_DAYS / len(port)) - 1
    wealth = (1 + port).cumprod()
    drawdown = wealth / wealth.cummax() - 1

    per_ticker = pd.DataFrame({
        "Weight": pd.Series(weights),
        "Ann. volatility": returns[holdings].std() * rd.TRADING_DAYS ** 0.5,
        "Beta": pd.Series({t: linregress(bench, returns[t]).slope for t in holdings}),
    })

    return {
        "ann_vol": ann_vol,
        "ann_return": ann_return,
        "max_drawdown": drawdown.min(),
        "beta": linregress(bench, port).slope,
        "sharpe": (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol else None,
        "var_pct": rd.VAR_Z[VAR_CONFIDENCE] * daily_vol,
        "cumulative": pd.DataFrame({"Portfolio": (1 + port).cumprod() - 1,
                                    benchmark: (1 + bench).cumprod() - 1}),
        "corr": returns[holdings].corr(),
        "per_ticker": per_ticker,
        "days": len(port),
    }


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------

def cumulative_chart(cum: pd.DataFrame, c: dict) -> alt.LayerChart:
    series = list(cum.columns)
    long = cum.reset_index(names="Date").melt("Date", var_name="Series", value_name="Return")
    wide = cum.reset_index(names="Date")
    color = alt.Color("Series:N", scale=alt.Scale(domain=series, range=[c["portfolio"], c["benchmark"]]),
                      legend=alt.Legend(orient="top", title=None))
    x = alt.X("Date:T", title=None)
    y = alt.Y("Return:Q", title="Cumulative return", axis=alt.Axis(format="%", grid=True))

    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(x=x, y=y, color=color)

    hover = alt.selection_point(fields=["Date"], nearest=True, on="pointerover", empty=False, clear="pointerout")
    rule = alt.Chart(wide).mark_rule(color=c["muted"], strokeWidth=1).encode(
        x="Date:T",
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("Date:T", format="%b %d, %Y")]
                + [alt.Tooltip(f"{s}:Q", format="+.1%") for s in series],
    ).add_params(hover)
    points = alt.Chart(long).mark_point(filled=True, size=70, strokeWidth=2).encode(
        x=x, y=y, color=color, opacity=alt.condition(hover, alt.value(1), alt.value(0)),
    ).transform_filter(hover)

    last = long[long["Date"] == long["Date"].max()]
    end_labels = alt.Chart(last).mark_text(align="left", dx=6, fontWeight="bold", color=c["ink"]).encode(
        x=x, y=y, text=alt.Text("Return:Q", format="+.1%"),
    )
    return (lines + rule + points + end_labels).properties(height=340)


def correlation_chart(corr: pd.DataFrame, c: dict) -> alt.LayerChart:
    """Lower triangle only -- the upper half repeats it and the diagonal is always 1."""
    tickers = list(corr.columns)
    cells = [{"Row": a, "Col": b, "Correlation": corr.loc[a, b]}
             for i, a in enumerate(tickers) for b in tickers[:i]]
    df = pd.DataFrame(cells)
    band = alt.Scale(paddingInner=0.04)  # small surface-colored gap between cells
    x = alt.X("Col:N", sort=tickers[:-1], title=None, scale=band, axis=alt.Axis(labelAngle=0, orient="top"))
    y = alt.Y("Row:N", sort=tickers[1:], title=None, scale=band)
    base = alt.Chart(df).encode(x=x, y=y)

    rect = base.mark_rect(stroke=None, cornerRadius=4).encode(
        color=alt.Color("Correlation:Q",
                        scale=alt.Scale(domain=[-1, 0, 1], range=[c["neg"], c["mid"], c["pos"]], interpolate="lab"),
                        legend=alt.Legend(title="Correlation", format=".1f", orient="bottom",
                                         direction="horizontal", gradientLength=240)),
        tooltip=[alt.Tooltip("Row:N", title="Ticker"), alt.Tooltip("Col:N", title="vs"),
                 alt.Tooltip("Correlation:Q", format=".2f")],
    )
    # Dark ink only works on the pale cells near 0 in light mode; everywhere else the cells are dark.
    light_cell_ink = "#0b0b0b" if c["ink"] == "#0b0b0b" else "#ffffff"
    text = base.mark_text(fontSize=13).encode(
        text=alt.Text("Correlation:Q", format=".2f"),
        color=alt.condition("abs(datum.Correlation) > 0.55", alt.value("#ffffff"), alt.value(light_cell_ink)),
    )
    return (rect + text).properties(width=alt.Step(64), height=alt.Step(64))


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

def read_portfolio(edited: pd.DataFrame) -> tuple[dict, list[str]]:
    """Clean the sidebar table into {ticker: weight fraction}. Returns the
    weights plus any problems worth telling the user about."""
    problems, weights = [], {}
    for _, row in edited.iterrows():
        ticker = str(row.get("Ticker") or "").strip().upper()
        weight = row.get("Weight (%)")
        if not ticker:
            continue
        if pd.isna(weight) or weight <= 0:
            problems.append(f"{ticker} has no positive weight, so it's left out.")
            continue
        weights[ticker] = weights.get(ticker, 0) + float(weight)
    return weights, problems


def main() -> None:
    st.set_page_config(page_title="Portfolio Risk", layout="wide")
    theme = "dark" if getattr(st.context.theme, "type", "light") == "dark" else "light"
    c = COLORS[theme]

    st.sidebar.header("Portfolio")
    benchmark = st.sidebar.text_input("Benchmark", rd.DEFAULT_BENCHMARK).strip().upper() or rd.DEFAULT_BENCHMARK
    with st.spinner("Loading sample portfolio..."):
        default = sample_portfolio()
    edited = st.sidebar.data_editor(
        default, num_rows="dynamic", hide_index=True, width="stretch",
        column_config={
            "Ticker": st.column_config.TextColumn(required=True),
            "Weight (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5, format="%.1f%%"),
        },
    )
    st.sidebar.caption("Weights are rescaled to add up to 100%. The sample portfolio is "
                       "positions.csv valued at the latest close.")

    st.title("Portfolio risk")
    raw_weights, problems = read_portfolio(edited)
    for p in problems:
        st.warning(p)
    if not raw_weights:
        st.info("Add at least one ticker with a positive weight in the sidebar.")
        st.stop()

    closes, bad = {}, []
    with st.spinner("Fetching price history..."):
        for t in list(raw_weights) + [benchmark]:
            try:
                closes[t] = get_closes(t)
            except ValueError:
                bad.append(t)

    if benchmark in bad:
        st.error(f"Couldn't find price data for the benchmark **{benchmark}**. Check the symbol in the sidebar.")
        st.stop()
    if bad:
        st.error(f"Couldn't find price data for **{', '.join(bad)}**, so "
                 f"{'it is' if len(bad) == 1 else 'they are'} left out. Check the "
                 f"symbol{'s' if len(bad) > 1 else ''} (e.g. BRK-B, not BRK.B).")
    raw_weights = {t: w for t, w in raw_weights.items() if t not in bad}
    if not raw_weights:
        st.stop()

    total = sum(raw_weights.values())
    weights = {t: w / total for t, w in raw_weights.items()}
    if abs(total - 100) > 0.05:
        st.info(f"Weights add up to {total:.1f}%, so they've been rescaled to 100%.")

    prices = pd.DataFrame({t: closes[t] for t in list(weights) + [benchmark] if t in closes})
    returns = prices.dropna().pct_change().dropna()
    if len(returns) < 20:
        st.error("Not enough overlapping trading days across these tickers to measure risk.")
        st.stop()
    m = compute_metrics(returns, weights, benchmark)

    cols = st.columns(5)
    cols[0].metric("Annualized volatility", f"{m['ann_vol']:.1%}",
                   help="Standard deviation of daily portfolio returns, scaled to a year.")
    cols[1].metric("Max drawdown", f"{m['max_drawdown']:.1%}",
                   help="Largest peak-to-trough fall in portfolio value over the period.")
    cols[2].metric(f"Beta vs {benchmark}", f"{m['beta']:.2f}",
                   help=f"How much the portfolio tends to move for a 1% move in {benchmark}.")
    cols[3].metric("Sharpe ratio", f"{m['sharpe']:.2f}" if m["sharpe"] is not None else "N/A",
                   help=f"Annualized return minus a {RISK_FREE_RATE:.1%} risk-free rate, divided by volatility.")
    cols[4].metric(f"1-day VaR ({VAR_CONFIDENCE:.0%})", f"{m['var_pct']:.1%}",
                   help="Parametric (normal) estimate: on 95% of days the loss should be smaller "
                        "than this. Real returns have fatter tails, so treat it as a floor.")
    st.caption(f"Based on {m['days']} overlapping trading days over the last "
               f"{rd.HISTORY_PERIOD}. Annualized return: {m['ann_return']:+.1%}. Not investment advice.")

    concentrated = [t for t, w in weights.items() if w > rd.CONCENTRATION_WARN]
    if concentrated:
        st.warning(f"⚠ Over {rd.CONCENTRATION_WARN:.0%} of the portfolio: " +
                   ", ".join(f"{t} ({weights[t]:.1%})" for t in concentrated))

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader(f"Cumulative return vs {benchmark}")
        st.altair_chart(cumulative_chart(m["cumulative"], c), width="stretch")
        with st.expander("Show data"):
            st.dataframe(m["cumulative"].style.format("{:+.2%}"), width="stretch")
    with right:
        st.subheader("Correlation between holdings")
        if len(weights) < 2:
            st.info("Add a second ticker to see correlations.")
        else:
            st.altair_chart(correlation_chart(m["corr"], c), width="content")
            with st.expander("Show data"):
                st.dataframe(m["corr"].style.format("{:.2f}"), width="stretch")

    st.subheader("Holdings")
    st.dataframe(
        m["per_ticker"].style.format({"Weight": "{:.1%}", "Ann. volatility": "{:.1%}", "Beta": "{:.2f}"}),
        width="stretch",
    )


if __name__ == "__main__":
    main()
