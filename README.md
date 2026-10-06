# risk-dashboard

Portfolio risk report from a `ticker,shares` CSV: concentration, annualized volatility, beta, and 1-day Value-at-Risk computed from a year of daily returns.

## Run

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python3 risk_dashboard.py                    # reads positions.csv (sample portfolio included)
python3 risk_dashboard.py --file my.csv --benchmark QQQ --var-confidence 0.99
```

Edit `positions.csv` with your own holdings. Educational tool, not investment advice.
