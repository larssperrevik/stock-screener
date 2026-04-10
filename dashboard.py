"""Interactive dashboard for the stock screener and backtester.

Generates an HTML dashboard with:
- Backtest equity curve vs benchmark
- Rolling returns comparison
- Drawdown chart
- Current holdings table
- Latest screen results with returns since screening
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

from data.simfin_loader import (
    load_prices, get_all_fundamentals_at_date, get_tradeable_tickers_at_date,
    load_companies, load_industries,
)
from data.live_prices import get_live_prices
from screener.criteria import ScreenCriteria, apply_screen
from backtest.engine import BacktestEngine
from metrics.performance import (
    cagr, sharpe_ratio, sortino_ratio, max_drawdown, ulcer_index, full_report
)


def _cumulative_returns_json(returns, label):
    cum = (1 + returns).cumprod()
    dates = [d.strftime("%Y-%m-%d") for d in cum.index]
    values = [round(float(v), 4) for v in cum.values]
    return {"label": label, "dates": dates, "values": values}


def _drawdown_json(returns, label):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    dates = [d.strftime("%Y-%m-%d") for d in dd.index]
    values = [round(float(v), 4) for v in dd.values]
    return {"label": label, "dates": dates, "values": values}


def _rolling_returns_json(returns, window, label):
    roll = (1 + returns).rolling(window).apply(lambda x: x.prod() ** (252/window) - 1, raw=True)
    roll = roll.dropna()
    dates = [d.strftime("%Y-%m-%d") for d in roll.index]
    values = [round(float(v), 4) for v in roll.values]
    return {"label": label, "dates": dates, "values": values}


def latest_screen_with_returns(criteria=None, screen_date="2024-07-01"):
    """Run screen at a date and compute returns to today using live FMP prices."""
    if criteria is None:
        criteria = ScreenCriteria()

    fundamentals = get_all_fundamentals_at_date(screen_date)
    tradeable = get_tradeable_tickers_at_date(screen_date)
    fundamentals = fundamentals[fundamentals["Ticker"].isin(tradeable)]
    screened = apply_screen(fundamentals, criteria)

    if screened.empty:
        return pd.DataFrame()

    top_tickers = screened.head(30)["Ticker"].tolist()

    # Get SimFin prices for the screen date baseline
    all_prices = load_prices()
    price_matrix = all_prices.pivot_table(
        index="Date", columns="Ticker", values="Adj. Close", aggfunc="first"
    )

    # Fetch live prices from FMP (screen date to today)
    print(f"  Fetching live prices for {len(top_tickers)} tickers...")
    live = get_live_prices(top_tickers, start=screen_date)

    companies = load_companies()
    industries = load_industries()

    results = []
    for _, row in screened.head(30).iterrows():
        ticker = row["Ticker"]

        # Get baseline price at screen date from SimFin
        if ticker in price_matrix.columns:
            simfin_prices = price_matrix[ticker].dropna()
            baseline_prices = simfin_prices[simfin_prices.index >= screen_date]
            if len(baseline_prices) > 0:
                baseline_price = baseline_prices.iloc[0]
            else:
                continue
        else:
            continue

        # Get latest price: prefer live FMP data, fall back to SimFin
        if not live.empty and ticker in live.columns:
            live_col = live[ticker].dropna()
            if len(live_col) > 0:
                last_price = live_col.iloc[-1]
                last_date = live_col.index[-1].strftime("%Y-%m-%d")
            else:
                last_price = baseline_prices.iloc[-1]
                last_date = baseline_prices.index[-1].strftime("%Y-%m-%d")
        else:
            last_price = baseline_prices.iloc[-1] if len(baseline_prices) > 0 else baseline_price
            last_date = baseline_prices.index[-1].strftime("%Y-%m-%d") if len(baseline_prices) > 0 else screen_date

        ret_since = (last_price / baseline_price) - 1

        co = companies[companies["Ticker"] == ticker]
        co_name = co.iloc[0]["Company Name"] if not co.empty else ""
        ind_id = co.iloc[0].get("IndustryId") if not co.empty else None
        sector = ""
        if pd.notna(ind_id):
            ind = industries[industries["IndustryId"] == ind_id]
            if not ind.empty:
                sector = ind.iloc[0].get("Sector", "")

        results.append({
            "ticker": ticker,
            "name": co_name,
            "sector": sector,
            "roe": row.get("Return on Equity"),
            "roic": row.get("Return On Invested Capital"),
            "piotroski": row.get("Piotroski F-Score"),
            "gross_margin": row.get("Gross Profit Margin"),
            "op_margin": row.get("Operating Margin"),
            "debt_ratio": row.get("Debt Ratio"),
            "composite_rank": row.get("composite_rank"),
            "return_since_screen": ret_since,
            "last_price": last_price,
            "price_date": last_date,
        })

    return pd.DataFrame(results)


def generate_dashboard(
    criteria=None,
    start="2005-01-01",
    end="2024-09-30",
    freq="quarterly",
    top_n=20,
    output="dashboard.html",
):
    if criteria is None:
        criteria = ScreenCriteria()

    engine = BacktestEngine(
        criteria=criteria, rebalance_freq=freq, top_n=top_n,
        start_date=start, end_date=end,
    )
    result = engine.run()

    port_ret = result["portfolio_returns"]
    bench_ret = result["benchmark_returns"]
    report = result["report"]
    holdings = result["holdings_history"]

    last_rebal = holdings[-1]["date"].strftime("%Y-%m-%d")
    print(f"\nRunning latest screen at {last_rebal} with forward returns...")
    latest = latest_screen_with_returns(criteria, screen_date=last_rebal)

    cum_port = _cumulative_returns_json(port_ret, "Portfolio")
    cum_bench = _cumulative_returns_json(bench_ret, "SPY") if bench_ret is not None else None
    dd_port = _drawdown_json(port_ret, "Portfolio")
    dd_bench = _drawdown_json(bench_ret, "SPY") if bench_ret is not None else None
    roll_port = _rolling_returns_json(port_ret, 252, "Portfolio (1Y)")
    roll_bench = _rolling_returns_json(bench_ret, 252, "SPY (1Y)") if bench_ret is not None else None

    holdings_data = []
    for h in holdings:
        holdings_data.append({
            "date": h["date"].strftime("%Y-%m-%d"),
            "tickers": h["tickers"],
        })

    latest_data = latest.to_dict("records") if not latest.empty else []

    chart_data = {
        "cumulative": [cum_port] + ([cum_bench] if cum_bench else []),
        "drawdown": [dd_port] + ([dd_bench] if dd_bench else []),
        "rolling": [roll_port] + ([roll_bench] if roll_bench else []),
        "report": {k: round(v, 4) if isinstance(v, float) else v for k, v in report.items()},
        "holdings": holdings_data,
        "latest_screen": latest_data,
        "last_rebalance": last_rebal,
        "config": {
            "start": start, "end": end, "freq": freq, "top_n": top_n,
            "min_roe": criteria.min_roe, "min_roic": criteria.min_roic,
            "min_piotroski": criteria.min_piotroski,
        },
    }

    html = _build_html(chart_data)
    with open(output, "w") as f:
        f.write(html)
    print(f"\nDashboard saved to {output}")


def _pct(val, digits=1):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{digits}%}"


def _build_html(data):
    report = data["report"]
    config = data["config"]

    screen_rows = ""
    for s in data["latest_screen"]:
        ret = s.get("return_since_screen", 0) or 0
        ret_class = "positive" if ret > 0 else "negative"
        screen_rows += (
            "<tr>"
            f"<td><strong>{s['ticker']}</strong></td>"
            f"<td>{(s.get('name') or '')[:30]}</td>"
            f"<td>{s.get('sector','')}</td>"
            f"<td>{_pct(s.get('roe'))}</td>"
            f"<td>{_pct(s.get('roic'))}</td>"
            f"<td>{int(s.get('piotroski') or 0)}</td>"
            f"<td>{_pct(s.get('gross_margin'))}</td>"
            f"<td>{_pct(s.get('op_margin'))}</td>"
            f"<td>{_pct(s.get('debt_ratio'))}</td>"
            f'<td class="{ret_class}">{_pct(ret)}</td>'
            f"<td>{s.get('price_date','')}</td>"
            "</tr>\n"
        )

    holdings_rows = ""
    for h in reversed(data["holdings"][-12:]):
        tickers_str = ", ".join(h["tickers"][:10])
        more = f" +{len(h['tickers'])-10}" if len(h["tickers"]) > 10 else ""
        holdings_rows += f"<tr><td>{h['date']}</td><td>{tickers_str}{more}</td></tr>\n"

    excess = report.get("excess_return", 0)
    excess_class = "highlight" if excess > 0 else "danger"

    cum_json = json.dumps(data["cumulative"])
    dd_json = json.dumps(data["drawdown"])
    roll_json = json.dumps(data["rolling"])

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Value Screener Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f1117; color: #e0e0e0; padding: 20px; }}
  h1 {{ color: #58a6ff; margin-bottom: 5px; }}
  h2 {{ color: #8b949e; margin: 20px 0 10px; font-size: 1.1em; text-transform: uppercase; letter-spacing: 1px; }}
  .subtitle {{ color: #8b949e; margin-bottom: 20px; font-size: 0.9em; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }}
  .metric {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; text-align: center; }}
  .metric .value {{ font-size: 1.8em; font-weight: bold; color: #58a6ff; }}
  .metric .label {{ font-size: 0.8em; color: #8b949e; margin-top: 5px; }}
  .metric.highlight .value {{ color: #3fb950; }}
  .metric.warn .value {{ color: #d29922; }}
  .metric.danger .value {{ color: #f85149; }}
  .chart-container {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                      padding: 15px; margin: 15px 0; }}
  canvas {{ max-height: 350px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
  th {{ background: #21262d; color: #8b949e; padding: 8px 10px; text-align: left;
       border-bottom: 2px solid #30363d; position: sticky; top: 0; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #21262d; }}
  tr:hover {{ background: #1c2128; }}
  .positive {{ color: #3fb950; font-weight: bold; }}
  .negative {{ color: #f85149; font-weight: bold; }}
  .table-container {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                      padding: 15px; margin: 15px 0; max-height: 500px; overflow-y: auto; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
  @media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<h1>Value Investing Screener</h1>
<p class="subtitle">Buffett/Munger-style quality screen &middot; {config['start']} to {config['end']} &middot;
   {config['freq']} rebalance &middot; Top {config['top_n']} stocks &middot;
   Min ROE {_pct(config['min_roe'], 0)} &middot; Min ROIC {_pct(config['min_roic'], 0)} &middot; Min Piotroski {config['min_piotroski']}</p>

<h2>Performance Summary</h2>
<div class="grid">
  <div class="metric highlight"><div class="value">{_pct(report['cagr'])}</div><div class="label">CAGR</div></div>
  <div class="metric"><div class="value">{_pct(report.get('benchmark_cagr', 0))}</div><div class="label">SPY CAGR</div></div>
  <div class="metric {excess_class}">
    <div class="value">{_pct(excess)}</div><div class="label">Excess Return</div></div>
  <div class="metric"><div class="value">{_pct(report['total_return'], 0)}</div><div class="label">Total Return</div></div>
  <div class="metric"><div class="value">{report['sharpe']:.2f}</div><div class="label">Sharpe</div></div>
  <div class="metric"><div class="value">{report['sortino']:.2f}</div><div class="label">Sortino</div></div>
  <div class="metric danger"><div class="value">{_pct(report['max_drawdown'])}</div><div class="label">Max Drawdown</div></div>
  <div class="metric"><div class="value">{_pct(report['volatility'])}</div><div class="label">Volatility</div></div>
</div>

<div class="two-col">
  <div class="chart-container"><canvas id="cumChart"></canvas></div>
  <div class="chart-container"><canvas id="ddChart"></canvas></div>
</div>
<div class="chart-container"><canvas id="rollChart"></canvas></div>

<h2>Latest Screen: {data['last_rebalance']} (with returns since)</h2>
<div class="table-container">
<table>
<thead><tr>
  <th>Ticker</th><th>Name</th><th>Sector</th><th>ROE</th><th>ROIC</th>
  <th>F-Score</th><th>Gross Mgn</th><th>Op Mgn</th><th>Debt Ratio</th>
  <th>Return Since</th><th>Price Date</th>
</tr></thead>
<tbody>{screen_rows}</tbody>
</table>
</div>

<h2>Recent Holdings</h2>
<div class="table-container">
<table>
<thead><tr><th>Rebalance Date</th><th>Holdings</th></tr></thead>
<tbody>{holdings_rows}</tbody>
</table>
</div>

<script>
const chartColors = {{
  portfolio: 'rgb(88, 166, 255)',
  benchmark: 'rgb(139, 148, 158)',
  portfolioFill: 'rgba(88, 166, 255, 0.1)',
  benchmarkFill: 'rgba(139, 148, 158, 0.05)',
  red: 'rgba(248, 81, 73, 0.5)',
  redLine: 'rgb(248, 81, 73)',
}};

const defaultOpts = {{
  responsive: true,
  interaction: {{ mode: 'index', intersect: false }},
  plugins: {{ legend: {{ labels: {{ color: '#8b949e' }} }} }},
  scales: {{
    x: {{ type: 'time', time: {{ unit: 'year' }}, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
    y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }}
  }}
}};

const cumData = {cum_json};
const ddData = {dd_json};
const rollData = {roll_json};

new Chart(document.getElementById('cumChart'), {{
  type: 'line',
  data: {{
    labels: cumData[0].dates,
    datasets: cumData.map((d, i) => ({{
      label: d.label,
      data: d.values,
      borderColor: i === 0 ? chartColors.portfolio : chartColors.benchmark,
      backgroundColor: i === 0 ? chartColors.portfolioFill : chartColors.benchmarkFill,
      fill: true, borderWidth: 1.5, pointRadius: 0,
    }}))
  }},
  options: {{ ...defaultOpts, plugins: {{ ...defaultOpts.plugins,
    title: {{ display: true, text: 'Cumulative Returns (Growth of $1)', color: '#e0e0e0' }} }} }}
}});

new Chart(document.getElementById('ddChart'), {{
  type: 'line',
  data: {{
    labels: ddData[0].dates,
    datasets: ddData.map((d, i) => ({{
      label: d.label,
      data: d.values,
      borderColor: i === 0 ? chartColors.redLine : chartColors.benchmark,
      backgroundColor: i === 0 ? chartColors.red : 'transparent',
      fill: true, borderWidth: 1.5, pointRadius: 0,
    }}))
  }},
  options: {{ ...defaultOpts, plugins: {{ ...defaultOpts.plugins,
    title: {{ display: true, text: 'Drawdown', color: '#e0e0e0' }} }},
    scales: {{ ...defaultOpts.scales, y: {{ ...defaultOpts.scales.y,
      ticks: {{ ...defaultOpts.scales.y.ticks, callback: v => (v*100).toFixed(0)+'%' }} }} }}
  }}
}});

new Chart(document.getElementById('rollChart'), {{
  type: 'line',
  data: {{
    labels: rollData[0].dates,
    datasets: rollData.map((d, i) => ({{
      label: d.label,
      data: d.values,
      borderColor: i === 0 ? chartColors.portfolio : chartColors.benchmark,
      borderWidth: 1.5, pointRadius: 0, fill: false,
    }}))
  }},
  options: {{ ...defaultOpts, plugins: {{ ...defaultOpts.plugins,
    title: {{ display: true, text: 'Rolling 1-Year Annualized Returns', color: '#e0e0e0' }} }},
    scales: {{ ...defaultOpts.scales, y: {{ ...defaultOpts.scales.y,
      ticks: {{ ...defaultOpts.scales.y.ticks, callback: v => (v*100).toFixed(0)+'%' }} }} }}
  }}
}});
</script>
</body>
</html>"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate dashboard")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="today")
    parser.add_argument("--freq", default="quarterly")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-roe", type=float, default=0.15)
    parser.add_argument("--min-roic", type=float, default=0.10)
    parser.add_argument("--min-piotroski", type=int, default=5)
    parser.add_argument("-o", "--output", default="dashboard.html")
    args = parser.parse_args()

    criteria = ScreenCriteria(
        min_roe=args.min_roe, min_roic=args.min_roic, min_piotroski=args.min_piotroski,
    )
    end = datetime.now().strftime("%Y-%m-%d") if args.end == "today" else args.end
    generate_dashboard(
        criteria=criteria, start=args.start, end=end,
        freq=args.freq, top_n=args.top_n, output=args.output,
    )
