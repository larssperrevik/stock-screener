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
from backtest.event_engine import EventDrivenEngine
from sector_analysis import run_with_holdings_tracking, build_sector_report
from metrics.performance import (
    cagr, sharpe_ratio, sortino_ratio, max_drawdown, ulcer_index, full_report
)


def _to_xy_json(index, values):
    """Convert to [{x: date, y: value}, ...] for Chart.js time axis."""
    return [{"x": d.strftime("%Y-%m-%d"), "y": round(float(v), 4)}
            for d, v in zip(index, values)]


def _cumulative_returns_json(returns, label):
    cum = (1 + returns).cumprod()
    return {"label": label, "data": _to_xy_json(cum.index, cum.values)}


def _drawdown_json(returns, label):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return {"label": label, "data": _to_xy_json(dd.index, dd.values)}


def _rolling_returns_json(returns, window, label):
    roll = (1 + returns).rolling(window).apply(lambda x: x.prod() ** (252/window) - 1, raw=True)
    roll = roll.dropna()
    return {"label": label, "data": _to_xy_json(roll.index, roll.values)}


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

    # === Run simple screener backtest ===
    engine = BacktestEngine(
        criteria=criteria, rebalance_freq=freq, top_n=top_n,
        start_date=start, end_date=end,
    )
    result = engine.run()

    port_ret = result["portfolio_returns"]
    bench_ret = result["benchmark_returns"]
    report = result["report"]
    holdings = result["holdings_history"]

    # === Run event-driven strategy ===
    combined_start = "2011-01-01"
    print("\n\nRunning event-driven strategy...")
    try:
        event_engine = EventDrivenEngine(
            criteria=criteria, max_positions=15, max_hold_days=540,
            min_hold_days=60, buy_threshold=40, sell_threshold=20,
            start_date=combined_start, end_date=end,
        )
        event_result = event_engine.run()
        comb_ret = event_result["portfolio_returns"]
        comb_report = event_result["report"]
        event_trades = event_result.get("trades", [])
        has_combined = True

        # Build holdings list from trades for display
        comb_holdings = []
    except Exception as e:
        print(f"Event-driven strategy failed: {e}")
        has_combined = False
        comb_ret = None
        comb_report = {}
        event_trades = []
        comb_holdings = []

    # === Sector analysis (reuses event-driven simulation) ===
    sector_data = {}
    if has_combined:
        print("\nRunning sector analysis...")
        try:
            daily_holdings, _ = run_with_holdings_tracking(combined_start, end)
            sector_data = build_sector_report(daily_holdings)
        except Exception as e:
            print(f"Sector analysis failed: {e}")

    # === Latest screen with live returns ===
    last_rebal = holdings[-1]["date"].strftime("%Y-%m-%d")
    print(f"\nRunning latest screen at {last_rebal} with forward returns...")
    latest = latest_screen_with_returns(criteria, screen_date=last_rebal)

    # === Build chart data ===
    # Trim all series to start from combined_start so they're comparable
    chart_start = pd.Timestamp(combined_start) if has_combined else port_ret.index[0]
    port_ret_trimmed = port_ret[port_ret.index >= chart_start]
    bench_ret_trimmed = bench_ret[bench_ret.index >= chart_start] if bench_ret is not None else None

    # Recompute metrics from the same period for fair comparison
    report = full_report(port_ret_trimmed, bench_ret_trimmed)
    if bench_ret_trimmed is not None:
        spy_report = full_report(bench_ret_trimmed)
    else:
        spy_report = {}

    cum_port = _cumulative_returns_json(port_ret_trimmed, "Screener")
    cum_bench = _cumulative_returns_json(bench_ret_trimmed, "SPY") if bench_ret_trimmed is not None else None
    dd_port = _drawdown_json(port_ret_trimmed, "Screener")
    dd_bench = _drawdown_json(bench_ret_trimmed, "SPY") if bench_ret_trimmed is not None else None
    roll_port = _rolling_returns_json(port_ret_trimmed, 252, "Screener (1Y)")
    roll_bench = _rolling_returns_json(bench_ret_trimmed, 252, "SPY (1Y)") if bench_ret_trimmed is not None else None

    cum_combined = _cumulative_returns_json(comb_ret, "Event-Driven") if has_combined else None
    dd_combined = _drawdown_json(comb_ret, "Event-Driven") if has_combined else None
    roll_combined = _rolling_returns_json(comb_ret, 252, "Event-Driven (1Y)") if has_combined else None

    holdings_data = []
    for h in holdings:
        holdings_data.append({
            "date": h["date"].strftime("%Y-%m-%d"),
            "tickers": h["tickers"],
        })

    # Event-driven: show recent trades
    event_trades_data = []
    if has_combined and event_trades:
        for t in sorted(event_trades, key=lambda x: x.exit_date, reverse=True)[:20]:
            event_trades_data.append({
                "ticker": t.ticker,
                "entry": t.entry_date.strftime("%Y-%m-%d"),
                "exit": t.exit_date.strftime("%Y-%m-%d"),
                "return": t.return_pct,
                "hold_days": t.hold_days,
                "reason": t.reason,
            })

    latest_data = latest.to_dict("records") if not latest.empty else []

    chart_data = {
        "cumulative": [cum_port] + ([cum_combined] if cum_combined else []) + ([cum_bench] if cum_bench else []),
        "drawdown": [dd_port] + ([dd_combined] if dd_combined else []) + ([dd_bench] if dd_bench else []),
        "rolling": [roll_port] + ([roll_combined] if roll_combined else []) + ([roll_bench] if roll_bench else []),
        "report": {k: round(v, 4) if isinstance(v, float) else v for k, v in report.items()},
        "combined_report": {k: round(v, 4) if isinstance(v, float) else v for k, v in comb_report.items()} if has_combined else {},
        "spy_report": {k: round(v, 4) if isinstance(v, float) else v for k, v in spy_report.items()},
        "has_combined": has_combined,
        "sector_data": sector_data,
        "holdings": holdings_data,
        "event_trades": event_trades_data,
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
    comb_report = data.get("combined_report", {})
    spy_report = data.get("spy_report", {})
    has_combined = data.get("has_combined", False)
    sector_data = data.get("sector_data", {})
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

    # Screener holdings
    holdings_rows = ""
    for h in reversed(data["holdings"][-8:]):
        tickers_str = ", ".join(h["tickers"][:10])
        more = f" +{len(h['tickers'])-10}" if len(h["tickers"]) > 10 else ""
        holdings_rows += f"<tr><td>{h['date']}</td><td>{tickers_str}{more}</td></tr>\n"

    # Event-driven trades table
    event_trades_rows = ""
    for t in data.get("event_trades", []):
        ret = t.get("return", 0)
        ret_class = "positive" if ret > 0 else "negative"
        event_trades_rows += (
            f"<tr><td><strong>{t['ticker']}</strong></td>"
            f"<td>{t['entry']}</td><td>{t['exit']}</td>"
            f"<td>{t['hold_days']}d</td>"
            f'<td class="{ret_class}">{_pct(ret)}</td>'
            f"<td>{t['reason']}</td></tr>\n"
        )

    excess = report.get("excess_return", 0)
    excess_class = "highlight" if excess > 0 else "danger"
    comb_excess = comb_report.get("excess_return", 0)
    comb_excess_class = "highlight" if comb_excess > 0 else "danger"

    cum_json = json.dumps(data["cumulative"])
    dd_json = json.dumps(data["drawdown"])
    roll_json = json.dumps(data["rolling"])

    # Sector chart data
    sector_colors = [
        "#58a6ff", "#3fb950", "#d2a8ff", "#f0883e", "#f85149",
        "#db61a2", "#79c0ff", "#56d364", "#e3b341", "#8b949e",
        "#b392f0", "#ff7b72", "#7ee787", "#ffa657", "#d29922",
    ]
    sector_dates_json = json.dumps(sector_data.get("dates", []))
    sector_datasets = []
    for i, sector in enumerate(sector_data.get("sectors", [])):
        weights = sector_data.get("weights", {}).get(sector, [])
        if max(weights) < 0.01 if weights else True:
            continue
        color = sector_colors[i % len(sector_colors)]
        sector_datasets.append({
            "label": sector,
            "data": [round(w * 100, 1) for w in weights],
            "backgroundColor": color + "99",
            "borderColor": color,
            "borderWidth": 1,
            "fill": True,
        })
    sector_datasets_json = json.dumps(sector_datasets)

    rel_perf = sector_data.get("relative_performance", {})
    rel_perf_json = json.dumps(rel_perf)

    sector_stats_rows = ""
    for s in sector_data.get("stats", []):
        contrib_class = "positive" if s["total_contribution"] > 0 else "negative"
        eff = s["contribution_per_pct"]
        eff_class = "positive" if eff > 2.0 else ("negative" if eff < 0 else "")
        sector_stats_rows += (
            f"<tr><td>{s['sector']}</td>"
            f"<td>{s['avg_weight']:.1%}</td>"
            f'<td class="{contrib_class}">{s["total_contribution"]:.1%}</td>'
            f'<td class="{eff_class}">{eff:.2f}x</td></tr>\n'
        )

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
  .three-col {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; }}
  .ml-score {{ color: #d2a8ff; }}
  .warn {{ color: #d29922; }}
  .strategy-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; }}
  .strategy-card h3 {{ color: #58a6ff; margin-bottom: 15px; font-size: 1.1em; }}
  .strategy-card.combined h3 {{ color: #d2a8ff; }}
  .strategy-card .stat {{ display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #21262d; }}
  .strategy-card .stat .val {{ font-weight: bold; }}
  .tabs {{ display: flex; gap: 0; margin: 20px 0 0; border-bottom: 2px solid #30363d; }}
  .tab {{ padding: 10px 24px; cursor: pointer; color: #8b949e; border-bottom: 2px solid transparent;
          margin-bottom: -2px; font-size: 0.95em; transition: all 0.2s; }}
  .tab:hover {{ color: #e0e0e0; }}
  .tab.active {{ color: #58a6ff; border-bottom-color: #58a6ff; font-weight: bold; }}
  .tab-content {{ display: none; padding-top: 10px; }}
  .tab-content.active {{ display: block; }}
  @media (max-width: 900px) {{ .two-col, .three-col {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<h1>Value Investing Dashboard</h1>
<p class="subtitle">Buffett/Munger quality screen + event-driven engine &middot; {config['start']} to {config['end']} &middot;
   Screener: {config['freq']}, top {config['top_n']} &middot;
   Event: report-triggered, 15 positions, 540d max hold</p>

<div class="tabs">
  <div class="tab active" onclick="switchTab('performance')">Performance</div>
  <div class="tab" onclick="switchTab('sectors')">Sectors</div>
  <div class="tab" onclick="switchTab('holdings')">Holdings</div>
</div>

<div id="tab-performance" class="tab-content active">

<h2>Strategy Comparison</h2>
<div class="three-col">
  <div class="strategy-card">
    <h3>Screener Only</h3>
    <div class="stat"><span>CAGR</span><span class="val">{_pct(report['cagr'])}</span></div>
    <div class="stat"><span>Total Return</span><span class="val">{_pct(report['total_return'], 0)}</span></div>
    <div class="stat"><span>Sharpe</span><span class="val">{report['sharpe']:.2f}</span></div>
    <div class="stat"><span>Sortino</span><span class="val">{report['sortino']:.2f}</span></div>
    <div class="stat"><span>Max Drawdown</span><span class="val negative">{_pct(report['max_drawdown'])}</span></div>
    <div class="stat"><span>Volatility</span><span class="val">{_pct(report['volatility'])}</span></div>
    <div class="stat"><span>vs SPY</span><span class="val {excess_class}">{_pct(excess)}</span></div>
  </div>
  {'<div class="strategy-card combined"><h3>Event-Driven</h3>' + f"""
    <div class="stat"><span>CAGR</span><span class="val">{_pct(comb_report.get('cagr'))}</span></div>
    <div class="stat"><span>Total Return</span><span class="val">{_pct(comb_report.get('total_return'), 0)}</span></div>
    <div class="stat"><span>Sharpe</span><span class="val">{comb_report.get('sharpe', 0):.2f}</span></div>
    <div class="stat"><span>Sortino</span><span class="val">{comb_report.get('sortino', 0):.2f}</span></div>
    <div class="stat"><span>Max Drawdown</span><span class="val negative">{_pct(comb_report.get('max_drawdown'))}</span></div>
    <div class="stat"><span>Volatility</span><span class="val">{_pct(comb_report.get('volatility'))}</span></div>
    <div class="stat"><span>vs SPY</span><span class="val {comb_excess_class}">{_pct(comb_excess)}</span></div>
  </div>""" if has_combined else ''}
  <div class="strategy-card">
    <h3>SPY Benchmark</h3>
    <div class="stat"><span>CAGR</span><span class="val">{_pct(spy_report.get('cagr'))}</span></div>
    <div class="stat"><span>Total Return</span><span class="val">{_pct(spy_report.get('total_return'), 0)}</span></div>
    <div class="stat"><span>Sharpe</span><span class="val">{spy_report.get('sharpe', 0):.2f}</span></div>
    <div class="stat"><span>Sortino</span><span class="val">{spy_report.get('sortino', 0):.2f}</span></div>
    <div class="stat"><span>Max Drawdown</span><span class="val negative">{_pct(spy_report.get('max_drawdown'))}</span></div>
    <div class="stat"><span>Volatility</span><span class="val">{_pct(spy_report.get('volatility'))}</span></div>
  </div>
</div>

<div class="two-col">
  <div class="chart-container"><canvas id="cumChart"></canvas></div>
  <div class="chart-container"><canvas id="ddChart"></canvas></div>
</div>
<div class="chart-container"><canvas id="rollChart"></canvas></div>


</div><!-- end tab-performance -->

<div id="tab-sectors" class="tab-content">

<h2>Sector Allocation Over Time</h2>
<div class="chart-container">
  <canvas id="sectorChart" height="120"></canvas>
</div>

<h2>Sector Performance</h2>
<div class="table-container">
<table>
<thead><tr>
  <th>Sector</th><th>Avg Weight</th><th>Total Contribution</th><th>Efficiency</th>
</tr></thead>
<tbody>{sector_stats_rows}</tbody>
</table>
<p style="color: #8b949e; font-size: 0.8em; margin-top: 10px;">
  Efficiency = total return contribution / average weight. Higher = more return per unit of allocation.</p>
</div>

</div><!-- end tab-sectors -->

<div id="tab-holdings" class="tab-content">

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

<div class="two-col">
<div>
<h2>Screener Holdings</h2>
<div class="table-container">
<table>
<thead><tr><th>Date</th><th>Holdings</th></tr></thead>
<tbody>{holdings_rows}</tbody>
</table>
</div>
</div>
<div>
<h2>Event-Driven Trades (Recent)</h2>
<div class="table-container">
<table>
<thead><tr><th>Ticker</th><th>Entry</th><th>Exit</th><th>Hold</th><th>Return</th><th>Reason</th></tr></thead>
<tbody>{event_trades_rows}</tbody>
</table>
</div>
</div>
</div>

</div><!-- end tab-holdings -->

<script>
const colors = [
  {{ line: 'rgb(88, 166, 255)', fill: 'rgba(88, 166, 255, 0.1)' }},   // Screener - blue
  {{ line: 'rgb(210, 168, 255)', fill: 'rgba(210, 168, 255, 0.1)' }},  // Screener+ML - purple
  {{ line: 'rgb(139, 148, 158)', fill: 'rgba(139, 148, 158, 0.05)' }}, // SPY - gray
];
const ddColors = [
  {{ line: 'rgb(88, 166, 255)', fill: 'rgba(88, 166, 255, 0.3)' }},
  {{ line: 'rgb(210, 168, 255)', fill: 'rgba(210, 168, 255, 0.3)' }},
  {{ line: 'rgb(139, 148, 158)', fill: 'rgba(139, 148, 158, 0.1)' }},
];

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
    datasets: cumData.map((d, i) => ({{
      label: d.label,
      data: d.data,
      borderColor: colors[i % colors.length].line,
      backgroundColor: colors[i % colors.length].fill,
      fill: i === 0, borderWidth: 1.5, pointRadius: 0,
    }}))
  }},
  options: {{ ...defaultOpts, plugins: {{ ...defaultOpts.plugins,
    title: {{ display: true, text: 'Cumulative Returns (Growth of $1)', color: '#e0e0e0' }} }} }}
}});

new Chart(document.getElementById('ddChart'), {{
  type: 'line',
  data: {{
    datasets: ddData.map((d, i) => ({{
      label: d.label,
      data: d.data,
      borderColor: ddColors[i % ddColors.length].line,
      backgroundColor: ddColors[i % ddColors.length].fill,
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
    datasets: rollData.map((d, i) => ({{
      label: d.label,
      data: d.data,
      borderColor: colors[i % colors.length].line,
      borderWidth: 1.5, pointRadius: 0, fill: false,
    }}))
  }},
  options: {{ ...defaultOpts, plugins: {{ ...defaultOpts.plugins,
    title: {{ display: true, text: 'Rolling 1-Year Annualized Returns', color: '#e0e0e0' }} }},
    scales: {{ ...defaultOpts.scales, y: {{ ...defaultOpts.scales.y,
      ticks: {{ ...defaultOpts.scales.y.ticks, callback: v => (v*100).toFixed(0)+'%' }} }} }}
  }}
}});

// Sector chart with relative performance overlay
const sectorLabels = {sector_dates_json};
const sectorDatasets = {sector_datasets_json};
const relPerf = {rel_perf_json};
let sectorChart = null;

function initSectorChart() {{
  if (sectorChart) return;
  const ctx = document.getElementById('sectorChart');
  if (!ctx) return;

  // Add relative performance line on secondary axis (same labels as sectors)
  const allDatasets = [...sectorDatasets];
  if (relPerf.values && relPerf.values.length > 0) {{
    allDatasets.push({{
      label: 'vs S&P 500',
      data: relPerf.values,
      borderColor: '#ffffff',
      backgroundColor: 'transparent',
      borderWidth: 2.5,
      borderDash: [6, 3],
      fill: false,
      yAxisID: 'y2',
      pointRadius: 0,
      order: -1,
    }});
  }}

  const chartLabels = sectorLabels;

  sectorChart = new Chart(ctx, {{
    type: 'line',
    data: {{ labels: chartLabels, datasets: allDatasets }},
    options: {{
      responsive: true,
      interaction: {{ mode: 'index', intersect: false }},
      scales: {{
        x: {{ type: 'time', time: {{ unit: 'year' }}, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
        y: {{ stacked: true, min: 0, max: 100, position: 'left',
              ticks: {{ color: '#8b949e', callback: v => v + '%' }}, grid: {{ color: '#21262d' }},
              title: {{ display: true, text: 'Sector Weight', color: '#8b949e' }} }},
        y2: {{ position: 'right', grid: {{ drawOnChartArea: false }},
              ticks: {{ color: '#ffffff', callback: v => (v > 0 ? '+' : '') + v + '%' }},
              title: {{ display: true, text: 'vs S&P 500 (cumulative)', color: '#ffffff' }} }}
      }},
      plugins: {{
        legend: {{ labels: {{ color: '#8b949e' }}, position: 'right' }},
        title: {{ display: true, text: 'Sector Allocation & Relative Performance vs S&P 500', color: '#e0e0e0' }},
        tooltip: {{
          callbacks: {{
            label: function(ctx) {{
              if (ctx.dataset.yAxisID === 'y2') {{
                return ctx.dataset.label + ': ' + (ctx.raw > 0 ? '+' : '') + ctx.raw.toFixed(1) + '%';
              }}
              return ctx.dataset.label + ': ' + ctx.raw + '%';
            }}
          }}
        }}
      }},
      elements: {{ line: {{ tension: 0.3 }}, point: {{ radius: 0 }} }}
    }}
  }});
}}

// Tab switching
function switchTab(name) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');
  // Lazy init sector chart when tab is first shown
  if (name === 'sectors') setTimeout(initSectorChart, 50);
}}
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
