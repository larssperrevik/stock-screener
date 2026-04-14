"""Sector weight analysis for the event-driven strategy.

Reconstructs portfolio holdings over time, maps to sectors,
and analyzes how sector tilts relate to performance.

Outputs an HTML report with:
- Stacked area chart of sector weights over time
- Sector contribution to returns
- Performance during different sector regimes
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

from data.simfin_loader import load_derived_annual, load_prices, load_companies, load_industries
from screener.criteria import ScreenCriteria
from backtest.event_engine import EventDrivenEngine, compute_stock_score


def run_with_holdings_tracking(start="2011-01-01", end="2024-09-30"):
    """Run event engine and capture daily holdings snapshots."""
    engine = EventDrivenEngine(
        criteria=ScreenCriteria(),
        max_positions=15, max_hold_days=540,
        min_hold_days=60, buy_threshold=40, sell_threshold=20,
        start_date=start, end_date=end,
    )

    # We need to modify the run to capture daily holdings
    # Re-implement the core loop with holdings tracking
    derived = load_derived_annual()
    prices = load_prices()

    print("Building price matrix...")
    price_matrix = prices.pivot_table(
        index="Date", columns="Ticker", values="Adj. Close", aggfunc="first"
    ).ffill(limit=5)

    price_by_ticker = {}
    for ticker, group in prices.groupby("Ticker"):
        price_by_ticker[ticker] = group.set_index("Date").sort_index()

    derived_by_ticker = {}
    for ticker, group in derived.groupby("Ticker"):
        if "_delisted" in ticker or "_old" in ticker:
            continue
        derived_by_ticker[ticker] = group.sort_values("Publish Date")

    # Build sector map
    companies = load_companies()
    industries = load_industries()
    sector_map = {}
    for _, co in companies.iterrows():
        ticker = co["Ticker"]
        ind_id = co.get("IndustryId")
        if pd.notna(ind_id):
            ind = industries[industries["IndustryId"] == ind_id]
            if not ind.empty:
                sector_map[ticker] = ind.iloc[0].get("Sector", "Unknown")
            else:
                sector_map[ticker] = "Unknown"
        else:
            sector_map[ticker] = "Unknown"

    seen_reports = set()
    for _, row in derived[derived["Publish Date"] < pd.Timestamp(start)].iterrows():
        seen_reports.add((row["Ticker"], row["Publish Date"]))

    positions = {}
    trading_days = price_matrix.index[
        (price_matrix.index >= start) & (price_matrix.index <= end)
    ]

    # Daily snapshots
    daily_holdings = []  # [{date, holdings: {ticker: weight}, returns: float}]

    print(f"Simulating {len(trading_days)} trading days...")

    for day in trading_days:
        # Check new reports
        day_reports = derived[
            (derived["Publish Date"] >= day) & (derived["Publish Date"] <= day)
        ]
        tickers_to_evaluate = set()
        for _, row in day_reports.iterrows():
            ticker = row["Ticker"]
            if "_delisted" in ticker or "_old" in ticker:
                continue
            key = (ticker, row["Publish Date"])
            if key not in seen_reports:
                seen_reports.add(key)
                tickers_to_evaluate.add(ticker)

        for ticker, pos in list(positions.items()):
            if (day - pos["entry_date"]).days >= 540:
                tickers_to_evaluate.add(ticker)

        # Evaluate
        for ticker in tickers_to_evaluate:
            if ticker not in derived_by_ticker:
                continue
            ticker_derived = derived_by_ticker[ticker]
            available = ticker_derived[ticker_derived["Publish Date"] <= day]
            if available.empty:
                continue

            latest = available.iloc[-1]
            publish_date = latest["Publish Date"]
            days_since = (day - publish_date).days
            if days_since > 120:
                continue

            passes = engine._passes_screen(latest.to_dict())
            hist = price_by_ticker.get(ticker)
            if hist is not None:
                hist = hist[hist.index <= day]

            score, _ = compute_stock_score(ticker, latest.to_dict(), hist, available)
            if days_since < 30:
                score += 5
            elif days_since < 60:
                score += 3

            # Sell logic
            if ticker in positions:
                days_held = (day - positions[ticker]["entry_date"]).days
                if not passes and days_held >= 60:
                    del positions[ticker]
                elif score < 20 and days_held >= 60:
                    del positions[ticker]
                elif days_held >= 540:
                    if score >= 40 and passes:
                        positions[ticker]["entry_date"] = day
                    else:
                        del positions[ticker]
            # Buy logic
            elif passes and score >= 40 and len(positions) < 15:
                if ticker in price_matrix.columns:
                    p = price_matrix.loc[day, ticker]
                    if pd.notna(p) and p > 0:
                        positions[ticker] = {"entry_date": day, "entry_price": p, "score": score}

        # Record daily snapshot
        held = list(positions.keys())
        n = len(held)

        # Compute daily return
        prev_idx = price_matrix.index.get_loc(day)
        prev = price_matrix.index[prev_idx - 1] if prev_idx > 0 else None

        if held and prev is not None:
            rets = []
            for t in held:
                if t in price_matrix.columns:
                    p0, p1 = price_matrix.loc[prev, t], price_matrix.loc[day, t]
                    if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                        r = (p1 / p0) - 1
                        if abs(r) < 1:
                            rets.append(r)
            port_ret = np.mean(rets) if rets else 0.0
        else:
            port_ret = (1 + 0.04) ** (1/252) - 1

        # Sector weights (equal weight per position)
        sector_weights = defaultdict(float)
        for t in held:
            sector = sector_map.get(t, "Unknown")
            sector_weights[sector] += 1.0 / max(n, 1)

        # Per-sector return contribution
        sector_returns = defaultdict(float)
        if held and prev is not None:
            for t in held:
                if t in price_matrix.columns:
                    p0, p1 = price_matrix.loc[prev, t], price_matrix.loc[day, t]
                    if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                        r = (p1 / p0) - 1
                        if abs(r) < 1:
                            sector = sector_map.get(t, "Unknown")
                            sector_returns[sector] += r / max(n, 1)

        # SPY return for relative performance
        spy_ret = 0.0
        if "SPY" in price_matrix.columns and prev is not None:
            sp0 = price_matrix.loc[prev, "SPY"]
            sp1 = price_matrix.loc[day, "SPY"]
            if pd.notna(sp0) and pd.notna(sp1) and sp0 > 0:
                spy_ret = (sp1 / sp0) - 1

        daily_holdings.append({
            "date": day,
            "holdings": held,
            "n_positions": n,
            "port_return": port_ret,
            "spy_return": spy_ret,
            "sector_weights": dict(sector_weights),
            "sector_returns": dict(sector_returns),
        })

    return daily_holdings, sector_map


def _compute_sp500_sector_weights(monthly_dates, sector_map):
    """Compute S&P 500 sector weights at each monthly date using market caps."""
    from data.simfin_loader import get_sp500_tickers, SIMFIN_DIR

    sp500 = set(get_sp500_tickers())

    # Load market caps directly from SimFin source (has Market-Cap column)
    print("  Loading market cap data from SimFin source...")
    prices = pd.read_csv(
        SIMFIN_DIR / "us-derived-shareprices-daily.csv", sep=";",
        usecols=["Ticker", "Date", "Market-Cap"], parse_dates=["Date"],
        dtype={"Ticker": str}
    )
    mcap_data = prices[prices["Ticker"].isin(sp500)].copy()
    mcap_data = mcap_data.dropna(subset=["Market-Cap"])
    mcap_data["Market-Cap"] = pd.to_numeric(mcap_data["Market-Cap"], errors="coerce")

    sp_weights = {}
    for date_str in monthly_dates:
        date = pd.Timestamp(date_str)
        # Get market caps near this date
        window = mcap_data[
            (mcap_data["Date"] >= date - pd.Timedelta(days=10)) &
            (mcap_data["Date"] <= date + pd.Timedelta(days=5))
        ]
        if window.empty:
            sp_weights[date_str] = {}
            continue

        latest = window.sort_values("Date").groupby("Ticker").last()
        latest["Sector"] = latest.index.map(lambda t: sector_map.get(t, "Unknown"))
        sector_caps = latest.groupby("Sector")["Market-Cap"].sum()
        total = sector_caps.sum()
        if total > 0:
            sp_weights[date_str] = (sector_caps / total).to_dict()
        else:
            sp_weights[date_str] = {}

    return sp_weights


def build_sector_report(daily_holdings):
    """Analyze sector weights and performance."""
    dates = [d["date"] for d in daily_holdings]
    returns = [d["port_return"] for d in daily_holdings]

    # Collect all sectors
    all_sectors = set()
    for d in daily_holdings:
        all_sectors.update(d["sector_weights"].keys())
    all_sectors = sorted(all_sectors)

    # Build sector weight timeseries (monthly for chart)
    monthly_dates = []
    monthly_weights = {s: [] for s in all_sectors}
    monthly_returns = {s: [] for s in all_sectors}

    current_month = None
    month_weights_accum = {s: [] for s in all_sectors}
    month_returns_accum = {s: [] for s in all_sectors}

    for d in daily_holdings:
        month_key = (d["date"].year, d["date"].month)
        if month_key != current_month:
            if current_month is not None:
                monthly_dates.append(f"{current_month[0]}-{current_month[1]:02d}-01")
                for s in all_sectors:
                    vals = month_weights_accum[s]
                    monthly_weights[s].append(np.mean(vals) if vals else 0)
                    rets = month_returns_accum[s]
                    monthly_returns[s].append(sum(rets))
                month_weights_accum = {s: [] for s in all_sectors}
                month_returns_accum = {s: [] for s in all_sectors}
            current_month = month_key

        for s in all_sectors:
            month_weights_accum[s].append(d["sector_weights"].get(s, 0))
            month_returns_accum[s].append(d["sector_returns"].get(s, 0))

    # Last month
    if current_month:
        monthly_dates.append(f"{current_month[0]}-{current_month[1]:02d}-01")
        for s in all_sectors:
            monthly_weights[s].append(np.mean(month_weights_accum[s]) if month_weights_accum[s] else 0)
            monthly_returns[s].append(sum(month_returns_accum[s]))
        for s in all_sectors:
            monthly_weights[s].append(np.mean(month_weights_accum[s]) if month_weights_accum[s] else 0)
            monthly_returns[s].append(sum(month_returns_accum[s]))

    # Sector performance summary
    print("\n" + "=" * 70)
    print("SECTOR ANALYSIS")
    print("=" * 70)

    sector_stats = []
    for s in all_sectors:
        avg_weight = np.mean([d["sector_weights"].get(s, 0) for d in daily_holdings])
        total_contribution = sum(d["sector_returns"].get(s, 0) for d in daily_holdings)
        if avg_weight > 0.01:
            sector_stats.append({
                "sector": s,
                "avg_weight": avg_weight,
                "total_contribution": total_contribution,
                "contribution_per_pct": total_contribution / avg_weight if avg_weight > 0 else 0,
            })

    sector_stats.sort(key=lambda x: x["total_contribution"], reverse=True)

    print(f"\n{'Sector':<25} {'Avg Weight':>10} {'Total Contrib':>14} {'Contrib/Weight':>14}")
    print("-" * 65)
    for s in sector_stats:
        print(f"  {s['sector']:<23} {s['avg_weight']:>9.1%} {s['total_contribution']:>13.2%} "
              f"{s['contribution_per_pct']:>13.2f}")

    # Regime analysis: top sector weight > 30%
    print("\n\nSECTOR REGIME ANALYSIS (when a single sector > 30% weight)")
    print("-" * 65)
    regime_periods = defaultdict(list)
    for d in daily_holdings:
        for s, w in d["sector_weights"].items():
            if w > 0.30:
                regime_periods[s].append(d["port_return"])

    for sector in sorted(regime_periods.keys()):
        rets = regime_periods[sector]
        n_days = len(rets)
        ann_ret = (1 + np.mean(rets)) ** 252 - 1
        vol = np.std(rets) * np.sqrt(252)
        print(f"  {sector:<25} {n_days:>5} days  Ann.Return: {ann_ret:>7.1%}  Vol: {vol:>6.1%}")

    # Smoothed relative performance (portfolio vs SPY)
    # Cumulative daily, then resample to monthly, smooth with 3-month MA
    port_cum = []
    spy_cum = []
    cum_p = 1.0
    cum_s = 1.0
    daily_dates = []
    for d in daily_holdings:
        cum_p *= (1 + d["port_return"])
        cum_s *= (1 + d["spy_return"])
        daily_dates.append(d["date"])
        port_cum.append(cum_p)
        spy_cum.append(cum_s)

    # Relative: portfolio / SPY - 1 (0 = equal, positive = outperforming)
    relative_daily = pd.Series(
        [p / s - 1 if s > 0 else 0 for p, s in zip(port_cum, spy_cum)],
        index=daily_dates
    )
    # Resample to match monthly_dates exactly
    relative_monthly = relative_daily.resample("MS").last()
    relative_smoothed = relative_monthly.rolling(3, min_periods=1).mean()

    # Align to sector monthly_dates
    target_dates = pd.to_datetime(monthly_dates)
    rel_values = []
    for d in target_dates:
        # Find closest date in smoothed series
        mask = relative_smoothed.index <= d
        if mask.any():
            rel_values.append(round(float(relative_smoothed[mask].iloc[-1]) * 100, 2))
        else:
            rel_values.append(0.0)

    relative_performance = {
        "values": rel_values,
    }

    # S&P 500 sector weights for comparison
    # Build sector map from daily_holdings context
    companies = load_companies()
    industries_df = load_industries()
    sector_map_full = {}
    for _, co in companies.iterrows():
        t = co["Ticker"]
        ind_id = co.get("IndustryId")
        if pd.notna(ind_id):
            ind = industries_df[industries_df["IndustryId"] == ind_id]
            sector_map_full[t] = ind.iloc[0].get("Sector", "Unknown") if not ind.empty else "Unknown"
        else:
            sector_map_full[t] = "Unknown"

    sp500_weights = _compute_sp500_sector_weights(monthly_dates, sector_map_full)

    # Build monthly S&P 500 sector weight series
    sp500_monthly = {s: [] for s in all_sectors}
    for date_str in monthly_dates:
        sw = sp500_weights.get(date_str, {})
        for s in all_sectors:
            sp500_monthly[s].append(sw.get(s, 0))

    # Compute over/underweight per sector per month
    overweight_monthly = {s: [] for s in all_sectors}
    for i, date_str in enumerate(monthly_dates):
        for s in all_sectors:
            port_w = monthly_weights[s][i] if i < len(monthly_weights[s]) else 0
            sp_w = sp500_monthly[s][i] if i < len(sp500_monthly[s]) else 0
            overweight_monthly[s].append(port_w - sp_w)

    # Summary stats
    print("\n\nSECTOR OVER/UNDERWEIGHT vs S&P 500")
    print("-" * 65)
    print(f"  {'Sector':<23} {'Port Wt':>9} {'S&P Wt':>9} {'Over/Under':>11}")
    print("-" * 65)
    for s in all_sectors:
        avg_port = np.mean(monthly_weights[s])
        avg_sp = np.mean(sp500_monthly[s])
        diff = avg_port - avg_sp
        if avg_port > 0.01 or avg_sp > 0.01:
            diff_class = "+" if diff > 0 else ""
            print(f"  {s:<23} {avg_port:>8.1%} {avg_sp:>8.1%} {diff_class}{diff:>9.1%}")

    return {
        "dates": monthly_dates,
        "weights": monthly_weights,
        "returns": monthly_returns,
        "sectors": all_sectors,
        "stats": sector_stats,
        "relative_performance": relative_performance,
        "sp500_weights": sp500_monthly,
        "overweight": overweight_monthly,
    }


def generate_sector_html(sector_data, output="sector_analysis.html"):
    """Generate HTML with stacked area chart of sector weights."""
    dates_json = json.dumps(sector_data["dates"])

    # Colors for sectors
    colors = [
        "#58a6ff", "#3fb950", "#d2a8ff", "#f0883e", "#f85149",
        "#db61a2", "#79c0ff", "#56d364", "#e3b341", "#8b949e",
        "#b392f0", "#ff7b72", "#7ee787", "#ffa657", "#d29922",
    ]

    datasets = []
    for i, sector in enumerate(sector_data["sectors"]):
        weights = sector_data["weights"][sector]
        if max(weights) < 0.01:
            continue
        color = colors[i % len(colors)]
        datasets.append({
            "label": sector,
            "data": [round(w * 100, 1) for w in weights],
            "backgroundColor": color + "99",
            "borderColor": color,
            "borderWidth": 1,
            "fill": True,
        })

    datasets_json = json.dumps(datasets)

    # Stats table
    stats_rows = ""
    for s in sector_data["stats"]:
        contrib_class = "positive" if s["total_contribution"] > 0 else "negative"
        efficiency = s["contribution_per_pct"]
        eff_class = "positive" if efficiency > 0.5 else ("negative" if efficiency < 0 else "")
        stats_rows += (
            f"<tr>"
            f"<td>{s['sector']}</td>"
            f"<td>{s['avg_weight']:.1%}</td>"
            f'<td class="{contrib_class}">{s["total_contribution"]:.2%}</td>'
            f'<td class="{eff_class}">{efficiency:.2f}</td>'
            f"</tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Sector Analysis</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f1117; color: #e0e0e0; padding: 20px; }}
  h1 {{ color: #58a6ff; margin-bottom: 5px; }}
  h2 {{ color: #8b949e; margin: 25px 0 10px; font-size: 1.1em; text-transform: uppercase; letter-spacing: 1px; }}
  .subtitle {{ color: #8b949e; margin-bottom: 20px; font-size: 0.9em; }}
  .chart-container {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                      padding: 15px; margin: 15px 0; }}
  canvas {{ max-height: 400px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
  th {{ background: #21262d; color: #8b949e; padding: 10px 12px; text-align: left;
       border-bottom: 2px solid #30363d; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; }}
  tr:hover {{ background: #1c2128; }}
  .positive {{ color: #3fb950; font-weight: bold; }}
  .negative {{ color: #f85149; font-weight: bold; }}
  .table-container {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                      padding: 15px; margin: 15px 0; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
  @media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<h1>Sector Weight Analysis</h1>
<p class="subtitle">Event-driven strategy portfolio composition over time</p>

<h2>Sector Weights Over Time</h2>
<div class="chart-container">
  <canvas id="weightChart" height="120"></canvas>
</div>

<h2>Sector Contribution to Returns</h2>
<div class="chart-container">
  <canvas id="contribChart" height="80"></canvas>
</div>

<div class="two-col">
<div>
<h2>Sector Performance Summary</h2>
<div class="table-container">
<table>
<thead><tr>
  <th>Sector</th><th>Avg Weight</th><th>Total Contribution</th><th>Efficiency</th>
</tr></thead>
<tbody>{stats_rows}</tbody>
</table>
<p style="color: #8b949e; font-size: 0.8em; margin-top: 10px;">
  Efficiency = total return contribution / average weight. Higher = more return per unit of allocation.
</p>
</div>
</div>
</div>

<script>
const labels = {dates_json};
const datasets = {datasets_json};

new Chart(document.getElementById('weightChart'), {{
  type: 'line',
  data: {{ labels, datasets }},
  options: {{
    responsive: true,
    scales: {{
      x: {{ type: 'time', time: {{ unit: 'year' }}, ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
      y: {{ stacked: true, min: 0, max: 100,
            ticks: {{ color: '#8b949e', callback: v => v + '%' }}, grid: {{ color: '#21262d' }} }}
    }},
    plugins: {{
      legend: {{ labels: {{ color: '#8b949e' }}, position: 'right' }},
      title: {{ display: true, text: 'Portfolio Sector Allocation (%)', color: '#e0e0e0' }}
    }},
    elements: {{ line: {{ tension: 0.3 }}, point: {{ radius: 0 }} }}
  }}
}});

// Cumulative sector contribution chart
const contribDatasets = datasets.map(d => {{
  const cumReturns = [];
  let cum = 0;
  const sectorIdx = datasets.indexOf(d);
  // Need to get monthly returns for this sector
  return {{
    ...d,
    fill: false,
    backgroundColor: undefined,
    borderWidth: 2,
  }};
}});

</script>
</body>
</html>"""

    with open(output, "w") as f:
        f.write(html)
    print(f"\nSector analysis saved to {output}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2011-01-01")
    parser.add_argument("--end", default="2024-09-30")
    parser.add_argument("-o", "--output", default="sector_analysis.html")
    args = parser.parse_args()

    daily, sector_map = run_with_holdings_tracking(args.start, args.end)
    data = build_sector_report(daily)
    generate_sector_html(data, args.output)
