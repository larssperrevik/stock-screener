"""Walk-forward optimization of screener filter thresholds.

Uses continuous OOS simulation (positions carry over).
Fixed trading params from previous walk-forward: sell=15, min_hold=30, buy=40, max_hold=540.
Sweeps screener criteria: min_roic, min_piotroski, min_gross_margin.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import time
from itertools import product

from backtest.event_engine import EventDrivenEngine, Position, Trade, compute_stock_score
from screener.criteria import ScreenCriteria
from metrics.performance import full_report
from data.simfin_loader import (
    load_derived_annual, load_prices, load_companies, load_industries,
    get_sp500_tickers, SIMFIN_DIR,
)


# Fixed trading params (walk-forward validated)
TRADING_PARAMS = dict(
    max_positions=15,
    max_hold_days=540,
    min_hold_days=30,
    buy_threshold=40,
    sell_threshold=15,
    max_correlation=0.65,
    max_sector_overweight=0.40,
    trailing_stop=0,
    weight_by_score=False,
)

# Screener filter grid
FILTER_GRID = {
    "min_roic": [0.08, 0.10, 0.12, 0.15],
    "min_piotroski": [4, 5, 6, 7],
    "min_gross_margin": [0.20, 0.25, 0.30, 0.40],
}

TRAIN_YEARS = 5
TEST_YEARS = 1
FIRST_TRAIN_START = 2011
LAST_TEST_END = 2026


def build_filter_combos():
    keys = list(FILTER_GRID.keys())
    values = [FILTER_GRID[k] for k in keys]
    combos = []
    for vals in product(*values):
        combos.append(dict(zip(keys, vals)))
    return combos


def build_folds():
    folds = []
    year = FIRST_TRAIN_START
    while year + TRAIN_YEARS < LAST_TEST_END:
        train_start = f"{year}-01-01"
        train_end = f"{year + TRAIN_YEARS - 1}-12-31"
        test_start = f"{year + TRAIN_YEARS}-01-01"
        test_end_year = year + TRAIN_YEARS + TEST_YEARS - 1
        if test_end_year >= LAST_TEST_END:
            test_end = f"{LAST_TEST_END}-12-31"
        else:
            test_end = f"{test_end_year}-12-31"
        folds.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })
        year += 1
    return folds


def train_fold(combos, train_start, train_end):
    """Sweep all filter combos on training window, return best by Sortino."""
    best_sortino = -999
    best_combo = combos[0]
    best_result = None

    for ci, combo in enumerate(combos):
        criteria = ScreenCriteria(
            min_roic=combo["min_roic"],
            min_piotroski=combo["min_piotroski"],
            min_gross_margin=combo["min_gross_margin"],
        )
        engine = EventDrivenEngine(
            criteria=criteria,
            start_date=train_start,
            end_date=train_end,
            **TRADING_PARAMS,
        )
        result = engine.run(quiet=True)
        r = result["report"]
        sortino = r.get("sortino", 0)
        if np.isnan(sortino) or np.isinf(sortino):
            sortino = 0

        if sortino > best_sortino:
            best_sortino = sortino
            best_combo = combo
            best_result = r

    return best_combo, best_sortino, best_result


def run_continuous_oos(fold_params):
    """Run one continuous simulation switching screener filters at fold boundaries."""
    # Load all data once
    print("Loading data...")
    derived = load_derived_annual()
    prices = load_prices()
    companies = load_companies()
    industries_df = load_industries()

    print("Building price matrix...")
    price_matrix = prices.pivot_table(
        index="Date", columns="Ticker", values="Adj. Close", aggfunc="first"
    ).ffill(limit=5)

    print("Indexing data...")
    price_by_ticker = {}
    for ticker, group in prices.groupby("Ticker"):
        price_by_ticker[ticker] = group.set_index("Date").sort_index()

    derived_by_ticker = {}
    for ticker, group in derived.groupby("Ticker"):
        if "_delisted" in ticker or "_old" in ticker:
            continue
        derived_by_ticker[ticker] = group.sort_values("Publish Date")

    # Sector map
    sector_map = {}
    for _, co in companies.iterrows():
        ind_id = co.get("IndustryId")
        if pd.notna(ind_id):
            ind = industries_df[industries_df["IndustryId"] == ind_id]
            if not ind.empty:
                sector_map[co["Ticker"]] = ind.iloc[0].get("Sector", "Unknown")

    # S&P 500 data
    sp500_tickers = set(get_sp500_tickers())
    sp_mcap_data = None
    try:
        sp_prices = pd.read_csv(
            SIMFIN_DIR / "us-derived-shareprices-daily.csv", sep=";",
            usecols=["Ticker", "Date", "Market-Cap"], parse_dates=["Date"],
            dtype={"Ticker": str}
        )
        sp_mcap_data = sp_prices[sp_prices["Ticker"].isin(sp500_tickers)].copy()
        sp_mcap_data = sp_mcap_data.dropna(subset=["Market-Cap"])
        sp_mcap_data["Market-Cap"] = pd.to_numeric(sp_mcap_data["Market-Cap"], errors="coerce")
        sp_mcap_data["Sector"] = sp_mcap_data["Ticker"].map(sector_map)
    except Exception as e:
        print(f"Warning: could not load S&P market caps: {e}")

    sp_sector_cache = {}

    # Pre-populate seen reports
    oos_start = pd.Timestamp(fold_params[0][0])
    oos_end = pd.Timestamp(fold_params[-1][1])
    seen_reports = set()
    for _, row in derived[derived["Publish Date"] < oos_start].iterrows():
        seen_reports.add((row["Ticker"], row["Publish Date"]))

    # Build parameter schedule
    param_schedule = []
    for test_start, test_end, filter_params in fold_params:
        param_schedule.append((pd.Timestamp(test_start), filter_params))
    param_schedule.sort(key=lambda x: x[0])

    # State
    positions = {}
    trades = []
    daily_returns = []
    current_filters = param_schedule[0][1]
    next_param_idx = 1

    # Create engine with initial filters for helper methods
    current_criteria = ScreenCriteria(
        min_roic=current_filters["min_roic"],
        min_piotroski=current_filters["min_piotroski"],
        min_gross_margin=current_filters["min_gross_margin"],
    )
    temp_engine = EventDrivenEngine(criteria=current_criteria, **TRADING_PARAMS)

    trading_days = price_matrix.index[
        (price_matrix.index >= oos_start) &
        (price_matrix.index <= oos_end)
    ]

    print(f"\nRunning continuous OOS: {oos_start.strftime('%Y-%m-%d')} to {oos_end.strftime('%Y-%m-%d')}")
    print(f"  {len(trading_days)} trading days, {len(param_schedule)} filter switches")

    last_log_month = None

    for day in trading_days:
        # Check for filter switch
        if next_param_idx < len(param_schedule) and day >= param_schedule[next_param_idx][0]:
            current_filters = param_schedule[next_param_idx][1]
            next_param_idx += 1
            current_criteria = ScreenCriteria(
                min_roic=current_filters["min_roic"],
                min_piotroski=current_filters["min_piotroski"],
                min_gross_margin=current_filters["min_gross_margin"],
            )
            temp_engine.criteria = current_criteria
            print(f"  {day.strftime('%Y-%m-%d')}: FILTER SWITCH -> "
                  f"roic={current_filters['min_roic']} pio={current_filters['min_piotroski']} "
                  f"gm={current_filters['min_gross_margin']}")

        # New reports
        new_reports_today = []
        day_reports = derived[
            (derived["Publish Date"] >= day) &
            (derived["Publish Date"] <= day)
        ]
        for _, row in day_reports.iterrows():
            ticker = row["Ticker"]
            if "_delisted" in ticker or "_old" in ticker:
                continue
            key = (ticker, row["Publish Date"])
            if key not in seen_reports:
                seen_reports.add(key)
                new_reports_today.append(row)

        # Evaluate
        tickers_to_evaluate = set()
        for report in new_reports_today:
            tickers_to_evaluate.add(report["Ticker"])
        for ticker, pos in list(positions.items()):
            days_held = (day - pos.entry_date).days
            if days_held >= temp_engine.max_hold_days:
                tickers_to_evaluate.add(ticker)

        for ticker in tickers_to_evaluate:
            if ticker not in derived_by_ticker:
                continue
            ticker_derived = derived_by_ticker[ticker]
            available = ticker_derived[ticker_derived["Publish Date"] <= day]
            if available.empty:
                continue

            latest = available.iloc[-1]
            publish_date = latest["Publish Date"]
            days_since_publish = (day - publish_date).days
            if days_since_publish > temp_engine.freshness_days:
                continue

            latest_dict = latest.to_dict()
            passes_screen = temp_engine._passes_screen(latest_dict)

            ticker_prices = price_by_ticker.get(ticker)
            hist = ticker_prices[ticker_prices.index <= day] if ticker_prices is not None else None

            score, details = compute_stock_score(ticker, latest_dict, hist, available)
            if days_since_publish < 30:
                score += 5
            elif days_since_publish < 60:
                score += 3

            # Sell
            if ticker in positions:
                pos = positions[ticker]
                days_held = (day - pos.entry_date).days
                should_sell = False
                reason = ""

                if not passes_screen:
                    should_sell = True
                    reason = "failed_screen"
                elif score < temp_engine.sell_threshold and days_held >= temp_engine.min_hold_days:
                    should_sell = True
                    reason = f"low_score_{score:.0f}"
                elif days_held >= temp_engine.max_hold_days:
                    if score >= temp_engine.buy_threshold and passes_screen:
                        positions[ticker] = Position(
                            ticker=ticker, entry_date=day,
                            entry_price=pos.entry_price,
                            score=score, last_report_date=publish_date,
                            peak_price=pos.peak_price,
                        )
                        continue
                    else:
                        should_sell = True
                        reason = "max_hold"

                if should_sell and days_held >= temp_engine.min_hold_days:
                    current_price = price_matrix.loc[day, ticker] if ticker in price_matrix.columns else None
                    if current_price and pd.notna(current_price) and pos.entry_price > 0:
                        ret = (current_price / pos.entry_price) - 1
                        trades.append(Trade(
                            ticker=ticker, entry_date=pos.entry_date,
                            exit_date=day, entry_price=pos.entry_price,
                            exit_price=current_price, return_pct=ret,
                            hold_days=days_held, reason=reason,
                        ))
                    del positions[ticker]

            # Buy
            elif (passes_screen
                  and score >= temp_engine.buy_threshold
                  and ticker in price_matrix.columns):

                current_price = price_matrix.loc[day, ticker]
                if not (pd.notna(current_price) and current_price > 0):
                    continue

                if not temp_engine._check_sector_cap(ticker, positions, sector_map, day, sp_mcap_data, sp_sector_cache):
                    continue

                corr_conflict = temp_engine._find_correlated_holding(ticker, positions, price_matrix, day)
                if corr_conflict is not None:
                    old_pos = positions[corr_conflict]
                    if score > old_pos.score:
                        old_price = price_matrix.loc[day, corr_conflict] if corr_conflict in price_matrix.columns else None
                        if old_price and pd.notna(old_price) and old_pos.entry_price > 0:
                            days_held = (day - old_pos.entry_date).days
                            ret = (old_price / old_pos.entry_price) - 1
                            trades.append(Trade(
                                ticker=corr_conflict, entry_date=old_pos.entry_date,
                                exit_date=day, entry_price=old_pos.entry_price,
                                exit_price=old_price, return_pct=ret,
                                hold_days=days_held, reason="corr_swap",
                            ))
                        del positions[corr_conflict]
                        positions[ticker] = Position(
                            ticker=ticker, entry_date=day,
                            entry_price=current_price, score=score,
                            last_report_date=publish_date,
                            peak_price=current_price,
                        )
                    continue

                if len(positions) >= temp_engine.max_positions:
                    continue

                positions[ticker] = Position(
                    ticker=ticker, entry_date=day,
                    entry_price=current_price, score=score,
                    last_report_date=publish_date,
                    peak_price=current_price,
                )

        # Daily return (equal weight)
        held_tickers = [t for t in positions if t in price_matrix.columns]
        if held_tickers:
            prev_day_idx = price_matrix.index.get_loc(day)
            if prev_day_idx > 0:
                prev_day = price_matrix.index[prev_day_idx - 1]
                day_rets = []
                for t in held_tickers:
                    p0 = price_matrix.loc[prev_day, t]
                    p1 = price_matrix.loc[day, t]
                    if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                        r = (p1 / p0) - 1
                        if abs(r) < 1.0:
                            day_rets.append(r)
                port_ret = np.mean(day_rets) if day_rets else 0.0
            else:
                port_ret = 0.0
        else:
            port_ret = (1 + 0.04) ** (1/252) - 1

        daily_returns.append({"date": day, "return": port_ret, "n_positions": len(positions)})

        month_key = (day.year, day.month)
        if month_key != last_log_month:
            n_new = len(new_reports_today)
            held = ", ".join(sorted(positions.keys())[:8])
            more = f" +{len(positions)-8}" if len(positions) > 8 else ""
            if n_new > 0 or day.day <= 3:
                print(f"  {day.strftime('%Y-%m-%d')}: {len(positions)} positions | {held}{more}")
            last_log_month = month_key

    ret_df = pd.DataFrame(daily_returns).set_index("date")
    portfolio_returns = ret_df["return"]

    bench_returns = None
    if "SPY" in price_matrix.columns:
        bench_daily = price_matrix["SPY"].pct_change().clip(-1, 1).fillna(0)
        bench_returns = bench_daily.reindex(portfolio_returns.index).fillna(0)

    return portfolio_returns, bench_returns, trades, ret_df["n_positions"]


def main():
    combos = build_filter_combos()
    folds = build_folds()

    print("=" * 80)
    print("WALK-FORWARD: SCREENER FILTER OPTIMIZATION")
    print("=" * 80)
    print(f"Filter combos: {len(combos)}")
    print(f"Folds: {len(folds)}")
    print(f"Trading params (fixed): buy=40 sell=15 hold=30-540d pos=15")
    print(f"Fixed: correlation=0.65, sector_cap=0.40")
    print()
    for k, v in FILTER_GRID.items():
        print(f"  {k}: {v}")
    print()

    t0 = time.time()

    # Phase 1: Train
    fold_params = []
    for fold_idx, fold in enumerate(folds):
        fold_t0 = time.time()
        print(f"Training fold {fold_idx+1}/{len(folds)}: {fold['train_start']}..{fold['train_end']}")

        best_combo, best_sortino, best_result = train_fold(
            combos, fold["train_start"], fold["train_end"]
        )

        fold_time = time.time() - fold_t0
        print(f"  Best: roic={best_combo['min_roic']} pio={best_combo['min_piotroski']} "
              f"gm={best_combo['min_gross_margin']} | "
              f"Sortino={best_sortino:.2f} CAGR={best_result['cagr']:.2%} ({fold_time/60:.1f}m)")

        fold_params.append((fold["test_start"], fold["test_end"], best_combo))

    train_time = time.time() - t0
    print(f"\nTraining complete in {train_time/60:.1f} minutes")

    # Phase 2: Continuous OOS
    print(f"\n{'='*80}")
    print("CONTINUOUS OOS SIMULATION")
    print("=" * 80)

    oos_returns, bench_returns, trades, positions_ts = run_continuous_oos(fold_params)

    report = full_report(oos_returns, bench_returns)
    spy_report = full_report(bench_returns) if bench_returns is not None else {}

    print(f"\n{'='*80}")
    print("WALK-FORWARD OOS: FILTER OPTIMIZATION")
    print("=" * 80)
    print(f"  Period:      {oos_returns.index[0].strftime('%Y-%m-%d')} to {oos_returns.index[-1].strftime('%Y-%m-%d')}")
    print(f"  CAGR:        {report['cagr']:.2%}")
    print(f"  Sharpe:      {report['sharpe']:.2f}")
    print(f"  Sortino:     {report['sortino']:.2f}")
    print(f"  Max DD:      {report['max_drawdown']:.2%}")
    print(f"  Volatility:  {report['volatility']:.2%}")
    print(f"  vs SPY:      {report.get('excess_return', 0):.2%}")

    print(f"\n  SPY same period:")
    print(f"  CAGR:        {spy_report.get('cagr', 0):.2%}")
    print(f"  Sharpe:      {spy_report.get('sharpe', 0):.2f}")
    print(f"  Max DD:      {spy_report.get('max_drawdown', 0):.2%}")

    if trades:
        n_trades = len(trades)
        n_swaps = sum(1 for t in trades if t.reason == "corr_swap")
        win_rate = sum(1 for t in trades if t.return_pct > 0) / max(n_trades, 1)
        avg_hold = sum(t.hold_days for t in trades) / max(n_trades, 1)
        print(f"\n  Trades:      {n_trades} ({n_swaps} swaps)")
        print(f"  Win rate:    {win_rate:.0%}")
        print(f"  Avg hold:    {avg_hold:.0f} days")

    # Yearly breakdown
    print(f"\n{'='*80}")
    print("YEARLY OOS BREAKDOWN")
    print("=" * 80)
    print(f"  {'Year':>4} {'Filters':>30} | {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>7} {'vs SPY':>7}")
    print("-" * 80)

    for test_start, test_end, filters in fold_params:
        year_mask = (oos_returns.index >= test_start) & (oos_returns.index <= test_end)
        year_ret = oos_returns[year_mask]
        if len(year_ret) == 0:
            continue
        year_report = full_report(year_ret)
        bench_year = bench_returns[year_mask] if bench_returns is not None else None
        spy_year_report = full_report(bench_year) if bench_year is not None else {}
        year_excess = year_report.get("cagr", 0) - spy_year_report.get("cagr", 0)
        filters_str = f"roic={filters['min_roic']} pio={filters['min_piotroski']} gm={filters['min_gross_margin']}"
        year = test_start[:4]
        print(f"  {year}  {filters_str:>30} | "
              f"{year_report['cagr']:7.2%} {year_report.get('sharpe',0):7.2f} "
              f"{year_report['max_drawdown']:7.2%} {year_excess:+7.2%}")

    # Filter stability
    print(f"\n{'='*80}")
    print("FILTER STABILITY")
    print("=" * 80)
    for key in FILTER_GRID:
        counts = {}
        for _, _, filters in fold_params:
            val = filters[key]
            counts[val] = counts.get(val, 0) + 1
        print(f"  {key}:")
        for val in sorted(counts.keys()):
            bar = "#" * counts[val]
            print(f"    {val:>6}: {bar} ({counts[val]}/{len(fold_params)})")

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time/60:.1f} minutes")

    oos_returns.to_csv("data/walkforward_filters_oos.csv")
    print("Saved to data/walkforward_filters_oos.csv")


if __name__ == "__main__":
    main()
