"""Event-driven backtesting engine.

Simulates real-life trading: walk through every trading day, react to
new financial reports as they're published, and make buy/sell decisions
based on fresh data — not arbitrary calendar dates.

Design:
- Each trading day, check if new annual reports were published
- When new data arrives for a stock, recalculate its score
- BUY: passes screener + score above threshold
- SELL: fails screener, score drops, or max hold period reached
- Equal weight positions, max N holdings
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from data.simfin_loader import load_derived_annual, load_prices, load_companies, load_industries
from screener.criteria import ScreenCriteria, apply_screen
from metrics.performance import full_report, print_report


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    score: float
    last_report_date: pd.Timestamp


@dataclass
class Trade:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    return_pct: float
    hold_days: int
    reason: str


def compute_stock_score(ticker, derived_row, price_history, all_derived_ticker):
    """Score a stock based on quality, valuation-vs-history, and freshness.

    No momentum. Focus on what matters for quality investing:
    - How good is the business? (quality)
    - Is it cheap relative to itself? (valuation vs own history)
    - Is the moat stable/improving? (durability)
    - How fresh is the data? (information edge)
    """
    score = 0.0
    details = {}

    # === QUALITY (0-30 points) ===
    piotroski = derived_row.get("Piotroski F-Score", 0) or 0
    score += min(piotroski, 9) * 2  # 0-18 points
    details["piotroski"] = piotroski

    roic = derived_row.get("Return On Invested Capital", 0) or 0
    if roic > 0.25:
        score += 6
    elif roic > 0.15:
        score += 4
    elif roic > 0.10:
        score += 2
    details["roic"] = roic

    fcf_ni = derived_row.get("Free Cash Flow to Net Income", 0) or 0
    if fcf_ni > 1.0:
        score += 4
    elif fcf_ni > 0.8:
        score += 2
    details["fcf_conversion"] = fcf_ni

    gm = derived_row.get("Gross Profit Margin", 0) or 0
    if gm > 0.60:
        score += 2
    details["gross_margin"] = gm

    # === MOAT DURABILITY (0-20 points) ===
    if len(all_derived_ticker) >= 3:
        roic_series = all_derived_ticker["Return On Invested Capital"].dropna().tail(5)
        if len(roic_series) >= 3:
            roic_consistency = (roic_series > 0.10).mean()
            score += roic_consistency * 8  # 0-8 points
            details["roic_consistency"] = roic_consistency

            roic_trend = _slope(roic_series.values)
            if roic_trend > 0:
                score += 3
            details["roic_trend"] = roic_trend

        gm_series = all_derived_ticker["Gross Profit Margin"].dropna().tail(5)
        if len(gm_series) >= 3:
            gm_stability = 1.0 - min(gm_series.std() / max(gm_series.mean(), 0.01), 1.0)
            score += gm_stability * 5  # 0-5 points
            details["gm_stability"] = gm_stability

        # Improving or stable margins
        om_series = all_derived_ticker["Operating Margin"].dropna().tail(5)
        if len(om_series) >= 3:
            om_trend = _slope(om_series.values)
            if om_trend > 0:
                score += 4
            elif om_trend > -0.05:
                score += 2  # stable is ok
            details["om_trend"] = om_trend

    # === VALUATION VS OWN HISTORY (0-25 points) ===
    # Is this stock cheap relative to its own 5-year average?
    if price_history is not None and len(price_history) > 0:
        for ratio_col, weight in [
            ("Price to Earnings Ratio (ttm)", 8),
            ("Price to Free Cash Flow (ttm)", 8),
            ("EV/EBITDA", 5),
            ("Price to Book Value", 4),
        ]:
            if ratio_col in price_history.columns:
                vals = price_history[ratio_col].dropna()
                if len(vals) > 252:  # need ~1 year of data
                    current = vals.iloc[-1]
                    hist_median = vals.tail(252 * 3).median()  # 3-year median
                    if pd.notna(current) and pd.notna(hist_median) and hist_median > 0 and current > 0:
                        discount = 1.0 - (current / hist_median)
                        # Capped: max +weight if 50%+ discount, 0 if at median, negative if expensive
                        val_score = max(-weight, min(weight, discount * weight * 2))
                        score += val_score
                        details[f"val_{ratio_col[:10]}"] = discount

    # === MANAGEMENT (0-10 points) ===
    if len(all_derived_ticker) >= 2:
        # Share buyback
        eqps = all_derived_ticker["Equity Per Share"].dropna().tail(3)
        if len(eqps) >= 2:
            eq_growth = (eqps.iloc[-1] / eqps.iloc[0]) - 1
            if eq_growth > 0.1:  # growing equity per share
                score += 3
            details["eq_per_share_growth"] = eq_growth

        # Debt discipline
        debt = all_derived_ticker["Debt Ratio"].dropna().tail(3)
        if len(debt) >= 2:
            debt_change = debt.iloc[-1] - debt.iloc[0]
            if debt_change < -0.05:  # reducing debt
                score += 4
            elif debt_change < 0.02:  # stable
                score += 2
            details["debt_change"] = debt_change

        # Dividend discipline
        dpr = all_derived_ticker["Dividend Payout Ratio"].dropna()
        if len(dpr) >= 1:
            if 0 < dpr.iloc[-1] < 0.6:
                score += 3  # moderate payout = disciplined
            details["payout_ratio"] = dpr.iloc[-1]

    # === DATA FRESHNESS (0-5 points) ===
    # Handled externally (days since publish)

    details["total_score"] = score
    return score, details


def _slope(arr):
    y = np.array(arr, dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return 0.0
    y = y[mask]
    x = np.arange(len(y), dtype=float)
    return np.polyfit(x, y, 1)[0]


class EventDrivenEngine:
    def __init__(
        self,
        criteria=None,
        max_positions=20,
        max_hold_days=365,       # Re-evaluate after 1 year
        min_hold_days=60,        # Don't churn: hold at least 2 months
        buy_threshold=50.0,      # Minimum score to buy
        sell_threshold=30.0,     # Sell if score drops below this
        freshness_days=120,      # Only score stocks with data < 120 days old
        start_date="2011-01-01",
        end_date="2024-09-30",
        benchmark="SPY",
    ):
        self.criteria = criteria or ScreenCriteria()
        self.max_positions = max_positions
        self.max_hold_days = max_hold_days
        self.min_hold_days = min_hold_days
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.freshness_days = freshness_days
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.benchmark = benchmark

    def run(self, quiet=False):
        """Run the event-driven backtest."""
        derived = load_derived_annual()
        prices = load_prices()

        if not quiet:
            print("Building price matrix...")
        price_matrix = prices.pivot_table(
            index="Date", columns="Ticker", values="Adj. Close", aggfunc="first"
        ).ffill(limit=5)

        # Also keep the raw prices for valuation ratios
        if not quiet:
            print("Indexing price data by ticker...")
        price_by_ticker = {}
        for ticker, group in prices.groupby("Ticker"):
            price_by_ticker[ticker] = group.set_index("Date").sort_index()

        # Index derived data by ticker (exclude delisted/acquired)
        derived_by_ticker = {}
        for ticker, group in derived.groupby("Ticker"):
            if "_delisted" in ticker or "_old" in ticker:
                continue
            derived_by_ticker[ticker] = group.sort_values("Publish Date")

        # Track which reports we've already seen (to detect new ones)
        seen_reports = set()  # (ticker, publish_date) tuples

        # Pre-populate seen reports before start date
        for _, row in derived[derived["Publish Date"] < self.start_date].iterrows():
            seen_reports.add((row["Ticker"], row["Publish Date"]))

        # State
        positions: dict[str, Position] = {}
        trades: list[Trade] = []
        daily_returns = []

        trading_days = price_matrix.index[
            (price_matrix.index >= self.start_date) &
            (price_matrix.index <= self.end_date)
        ]

        if not quiet:
            print(f"\nSimulating {len(trading_days)} trading days from "
                  f"{self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
            print(f"  Max positions: {self.max_positions}, Hold: {self.min_hold_days}-{self.max_hold_days} days")
            print(f"  Buy threshold: {self.buy_threshold}, Sell threshold: {self.sell_threshold}")
            print()

        last_log_month = None
        new_reports_today = []

        for day in trading_days:
            # === CHECK FOR NEW REPORTS ===
            new_reports_today = []
            day_reports = derived[
                (derived["Publish Date"] >= day - pd.Timedelta(days=0)) &
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

            # === PROCESS NEW REPORTS: SCORE AND DECIDE ===
            tickers_to_evaluate = set()

            # New reports trigger evaluation
            for report in new_reports_today:
                tickers_to_evaluate.add(report["Ticker"])

            # Also evaluate held positions approaching max hold
            for ticker, pos in list(positions.items()):
                days_held = (day - pos.entry_date).days
                if days_held >= self.max_hold_days:
                    tickers_to_evaluate.add(ticker)

            # === EVALUATE AND TRADE ===
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

                # Skip if data is too stale
                if days_since_publish > self.freshness_days:
                    continue

                # Check if it passes screener
                latest_dict = latest.to_dict()
                passes_screen = self._passes_screen(latest_dict)

                # Get price history for valuation-vs-history
                ticker_prices = price_by_ticker.get(ticker)
                if ticker_prices is not None:
                    hist = ticker_prices[ticker_prices.index <= day]
                else:
                    hist = None

                # Score
                score, details = compute_stock_score(
                    ticker, latest_dict, hist, available
                )

                # Freshness bonus
                if days_since_publish < 30:
                    score += 5
                elif days_since_publish < 60:
                    score += 3

                # === SELL LOGIC ===
                if ticker in positions:
                    pos = positions[ticker]
                    days_held = (day - pos.entry_date).days

                    should_sell = False
                    reason = ""

                    if not passes_screen:
                        should_sell = True
                        reason = "failed_screen"
                    elif score < self.sell_threshold and days_held >= self.min_hold_days:
                        should_sell = True
                        reason = f"low_score_{score:.0f}"
                    elif days_held >= self.max_hold_days:
                        if score >= self.buy_threshold and passes_screen:
                            # Renew the position
                            positions[ticker] = Position(
                                ticker=ticker, entry_date=day,
                                entry_price=pos.entry_price,
                                score=score, last_report_date=publish_date,
                            )
                            continue
                        else:
                            should_sell = True
                            reason = "max_hold"

                    if should_sell and days_held >= self.min_hold_days:
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

                # === BUY LOGIC ===
                elif (passes_screen
                      and score >= self.buy_threshold
                      and len(positions) < self.max_positions
                      and ticker in price_matrix.columns):

                    current_price = price_matrix.loc[day, ticker]
                    if pd.notna(current_price) and current_price > 0:
                        positions[ticker] = Position(
                            ticker=ticker, entry_date=day,
                            entry_price=current_price, score=score,
                            last_report_date=publish_date,
                        )

            # === COMPUTE DAILY PORTFOLIO RETURN ===
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
                            if abs(r) < 1.0:  # filter extreme
                                day_rets.append(r)
                    port_ret = np.mean(day_rets) if day_rets else 0.0
                else:
                    port_ret = 0.0
            else:
                # Cash: risk-free rate
                port_ret = (1 + 0.04) ** (1/252) - 1

            daily_returns.append({"date": day, "return": port_ret, "n_positions": len(positions)})

            # Monthly logging
            if not quiet:
                month_key = (day.year, day.month)
                if month_key != last_log_month:
                    n_new = len(new_reports_today)
                    held = ", ".join(sorted(positions.keys())[:8])
                    more = f" +{len(positions)-8}" if len(positions) > 8 else ""
                    if n_new > 0 or day.day <= 3:
                        print(f"  {day.strftime('%Y-%m-%d')}: {len(positions)} positions, "
                              f"{n_new} new reports | {held}{more}")
                    last_log_month = month_key

        # === RESULTS ===
        ret_df = pd.DataFrame(daily_returns).set_index("date")
        portfolio_returns = ret_df["return"]

        # Benchmark
        bench_returns = None
        if self.benchmark in price_matrix.columns:
            bench_daily = price_matrix[self.benchmark].pct_change().clip(-1, 1).fillna(0)
            bench_returns = bench_daily.reindex(portfolio_returns.index).fillna(0)

        report = full_report(portfolio_returns, bench_returns)
        if not quiet:
            print_report(report)

        # Trade statistics
        if trades:
            trade_df = pd.DataFrame([t.__dict__ for t in trades])
            avg_hold = trade_df["hold_days"].mean()
            win_rate = (trade_df["return_pct"] > 0).mean()
            avg_win = trade_df.loc[trade_df["return_pct"] > 0, "return_pct"].mean()
            avg_loss = trade_df.loc[trade_df["return_pct"] <= 0, "return_pct"].mean()

            if not quiet:
                print(f"\n  Total trades: {len(trades)}")
                print(f"  Avg hold period: {avg_hold:.0f} days")
                print(f"  Win rate: {win_rate:.0%}")
                print(f"  Avg win: {avg_win:+.1%}")
                print(f"  Avg loss: {avg_loss:+.1%}")
                print(f"  Payoff ratio: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "")

                print(f"\n  Sell reasons:")
                for reason, count in trade_df["reason"].value_counts().items():
                    avg_ret = trade_df.loc[trade_df["reason"] == reason, "return_pct"].mean()
                    print(f"    {reason}: {count} trades, avg return {avg_ret:+.1%}")

        avg_pos = ret_df["n_positions"].mean()
        max_pos = ret_df["n_positions"].max()
        pct_invested = (ret_df["n_positions"] > 0).mean()
        if not quiet:
            print(f"\n  Avg positions: {avg_pos:.1f}")
            print(f"  Max positions: {max_pos}")
            print(f"  Time invested: {pct_invested:.0%}")

        return {
            "portfolio_returns": portfolio_returns,
            "benchmark_returns": bench_returns,
            "trades": trades,
            "report": report,
            "positions_ts": ret_df["n_positions"],
        }

    def _passes_screen(self, row):
        """Quick check if a single row passes screener criteria."""
        c = self.criteria
        gm = row.get("Gross Profit Margin")
        om = row.get("Operating Margin")
        roe = row.get("Return on Equity")
        roa = row.get("Return on Assets")
        cr = row.get("Current Ratio")
        dr = row.get("Debt Ratio")
        pf = row.get("Piotroski F-Score")
        roic = row.get("Return On Invested Capital")
        fcf = row.get("Free Cash Flow to Net Income")

        if gm is not None and gm < c.min_gross_margin:
            return False
        if om is not None and om < c.min_operating_margin:
            return False
        if roe is not None and roe < c.min_roe:
            return False
        if roa is not None and roa < c.min_roa:
            return False
        if cr is not None and cr < c.min_current_ratio:
            return False
        if dr is not None and dr > c.max_debt_ratio:
            return False
        if pf is not None and pf < c.min_piotroski:
            return False
        if roic is not None and roic < c.min_roic:
            return False
        if fcf is not None and fcf < c.min_fcf_to_net_income:
            return False
        return True


def sweep_thresholds():
    """Run parameter sweep to find optimal thresholds."""
    from itertools import product

    buy_thresholds = [40, 45, 50, 55, 60]
    sell_thresholds = [20, 25, 30, 35]
    max_positions_list = [15, 20, 25]
    max_hold_list = [270, 365, 540]

    criteria = ScreenCriteria()
    results = []

    total = len(buy_thresholds) * len(sell_thresholds) * len(max_positions_list) * len(max_hold_list)
    i = 0

    for buy_t, sell_t, max_pos, max_hold in product(
        buy_thresholds, sell_thresholds, max_positions_list, max_hold_list
    ):
        if sell_t >= buy_t:
            continue
        i += 1
        print(f"\n[{i}/{total}] buy={buy_t} sell={sell_t} pos={max_pos} hold={max_hold}d", end=" ")

        engine = EventDrivenEngine(
            criteria=criteria,
            max_positions=max_pos,
            max_hold_days=max_hold,
            buy_threshold=buy_t,
            sell_threshold=sell_t,
        )
        try:
            result = engine.run(quiet=True)
            r = result["report"]
            n_trades = len(result["trades"])
            print(f"=> CAGR={r['cagr']:.2%} Sharpe={r['sharpe']:.2f} "
                  f"Sortino={r['sortino']:.2f} MaxDD={r['max_drawdown']:.2%} "
                  f"Trades={n_trades}")
            results.append({
                "buy_threshold": buy_t,
                "sell_threshold": sell_t,
                "max_positions": max_pos,
                "max_hold_days": max_hold,
                "cagr": r["cagr"],
                "sharpe": r["sharpe"],
                "sortino": r["sortino"],
                "max_drawdown": r["max_drawdown"],
                "volatility": r["volatility"],
                "excess_return": r.get("excess_return", 0),
                "n_trades": n_trades,
            })
        except Exception as e:
            print(f"=> FAILED: {e}")

    if not results:
        print("No results!")
        return

    df = pd.DataFrame(results)

    # Rank by composite: high sortino + low drawdown
    df["composite"] = df["sortino"] - df["max_drawdown"].abs() * 2
    df = df.sort_values("composite", ascending=False)

    print("\n\n" + "=" * 80)
    print("TOP 10 PARAMETER COMBINATIONS (by Sortino + drawdown penalty)")
    print("=" * 80)
    for _, row in df.head(10).iterrows():
        print(f"  buy={row['buy_threshold']:.0f} sell={row['sell_threshold']:.0f} "
              f"pos={row['max_positions']:.0f} hold={row['max_hold_days']:.0f}d | "
              f"CAGR={row['cagr']:.2%} Sharpe={row['sharpe']:.2f} "
              f"Sortino={row['sortino']:.2f} MaxDD={row['max_drawdown']:.2%} "
              f"Trades={row['n_trades']:.0f}")

    best = df.iloc[0]
    print(f"\nBest: buy={best['buy_threshold']:.0f} sell={best['sell_threshold']:.0f} "
          f"pos={best['max_positions']:.0f} hold={best['max_hold_days']:.0f}d")

    df.to_csv("data/sweep_results.csv", index=False)
    print("Saved to data/sweep_results.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Event-driven backtest")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--start", default="2011-01-01")
    parser.add_argument("--end", default="2024-09-30")
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--max-hold-days", type=int, default=365)
    parser.add_argument("--min-hold-days", type=int, default=60)
    parser.add_argument("--buy-threshold", type=float, default=50)
    parser.add_argument("--sell-threshold", type=float, default=30)
    parser.add_argument("--min-roe", type=float, default=0.15)
    parser.add_argument("--min-roic", type=float, default=0.10)
    parser.add_argument("--min-piotroski", type=int, default=5)
    args = parser.parse_args()

    if args.sweep:
        sweep_thresholds()
    else:
        criteria = ScreenCriteria(
            min_roe=args.min_roe, min_roic=args.min_roic, min_piotroski=args.min_piotroski,
        )
        engine = EventDrivenEngine(
            criteria=criteria,
            max_positions=args.max_positions,
            max_hold_days=args.max_hold_days,
            min_hold_days=args.min_hold_days,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
            start_date=args.start,
            end_date=args.end,
        )
        engine.run()
