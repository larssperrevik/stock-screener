"""Backtesting engine using SimFin local data.

No API calls needed. Uses point-in-time fundamentals (Publish Date)
and includes delisted companies for survivorship-bias-free backtesting.
"""

import pandas as pd
import numpy as np
from data.simfin_loader import (
    load_prices, get_all_fundamentals_at_date, get_tradeable_tickers_at_date
)
from screener.criteria import ScreenCriteria, apply_screen
from metrics.performance import full_report, print_report


class BacktestEngine:
    def __init__(
        self,
        criteria=None,
        rebalance_freq="annual",
        top_n=20,
        start_date="2005-01-01",
        end_date="2024-09-30",
        benchmark="SPY",
    ):
        self.criteria = criteria or ScreenCriteria()
        self.rebalance_freq = rebalance_freq
        self.top_n = top_n
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.benchmark = benchmark

    def _rebalance_dates(self):
        freq_map = {"annual": "YS", "quarterly": "QS", "monthly": "MS"}
        freq = freq_map.get(self.rebalance_freq, "YS")
        return list(pd.date_range(self.start_date, self.end_date, freq=freq))

    def run(self):
        """Run the backtest. All data is local — no API calls."""
        rebalance_dates = self._rebalance_dates()

        all_prices = load_prices()

        # Build price matrix using Adj. Close
        print("Building price matrix...")
        price_matrix = all_prices.pivot_table(
            index="Date", columns="Ticker", values="Adj. Close", aggfunc="first"
        )

        # Compute returns only where we have consecutive trading days
        # (forward-fill up to 5 days for weekends/holidays, then compute returns)
        price_filled = price_matrix.ffill(limit=5)
        daily_returns = price_filled.pct_change()

        # Clip extreme returns — anything beyond +/- 100% in a day is data noise
        daily_returns = daily_returns.clip(-1.0, 1.0)

        # Replace inf/nan with 0
        daily_returns = daily_returns.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        holdings_history = []

        print(f"\nRunning screen at {len(rebalance_dates)} rebalance dates...")
        for date in rebalance_dates:
            date_str = date.strftime("%Y-%m-%d")

            tradeable = get_tradeable_tickers_at_date(date)
            fundamentals = get_all_fundamentals_at_date(date)
            fundamentals = fundamentals[fundamentals["Ticker"].isin(tradeable)]
            screened = apply_screen(fundamentals, self.criteria)

            if screened.empty:
                holdings_history.append({"date": date, "tickers": []})
                print(f"  {date_str}: 0 stocks")
                continue

            top = screened.head(self.top_n)
            tickers = top["Ticker"].tolist()
            holdings_history.append({"date": date, "tickers": tickers})
            print(f"  {date_str}: {len(tickers)} stocks (from {len(screened)} passed)")

        # Build portfolio returns
        print("\nComputing portfolio returns...")
        portfolio_returns = pd.Series(0.0, index=daily_returns.index, dtype=float)

        for i, entry in enumerate(holdings_history):
            start = entry["date"]
            end = holdings_history[i + 1]["date"] if i + 1 < len(holdings_history) else self.end_date

            mask = (daily_returns.index >= start) & (daily_returns.index < end)
            tickers = [t for t in entry["tickers"] if t in daily_returns.columns]

            if tickers:
                period_rets = daily_returns.loc[mask, tickers]
                portfolio_returns.loc[mask] = period_rets.mean(axis=1)

        # Trim to date range
        portfolio_returns = portfolio_returns.loc[
            (portfolio_returns.index >= self.start_date) &
            (portfolio_returns.index <= self.end_date)
        ]
        portfolio_returns = portfolio_returns[portfolio_returns.index.isin(daily_returns.index)]

        # Benchmark
        bench_returns = None
        if self.benchmark in daily_returns.columns:
            bench_returns = daily_returns[self.benchmark].reindex(portfolio_returns.index).fillna(0.0)

        report = full_report(portfolio_returns, bench_returns)
        print_report(report)

        # Print holdings for inspection
        print("\nHoldings at each rebalance:")
        for entry in holdings_history:
            d = entry["date"].strftime("%Y-%m-%d")
            t = ", ".join(entry["tickers"][:10])
            more = f" +{len(entry['tickers'])-10} more" if len(entry["tickers"]) > 10 else ""
            print(f"  {d}: {t}{more}")

        return {
            "portfolio_returns": portfolio_returns,
            "benchmark_returns": bench_returns,
            "holdings_history": holdings_history,
            "report": report,
        }
