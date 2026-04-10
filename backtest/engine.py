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
        rebalance_freq="annual",   # annual, quarterly, monthly
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

        # Pre-load price data once
        all_prices = load_prices()

        # Build a pivot table of adjusted close prices
        print("Building price matrix...")
        price_matrix = all_prices.pivot_table(
            index="Date", columns="Ticker", values="Adj. Close", aggfunc="first"
        )
        daily_returns = price_matrix.pct_change()

        holdings_history = []

        print(f"\nRunning screen at {len(rebalance_dates)} rebalance dates...")
        for date in rebalance_dates:
            date_str = date.strftime("%Y-%m-%d")

            # Get tradeable tickers at this date (survivorship-bias free)
            tradeable = get_tradeable_tickers_at_date(date)

            # Get point-in-time fundamentals
            fundamentals = get_all_fundamentals_at_date(date)

            # Filter to tradeable tickers only
            fundamentals = fundamentals[fundamentals["Ticker"].isin(tradeable)]

            # Apply the screen
            screened = apply_screen(fundamentals, self.criteria)

            if screened.empty:
                holdings_history.append({"date": date, "tickers": []})
                print(f"  {date_str}: 0 stocks (from {len(tradeable)} tradeable)")
                continue

            top = screened.head(self.top_n)
            tickers = top["Ticker"].tolist()
            holdings_history.append({"date": date, "tickers": tickers})
            print(f"  {date_str}: {len(tickers)} stocks selected (from {len(tradeable)} tradeable, {len(screened)} passed screen)")

        # Build portfolio returns
        print("\nComputing portfolio returns...")
        portfolio_returns = pd.Series(0.0, index=daily_returns.index, dtype=float)

        for i, entry in enumerate(holdings_history):
            start = entry["date"]
            end = holdings_history[i + 1]["date"] if i + 1 < len(holdings_history) else self.end_date

            mask = (daily_returns.index >= start) & (daily_returns.index < end)
            tickers = [t for t in entry["tickers"] if t in daily_returns.columns]

            if tickers:
                # Equal weight
                period_rets = daily_returns.loc[mask, tickers]
                portfolio_returns.loc[mask] = period_rets.mean(axis=1)

        # Trim to date range
        portfolio_returns = portfolio_returns.loc[
            (portfolio_returns.index >= self.start_date) &
            (portfolio_returns.index <= self.end_date)
        ].dropna()

        # Benchmark
        bench_returns = None
        if self.benchmark in daily_returns.columns:
            bench_returns = daily_returns[self.benchmark].loc[portfolio_returns.index]

        report = full_report(portfolio_returns, bench_returns)
        print_report(report)

        return {
            "portfolio_returns": portfolio_returns,
            "benchmark_returns": bench_returns,
            "holdings_history": holdings_history,
            "report": report,
        }
