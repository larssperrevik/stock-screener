"""Backtesting engine with survivorship-bias awareness."""

import pandas as pd
from datetime import datetime
from data.fetcher import get_prices
from metrics.performance import full_report, print_report


class BacktestEngine:
    def __init__(self, screen_fn, rebalance_freq="annual", top_n=20,
                 start_date="2010-01-01", end_date=None, benchmark="SPY",
                 initial_capital=100000):
        self.screen_fn = screen_fn
        self.rebalance_freq = rebalance_freq
        self.top_n = top_n
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date or datetime.now().strftime("%Y-%m-%d"))
        self.benchmark = benchmark
        self.initial_capital = initial_capital

    def _rebalance_dates(self):
        freq_map = {"annual": "YS", "quarterly": "QS", "monthly": "MS"}
        freq = freq_map.get(self.rebalance_freq, "YS")
        return list(pd.date_range(self.start_date, self.end_date, freq=freq))

    def _get_price_data(self, tickers):
        all_prices = {}
        for ticker in tickers:
            try:
                prices = get_prices(ticker, start=self.start_date.strftime("%Y-%m-%d"),
                                    end=self.end_date.strftime("%Y-%m-%d"))
                if not prices.empty:
                    if isinstance(prices.columns, pd.MultiIndex):
                        close = prices["Close"].iloc[:, 0]
                    else:
                        close = prices["Close"]
                    all_prices[ticker] = close
            except Exception:
                continue
        return pd.DataFrame(all_prices)

    def run(self, universe_by_date=None):
        """Run the backtest.

        universe_by_date: dict mapping date strings to tradeable ticker lists.
        This is the "time machine" for survivorship-bias-free backtesting.
        """
        rebalance_dates = self._rebalance_dates()
        all_tickers = set()
        holdings_history = []

        print(f"Running screen at {len(rebalance_dates)} rebalance dates...")
        for date in rebalance_dates:
            date_str = date.strftime("%Y-%m-%d")
            if universe_by_date:
                universe = universe_by_date.get(date_str, [])
            else:
                universe = None

            try:
                screened = self.screen_fn(date=date_str, universe=universe)
                if screened.empty:
                    holdings_history.append({"date": date, "tickers": []})
                    continue
                top = screened.head(self.top_n)
                tickers = top["ticker"].tolist()
                all_tickers.update(tickers)
                holdings_history.append({"date": date, "tickers": tickers})
                print(f"  {date_str}: {len(tickers)} stocks selected")
            except Exception as e:
                print(f"  {date_str}: screen failed -- {e}")
                holdings_history.append({"date": date, "tickers": []})

        if not all_tickers:
            print("No stocks selected at any rebalance date.")
            return {}

        print(f"\nFetching prices for {len(all_tickers)} unique tickers...")
        all_tickers.add(self.benchmark)
        prices = self._get_price_data(list(all_tickers))

        if prices.empty:
            print("No price data available.")
            return {}

        daily_returns = prices.pct_change().dropna()
        portfolio_returns = pd.Series(0.0, index=daily_returns.index, dtype=float)

        for i, entry in enumerate(holdings_history):
            start = entry["date"]
            end = holdings_history[i + 1]["date"] if i + 1 < len(holdings_history) else self.end_date
            mask = (daily_returns.index >= start) & (daily_returns.index < end)
            period_returns = daily_returns.loc[mask]
            tickers = [t for t in entry["tickers"] if t in daily_returns.columns]
            if tickers:
                portfolio_returns.loc[mask] = period_returns[tickers].mean(axis=1)

        portfolio_returns = portfolio_returns.loc[portfolio_returns.index.isin(daily_returns.index)]

        bench_returns = None
        if self.benchmark in daily_returns.columns:
            bench_returns = daily_returns[self.benchmark]
            bench_returns = bench_returns.loc[bench_returns.index.isin(portfolio_returns.index)]

        report = full_report(portfolio_returns, bench_returns)
        print_report(report)

        return {
            "portfolio_returns": portfolio_returns,
            "benchmark_returns": bench_returns,
            "holdings_history": holdings_history,
            "prices": prices,
            "report": report,
        }
