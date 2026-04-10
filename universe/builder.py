"""Universe builder for survivorship-bias-free backtesting."""

import pandas as pd
from data.fetcher import get_sp500_tickers, get_prices


def current_sp500():
    return get_sp500_tickers()


def filter_tradeable_at_date(tickers, date):
    """Filter to tickers that had trading data at a given date."""
    tradeable = []
    check_start = (pd.Timestamp(date) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    check_end = (pd.Timestamp(date) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    for ticker in tickers:
        try:
            prices = get_prices(ticker, start=check_start, end=check_end)
            if not prices.empty and len(prices) > 5:
                tradeable.append(ticker)
        except Exception:
            continue
    return tradeable


def build_universe_timeline(tickers, start_date, end_date, freq="YS"):
    """Build a date -> tradeable tickers mapping (the "time machine")."""
    dates = pd.date_range(start_date, end_date, freq=freq)
    timeline = {}
    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        print(f"Building universe for {date_str}...")
        tradeable = filter_tradeable_at_date(tickers, date_str)
        timeline[date_str] = tradeable
        print(f"  {len(tradeable)} tradeable stocks")
    return timeline
