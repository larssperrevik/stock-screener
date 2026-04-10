"""Universe builder for survivorship-bias-free backtesting.

FMP provides historical S&P 500 additions/removals, which lets us
reconstruct who was in the index at any point in time.
"""

import pandas as pd
from data.fetcher import get_sp500_tickers, get_historical_sp500, get_prices


def current_sp500():
    return get_sp500_tickers()


def build_historical_sp500_universe(start_date, end_date):
    """Reconstruct S&P 500 membership at each year using FMP historical data.

    Returns a dict mapping date strings to lists of tickers that were
    in the S&P 500 at that time. This eliminates survivorship bias.
    """
    current = set(get_sp500_tickers())
    changes = get_historical_sp500()

    if changes.empty:
        print("Warning: no historical S&P 500 data, falling back to current constituents")
        dates = pd.date_range(start_date, end_date, freq="YS")
        return {d.strftime("%Y-%m-%d"): sorted(current) for d in dates}

    changes["dateAdded"] = pd.to_datetime(changes.get("dateAdded"), errors="coerce")
    changes["dateRemoved"] = pd.to_datetime(changes.get("date"), errors="coerce")

    dates = pd.date_range(start_date, end_date, freq="YS")
    timeline = {}

    for date in dates:
        # Start with current members
        members = set(current)

        # Reverse changes that happened after this date:
        # - If a stock was added after this date, remove it
        # - If a stock was removed after this date, add it back
        for _, row in changes.iterrows():
            symbol = row.get("symbol")
            if not symbol:
                continue
            added = row.get("dateAdded")
            removed = row.get("dateRemoved")

            if pd.notna(added) and added > date:
                members.discard(symbol)
            if pd.notna(removed) and removed > date:
                removed_ticker = row.get("removedTicker", symbol)
                if removed_ticker:
                    members.add(removed_ticker)

        timeline[date.strftime("%Y-%m-%d")] = sorted(members)
        print(f"  {date.strftime('%Y-%m-%d')}: {len(members)} members")

    return timeline


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
