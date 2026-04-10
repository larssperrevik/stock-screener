"""Load SimFin bulk CSV data into pandas DataFrames.

This replaces the FMP API entirely for backtesting. All data is local,
includes delisted companies, and has publish dates for point-in-time queries.
"""

import pandas as pd
from pathlib import Path
from functools import lru_cache

SIMFIN_DIR = Path(__file__).parent / "simfin"


@lru_cache(maxsize=1)
def load_companies():
    df = pd.read_csv(SIMFIN_DIR / "us-companies.csv", sep=";")
    return df


@lru_cache(maxsize=1)
def load_industries():
    df = pd.read_csv(SIMFIN_DIR / "industries.csv", sep=";")
    return df


@lru_cache(maxsize=1)
def load_derived_annual():
    df = pd.read_csv(SIMFIN_DIR / "us-derived-annual.csv", sep=";",
                      parse_dates=["Report Date", "Publish Date", "Restated Date"])
    df["Fiscal Year"] = df["Fiscal Year"].astype(int)
    return df


@lru_cache(maxsize=1)
def load_prices():
    """Load daily price data. ~3.4GB CSV, takes a moment."""
    print("Loading price data (this may take a minute on first load)...")
    df = pd.read_csv(SIMFIN_DIR / "us-derived-shareprices-daily.csv", sep=";",
                      parse_dates=["Date"],
                      dtype={"Ticker": str, "SimFinId": int})
    print(f"  Loaded {len(df):,} price records for {df['Ticker'].nunique():,} tickers")
    return df


@lru_cache(maxsize=1)
def load_income_annual():
    return pd.read_csv(SIMFIN_DIR / "us-income-annual-full.csv", sep=";",
                        parse_dates=["Report Date", "Publish Date", "Restated Date"])


@lru_cache(maxsize=1)
def load_balance_annual():
    return pd.read_csv(SIMFIN_DIR / "us-balance-annual-full.csv", sep=";",
                        parse_dates=["Report Date", "Publish Date", "Restated Date"])


@lru_cache(maxsize=1)
def load_cashflow_annual():
    return pd.read_csv(SIMFIN_DIR / "us-cashflow-annual-full.csv", sep=";",
                        parse_dates=["Report Date", "Publish Date", "Restated Date"])


def get_company_info(ticker):
    """Get company metadata."""
    companies = load_companies()
    industries = load_industries()
    row = companies[companies["Ticker"] == ticker]
    if row.empty:
        return None
    info = row.iloc[0].to_dict()
    if pd.notna(info.get("IndustryId")):
        ind = industries[industries["IndustryId"] == info["IndustryId"]]
        if not ind.empty:
            info["Sector"] = ind.iloc[0].get("Sector", "")
            info["Industry"] = ind.iloc[0].get("Industry", "")
    return info


def get_fundamentals_at_date(ticker, date):
    """Get the most recent fundamentals known BEFORE a given date.

    Uses Publish Date to ensure point-in-time correctness (no look-ahead bias).
    """
    derived = load_derived_annual()
    mask = (derived["Ticker"] == ticker) & (derived["Publish Date"] <= date)
    rows = derived[mask].sort_values("Publish Date", ascending=False)
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def get_all_fundamentals_at_date(date):
    """Get the most recent fundamentals for ALL tickers known before a date.

    This is the core "time machine" function — returns only data that
    would have been publicly available at the given date.
    """
    derived = load_derived_annual()
    mask = derived["Publish Date"] <= date
    available = derived[mask].sort_values("Publish Date")
    # Keep only the most recent record per ticker
    latest = available.groupby("Ticker").last().reset_index()
    return latest


def get_prices_for_ticker(ticker, start=None, end=None):
    """Get daily prices for a single ticker."""
    prices = load_prices()
    mask = prices["Ticker"] == ticker
    if start:
        mask &= prices["Date"] >= start
    if end:
        mask &= prices["Date"] <= end
    return prices[mask].set_index("Date").sort_index()


def get_tradeable_tickers_at_date(date):
    """Get all tickers that had trading activity around a date.

    This handles survivorship bias — delisted companies will have price
    data up until their delisting, so they'll be included for historical dates.
    """
    prices = load_prices()
    window_start = pd.Timestamp(date) - pd.Timedelta(days=10)
    window_end = pd.Timestamp(date) + pd.Timedelta(days=5)
    mask = (prices["Date"] >= window_start) & (prices["Date"] <= window_end)
    return sorted(prices[mask]["Ticker"].unique().tolist())


def get_sp500_tickers():
    """Get S&P 500 tickers from local file (for current screening)."""
    import json
    ticker_file = Path(__file__).parent / "sp500_tickers.json"
    with open(ticker_file) as f:
        return json.load(f)
