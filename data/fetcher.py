"""Fetch fundamental and price data from Yahoo Finance."""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(ticker, data_type):
    return CACHE_DIR / f"{ticker}_{data_type}.parquet"


def get_fundamentals(ticker, use_cache=True):
    """Get key fundamental data for a ticker."""
    cache = CACHE_DIR / f"{ticker}_info.json"
    if use_cache and cache.exists():
        age = datetime.now().timestamp() - cache.stat().st_mtime
        if age < 86400:
            with open(cache) as f:
                return json.load(f)

    stock = yf.Ticker(ticker)
    info = stock.info

    fundamentals = {
        "ticker": ticker,
        "name": info.get("longName", ""),
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "market_cap": info.get("marketCap", 0),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "pb_ratio": info.get("priceToBook"),
        "ps_ratio": info.get("priceToSalesTrailing12Months"),
        "dividend_yield": info.get("dividendYield"),
        "roe": info.get("returnOnEquity"),
        "roa": info.get("returnOnAssets"),
        "debt_to_equity": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "free_cash_flow": info.get("freeCashflow"),
        "operating_cash_flow": info.get("operatingCashflow"),
        "total_revenue": info.get("totalRevenue"),
        "net_income": info.get("netIncomeToCommon"),
        "total_debt": info.get("totalDebt"),
        "total_cash": info.get("totalCash"),
        "earnings_growth": info.get("earningsGrowth"),
        "revenue_growth": info.get("revenueGrowth"),
        "gross_margins": info.get("grossMargins"),
        "operating_margins": info.get("operatingMargins"),
        "profit_margins": info.get("profitMargins"),
        "enterprise_value": info.get("enterpriseValue"),
        "ebitda": info.get("ebitda"),
    }

    with open(cache, "w") as f:
        json.dump(fundamentals, f)

    return fundamentals


def get_financials(ticker):
    """Get income statement, balance sheet, cash flow."""
    stock = yf.Ticker(ticker)
    return {
        "income": stock.financials,
        "balance": stock.balance_sheet,
        "cashflow": stock.cashflow,
    }


def get_prices(ticker, start="2000-01-01", end=None):
    """Get historical daily prices."""
    cache = _cache_path(ticker, "prices")
    if cache.exists():
        df = pd.read_parquet(cache)
        last_date = df.index.max()
        if pd.Timestamp.now() - last_date < timedelta(days=1):
            return df

    end = end or datetime.now().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if not df.empty:
        df.to_parquet(cache)
    return df


def get_sp500_tickers():
    """Get current S&P 500 constituents from Wikipedia."""
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = table[0]
    return sorted(df["Symbol"].str.replace(".", "-", regex=False).tolist())
