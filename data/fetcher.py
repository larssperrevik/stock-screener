"""Fetch fundamental and price data from Financial Modeling Prep."""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json
import os

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

FMP_BASE = "https://financialmodelingprep.com/api/v3"


def _api_key():
    key = os.environ.get("FMP_API_KEY", "")
    if not key:
        cfg = Path(__file__).parent.parent / "config.json"
        if cfg.exists():
            with open(cfg) as f:
                key = json.load(f).get("fmp_api_key", "")
    if not key:
        raise ValueError("Set FMP_API_KEY env var or add fmp_api_key to config.json")
    return key


def _fmp_get(endpoint, params=None):
    params = params or {}
    params["apikey"] = _api_key()
    r = requests.get(f"{FMP_BASE}/{endpoint}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _cache_path(ticker, data_type):
    return CACHE_DIR / f"{ticker}_{data_type}.json"


def get_fundamentals(ticker, use_cache=True):
    """Get key fundamental data for a ticker."""
    cache = _cache_path(ticker, "fundamentals")
    if use_cache and cache.exists():
        age = datetime.now().timestamp() - cache.stat().st_mtime
        if age < 86400:
            with open(cache) as f:
                return json.load(f)

    # Profile gives us most of what we need
    profile = _fmp_get(f"profile/{ticker}")
    if not profile:
        raise ValueError(f"No data for {ticker}")
    p = profile[0]

    # Key metrics for ratios not in profile
    metrics = _fmp_get(f"key-metrics-ttm/{ticker}")
    m = metrics[0] if metrics else {}

    # Ratios
    ratios = _fmp_get(f"ratios-ttm/{ticker}")
    rt = ratios[0] if ratios else {}

    fundamentals = {
        "ticker": ticker,
        "name": p.get("companyName", ""),
        "sector": p.get("sector", ""),
        "industry": p.get("industry", ""),
        "market_cap": p.get("mktCap"),
        "pe_ratio": rt.get("peRatioTTM"),
        "forward_pe": p.get("dcf"),  # not exact but a proxy
        "pb_ratio": rt.get("priceToBookRatioTTM"),
        "ps_ratio": rt.get("priceToSalesRatioTTM"),
        "dividend_yield": rt.get("dividendYieldTTM"),
        "roe": rt.get("returnOnEquityTTM"),
        "roa": rt.get("returnOnAssetsTTM"),
        "debt_to_equity": rt.get("debtEquityRatioTTM"),
        "current_ratio": rt.get("currentRatioTTM"),
        "free_cash_flow": m.get("freeCashFlowPerShareTTM"),
        "operating_cash_flow": m.get("operatingCashFlowPerShareTTM"),
        "total_revenue": m.get("revenuePerShareTTM"),
        "net_income": m.get("netIncomePerShareTTM"),
        "total_debt": None,  # not directly in TTM endpoint
        "total_cash": m.get("cashPerShareTTM"),
        "earnings_growth": m.get("earningsYieldTTM"),
        "revenue_growth": rt.get("revenuePerShareTTM"),
        "gross_margins": rt.get("grossProfitMarginTTM"),
        "operating_margins": rt.get("operatingProfitMarginTTM"),
        "profit_margins": rt.get("netProfitMarginTTM"),
        "enterprise_value": m.get("enterpriseValueTTM"),
        "ebitda": m.get("ebitdaPerShareTTM"),
        "roic": rt.get("returnOnCapitalEmployedTTM"),
        "fcf_yield": m.get("freeCashFlowYieldTTM"),
    }

    with open(cache, "w") as f:
        json.dump(fundamentals, f)

    return fundamentals


def get_financials(ticker):
    """Get income statement, balance sheet, cash flow as DataFrames."""
    income = _fmp_get(f"income-statement/{ticker}", {"period": "annual", "limit": 10})
    balance = _fmp_get(f"balance-sheet-statement/{ticker}", {"period": "annual", "limit": 10})
    cashflow = _fmp_get(f"cash-flow-statement/{ticker}", {"period": "annual", "limit": 10})
    return {
        "income": pd.DataFrame(income),
        "balance": pd.DataFrame(balance),
        "cashflow": pd.DataFrame(cashflow),
    }


def get_prices(ticker, start="2000-01-01", end=None):
    """Get historical daily prices."""
    cache = CACHE_DIR / f"{ticker}_prices.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        last_date = df.index.max()
        if pd.Timestamp.now() - last_date < timedelta(days=1):
            if start:
                df = df.loc[df.index >= start]
            if end:
                df = df.loc[df.index <= end]
            return df

    end = end or datetime.now().strftime("%Y-%m-%d")
    data = _fmp_get(f"historical-price-full/{ticker}", {"from": start, "to": end})
    if not data or "historical" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["historical"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    # Rename to standard columns
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "adjClose": "Adj Close", "volume": "Volume",
    })
    if not df.empty:
        df.to_parquet(cache)
    return df


def get_sp500_tickers():
    """Get current S&P 500 constituents from FMP."""
    data = _fmp_get("sp500_constituent")
    return sorted([d["symbol"] for d in data])


def get_historical_sp500():
    """Get historical S&P 500 additions/removals for survivorship bias handling."""
    data = _fmp_get("historical/sp500_constituent")
    return pd.DataFrame(data)
