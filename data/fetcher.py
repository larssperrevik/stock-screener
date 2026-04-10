"""Fetch fundamental and price data from Financial Modeling Prep (stable API)."""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json
import os

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

FMP_BASE = "https://financialmodelingprep.com/stable"


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

    profile = _fmp_get("profile", {"symbol": ticker})
    if not profile:
        raise ValueError(f"No data for {ticker}")
    p = profile[0]

    metrics = _fmp_get("key-metrics-ttm", {"symbol": ticker})
    m = metrics[0] if metrics else {}

    ratios = _fmp_get("ratios-ttm", {"symbol": ticker})
    rt = ratios[0] if ratios else {}

    fundamentals = {
        "ticker": ticker,
        "name": p.get("companyName", ""),
        "sector": p.get("sector", ""),
        "industry": p.get("industry", ""),
        "market_cap": m.get("marketCap") or p.get("marketCap"),
        "pe_ratio": rt.get("priceToEarningsRatioTTM"),
        "pb_ratio": rt.get("priceToBookRatioTTM"),
        "ps_ratio": rt.get("priceToSalesRatioTTM"),
        "dividend_yield": rt.get("dividendYieldTTM"),
        "roe": m.get("returnOnEquityTTM"),
        "roa": m.get("returnOnAssetsTTM"),
        "roic": m.get("returnOnInvestedCapitalTTM"),
        "debt_to_equity": rt.get("debtToEquityRatioTTM"),
        "current_ratio": rt.get("currentRatioTTM"),
        "gross_margins": rt.get("grossProfitMarginTTM"),
        "operating_margins": rt.get("operatingProfitMarginTTM"),
        "profit_margins": rt.get("netProfitMarginTTM"),
        "enterprise_value": m.get("enterpriseValueTTM"),
        "ev_ebitda": m.get("evToEBITDATTM"),
        "fcf_yield": m.get("freeCashFlowYieldTTM"),
        "earnings_yield": m.get("earningsYieldTTM"),
        "net_income": m.get("netIncomePerShareTTM") if "netIncomePerShareTTM" in m else None,
        "operating_cash_flow": m.get("operatingCashFlowPerShareTTM") if "operatingCashFlowPerShareTTM" in m else None,
        "free_cash_flow": m.get("freeCashFlowPerShareTTM") if "freeCashFlowPerShareTTM" in m else None,
        "peg_ratio": rt.get("priceToEarningsGrowthRatioTTM"),
        "return_on_capital_employed": m.get("returnOnCapitalEmployedTTM"),
        "income_quality": m.get("incomeQualityTTM"),
        "graham_number": m.get("grahamNumberTTM"),
    }

    with open(cache, "w") as f:
        json.dump(fundamentals, f)

    return fundamentals


def get_financials(ticker):
    """Get income statement, balance sheet, cash flow as DataFrames."""
    income = _fmp_get("income-statement", {"symbol": ticker, "period": "annual", "limit": 10})
    balance = _fmp_get("balance-sheet-statement", {"symbol": ticker, "period": "annual", "limit": 10})
    cashflow = _fmp_get("cash-flow-statement", {"symbol": ticker, "period": "annual", "limit": 10})
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
    params = {"symbol": ticker, "from": start, "to": end}
    data = _fmp_get("historical-price-eod/full", params)
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    if not df.empty:
        df.to_parquet(cache)
    return df


def get_sp500_tickers():
    """Get S&P 500 constituents from local file."""
    ticker_file = Path(__file__).parent / "sp500_tickers.json"
    with open(ticker_file) as f:
        return json.load(f)
