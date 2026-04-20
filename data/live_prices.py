"""Fetch live/recent prices from FMP to supplement SimFin historical data."""

import requests
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime


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


def get_live_prices(tickers, start="2024-09-01"):
    """Fetch recent daily prices for a list of tickers from FMP.

    Returns a DataFrame with Date index, ticker columns, adjusted close values.
    """
    end = datetime.now().strftime("%Y-%m-%d")
    key = _api_key()
    all_data = {}

    for ticker in tickers:
        try:
            r = requests.get(
                f"{FMP_BASE}/historical-price-eod/full",
                params={"symbol": ticker, "from": start, "to": end, "apikey": key},
                timeout=15,
            )
            if r.status_code == 402:
                print(f"  {ticker}: rate limit hit, stopping")
                break
            r.raise_for_status()
            data = r.json()
            if data:
                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
                # Use adjClose if available, otherwise close
                col = "adjClose" if "adjClose" in df.columns else "close"
                all_data[ticker] = df[col]
        except Exception as e:
            print(f"  {ticker}: {e}")

    if not all_data:
        return pd.DataFrame()

    return pd.DataFrame(all_data)


def get_current_price(ticker):
    """Get the current/latest price for a single ticker."""
    key = _api_key()
    r = requests.get(
        f"{FMP_BASE}/profile",
        params={"symbol": ticker, "apikey": key},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    if data:
        return {"price": data[0].get("price"), "name": data[0].get("companyName")}
    return None
