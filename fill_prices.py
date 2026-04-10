"""Fill the gap between SimFin data (ends Oct 2024) and today using FMP.

Strategy:
- Load SimFin prices to find the last date per ticker
- For tickers we care about (recent screen + benchmark + S&P 500 top holdings),
  fetch daily prices from FMP starting where SimFin left off
- Save as a supplement CSV that the loader merges automatically
- Tracks progress so it can resume across multiple days if needed

Usage:
    python3 fill_prices.py              # Fill all screened tickers + SPY
    python3 fill_prices.py --all-sp500  # Fill all S&P 500 tickers (multi-day)
"""

import argparse
import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

from data.simfin_loader import load_prices, get_all_fundamentals_at_date, get_sp500_tickers
from screener.criteria import ScreenCriteria, apply_screen

FMP_BASE = "https://financialmodelingprep.com/stable"
SUPPLEMENT_FILE = Path("data/simfin/price_supplement.csv")
PROGRESS_FILE = Path("data/simfin/fill_progress.json")


def _api_key():
    cfg = Path("config.json")
    if cfg.exists():
        with open(cfg) as f:
            return json.load(f).get("fmp_api_key", "")
    import os
    return os.environ.get("FMP_API_KEY", "")


def _load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "last_run": None}


def _save_progress(progress):
    progress["last_run"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def _get_simfin_last_date():
    """Find the last date in SimFin price data."""
    prices = load_prices()
    return prices["Date"].max()


def _fetch_fmp_prices(ticker, start_date, api_key):
    """Fetch daily prices from FMP for one ticker."""
    end = datetime.now().strftime("%Y-%m-%d")
    r = requests.get(
        f"{FMP_BASE}/historical-price-eod/full",
        params={"symbol": ticker, "from": start_date, "to": end, "apikey": api_key},
        timeout=15,
    )
    if r.status_code == 402:
        return None  # rate limit
    r.raise_for_status()
    data = r.json()
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df.rename(columns={
        "date": "Date", "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "adjClose": "Adj. Close", "volume": "Volume",
    })
    df["Ticker"] = ticker
    df["Date"] = pd.to_datetime(df["Date"])

    # Keep only columns that match SimFin format
    keep = ["Ticker", "Date", "Open", "High", "Low", "Close", "Adj. Close", "Volume"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].sort_values("Date")


def _get_priority_tickers():
    """Get tickers we care most about: latest screen + benchmark."""
    tickers = {"SPY"}  # always include benchmark

    # Latest screen
    print("Running screen to determine priority tickers...")
    fundamentals = get_all_fundamentals_at_date("2024-09-30")
    screened = apply_screen(fundamentals, ScreenCriteria())
    if not screened.empty:
        screen_tickers = screened.head(40)["Ticker"].tolist()
        tickers.update(screen_tickers)
        print(f"  {len(screen_tickers)} from latest screen")

    # Also grab a few recent screens for backtest continuity
    for date in ["2024-07-01", "2024-04-01", "2024-01-01"]:
        f = get_all_fundamentals_at_date(date)
        s = apply_screen(f, ScreenCriteria())
        if not s.empty:
            tickers.update(s.head(20)["Ticker"].tolist())

    print(f"  {len(tickers)} priority tickers total")
    return sorted(tickers)


def _get_all_active_tickers():
    """Get all tickers that were actively trading near SimFin's last date."""
    prices = load_prices()
    last_date = prices["Date"].max()
    cutoff = last_date - pd.Timedelta(days=30)
    recent = prices[prices["Date"] >= cutoff]
    tickers = sorted(recent["Ticker"].unique().tolist())
    # Filter out delisted tickers (they have _delisted suffix in SimFin)
    tickers = [t for t in tickers if "_delisted" not in t]
    return tickers


def fill_prices(tickers=None, all_sp500=False, priority_only=False):
    api_key = _api_key()
    if not api_key:
        print("Error: No FMP API key configured")
        return

    simfin_last = _get_simfin_last_date()
    start_date = (simfin_last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"SimFin data ends: {simfin_last.strftime('%Y-%m-%d')}")
    print(f"Fetching from: {start_date}")

    if tickers is None:
        if priority_only:
            tickers = _get_priority_tickers()
        elif all_sp500:
            tickers = get_sp500_tickers()
            print(f"Filling all S&P 500: {len(tickers)} tickers")
        else:
            tickers = _get_all_active_tickers()
            print(f"Filling all {len(tickers)} active tickers")

    # Load existing supplement if any
    if SUPPLEMENT_FILE.exists():
        existing = pd.read_csv(SUPPLEMENT_FILE, parse_dates=["Date"])
        existing_tickers = set(existing["Ticker"].unique())
        print(f"Existing supplement: {len(existing)} rows, {len(existing_tickers)} tickers")
    else:
        existing = pd.DataFrame()
        existing_tickers = set()

    # Load progress
    progress = _load_progress()
    completed = set(progress["completed"])

    # Filter to tickers not yet completed
    remaining = [t for t in tickers if t not in completed]
    print(f"Already completed: {len(completed)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All tickers filled!")
        return

    new_frames = [existing] if not existing.empty else []
    api_calls = 0
    session_done = 0

    for ticker in remaining:
        print(f"  [{len(completed) + session_done + 1}/{len(tickers)}] {ticker}...", end=" ", flush=True)

        result = _fetch_fmp_prices(ticker, start_date, api_key)
        if result is None:
            print("rate limit!")
            break

        api_calls += 1
        if result.empty:
            print("no new data")
        else:
            new_frames.append(result)
            print(f"{len(result)} days")

        progress["completed"].append(ticker)
        session_done += 1
        _save_progress(progress)
        time.sleep(0.2)

    # Save combined supplement
    if new_frames:
        combined = pd.concat(new_frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["Ticker", "Date"])
        combined = combined.sort_values(["Ticker", "Date"])
        combined.to_csv(SUPPLEMENT_FILE, index=False)
        print(f"\nSaved supplement: {len(combined)} rows, {combined['Ticker'].nunique()} tickers")

    print(f"Session: {session_done} tickers, {api_calls} API calls")
    total_done = len(completed) + session_done
    print(f"Total progress: {total_done}/{len(tickers)}")
    if total_done < len(tickers):
        print("Run again tomorrow to continue.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fill SimFin price gap with FMP data")
    parser.add_argument("--all-sp500", action="store_true",
                        help="Fill S&P 500 tickers only")
    parser.add_argument("--priority-only", action="store_true",
                        help="Fill only screened tickers + SPY")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to fill")
    args = parser.parse_args()

    fill_prices(tickers=args.tickers, all_sp500=args.all_sp500,
                priority_only=args.priority_only)
