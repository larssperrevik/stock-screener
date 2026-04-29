"""Native price filler on .113 using Yahoo v8 chart endpoint.

Replaces the older Windows-side win_incremental_prices.py + scp workflow.
The v8 endpoint with a UA header works from .113 directly (verified
2026-04-29). Reads the existing price_supplement.csv, fetches incremental
data per ticker since its last date, appends, and saves back in place.

Idempotent and resume-safe via the existing supplement file.
"""
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time
import sys

SIMFIN_DIR = Path(__file__).resolve().parent / "data" / "simfin"
SUPPLEMENT = SIMFIN_DIR / "price_supplement.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Mozilla/5.0"}
BASE = "https://query1.finance.yahoo.com/v8/finance/chart"


def fetch_one(ticker, period1_ts, period2_ts):
    """Fetch daily OHLCV+adjclose for one ticker between two unix timestamps."""
    url = f"{BASE}/{ticker}"
    params = {
        "period1": period1_ts,
        "period2": period2_ts,
        "interval": "1d",
        "events": "split,div",
    }
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=15)
    except Exception as e:
        return None, f"network_error: {e}"
    if r.status_code == 429:
        return None, "rate_limit"
    if r.status_code != 200:
        return None, f"status_{r.status_code}"
    try:
        data = r.json()
    except Exception:
        return None, "bad_json"
    chart = data.get("chart") or {}
    err = chart.get("error")
    if err:
        return None, f"yahoo_error: {err.get('code','')}"
    result = chart.get("result")
    if not result:
        return None, "empty_result"
    res = result[0]
    ts = res.get("timestamp")
    if not ts:
        return None, "no_timestamps"
    quote = (res.get("indicators") or {}).get("quote", [{}])[0]
    adj = (res.get("indicators") or {}).get("adjclose", [{}])[0].get("adjclose") or [None] * len(ts)
    rows = []
    for i, t in enumerate(ts):
        d = pd.Timestamp(t, unit="s").normalize()
        rows.append({
            "Ticker": ticker,
            "Date": d,
            "Open": quote.get("open", [None]*len(ts))[i],
            "High": quote.get("high", [None]*len(ts))[i],
            "Low": quote.get("low", [None]*len(ts))[i],
            "Close": quote.get("close", [None]*len(ts))[i],
            "Adj. Close": adj[i] if adj else quote.get("close", [None]*len(ts))[i],
            "Volume": quote.get("volume", [None]*len(ts))[i],
        })
    return pd.DataFrame(rows), None


def main():
    if not SUPPLEMENT.exists():
        print(f"Supplement not found at {SUPPLEMENT}; nothing to incrementally extend.")
        return
    df = pd.read_csv(SUPPLEMENT, parse_dates=["Date"])
    last_global = df["Date"].max()
    last_per_ticker = df.groupby("Ticker")["Date"].max()
    today = pd.Timestamp.now().normalize()
    print(f"Supplement: {len(df):,} rows, {df['Ticker'].nunique():,} tickers, latest {last_global.date()}")
    print(f"Today: {today.date()}; fetching gaps per ticker.")

    new_rows = []
    failed = []
    rate_limited = []
    n_done = 0
    tickers = sorted(last_per_ticker.index.tolist())
    for i, t in enumerate(tickers, 1):
        last = last_per_ticker[t]
        if last >= today - pd.Timedelta(days=1):
            n_done += 1
            continue
        period1 = int((last + pd.Timedelta(days=1)).timestamp())
        period2 = int((today + pd.Timedelta(days=1)).timestamp())
        out, err = fetch_one(t, period1, period2)
        if err == "rate_limit":
            rate_limited.append(t)
            time.sleep(30)  # backoff
            continue
        if err is not None or out is None or out.empty:
            failed.append((t, err or "empty"))
        else:
            new_rows.append(out)
            n_done += 1
        if i % 200 == 0:
            print(f"  [{i}/{len(tickers)}] done={n_done} fail={len(failed)} rl={len(rate_limited)}")
        time.sleep(0.15)  # ~6 req/s, well under typical Yahoo throttle

    if not new_rows:
        print("No new rows fetched. Already current.")
        return

    new_df = pd.concat(new_rows, ignore_index=True)
    print(f"Fetched {len(new_df):,} new rows across {new_df['Ticker'].nunique()} tickers.")
    print(f"  Date range new data: {new_df['Date'].min().date()} to {new_df['Date'].max().date()}")
    combined = pd.concat([df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Ticker", "Date"], keep="last")
    combined = combined.sort_values(["Ticker", "Date"])
    combined.to_csv(SUPPLEMENT, index=False)
    print(f"Saved: {len(combined):,} total rows, latest {combined['Date'].max().date()}.")
    if failed:
        print(f"Failed: {len(failed)} tickers (sample): {failed[:10]}")
    if rate_limited:
        print(f"Rate-limited: {len(rate_limited)} (will retry on next run)")


if __name__ == "__main__":
    main()
