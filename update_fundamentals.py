"""Update SimFin fundamentals via API for data newer than bulk file.

Fetches derived annual data for all tickers, keeping only records
with Report Date after the bulk file's last date. Appends to the
existing derived-annual CSV.

Free tier: 1 request/sec. ~4400 tickers = ~75 min.
"""

import requests
import pandas as pd
import json
import time
import sys
from pathlib import Path
from datetime import datetime

SIMFIN_DIR = Path("data/simfin")
DERIVED_FILE = SIMFIN_DIR / "us-derived-annual.csv"
SUPPLEMENT_DERIVED = SIMFIN_DIR / "derived_supplement.csv"
PROGRESS_FILE = SIMFIN_DIR / "fundamentals_progress.json"

API_BASE = "https://backend.simfin.com/api/v3"


def _load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "last_run": None}


def _save_progress(progress):
    progress["last_run"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def fetch_derived(ticker, api_key):
    """Fetch derived annual data for one ticker from SimFin API."""
    for attempt in range(3):
        r = requests.get(
            f"{API_BASE}/companies/statements/compact",
            headers={"Authorization": f"api-key {api_key}"},
            params={"ticker": ticker, "statements": "derived", "period": "fy"},
            timeout=15,
        )
        if r.status_code == 429:
            if attempt < 2:
                time.sleep(3)  # short retry
                continue
            return "rate_limit"
        break
    if r.status_code != 200:
        return None

    data = r.json()
    if not data or not data[0].get("statements"):
        return None

    company = data[0]
    stmt = company["statements"][0]
    columns = stmt["columns"]
    rows = stmt["data"]

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=columns)
    df["Ticker"] = ticker
    df["SimFinId"] = company.get("id", 0)
    df["Currency"] = company.get("currency", "USD")

    return df


def fetch_pl_publish_dates(ticker, api_key):
    """Fetch income statement to get Publish Date (not in derived endpoint)."""
    for attempt in range(3):
        r = requests.get(
            f"{API_BASE}/companies/statements/compact",
            headers={"Authorization": f"api-key {api_key}"},
            params={"ticker": ticker, "statements": "pl", "period": "fy"},
            timeout=15,
        )
        if r.status_code == 429:
            if attempt < 2:
                time.sleep(3)
                continue
            return "rate_limit"
        break
    if r.status_code != 200:
        return None

    data = r.json()
    if not data or not data[0].get("statements"):
        return None

    stmt = data[0]["statements"][0]
    columns = stmt["columns"]
    rows = stmt["data"]

    if not rows or "Publish Date" not in columns:
        return None

    df = pd.DataFrame(rows, columns=columns)
    # Only need fiscal year -> publish date mapping
    return df[["Fiscal Year", "Fiscal Period", "Publish Date"]].drop_duplicates()


def main():
    api_key = sys.argv[1] if len(sys.argv) > 1 else "6a5e0dc7-b994-471c-9ece-296d2235a839"

    # Get the last date in our bulk derived data
    print("Checking existing data...")
    existing = pd.read_csv(DERIVED_FILE, sep=";", usecols=["Ticker", "Report Date"],
                           parse_dates=["Report Date"])
    last_report = existing["Report Date"].max()
    print(f"Bulk data last report: {last_report.strftime('%Y-%m-%d')}")

    # Use our known active tickers from SimFin bulk data
    active_file = SIMFIN_DIR / "active_tickers.json"
    if active_file.exists():
        with open(active_file) as f:
            us_tickers = [t for t in json.load(f)
                          if "_delisted" not in t and "_old" not in t]
    else:
        # Fallback: get from bulk derived data
        us_tickers = sorted(existing["Ticker"].unique().tolist())
        us_tickers = [t for t in us_tickers if "_delisted" not in t and "_old" not in t]
    print(f"{len(us_tickers)} tickers to check")

    progress = _load_progress()
    completed = set(progress["completed"])
    failed = set(progress["failed"])
    remaining = [t for t in us_tickers if t not in completed and t not in failed]
    print(f"Already done: {len(completed)}, failed: {len(failed)}, remaining: {len(remaining)}")

    if not remaining:
        print("All tickers fetched!")
        return

    new_rows = []
    session = 0

    for ticker in remaining:
        session += 1
        total = len(completed) + session
        print(f"  [{total}/{len(us_tickers)}] {ticker}...", end=" ", flush=True)

        # Fetch derived
        result = fetch_derived(ticker, api_key)
        if isinstance(result, str) and result == "rate_limit":
            print("rate limit! Stopping.")
            break
        time.sleep(1.5)  # respect rate limit with margin

        if result is None or (isinstance(result, pd.DataFrame) and result.empty):
            print("no data")
            progress["failed"].append(ticker)
            _save_progress(progress)
            continue

        # Fetch publish dates from PL
        pub_dates = fetch_pl_publish_dates(ticker, api_key)
        if isinstance(pub_dates, str) and pub_dates == "rate_limit":
            print("rate limit on PL! Stopping.")
            # Still save the derived data without publish dates
            new_rows.append(result)
            progress["completed"].append(ticker)
            _save_progress(progress)
            break
        time.sleep(1.1)

        # Merge publish dates into derived
        if pub_dates is not None and not pub_dates.empty:
            result = result.merge(pub_dates, on=["Fiscal Year", "Fiscal Period"], how="left")
            result["Publish Date"] = pd.to_datetime(result["Publish Date"], errors="coerce")

        # Only keep records newer than our bulk data
        if "Report Date" in result.columns:
            result["Report Date"] = pd.to_datetime(result["Report Date"], errors="coerce")
            new_records = result[result["Report Date"] > last_report]
        else:
            new_records = result  # keep all if no report date

        if new_records.empty:
            print("no new records")
        else:
            print(f"{len(new_records)} new records")
            new_rows.append(new_records)

        progress["completed"].append(ticker)

        if session % 50 == 0:
            _save_progress(progress)
            _save_supplement(new_rows)
            print(f"  --- checkpoint: {len(completed) + session} done ---")

    _save_progress(progress)
    _save_supplement(new_rows)

    total_done = len(completed) + session
    print(f"\nSession: {session} tickers processed")
    print(f"Total: {total_done}/{len(us_tickers)}")
    if total_done < len(us_tickers):
        print("Run again to continue (rate limit resets).")


def _save_supplement(new_rows):
    if not new_rows:
        return
    combined = pd.concat(new_rows, ignore_index=True)
    combined = combined.drop_duplicates(subset=["Ticker", "Fiscal Year", "Fiscal Period"])

    # Load existing supplement if any
    if SUPPLEMENT_DERIVED.exists():
        existing = pd.read_csv(SUPPLEMENT_DERIVED, parse_dates=["Report Date", "Publish Date"])
        combined = pd.concat([existing, combined], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Ticker", "Fiscal Year", "Fiscal Period"])

    combined.to_csv(SUPPLEMENT_DERIVED, index=False)
    print(f"  Saved derived supplement: {len(combined)} records, {combined['Ticker'].nunique()} tickers")


if __name__ == "__main__":
    main()
