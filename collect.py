"""Incremental data collector for S&P 500 historical data.

Run this repeatedly — it picks up where it left off and stops
when it hits the API rate limit (402). Each ticker costs 3 API calls
(profile + key-metrics-ttm + ratios-ttm) plus 1 for prices.
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime
from data.fetcher import get_fundamentals, get_prices, get_sp500_tickers, _fmp_get

PROGRESS_FILE = Path("data/cache/collect_progress.json")


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "failed": {}, "last_run": None}


def save_progress(progress):
    progress["last_run"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def collect_ticker(ticker):
    """Collect fundamentals and full price history for one ticker."""
    errors = []

    # Fundamentals (3 API calls)
    try:
        get_fundamentals(ticker, use_cache=False)
    except Exception as e:
        if "402" in str(e):
            raise  # rate limit — stop entirely
        errors.append(f"fundamentals: {e}")

    # Price history (1 API call)
    try:
        get_prices(ticker, start="2000-01-01")
    except Exception as e:
        if "402" in str(e):
            raise
        errors.append(f"prices: {e}")

    return errors


def main():
    print("Fetching S&P 500 ticker list from Wikipedia...")
    tickers = get_sp500_tickers()
    print(f"  {len(tickers)} tickers")

    progress = load_progress()
    completed = set(progress["completed"])
    remaining = [t for t in tickers if t not in completed]

    print(f"  {len(completed)} already collected")
    print(f"  {len(remaining)} remaining")
    if not remaining:
        print("All done!")
        return

    api_calls = 0
    session_completed = 0

    for i, ticker in enumerate(remaining):
        print(f"\n[{len(completed) + session_completed + 1}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        try:
            errors = collect_ticker(ticker)
            if errors:
                progress["failed"][ticker] = errors
                print(f"partial ({', '.join(errors)})")
            else:
                print("ok")

            progress["completed"].append(ticker)
            session_completed += 1
            api_calls += 4  # 3 fundamentals + 1 prices

            # Save progress every ticker
            save_progress(progress)

            # Small delay to be nice to the API
            time.sleep(0.3)

        except Exception as e:
            if "402" in str(e):
                print(f"\nRate limit hit after {session_completed} tickers ({api_calls} API calls)")
                print(f"Total progress: {len(completed) + session_completed}/{len(tickers)}")
                save_progress(progress)
                print(f"\nRun this script again later to continue.")
                return
            else:
                print(f"error: {e}")
                progress["failed"][ticker] = [str(e)]
                save_progress(progress)

    print(f"\nDone! Collected all {len(tickers)} tickers.")
    save_progress(progress)


if __name__ == "__main__":
    main()
