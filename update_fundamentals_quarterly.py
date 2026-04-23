'''Fetch quarterly SimFin derived + publish dates for data past bulk cutoff.

Bulk quarterly CSV stops ~2024-12-12. This script fills 2025 Q1-Q4 and
Q1 2026 via SimFin API free tier (2 req/sec).

Per ticker: 4 quarters × 2 statements (derived + pl) = 8 API calls.
~4763 tickers × 8 calls / 2 req/sec ≈ 5-6 hours.

Resume-safe: progress saved every 25 tickers.
'''
import requests, pandas as pd, json, time, sys
from pathlib import Path
from datetime import datetime

SIMFIN_DIR = Path('data/simfin')
BULK_DERIVED = SIMFIN_DIR / 'us-derived-quarterly.csv'
SUPPLEMENT = SIMFIN_DIR / 'derived_quarterly_supplement.csv'
PROGRESS = SIMFIN_DIR / 'fundamentals_quarterly_progress.json'
API_BASE = 'https://backend.simfin.com/api/v3'
QUARTERS = ['q1','q2','q3','q4']


def fetch_statement(ticker, api_key, statements, period):
    '''Fetch one statement type for one period, with retry on 429.'''
    for attempt in range(3):
        r = requests.get(
            f'{API_BASE}/companies/statements/compact',
            headers={'Authorization': f'api-key {api_key}'},
            params={'ticker': ticker, 'statements': statements, 'period': period},
            timeout=15,
        )
        if r.status_code == 429:
            if attempt < 2:
                time.sleep(3); continue
            return 'rate_limit'
        break
    if r.status_code != 200:
        return None
    data = r.json()
    if not data or not data[0].get('statements'):
        return None
    co = data[0]
    stmt = co['statements'][0]
    if not stmt['data']:
        return None
    df = pd.DataFrame(stmt['data'], columns=stmt['columns'])
    df['Ticker'] = ticker
    df['SimFinId'] = co.get('id', 0)
    df['Currency'] = co.get('currency', 'USD')
    return df


def load_progress():
    if PROGRESS.exists():
        return json.load(open(PROGRESS))
    return {'completed': [], 'failed': [], 'last_run': None}


def save_progress(p):
    p['last_run'] = datetime.now().isoformat()
    json.dump(p, open(PROGRESS, 'w'))


def main():
    api_key = sys.argv[1] if len(sys.argv)>1 else '6a5e0dc7-b994-471c-9ece-296d2235a839'

    # Cutoff: last publish date in bulk quarterly
    bulk = pd.read_csv(BULK_DERIVED, sep=';', usecols=['Publish Date'], parse_dates=['Publish Date'])
    cutoff = bulk['Publish Date'].max()
    print(f'Bulk quarterly cutoff: {cutoff.date()}')

    active = SIMFIN_DIR / 'active_tickers.json'
    tickers = [t for t in json.load(open(active)) if '_delisted' not in t and '_old' not in t]
    print(f'{len(tickers)} active tickers')

    prog = load_progress()
    done = set(prog['completed']); bad = set(prog['failed'])
    todo = [t for t in tickers if t not in done and t not in bad]
    print(f'Done: {len(done)}, failed: {len(bad)}, remaining: {len(todo)}')
    if not todo:
        print('All done!'); return

    # Load existing supplement to append to
    all_rows = []
    if SUPPLEMENT.exists():
        existing = pd.read_csv(SUPPLEMENT, parse_dates=['Publish Date','Report Date'])
        all_rows.append(existing)
        print(f'Existing supplement: {len(existing):,} rows')

    session_rows = []
    session = 0
    for ticker in todo:
        session += 1
        total = len(done) + session
        derived_frames = []
        pl_pubs = []
        rate_limited = False
        for q in QUARTERS:
            d = fetch_statement(ticker, api_key, 'derived', q)
            time.sleep(1.1)  # stay under 2/s
            if isinstance(d, str) and d == 'rate_limit':
                rate_limited = True; break
            if isinstance(d, pd.DataFrame):
                derived_frames.append(d)
            p = fetch_statement(ticker, api_key, 'pl', q)
            time.sleep(1.1)
            if isinstance(p, str) and p == 'rate_limit':
                rate_limited = True; break
            if isinstance(p, pd.DataFrame) and 'Publish Date' in p.columns:
                pl_pubs.append(p[['Fiscal Year','Fiscal Period','Publish Date']].drop_duplicates())
        if rate_limited:
            print(f'  [{total}/{len(tickers)}] {ticker}: rate limit, skipping and backing off 60s')
            time.sleep(60)
            # Don't mark as failed - leave for next run to retry
            continue
        if not derived_frames:
            print(f'  [{total}/{len(tickers)}] {ticker}: no data')
            bad.add(ticker); prog['failed'].append(ticker)
        else:
            merged = pd.concat(derived_frames, ignore_index=True)
            if pl_pubs:
                pubs = pd.concat(pl_pubs, ignore_index=True)
                merged = merged.merge(pubs, on=['Fiscal Year','Fiscal Period'], how='left')
            merged['Publish Date'] = pd.to_datetime(merged.get('Publish Date'), errors='coerce')
            merged['Report Date']  = pd.to_datetime(merged.get('Report Date'),  errors='coerce')
            merged = merged[merged['Publish Date'] > cutoff]
            if not merged.empty:
                session_rows.append(merged)
            done.add(ticker); prog['completed'].append(ticker)
            print(f'  [{total}/{len(tickers)}] {ticker}: +{len(merged)} new quarterly rows')
        # Checkpoint every 25
        if session % 25 == 0:
            save_progress(prog)
            if session_rows:
                combined = pd.concat(all_rows + session_rows, ignore_index=True)
                combined = combined.drop_duplicates(subset=['Ticker','Fiscal Year','Fiscal Period'], keep='last')
                combined.to_csv(SUPPLEMENT, index=False)
                all_rows = [combined]; session_rows = []
                print(f'  --- checkpoint: supplement now {len(combined):,} rows ---')

    save_progress(prog)
    if session_rows or all_rows:
        combined = pd.concat(all_rows + session_rows, ignore_index=True) if session_rows else all_rows[0]
        combined = combined.drop_duplicates(subset=['Ticker','Fiscal Year','Fiscal Period'], keep='last')
        combined.to_csv(SUPPLEMENT, index=False)
        print(f'Final supplement: {len(combined):,} rows across {combined["Ticker"].nunique()} tickers')

if __name__ == '__main__':
    main()
