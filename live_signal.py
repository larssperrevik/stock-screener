"""Live signal generator: what to buy/hold/sell right now.

Runs the event-driven scoring engine on the latest available data
and outputs actionable signals. This is what you check before trading.

Usage:
    python3 live_signal.py                # Show current signals
    python3 live_signal.py --portfolio    # Score an existing portfolio
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from data.simfin_loader import load_derived_annual, load_prices, load_companies, load_industries
from screener.criteria import ScreenCriteria, apply_screen
from backtest.event_engine import compute_stock_score


def get_current_signals(criteria=None, buy_threshold=40, top_n=30):
    """Score all screener-eligible stocks and return ranked buy candidates."""
    if criteria is None:
        criteria = ScreenCriteria()

    derived = load_derived_annual()
    prices = load_prices()
    companies = load_companies()
    industries = load_industries()

    today = prices["Date"].max()
    print(f"Data as of: {today.strftime('%Y-%m-%d')}")
    print(f"Buy threshold: {buy_threshold}\n")

    # Get all fundamentals available now
    fundamentals = derived[derived["Publish Date"] <= today]
    latest_per_ticker = fundamentals.sort_values("Publish Date").groupby("Ticker").last().reset_index()

    # Filter out delisted
    latest_per_ticker = latest_per_ticker[
        ~latest_per_ticker["Ticker"].str.contains("_delisted|_old", na=False)
    ]

    # Apply screener
    screened = apply_screen(latest_per_ticker, criteria)
    if screened.empty:
        print("No stocks pass the screener.")
        return pd.DataFrame()

    print(f"{len(screened)} stocks pass the quality screener\n")

    # Index data for scoring
    derived_by_ticker = {}
    for ticker, group in derived.groupby("Ticker"):
        derived_by_ticker[ticker] = group.sort_values("Publish Date")

    price_by_ticker = {}
    for ticker, group in prices.groupby("Ticker"):
        price_by_ticker[ticker] = group.set_index("Date").sort_index()

    # Score each screened stock
    results = []
    for _, row in screened.iterrows():
        ticker = row["Ticker"]
        if ticker not in derived_by_ticker:
            continue

        ticker_derived = derived_by_ticker[ticker]
        available = ticker_derived[ticker_derived["Publish Date"] <= today]
        if available.empty:
            continue

        latest = available.iloc[-1]
        publish_date = latest["Publish Date"]
        days_since = (today - publish_date).days

        # In live mode, don't skip on staleness — we show what we have
        # The freshness bonus still rewards recent reports

        # Get price history for valuation-vs-history
        ticker_prices = price_by_ticker.get(ticker)
        if ticker_prices is not None:
            hist = ticker_prices[ticker_prices.index <= today]
        else:
            hist = None

        score, details = compute_stock_score(ticker, latest.to_dict(), hist, available)

        # Freshness bonus
        if days_since < 30:
            score += 5
        elif days_since < 60:
            score += 3

        # Company info
        co = companies[companies["Ticker"] == ticker]
        co_name = co.iloc[0]["Company Name"] if not co.empty else ""
        ind_id = co.iloc[0].get("IndustryId") if not co.empty else None
        sector = ""
        if pd.notna(ind_id):
            ind = industries[industries["IndustryId"] == ind_id]
            if not ind.empty:
                sector = ind.iloc[0].get("Sector", "")

        # Current price
        if ticker_prices is not None and not ticker_prices.empty:
            last_row = ticker_prices.iloc[-1]
            current_price = last_row.get("Close", last_row.get("Adj. Close"))
        else:
            current_price = None

        results.append({
            "ticker": ticker,
            "name": co_name[:35],
            "sector": sector,
            "score": score,
            "signal": "BUY" if score >= buy_threshold else "WATCH",
            "publish_date": publish_date.strftime("%Y-%m-%d"),
            "days_since_report": days_since,
            "price": current_price,
            "roic": latest.get("Return On Invested Capital"),
            "piotroski": latest.get("Piotroski F-Score"),
            "gross_margin": latest.get("Gross Profit Margin"),
            "op_margin": latest.get("Operating Margin"),
            "debt_ratio": latest.get("Debt Ratio"),
            "fcf_to_ni": latest.get("Free Cash Flow to Net Income"),
            "val_discount": details.get("val_Price to E", details.get("val_Price to F", None)),
        })

    df = pd.DataFrame(results).sort_values("score", ascending=False)

    # Print results
    buys = df[df["signal"] == "BUY"]
    watches = df[df["signal"] == "WATCH"].head(10)

    print("=" * 90)
    print(f"BUY SIGNALS ({len(buys)} stocks above threshold {buy_threshold})")
    print("=" * 90)
    if buys.empty:
        print("  None — no stocks score high enough right now.")
    else:
        for _, r in buys.head(top_n).iterrows():
            print(f"  {r['signal']:5s}  {r['ticker']:8s} {r['name']:35s}  "
                  f"Score: {r['score']:5.1f}  ROIC: {r['roic']:.0%}  "
                  f"F-Score: {int(r['piotroski'] or 0)}  "
                  f"Report: {r['publish_date']} ({r['days_since_report']}d ago)")

    print(f"\nWATCHLIST (top 10 below threshold)")
    print("-" * 90)
    for _, r in watches.iterrows():
        print(f"  {r['signal']:5s}  {r['ticker']:8s} {r['name']:35s}  "
              f"Score: {r['score']:5.1f}  ROIC: {r['roic']:.0%}  "
              f"F-Score: {int(r['piotroski'] or 0)}  "
              f"Report: {r['publish_date']} ({r['days_since_report']}d ago)")

    # Entry plan
    if not buys.empty:
        n_buys = min(len(buys), 15)
        top_buys = buys.head(n_buys)
        weight = 1.0 / n_buys

        print(f"\n{'=' * 90}")
        print(f"ENTRY PLAN: Equal-weight {n_buys} positions ({weight:.1%} each)")
        print(f"{'=' * 90}")
        print(f"  For $100,000 portfolio:\n")
        for _, r in top_buys.iterrows():
            alloc = 100000 * weight
            if r["price"] and r["price"] > 0:
                shares = int(alloc / r["price"])
                print(f"    {r['ticker']:8s}  ${alloc:,.0f}  ~{shares} shares @ ${r['price']:.2f}  "
                      f"(score {r['score']:.0f})")
            else:
                print(f"    {r['ticker']:8s}  ${alloc:,.0f}  (price unavailable, score {r['score']:.0f})")

        print(f"\n  Review triggers:")
        print(f"    - Next report published -> re-score")
        print(f"    - Score drops below 20 -> consider selling")
        print(f"    - Screener fail (margins/ROIC collapse) -> sell")
        print(f"    - After 540 days -> re-evaluate, renew if still scoring well")
        print(f"    - New BUY signal when at <15 positions -> add it")

    return df


def score_portfolio(tickers, criteria=None, buy_threshold=40):
    """Score an existing portfolio of tickers."""
    if criteria is None:
        criteria = ScreenCriteria()

    derived = load_derived_annual()
    prices = load_prices()
    companies = load_companies()

    today = prices["Date"].max()
    print(f"Scoring portfolio as of {today.strftime('%Y-%m-%d')}\n")

    derived_by_ticker = {}
    for ticker, group in derived.groupby("Ticker"):
        derived_by_ticker[ticker] = group.sort_values("Publish Date")

    price_by_ticker = {}
    for ticker, group in prices.groupby("Ticker"):
        price_by_ticker[ticker] = group.set_index("Date").sort_index()

    for ticker in tickers:
        ticker = ticker.upper()
        if ticker not in derived_by_ticker:
            print(f"  {ticker}: NO DATA")
            continue

        ticker_derived = derived_by_ticker[ticker]
        available = ticker_derived[ticker_derived["Publish Date"] <= today]
        if available.empty:
            print(f"  {ticker}: NO DATA")
            continue

        latest = available.iloc[-1]
        publish_date = latest["Publish Date"]
        days_since = (today - publish_date).days

        hist = price_by_ticker.get(ticker)
        if hist is not None:
            hist = hist[hist.index <= today]

        score, details = compute_stock_score(ticker, latest.to_dict(), hist, available)
        if days_since < 30:
            score += 5
        elif days_since < 60:
            score += 3

        passes = "PASS" if score >= buy_threshold else "FAIL"
        co = companies[companies["Ticker"] == ticker]
        name = co.iloc[0]["Company Name"][:30] if not co.empty else ""

        signal = "HOLD" if score >= 20 else "SELL"
        if score >= buy_threshold:
            signal = "STRONG HOLD"

        print(f"  {ticker:8s} {name:30s}  Score: {score:5.1f}  {signal}")
        print(f"           ROIC: {latest.get('Return On Invested Capital', 0):.0%}  "
              f"Piotroski: {int(latest.get('Piotroski F-Score', 0) or 0)}  "
              f"Report: {publish_date.strftime('%Y-%m-%d')} ({days_since}d ago)")
        print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live trading signals")
    parser.add_argument("--portfolio", nargs="+", help="Score specific tickers")
    parser.add_argument("--buy-threshold", type=float, default=40)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    if args.portfolio:
        score_portfolio(args.portfolio, buy_threshold=args.buy_threshold)
    else:
        get_current_signals(buy_threshold=args.buy_threshold, top_n=args.top)
