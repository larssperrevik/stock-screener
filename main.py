"""Stock Screener -- run a value screen and optional backtest."""

import argparse
from data.fetcher import get_fundamentals, get_sp500_tickers
from screener.criteria import ScreenCriteria, apply_screen
from backtest.engine import BacktestEngine


def run_screen(tickers=None, criteria=None):
    """Run the screener on current data and print results."""
    if criteria is None:
        criteria = ScreenCriteria()
    if tickers is None:
        print("Fetching S&P 500 tickers...")
        tickers = get_sp500_tickers()

    print(f"Screening {len(tickers)} stocks...")
    fundamentals = []
    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            print(f"  Fetching {i}/{len(tickers)}...")
        try:
            f = get_fundamentals(ticker)
            fundamentals.append(f)
        except Exception as e:
            print(f"  {ticker}: failed -- {e}")

    results = apply_screen(fundamentals, criteria)

    if results.empty:
        print("No stocks passed the screen.")
        return

    display_cols = ["ticker", "name", "sector", "pe_ratio", "roe", "fcf_yield",
                    "debt_to_equity", "operating_margins", "piotroski", "composite_rank"]
    display_cols = [c for c in display_cols if c in results.columns]
    print(f"\n{len(results)} stocks passed the screen:\n")
    print(results[display_cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Value investing stock screener")
    parser.add_argument("--screen", action="store_true", help="Run the screener")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers (default: S&P 500)")
    parser.add_argument("--max-pe", type=float, default=25.0)
    parser.add_argument("--min-roe", type=float, default=0.15)
    parser.add_argument("--min-fcf-yield", type=float, default=0.04)
    args = parser.parse_args()

    if args.screen:
        criteria = ScreenCriteria(
            max_pe_ratio=args.max_pe,
            min_roe=args.min_roe,
            min_fcf_yield=args.min_fcf_yield,
        )
        run_screen(tickers=args.tickers, criteria=criteria)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
