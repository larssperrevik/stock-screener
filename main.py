"""Stock Screener and Backtester using SimFin data."""

import argparse
from screener.criteria import ScreenCriteria, apply_screen
from data.simfin_loader import get_all_fundamentals_at_date, get_sp500_tickers
from backtest.engine import BacktestEngine


def run_screen(date=None, criteria=None):
    """Run the screener using SimFin local data."""
    if criteria is None:
        criteria = ScreenCriteria()
    if date is None:
        date = "2024-09-30"  # latest in dataset

    print(f"Screening at date: {date}")
    fundamentals = get_all_fundamentals_at_date(date)
    print(f"  {len(fundamentals)} companies with data")

    results = apply_screen(fundamentals, criteria)

    if results.empty:
        print("No stocks passed the screen.")
        return results

    display_cols = [
        "Ticker", "Fiscal Year", "Return on Equity", "Return On Invested Capital",
        "Piotroski F-Score", "Gross Profit Margin", "Operating Margin",
        "Debt Ratio", "Current Ratio", "Free Cash Flow to Net Income",
        "composite_rank",
    ]
    display_cols = [c for c in display_cols if c in results.columns]
    print(f"\n{len(results)} stocks passed the screen:\n")

    pd_opts = {"max_columns": 12, "width": 200, "float_format": "{:.3f}".format}
    import pandas as pd
    with pd.option_context("display.max_columns", 12, "display.width", 200,
                           "display.float_format", "{:.3f}".format):
        print(results[display_cols].head(40).to_string(index=False))

    return results


def run_backtest(criteria=None, freq="annual", top_n=20,
                 start="2005-01-01", end="2024-09-30"):
    """Run the backtester."""
    engine = BacktestEngine(
        criteria=criteria,
        rebalance_freq=freq,
        top_n=top_n,
        start_date=start,
        end_date=end,
    )
    return engine.run()


def main():
    parser = argparse.ArgumentParser(description="Value investing stock screener & backtester")
    sub = parser.add_subparsers(dest="command")

    # Screen command
    sp = sub.add_parser("screen", help="Run the screener")
    sp.add_argument("--date", default="2024-09-30", help="Screen date (YYYY-MM-DD)")
    sp.add_argument("--max-pe", type=float, default=25.0)
    sp.add_argument("--min-roe", type=float, default=0.15)
    sp.add_argument("--min-roic", type=float, default=0.10)
    sp.add_argument("--min-piotroski", type=int, default=5)
    sp.add_argument("--top", type=int, default=40, help="Show top N results")

    # Backtest command
    bp = sub.add_parser("backtest", help="Run a backtest")
    bp.add_argument("--start", default="2005-01-01")
    bp.add_argument("--end", default="2024-09-30")
    bp.add_argument("--freq", default="annual", choices=["annual", "quarterly", "monthly"])
    bp.add_argument("--top-n", type=int, default=20, help="Number of stocks to hold")
    bp.add_argument("--min-roe", type=float, default=0.15)
    bp.add_argument("--min-roic", type=float, default=0.10)
    bp.add_argument("--min-piotroski", type=int, default=5)

    args = parser.parse_args()

    if args.command == "screen":
        criteria = ScreenCriteria(
            max_pe_ratio=args.max_pe,
            min_roe=args.min_roe,
            min_roic=args.min_roic,
            min_piotroski=args.min_piotroski,
        )
        run_screen(date=args.date, criteria=criteria)

    elif args.command == "backtest":
        criteria = ScreenCriteria(
            min_roe=args.min_roe,
            min_roic=args.min_roic,
            min_piotroski=args.min_piotroski,
        )
        run_backtest(criteria=criteria, freq=args.freq, top_n=args.top_n,
                     start=args.start, end=args.end)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
