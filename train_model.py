"""Train and evaluate the ML stock prediction model.

Usage:
    python3 train_model.py build          # Build training dataset (slow, ~30min)
    python3 train_model.py train          # Train model on saved dataset
    python3 train_model.py full           # Build + train end-to-end
    python3 train_model.py compare        # Compare ML vs simple screener
"""

import argparse
import sys
from pathlib import Path

from ml.features import build_training_dataset, save_dataset, load_dataset
from ml.model import WalkForwardModel, print_feature_importance


def cmd_build(args):
    print("Building training dataset...")
    print("This scans all tickers at each quarter — expect ~30 min.\n")
    df = build_training_dataset(
        start_year=args.start_year,
        end_year=args.end_year,
        forward_months=3,
    )
    save_dataset(df)
    print(f"\nDataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique tickers: {df['ticker'].nunique()}")
    print(f"Features: {len([c for c in df.columns if c not in ('ticker','date','forward_return_3m','target_quintile','target_buy','target_avoid','sector_name')])}")
    print(f"\nTarget distribution:")
    if "target_buy" in df.columns:
        print(f"  Buy (top quintile):   {df['target_buy'].mean():.1%}")
        print(f"  Avoid (bot quintile): {df['target_avoid'].mean():.1%}")


def cmd_train(args):
    print("Loading dataset...")
    df = load_dataset()
    print(f"Dataset: {len(df)} samples, {df['date'].nunique()} dates, "
          f"{df['ticker'].nunique()} tickers\n")

    model = WalkForwardModel(
        train_years=args.train_years,
        target_col="target_buy",
        n_estimators=args.n_estimators,
    )
    results = model.run(df, top_n=args.top_n)

    if results:
        print_feature_importance(results["feature_importance"])

        # Save results
        results["predictions"].to_parquet("data/ml_predictions.parquet", index=False)
        results["picks"].to_parquet("data/ml_picks.parquet", index=False)
        results["feature_importance"].to_csv("data/ml_feature_importance.csv")
        print("\nResults saved to data/ml_*.parquet")


def cmd_compare(args):
    """Compare ML model picks vs simple screener."""
    from screener.criteria import ScreenCriteria, apply_screen
    from data.simfin_loader import get_all_fundamentals_at_date, load_prices
    import pandas as pd
    import numpy as np

    print("Loading ML predictions...")
    picks = pd.read_parquet("data/ml_picks.parquet")

    print("Loading price data...")
    prices = load_prices()
    price_matrix = prices.pivot_table(
        index="Date", columns="Ticker", values="Adj. Close", aggfunc="first"
    ).ffill(limit=5)
    daily_returns = price_matrix.pct_change().clip(-1.0, 1.0).replace([np.inf, -np.inf], 0).fillna(0)

    dates = sorted(picks["date"].unique())
    criteria = ScreenCriteria()

    ml_quarterly = []
    screen_quarterly = []

    for i, date in enumerate(dates):
        end_date = dates[i + 1] if i + 1 < len(dates) else None
        if end_date is None:
            break

        # ML picks
        ml_tickers = picks[picks["date"] == date]["ticker"].tolist()
        ml_tickers = [t for t in ml_tickers if t in daily_returns.columns]

        # Screener picks
        fundamentals = get_all_fundamentals_at_date(date)
        screened = apply_screen(fundamentals, criteria)
        screen_tickers = screened.head(20)["Ticker"].tolist() if not screened.empty else []
        screen_tickers = [t for t in screen_tickers if t in daily_returns.columns]

        mask = (daily_returns.index >= date) & (daily_returns.index < end_date)

        if ml_tickers:
            ml_ret = daily_returns.loc[mask, ml_tickers].mean(axis=1)
            ml_quarterly.append({"date": date, "return": (1 + ml_ret).prod() - 1})

        if screen_tickers:
            sc_ret = daily_returns.loc[mask, screen_tickers].mean(axis=1)
            screen_quarterly.append({"date": date, "return": (1 + sc_ret).prod() - 1})

    ml_df = pd.DataFrame(ml_quarterly)
    sc_df = pd.DataFrame(screen_quarterly)

    print("\n" + "=" * 60)
    print("ML MODEL vs SIMPLE SCREENER")
    print("=" * 60)

    if not ml_df.empty:
        ml_total = (1 + ml_df["return"]).prod()
        ml_years = len(ml_df) / 4
        ml_cagr = ml_total ** (1/ml_years) - 1 if ml_years > 0 else 0
        print(f"\nML Model:")
        print(f"  Quarters: {len(ml_df)}")
        print(f"  Avg quarterly: {ml_df['return'].mean():+.2%}")
        print(f"  CAGR: {ml_cagr:.2%}")
        print(f"  Total: {ml_total - 1:.0%}")

    if not sc_df.empty:
        sc_total = (1 + sc_df["return"]).prod()
        sc_years = len(sc_df) / 4
        sc_cagr = sc_total ** (1/sc_years) - 1 if sc_years > 0 else 0
        print(f"\nSimple Screener:")
        print(f"  Quarters: {len(sc_df)}")
        print(f"  Avg quarterly: {sc_df['return'].mean():+.2%}")
        print(f"  CAGR: {sc_cagr:.2%}")
        print(f"  Total: {sc_total - 1:.0%}")

    if not ml_df.empty and not sc_df.empty:
        print(f"\n  ML excess CAGR: {ml_cagr - sc_cagr:+.2%}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="ML stock prediction model")
    sub = parser.add_subparsers(dest="command")

    bp = sub.add_parser("build", help="Build training dataset")
    bp.add_argument("--start-year", type=int, default=2006)
    bp.add_argument("--end-year", type=int, default=2024)

    tp = sub.add_parser("train", help="Train and evaluate model")
    tp.add_argument("--train-years", type=int, default=5)
    tp.add_argument("--top-n", type=int, default=20)
    tp.add_argument("--n-estimators", type=int, default=300)

    sub.add_parser("compare", help="Compare ML vs screener")

    fp = sub.add_parser("full", help="Build + train end-to-end")
    fp.add_argument("--start-year", type=int, default=2006)
    fp.add_argument("--end-year", type=int, default=2024)
    fp.add_argument("--train-years", type=int, default=5)
    fp.add_argument("--top-n", type=int, default=20)

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "full":
        args.n_estimators = 300
        cmd_build(args)
        cmd_train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
