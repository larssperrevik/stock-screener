"""Combined strategy: Screener filters quality, ML ranks timing.

Flow:
1. Screener passes ~100-200 quality companies (Buffett/Munger filter)
2. ML model scores each on 3-month forward probability
3. Only buy the top N if they exceed a confidence threshold
4. If nothing is confident enough, stay in cash (or benchmark)

This is the final strategy we backtest and use live.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from data.simfin_loader import load_prices, load_derived_annual, get_all_fundamentals_at_date
from screener.criteria import ScreenCriteria, apply_screen
from ml.features import build_fundamental_features, build_price_features
from ml.model import get_feature_cols, save_dataset, load_dataset
from metrics.performance import full_report, print_report

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


class CombinedStrategy:
    def __init__(
        self,
        criteria=None,
        train_years=5,
        top_n=20,
        min_confidence=0.3,
        rebalance_freq="quarterly",
        start_date="2011-01-01",
        end_date="2024-09-30",
        benchmark="SPY",
    ):
        self.criteria = criteria or ScreenCriteria()
        self.train_years = train_years
        self.top_n = top_n
        self.min_confidence = min_confidence
        self.rebalance_freq = rebalance_freq
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.benchmark = benchmark

    def run(self):
        """Run the combined screener + ML backtest."""
        if lgb is None:
            raise ImportError("lightgbm required")

        # Load all data
        derived = load_derived_annual()
        prices = load_prices()

        print("Building price matrix...")
        price_matrix = prices.pivot_table(
            index="Date", columns="Ticker", values="Adj. Close", aggfunc="first"
        )
        price_matrix = price_matrix.ffill(limit=5)
        daily_returns = price_matrix.pct_change().clip(-1.0, 1.0).replace(
            [np.inf, -np.inf], 0.0
        ).fillna(0.0)

        # Load pre-built ML dataset for training data
        print("Loading ML dataset...")
        ml_dataset = load_dataset()
        feature_cols = get_feature_cols(ml_dataset)

        # Rebalance dates
        freq_map = {"annual": "YS", "quarterly": "QS", "monthly": "MS"}
        freq = freq_map.get(self.rebalance_freq, "QS")
        rebalance_dates = pd.date_range(self.start_date, self.end_date, freq=freq)

        holdings_history = []
        cash_quarters = 0

        print(f"\nRunning combined strategy: {len(rebalance_dates)} rebalance dates")
        print(f"  Screener -> ML ranking -> top {self.top_n} above {self.min_confidence:.0%} confidence")
        print()

        for date in rebalance_dates:
            date_str = date.strftime("%Y-%m-%d")

            # === STEP 1: Screen for quality ===
            fundamentals = get_all_fundamentals_at_date(date)
            screened = apply_screen(fundamentals, self.criteria)

            if screened.empty:
                holdings_history.append({"date": date, "tickers": [], "scores": {}, "cash": True})
                cash_quarters += 1
                print(f"  {date_str}: CASH (0 passed screen)")
                continue

            screen_tickers = screened["Ticker"].tolist()

            # === STEP 2: Train ML model on data up to this date ===
            train_start = date - pd.DateOffset(years=self.train_years)
            train_mask = (
                (ml_dataset["date"] >= train_start.strftime("%Y-%m-%d")) &
                (ml_dataset["date"] < date_str)
            )
            train_data = ml_dataset[train_mask]

            if len(train_data) < 200:
                holdings_history.append({"date": date, "tickers": screen_tickers[:self.top_n],
                                         "scores": {}, "cash": False})
                print(f"  {date_str}: {len(screen_tickers)} screened, ML skipped (insufficient training data)")
                continue

            X_train = train_data[feature_cols]
            y_train = train_data["target_buy"]

            model = lgb.LGBMClassifier(
                objective="binary", metric="auc", n_estimators=300,
                learning_rate=0.05, max_depth=6, min_child_samples=50,
                subsample=0.8, colsample_bytree=0.8, verbose=-1,
                n_jobs=-1, random_state=42,
            )
            model.fit(X_train, y_train)

            # === STEP 3: Score screened stocks ===
            fund_features = build_fundamental_features(derived, date)
            price_features = build_price_features(prices, price_matrix, date)
            all_features = fund_features.join(price_features, how="inner")

            # Filter to screened tickers that have features
            scoreable = [t for t in screen_tickers if t in all_features.index]
            if not scoreable:
                holdings_history.append({"date": date, "tickers": [], "scores": {}, "cash": True})
                cash_quarters += 1
                print(f"  {date_str}: CASH ({len(screen_tickers)} screened but no ML features)")
                continue

            X_score = all_features.loc[scoreable]
            # Align columns with training features
            for col in feature_cols:
                if col not in X_score.columns:
                    X_score[col] = np.nan
            X_score = X_score[feature_cols]

            proba = model.predict_proba(X_score)[:, 1]
            scores = pd.Series(proba, index=scoreable)
            scores = scores.sort_values(ascending=False)

            # === STEP 4: Apply confidence threshold ===
            confident = scores[scores >= self.min_confidence]

            if confident.empty:
                holdings_history.append({"date": date, "tickers": [], "scores": scores.to_dict(),
                                         "cash": True})
                cash_quarters += 1
                print(f"  {date_str}: CASH ({len(screen_tickers)} screened, "
                      f"{len(scoreable)} scored, none above {self.min_confidence:.0%})")
                continue

            picks = confident.head(self.top_n)
            holdings_history.append({
                "date": date,
                "tickers": picks.index.tolist(),
                "scores": picks.to_dict(),
                "cash": False,
            })

            print(f"  {date_str}: {len(picks)} picks "
                  f"(from {len(screen_tickers)} screened, {len(confident)} confident) "
                  f"top={picks.iloc[0]:.2f} bottom={picks.iloc[-1]:.2f}")

        # === Build portfolio returns ===
        print(f"\nComputing returns... ({cash_quarters} cash quarters)")
        portfolio_returns = pd.Series(0.0, index=daily_returns.index, dtype=float)

        for i, entry in enumerate(holdings_history):
            start = entry["date"]
            end = holdings_history[i + 1]["date"] if i + 1 < len(holdings_history) else self.end_date
            mask = (daily_returns.index >= start) & (daily_returns.index < end)

            if entry["cash"]:
                # Cash: assume risk-free rate (~4% annualized)
                n_days = mask.sum()
                if n_days > 0:
                    daily_rf = (1 + 0.04) ** (1/252) - 1
                    portfolio_returns.loc[mask] = daily_rf
            else:
                tickers = [t for t in entry["tickers"] if t in daily_returns.columns]
                if tickers:
                    portfolio_returns.loc[mask] = daily_returns.loc[mask, tickers].mean(axis=1)

        portfolio_returns = portfolio_returns.loc[
            (portfolio_returns.index >= self.start_date) &
            (portfolio_returns.index <= self.end_date)
        ]
        portfolio_returns = portfolio_returns[portfolio_returns.index.isin(daily_returns.index)]

        # Benchmark
        bench_returns = None
        if self.benchmark in daily_returns.columns:
            bench_returns = daily_returns[self.benchmark].reindex(portfolio_returns.index).fillna(0.0)

        report = full_report(portfolio_returns, bench_returns)
        print_report(report)

        # Summary
        total_q = len(holdings_history)
        invested_q = total_q - cash_quarters
        print(f"\n  Invested: {invested_q}/{total_q} quarters ({invested_q/total_q:.0%})")
        print(f"  Cash:     {cash_quarters}/{total_q} quarters ({cash_quarters/total_q:.0%})")

        # Show holdings
        print("\nHoldings:")
        for entry in holdings_history:
            d = entry["date"].strftime("%Y-%m-%d")
            if entry["cash"]:
                print(f"  {d}: CASH")
            else:
                tickers = entry["tickers"][:8]
                scores = entry["scores"]
                parts = [f"{t}({scores.get(t, 0):.2f})" for t in tickers]
                more = f" +{len(entry['tickers']) - 8}" if len(entry["tickers"]) > 8 else ""
                print(f"  {d}: {', '.join(parts)}{more}")

        return {
            "portfolio_returns": portfolio_returns,
            "benchmark_returns": bench_returns,
            "holdings_history": holdings_history,
            "report": report,
            "cash_quarters": cash_quarters,
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Combined screener + ML strategy")
    parser.add_argument("--start", default="2011-01-01")
    parser.add_argument("--end", default="2024-09-30")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-confidence", type=float, default=0.3,
                        help="Min ML probability to buy (0-1). Higher = more selective.")
    parser.add_argument("--min-roe", type=float, default=0.15)
    parser.add_argument("--min-roic", type=float, default=0.10)
    parser.add_argument("--min-piotroski", type=int, default=5)
    args = parser.parse_args()

    criteria = ScreenCriteria(
        min_roe=args.min_roe, min_roic=args.min_roic, min_piotroski=args.min_piotroski,
    )
    strategy = CombinedStrategy(
        criteria=criteria,
        top_n=args.top_n,
        min_confidence=args.min_confidence,
        start_date=args.start,
        end_date=args.end,
    )
    strategy.run()
