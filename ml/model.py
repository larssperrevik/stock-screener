"""ML model for stock prediction using LightGBM.

Walk-forward validation: train on past, predict future.
Never uses future data for training.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
    print("Warning: lightgbm not installed. Run: pip install lightgbm")

from metrics.performance import full_report, print_report


# Feature columns (exclude metadata and targets)
EXCLUDE_COLS = {
    "ticker", "date", "sector_name",
    "forward_return_3m", "target_quintile", "target_buy", "target_avoid",
}


def get_feature_cols(df):
    return [c for c in df.columns if c not in EXCLUDE_COLS and df[c].dtype in ("float64", "float32", "int64", "int32")]


class WalkForwardModel:
    """Walk-forward stock prediction model.

    Train on N years of history, predict the next quarter.
    Slide forward one quarter at a time.
    """

    def __init__(
        self,
        train_years=5,
        target_col="target_buy",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
    ):
        self.train_years = train_years
        self.target_col = target_col
        self.params = {
            "objective": "binary",
            "metric": "auc",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "verbose": -1,
            "n_jobs": -1,
            "random_state": 42,
        }

    def run(self, dataset, top_n=20):
        """Run walk-forward prediction and build a portfolio.

        Returns predictions, portfolio picks, and performance metrics.
        """
        if lgb is None:
            raise ImportError("lightgbm required. pip install lightgbm")

        feature_cols = get_feature_cols(dataset)
        print(f"Features: {len(feature_cols)}")
        print(f"Dataset: {len(dataset)} samples")

        dates = sorted(dataset["date"].unique())
        train_cutoff_idx = 0

        # Find first date where we have enough training history
        for i, d in enumerate(dates):
            if (pd.Timestamp(d) - pd.Timestamp(dates[0])).days > self.train_years * 365:
                train_cutoff_idx = i
                break

        if train_cutoff_idx == 0:
            print("Not enough history for walk-forward validation")
            return {}

        all_predictions = []
        all_picks = []
        models = []
        feature_importance_sum = pd.Series(0.0, index=feature_cols)

        print(f"\nWalk-forward from {dates[train_cutoff_idx]} to {dates[-1]}")
        print(f"Training window: {self.train_years} years rolling\n")

        for pred_idx in range(train_cutoff_idx, len(dates)):
            pred_date = dates[pred_idx]
            train_start = pd.Timestamp(pred_date) - pd.DateOffset(years=self.train_years)

            # Train set: all data before prediction date within the rolling window
            train_mask = (
                (dataset["date"] >= train_start.strftime("%Y-%m-%d")) &
                (dataset["date"] < pred_date)
            )
            test_mask = dataset["date"] == pred_date

            train = dataset[train_mask]
            test = dataset[test_mask]

            if len(train) < 100 or len(test) < 10:
                continue

            X_train = train[feature_cols]
            y_train = train[self.target_col]
            X_test = test[feature_cols]

            # Train
            model = lgb.LGBMClassifier(**self.params)
            model.fit(X_train, y_train)

            # Predict probability of being a "buy"
            proba = model.predict_proba(X_test)[:, 1]

            # Store predictions
            preds = test[["ticker", "date", "forward_return_3m"]].copy()
            preds["ml_score"] = proba
            preds["ml_rank"] = preds["ml_score"].rank(ascending=False)
            all_predictions.append(preds)

            # Top N picks for this period
            top = preds.nsmallest(top_n, "ml_rank")
            all_picks.append(top)
            avg_ret = top["forward_return_3m"].mean()
            all_ret = preds["forward_return_3m"].mean()
            print(f"  {pred_date}: top {top_n} avg return {avg_ret:+.1%} "
                  f"(universe avg {all_ret:+.1%}, "
                  f"samples: {len(train)} train, {len(test)} test)")

            # Feature importance
            fi = pd.Series(model.feature_importances_, index=feature_cols)
            feature_importance_sum += fi
            models.append(model)

        if not all_predictions:
            print("No predictions generated")
            return {}

        predictions = pd.concat(all_predictions, ignore_index=True)
        picks = pd.concat(all_picks, ignore_index=True)

        # Feature importance
        fi_avg = feature_importance_sum / len(models)
        fi_avg = fi_avg.sort_values(ascending=False)

        # Performance analysis
        results = self._analyze_performance(predictions, picks, top_n)
        results["predictions"] = predictions
        results["picks"] = picks
        results["feature_importance"] = fi_avg
        results["models"] = models

        return results

    def _analyze_performance(self, predictions, picks, top_n):
        """Analyze prediction quality and portfolio performance."""
        print("\n" + "=" * 60)
        print("ML MODEL RESULTS")
        print("=" * 60)

        # Quintile analysis: do higher ML scores predict higher returns?
        predictions["ml_quintile"] = predictions.groupby("date")["ml_score"].transform(
            lambda x: pd.qcut(x, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
        )
        quintile_returns = predictions.groupby("ml_quintile")["forward_return_3m"].mean()
        print("\nQuintile Analysis (avg 3-month return by ML score quintile):")
        for q, r in quintile_returns.items():
            bar = "+" * max(0, int(r * 100)) if r > 0 else "-" * max(0, int(-r * 100))
            print(f"  Q{q}: {r:+.2%}  {bar}")

        spread = quintile_returns.iloc[-1] - quintile_returns.iloc[0] if len(quintile_returns) >= 2 else 0
        print(f"\n  Q5-Q1 spread: {spread:+.2%} per quarter")

        # Top N picks performance
        pick_returns = picks.groupby("date")["forward_return_3m"].mean()
        universe_returns = predictions.groupby("date")["forward_return_3m"].mean()

        print(f"\nTop {top_n} Picks vs Universe:")
        print(f"  Avg quarterly return (picks):    {pick_returns.mean():+.2%}")
        print(f"  Avg quarterly return (universe): {universe_returns.mean():+.2%}")
        print(f"  Excess per quarter:              {(pick_returns.mean() - universe_returns.mean()):+.2%}")

        # Win rate
        outperform = (pick_returns > universe_returns).mean()
        print(f"  Quarters outperforming universe: {outperform:.0%}")

        # Annualized
        n_quarters = len(pick_returns)
        if n_quarters > 0:
            total_pick = (1 + pick_returns).prod()
            total_universe = (1 + universe_returns).prod()
            years = n_quarters / 4
            pick_cagr = total_pick ** (1 / years) - 1 if years > 0 else 0
            universe_cagr = total_universe ** (1 / years) - 1 if years > 0 else 0
            print(f"\n  Annualized (picks):    {pick_cagr:.2%}")
            print(f"  Annualized (universe): {universe_cagr:.2%}")
            print(f"  Annualized excess:     {pick_cagr - universe_cagr:+.2%}")

        print("=" * 60)

        return {
            "quintile_returns": quintile_returns,
            "pick_quarterly_returns": pick_returns,
            "universe_quarterly_returns": universe_returns,
            "win_rate": outperform,
        }


def print_feature_importance(fi, top_n=25):
    """Pretty-print feature importance."""
    print(f"\nTop {top_n} Features:")
    print("-" * 50)
    max_imp = fi.iloc[0] if len(fi) > 0 else 1
    for i, (name, imp) in enumerate(fi.head(top_n).items()):
        bar = "#" * int(30 * imp / max_imp)
        print(f"  {i+1:2d}. {name:35s} {imp:6.0f}  {bar}")


def save_dataset(df, path="data/ml_dataset.parquet"):
    """Save the training dataset for reuse."""
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} samples to {path}")


def load_dataset(path="data/ml_dataset.parquet"):
    """Load a previously built dataset."""
    return pd.read_parquet(path)
