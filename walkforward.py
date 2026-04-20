"""Walk-forward optimization for the event-driven strategy.

Avoids curve fitting by training on past data only and testing on unseen data.

Design:
- 5-year training window, 1-year test window, rolling forward annually
- ~48 parameter combos swept on each training fold
- Best combo (by Sortino) is applied to the test window
- OOS test results are stitched together for true performance estimate
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import time
import json
from itertools import product
from pathlib import Path

from backtest.event_engine import EventDrivenEngine
from screener.criteria import ScreenCriteria
from metrics.performance import full_report, cagr, sharpe_ratio, sortino_ratio, max_drawdown


# === PARAMETER GRID (~48 combos) ===
GRID = {
    "buy_threshold": [35, 40, 50],
    "sell_threshold": [15, 20],
    "max_positions": [15],
    "max_hold_days": [180, 365, 540, 720],
    "min_hold_days": [30, 60],
}

# Fixed features
FIXED = {
    "max_correlation": 0.65,
    "max_sector_overweight": 0.40,
}

# Walk-forward config
TRAIN_YEARS = 5
TEST_YEARS = 1
FIRST_TRAIN_START = 2011
LAST_TEST_END = 2026  # inclusive (partial year ok)


def build_combos():
    keys = list(GRID.keys())
    values = [GRID[k] for k in keys]
    combos = []
    for vals in product(*values):
        params = dict(zip(keys, vals))
        if params["sell_threshold"] >= params["buy_threshold"]:
            continue
        if params["min_hold_days"] >= params["max_hold_days"]:
            continue
        combos.append(params)
    return combos


def build_folds():
    folds = []
    year = FIRST_TRAIN_START
    while year + TRAIN_YEARS < LAST_TEST_END:
        train_start = f"{year}-01-01"
        train_end = f"{year + TRAIN_YEARS - 1}-12-31"
        test_start = f"{year + TRAIN_YEARS}-01-01"
        test_end_year = year + TRAIN_YEARS + TEST_YEARS - 1
        if test_end_year >= LAST_TEST_END:
            test_end = f"{LAST_TEST_END}-12-31"
        else:
            test_end = f"{test_end_year}-12-31"
        folds.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })
        year += 1
    return folds


def run_engine(params, start, end):
    """Run event engine with given params, return report + returns."""
    criteria = ScreenCriteria()
    engine = EventDrivenEngine(
        criteria=criteria,
        start_date=start,
        end_date=end,
        **params,
        **FIXED,
    )
    result = engine.run(quiet=True)
    return result


def main():
    combos = build_combos()
    folds = build_folds()

    print("=" * 80)
    print("WALK-FORWARD OPTIMIZATION")
    print("=" * 80)
    print(f"Parameter combos: {len(combos)}")
    print(f"Folds: {len(folds)}")
    print(f"Train window: {TRAIN_YEARS} years, Test window: {TEST_YEARS} year")
    print(f"Total runs (train): {len(combos) * len(folds)}")
    print(f"Fixed: correlation={FIXED['max_correlation']}, sector_cap={FIXED['max_sector_overweight']}")
    print()
    for i, f in enumerate(folds):
        print(f"  Fold {i+1}: train {f['train_start']}..{f['train_end']} | test {f['test_start']}..{f['test_end']}")
    print()

    # Print grid
    for k, v in GRID.items():
        print(f"  {k}: {v}")
    print()

    t0 = time.time()
    fold_results = []
    all_oos_returns = []

    for fold_idx, fold in enumerate(folds):
        fold_t0 = time.time()
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx+1}/{len(folds)}: "
              f"Train {fold['train_start']}..{fold['train_end']} | "
              f"Test {fold['test_start']}..{fold['test_end']}")
        print("=" * 80)

        # === TRAINING: sweep all combos ===
        train_results = []
        for ci, combo in enumerate(combos):
            elapsed = time.time() - fold_t0
            runs_done = ci
            if runs_done > 0:
                eta_fold = (elapsed / runs_done) * (len(combos) - runs_done)
            else:
                eta_fold = 0
            print(f"  [{ci+1}/{len(combos)}] buy={combo['buy_threshold']} "
                  f"sell={combo['sell_threshold']} pos={combo['max_positions']} "
                  f"hold={combo['min_hold_days']}-{combo['max_hold_days']}d "
                  f"(ETA fold: {eta_fold/60:.0f}m)", end="")

            try:
                result = run_engine(combo, fold["train_start"], fold["train_end"])
                r = result["report"]
                trades = result["trades"]
                n_trades = len(trades)
                sortino = r.get("sortino", 0)
                if np.isnan(sortino) or np.isinf(sortino):
                    sortino = 0

                print(f" => Sortino={sortino:.2f} CAGR={r['cagr']:.2%} "
                      f"MaxDD={r['max_drawdown']:.2%} Trades={n_trades}")

                train_results.append({
                    "combo": combo,
                    "sortino": sortino,
                    "sharpe": r.get("sharpe", 0),
                    "cagr": r["cagr"],
                    "max_drawdown": r["max_drawdown"],
                    "n_trades": n_trades,
                })
            except Exception as e:
                print(f" => FAILED: {e}")
                train_results.append({
                    "combo": combo,
                    "sortino": -999,
                    "sharpe": 0,
                    "cagr": 0,
                    "max_drawdown": -1,
                    "n_trades": 0,
                })

        # === PICK BEST by Sortino ===
        train_results.sort(key=lambda x: x["sortino"], reverse=True)
        best = train_results[0]
        best_combo = best["combo"]

        print(f"\n  BEST (in-sample): buy={best_combo['buy_threshold']} "
              f"sell={best_combo['sell_threshold']} pos={best_combo['max_positions']} "
              f"hold={best_combo['min_hold_days']}-{best_combo['max_hold_days']}d | "
              f"Sortino={best['sortino']:.2f} CAGR={best['cagr']:.2%}")

        # === TEST: run best combo on OOS window ===
        print(f"\n  Running OOS test {fold['test_start']}..{fold['test_end']}...")
        try:
            oos_result = run_engine(best_combo, fold["test_start"], fold["test_end"])
            oos_report = oos_result["report"]
            oos_returns = oos_result["portfolio_returns"]
            oos_bench = oos_result.get("benchmark_returns")
            oos_trades = oos_result.get("trades", [])

            oos_sortino = oos_report.get("sortino", 0)
            if np.isnan(oos_sortino) or np.isinf(oos_sortino):
                oos_sortino = 0

            print(f"  OOS RESULT: Sortino={oos_sortino:.2f} "
                  f"CAGR={oos_report['cagr']:.2%} "
                  f"MaxDD={oos_report['max_drawdown']:.2%} "
                  f"Trades={len(oos_trades)}")

            all_oos_returns.append(oos_returns)

            fold_results.append({
                "fold": fold_idx + 1,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "best_params": best_combo,
                "train_sortino": best["sortino"],
                "train_cagr": best["cagr"],
                "oos_sortino": oos_sortino,
                "oos_cagr": oos_report["cagr"],
                "oos_sharpe": oos_report.get("sharpe", 0),
                "oos_max_drawdown": oos_report["max_drawdown"],
                "oos_excess": oos_report.get("excess_return", 0),
                "oos_trades": len(oos_trades),
            })
        except Exception as e:
            print(f"  OOS FAILED: {e}")
            fold_results.append({
                "fold": fold_idx + 1,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "best_params": best_combo,
                "train_sortino": best["sortino"],
                "train_cagr": best["cagr"],
                "oos_sortino": 0,
                "oos_cagr": 0,
                "oos_sharpe": 0,
                "oos_max_drawdown": 0,
                "oos_excess": 0,
                "oos_trades": 0,
            })

        fold_time = time.time() - fold_t0
        remaining_folds = len(folds) - fold_idx - 1
        print(f"\n  Fold time: {fold_time/60:.1f}m | "
              f"Remaining: ~{remaining_folds * fold_time / 60:.0f}m")

    # === STITCHED OOS RESULTS ===
    print("\n\n" + "=" * 80)
    print("WALK-FORWARD RESULTS (Out-of-Sample)")
    print("=" * 80)

    print(f"\n{'Fold':>4} {'Test Period':>21} {'Params':>30} | "
          f"{'IS Sort':>7} {'OOS Sort':>8} {'OOS CAGR':>9} {'OOS DD':>7} {'OOS xs':>7} {'Trades':>6}")
    print("-" * 115)

    for fr in fold_results:
        p = fr["best_params"]
        params_str = (f"b={p['buy_threshold']} s={p['sell_threshold']} "
                      f"p={p['max_positions']} h={p['min_hold_days']}-{p['max_hold_days']}d")
        print(f"  {fr['fold']:2d}  {fr['test_start']}..{fr['test_end']}  "
              f"{params_str:>30} | "
              f"{fr['train_sortino']:7.2f} {fr['oos_sortino']:8.2f} "
              f"{fr['oos_cagr']:9.2%} {fr['oos_max_drawdown']:7.2%} "
              f"{fr['oos_excess']:7.2%} {fr['oos_trades']:6d}")

    # Stitch OOS returns
    if all_oos_returns:
        stitched = pd.concat(all_oos_returns)
        # Remove overlaps (keep first)
        stitched = stitched[~stitched.index.duplicated(keep="first")]
        stitched = stitched.sort_index()

        stitched_report = full_report(stitched)

        print(f"\n{'='*80}")
        print("STITCHED OOS PERFORMANCE (what you'd actually get)")
        print("=" * 80)
        print(f"  Period:      {stitched.index[0].strftime('%Y-%m-%d')} to {stitched.index[-1].strftime('%Y-%m-%d')}")
        print(f"  CAGR:        {stitched_report['cagr']:.2%}")
        print(f"  Sharpe:      {stitched_report['sharpe']:.2f}")
        print(f"  Sortino:     {stitched_report['sortino']:.2f}")
        print(f"  Max DD:      {stitched_report['max_drawdown']:.2%}")
        print(f"  Volatility:  {stitched_report['volatility']:.2%}")
        if "excess_return" in stitched_report:
            print(f"  vs SPY:      {stitched_report['excess_return']:.2%}")

        # Compare to SPY over same period
        from data.simfin_loader import load_prices
        prices = load_prices()
        spy = prices[prices["Ticker"] == "SPY"].set_index("Date")["Adj. Close"]
        spy_ret = spy.pct_change().dropna()
        spy_oos = spy_ret.reindex(stitched.index).fillna(0)
        spy_report = full_report(spy_oos)

        print(f"\n  SPY same period:")
        print(f"  CAGR:        {spy_report['cagr']:.2%}")
        print(f"  Sharpe:      {spy_report['sharpe']:.2f}")
        print(f"  Max DD:      {spy_report['max_drawdown']:.2%}")

    # Parameter stability analysis
    print(f"\n{'='*80}")
    print("PARAMETER STABILITY (how often each value was chosen)")
    print("=" * 80)
    for key in GRID:
        counts = {}
        for fr in fold_results:
            val = fr["best_params"][key]
            counts[val] = counts.get(val, 0) + 1
        print(f"  {key}:")
        for val in sorted(counts.keys()):
            bar = "#" * counts[val]
            print(f"    {val:>6}: {bar} ({counts[val]}/{len(fold_results)})")

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time/60:.1f} minutes")

    # Save results
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv("data/walkforward_results.csv", index=False)
    print("Saved to data/walkforward_results.csv")

    if all_oos_returns:
        stitched.to_csv("data/walkforward_oos_returns.csv")
        print("Saved OOS returns to data/walkforward_oos_returns.csv")


if __name__ == "__main__":
    main()
