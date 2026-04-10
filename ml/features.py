"""Feature engineering for the ML model.

Builds features from SimFin data across multiple signal groups:
- Quality: ROIC, ROE, Piotroski, margins
- Moat: margin stability, share count trends, ROIC consistency
- Management: capital allocation, buyback rate
- Valuation: P/E, P/B, EV/EBITDA, earnings yield
- Momentum: 3m/6m/12m price returns, relative strength

Optimized for bulk computation using vectorized pandas operations.
"""

import pandas as pd
import numpy as np
from data.simfin_loader import load_derived_annual, load_prices, load_companies, load_industries


def _linear_slope(arr):
    """Simple linear regression slope, normalized. Works on arrays."""
    y = np.array(arr, dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    y_clean = y[mask]
    x = np.arange(len(y_clean), dtype=float)
    slope = np.polyfit(x, y_clean, 1)[0]
    mean = np.abs(y_clean).mean()
    return slope / mean if mean != 0 else 0.0


def _rolling_stat(group, col, n_years, stat):
    """Compute a rolling statistic over the last n_years of annual data."""
    vals = group[col].dropna().tail(n_years)
    if len(vals) < 2:
        return np.nan
    if stat == "mean":
        return vals.mean()
    elif stat == "std":
        return vals.std()
    elif stat == "min":
        return vals.min()
    elif stat == "slope":
        return _linear_slope(vals.values)
    elif stat == "positive_pct":
        return (vals > 0).mean()
    elif stat == "above_threshold":
        return (vals > 0.10).mean()
    return np.nan


def build_fundamental_features(derived, as_of_date):
    """Build quality + moat + management features for ALL tickers at once.

    Returns a DataFrame indexed by Ticker with all fundamental features.
    """
    mask = derived["Publish Date"] <= as_of_date
    available = derived[mask].copy()

    # Group by ticker, sorted by fiscal year
    available = available.sort_values(["Ticker", "Fiscal Year"])

    # Latest record per ticker (for quality features)
    latest = available.groupby("Ticker").last()

    features = pd.DataFrame(index=latest.index)

    # === QUALITY ===
    for src, dst in [
        ("Return on Equity", "q_roe"),
        ("Return on Assets", "q_roa"),
        ("Return On Invested Capital", "q_roic"),
        ("Piotroski F-Score", "q_piotroski"),
        ("Gross Profit Margin", "q_gross_margin"),
        ("Operating Margin", "q_op_margin"),
        ("Net Profit Margin", "q_net_margin"),
        ("Current Ratio", "q_current_ratio"),
        ("Debt Ratio", "q_debt_ratio"),
        ("Free Cash Flow to Net Income", "q_fcf_to_ni"),
    ]:
        if src in latest.columns:
            features[dst] = latest[src]

    # === MOAT (rolling stats over last 5 years) ===
    def _compute_rolling_features(group):
        result = {}
        for col, prefix in [
            ("Gross Profit Margin", "moat_gm"),
            ("Operating Margin", "moat_om"),
            ("Return On Invested Capital", "moat_roic"),
        ]:
            vals = group[col].dropna().tail(5)
            if len(vals) >= 3:
                result[f"{prefix}_mean_5y"] = vals.mean()
                result[f"{prefix}_std_5y"] = vals.std()
                result[f"{prefix}_min_5y"] = vals.min()
                result[f"{prefix}_trend"] = _linear_slope(vals.values)

        # ROIC above 10% frequency
        roic = group["Return On Invested Capital"].dropna().tail(5)
        if len(roic) >= 3:
            result["moat_roic_above_10_pct"] = (roic > 0.10).mean()

        # Revenue growth
        rev = group["Sales Per Share"].dropna().tail(6)
        if len(rev) >= 3:
            growth = rev.pct_change().dropna().replace([np.inf, -np.inf], np.nan).dropna()
            if len(growth) >= 2:
                result["moat_rev_growth_mean"] = growth.tail(5).mean()
                result["moat_rev_growth_std"] = growth.tail(5).std()
                result["moat_rev_positive_years"] = (growth.tail(5) > 0).mean()

        # EPS growth
        eps = group["Earnings Per Share, Diluted"].dropna().tail(6)
        if len(eps) >= 3:
            growth = eps.pct_change().dropna().replace([np.inf, -np.inf], np.nan).dropna()
            if len(growth) >= 2:
                result["moat_eps_growth_mean"] = growth.tail(5).mean()
                result["moat_eps_growth_positive"] = (growth.tail(5) > 0).mean()

        # === MANAGEMENT ===
        # Equity per share growth (buyback proxy)
        eqps = group["Equity Per Share"].dropna().tail(4)
        if len(eqps) >= 2:
            eq_growth = eqps.pct_change().dropna().replace([np.inf, -np.inf], np.nan).dropna()
            if len(eq_growth) >= 1:
                result["mgmt_equity_per_share_growth"] = eq_growth.tail(3).mean()

        # Dividend payout
        dpr = group["Dividend Payout Ratio"].dropna()
        if len(dpr) >= 1:
            result["mgmt_payout_ratio"] = dpr.iloc[-1]
            if len(dpr) >= 3:
                result["mgmt_payout_stability"] = dpr.tail(3).std()

        # FCF conversion
        fcf_ni = group["Free Cash Flow to Net Income"].dropna().tail(5)
        if len(fcf_ni) >= 3:
            result["mgmt_fcf_conversion_mean"] = fcf_ni.mean()
            result["mgmt_fcf_conversion_min"] = fcf_ni.min()

        # Debt trend
        debt = group["Debt Ratio"].dropna().tail(5)
        if len(debt) >= 3:
            result["mgmt_debt_trend"] = _linear_slope(debt.values)
            result["mgmt_debt_change_3y"] = debt.iloc[-1] - debt.iloc[-min(3, len(debt))]

        # ROIC trend
        roic_all = group["Return On Invested Capital"].dropna().tail(5)
        if len(roic_all) >= 3:
            result["mgmt_roic_trend"] = _linear_slope(roic_all.values)

        return pd.Series(result)

    print("  Computing fundamental features...", end=" ", flush=True)
    rolling_list = []
    for ticker, group in available.groupby("Ticker"):
        row = _compute_rolling_features(group)
        row.name = ticker
        rolling_list.append(row)
    if rolling_list:
        rolling_features = pd.DataFrame(rolling_list)
        rolling_features.index.name = "Ticker"
        features = features.join(rolling_features)
    print(f"{len(features)} tickers")

    return features


def build_price_features(prices, price_matrix, as_of_date):
    """Build valuation + momentum features for ALL tickers at once.

    Uses pre-built price_matrix for speed.
    """
    date = pd.Timestamp(as_of_date)

    # Get the closest trading day <= as_of_date
    valid_dates = price_matrix.index[price_matrix.index <= date]
    if valid_dates.empty:
        return pd.DataFrame()

    # === VALUATION (from raw prices with ratios) ===
    print("  Computing valuation features...", end=" ", flush=True)
    val_cols = {
        "Price to Earnings Ratio (ttm)": "val_pe_ttm",
        "Price to Sales Ratio (ttm)": "val_ps_ttm",
        "Price to Book Value": "val_pb",
        "Price to Free Cash Flow (ttm)": "val_pfcf_ttm",
        "EV/EBITDA": "val_ev_ebitda",
        "EV/Sales": "val_ev_sales",
        "Book to Market Value": "val_book_to_market",
        "Operating Income/EV": "val_op_income_ev",
        "Dividend Yield": "val_dividend_yield",
    }

    # Get the last row per ticker before as_of_date from raw prices
    mask = (prices["Date"] >= date - pd.Timedelta(days=10)) & (prices["Date"] <= date)
    recent = prices[mask].groupby("Ticker").last()

    val_features = pd.DataFrame(index=recent.index)
    for src, dst in val_cols.items():
        if src in recent.columns:
            val_features[dst] = recent[src]

    # Earnings yield
    if "val_pe_ttm" in val_features.columns:
        pe = val_features["val_pe_ttm"]
        val_features["val_earnings_yield"] = np.where(pe > 0, 1.0 / pe, np.nan)

    print(f"{len(val_features)} tickers")

    # === MOMENTUM ===
    print("  Computing momentum features...", end=" ", flush=True)
    # Work with the price matrix (already pivoted)
    pm = price_matrix.loc[price_matrix.index <= date]
    if len(pm) < 60:
        return val_features

    latest = pm.iloc[-1]
    mom_features = pd.DataFrame(index=pm.columns)

    for months, label in [(1, "1m"), (3, "3m"), (6, "6m"), (12, "12m")]:
        days = months * 21
        if len(pm) > days:
            past = pm.iloc[-days]
            mom_features[f"mom_{label}"] = (latest / past) - 1

    # 12-1 momentum (skip last month)
    if len(pm) > 252:
        mom_features["mom_12m_1m"] = (pm.iloc[-21] / pm.iloc[-252]) - 1

    # Volatility
    daily_ret = pm.pct_change()
    if len(daily_ret) > 63:
        mom_features["mom_vol_3m"] = daily_ret.tail(63).std() * np.sqrt(252)
    if len(daily_ret) > 252:
        mom_features["mom_vol_12m"] = daily_ret.tail(252).std() * np.sqrt(252)

    # Drawdown from 52w high
    if len(pm) > 252:
        high_52w = pm.tail(252).max()
        mom_features["mom_drawdown_52w"] = (latest / high_52w) - 1

    # SMA signals
    if len(pm) > 200:
        sma50 = pm.tail(50).mean()
        sma200 = pm.tail(200).mean()
        mom_features["mom_above_sma50"] = (latest > sma50).astype(float)
        mom_features["mom_above_sma200"] = (latest > sma200).astype(float)
        mom_features["mom_sma50_200_ratio"] = sma50 / sma200

    # Clip extreme momentum values
    for col in mom_features.columns:
        if col.startswith("mom_") and col != "mom_above_sma50" and col != "mom_above_sma200":
            mom_features[col] = mom_features[col].clip(-5, 5)

    print(f"{len(mom_features)} tickers")

    # Combine
    combined = val_features.join(mom_features, how="outer")
    return combined


def build_training_dataset(
    start_year=2006,
    end_year=2024,
    rebalance_freq="QS",
    forward_months=3,
    screener_only=False,
    criteria=None,
):
    """Build the full training dataset with features and forward returns.

    If screener_only=True, only includes stocks that pass the screener
    at each date. This trains the model to differentiate winners within
    the quality universe rather than the broad market.
    """
    from data.simfin_loader import get_all_fundamentals_at_date, get_tradeable_tickers_at_date

    if screener_only:
        from screener.criteria import ScreenCriteria, apply_screen
        if criteria is None:
            criteria = ScreenCriteria()
        print(f"Building SCREENER-FILTERED dataset (only quality stocks)")
    else:
        print(f"Building FULL UNIVERSE dataset")

    derived = load_derived_annual()
    prices = load_prices()

    print("Building price matrix...")
    price_matrix = prices.pivot_table(
        index="Date", columns="Ticker", values="Adj. Close", aggfunc="first"
    )
    price_matrix = price_matrix.ffill(limit=5)

    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq=rebalance_freq)

    all_rows = []
    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        print(f"\n{date_str}:")

        # Forward return target
        forward_date = date + pd.DateOffset(months=forward_months)
        price_at_date = price_matrix.loc[price_matrix.index >= date].head(1)
        price_at_forward = price_matrix.loc[price_matrix.index >= forward_date].head(1)

        if price_at_date.empty or price_at_forward.empty:
            print("  No forward data, skipping")
            continue

        p0 = price_at_date.iloc[0]
        p1 = price_at_forward.iloc[0]
        fwd_returns = ((p1 / p0) - 1).dropna()
        fwd_returns = fwd_returns[(fwd_returns > -1) & (fwd_returns < 5)]

        if fwd_returns.empty:
            continue

        # Filter to screener-eligible tickers if requested
        if screener_only:
            tradeable = set(get_tradeable_tickers_at_date(date))
            fundamentals = get_all_fundamentals_at_date(date)
            fundamentals = fundamentals[fundamentals["Ticker"].isin(tradeable)]
            screened = apply_screen(fundamentals, criteria)
            if screened.empty:
                print("  No stocks passed screen")
                continue
            eligible_tickers = set(screened["Ticker"].tolist())
            print(f"  {len(eligible_tickers)} passed screener,", end=" ")
        else:
            eligible_tickers = None

        # Build features
        fund_features = build_fundamental_features(derived, date)
        price_features = build_price_features(prices, price_matrix, date)

        # Combine
        combined = fund_features.join(price_features, how="inner")

        # Filter to eligible tickers
        if eligible_tickers is not None:
            combined = combined[combined.index.isin(eligible_tickers)]

        # Add forward returns
        tickers_with_data = combined.index.intersection(fwd_returns.index)
        combined = combined.loc[tickers_with_data]
        combined["forward_return_3m"] = fwd_returns[tickers_with_data]
        combined = combined.dropna(subset=["forward_return_3m"])

        combined["ticker"] = combined.index
        combined["date"] = date_str

        all_rows.append(combined)
        print(f"  => {len(combined)} samples with features + forward returns")

    if not all_rows:
        return pd.DataFrame()

    print("\nCombining all dates...")
    df = pd.concat(all_rows, ignore_index=True)

    # Add target labels
    if "forward_return_3m" in df.columns:
        def _safe_qcut(x):
            try:
                return pd.qcut(x, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
            except ValueError:
                # Not enough unique values for 5 bins — use rank percentile
                return pd.cut(x.rank(pct=True), bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                              labels=[1, 2, 3, 4, 5], include_lowest=True)

        df["target_quintile"] = df.groupby("date")["forward_return_3m"].transform(_safe_qcut)
        df["target_buy"] = (df["target_quintile"].astype(int) == 5).astype(int)
        df["target_avoid"] = (df["target_quintile"].astype(int) == 1).astype(int)

    print(f"\nTotal: {len(df)} samples, {df['ticker'].nunique()} tickers, "
          f"{df['date'].nunique()} dates")

    return df
