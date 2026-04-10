"""Feature engineering for the ML model.

Builds features from SimFin data across multiple signal groups:
- Quality: ROIC, ROE, Piotroski, margins
- Moat: margin stability, share count trends, ROIC consistency
- Management: capital allocation, buyback rate
- Valuation: P/E, P/B, EV/EBITDA, earnings yield
- Momentum: 3m/6m/12m price returns, relative strength
"""

import pandas as pd
import numpy as np
from data.simfin_loader import load_derived_annual, load_prices, load_companies, load_industries


def _safe_pct_change(series):
    """Percent change handling zeros and NaN."""
    return series.pct_change().replace([np.inf, -np.inf], np.nan)


def build_quality_features(derived, ticker, as_of_date):
    """Quality metrics from the most recent annual report."""
    mask = (derived["Ticker"] == ticker) & (derived["Publish Date"] <= as_of_date)
    rows = derived[mask].sort_values("Publish Date")
    if rows.empty:
        return {}

    latest = rows.iloc[-1]
    return {
        "q_roe": latest.get("Return on Equity"),
        "q_roa": latest.get("Return on Assets"),
        "q_roic": latest.get("Return On Invested Capital"),
        "q_piotroski": latest.get("Piotroski F-Score"),
        "q_gross_margin": latest.get("Gross Profit Margin"),
        "q_op_margin": latest.get("Operating Margin"),
        "q_net_margin": latest.get("Net Profit Margin"),
        "q_current_ratio": latest.get("Current Ratio"),
        "q_debt_ratio": latest.get("Debt Ratio"),
        "q_fcf_to_ni": latest.get("Free Cash Flow to Net Income"),
        "q_earnings_quality": latest.get("Free Cash Flow to Net Income"),
    }


def build_moat_features(derived, ticker, as_of_date):
    """Moat indicators: stability and durability of competitive advantage."""
    mask = (derived["Ticker"] == ticker) & (derived["Publish Date"] <= as_of_date)
    rows = derived[mask].sort_values("Fiscal Year")
    if len(rows) < 3:
        return {}

    features = {}

    # Gross margin stability (lower std = stronger moat)
    gm = rows["Gross Profit Margin"].dropna()
    if len(gm) >= 3:
        features["moat_gm_mean_5y"] = gm.tail(5).mean()
        features["moat_gm_std_5y"] = gm.tail(5).std()
        features["moat_gm_trend"] = _linear_slope(gm.tail(5))
        features["moat_gm_min_5y"] = gm.tail(5).min()

    # ROIC consistency (Buffett's key metric)
    roic = rows["Return On Invested Capital"].dropna()
    if len(roic) >= 3:
        features["moat_roic_mean_5y"] = roic.tail(5).mean()
        features["moat_roic_std_5y"] = roic.tail(5).std()
        features["moat_roic_min_5y"] = roic.tail(5).min()
        features["moat_roic_above_10_pct"] = (roic.tail(5) > 0.10).mean()

    # Operating margin stability
    om = rows["Operating Margin"].dropna()
    if len(om) >= 3:
        features["moat_om_mean_5y"] = om.tail(5).mean()
        features["moat_om_std_5y"] = om.tail(5).std()
        features["moat_om_trend"] = _linear_slope(om.tail(5))

    # Revenue growth consistency (proxy for pricing power)
    rev = rows["Sales Per Share"].dropna()
    if len(rev) >= 3:
        rev_growth = _safe_pct_change(rev).dropna()
        if len(rev_growth) >= 2:
            features["moat_rev_growth_mean"] = rev_growth.tail(5).mean()
            features["moat_rev_growth_std"] = rev_growth.tail(5).std()
            features["moat_rev_positive_years"] = (rev_growth.tail(5) > 0).mean()

    # Earnings growth consistency
    eps = rows["Earnings Per Share, Diluted"].dropna()
    if len(eps) >= 3:
        eps_growth = _safe_pct_change(eps).dropna()
        if len(eps_growth) >= 2:
            features["moat_eps_growth_mean"] = eps_growth.tail(5).mean()
            features["moat_eps_growth_positive"] = (eps_growth.tail(5) > 0).mean()

    return features


def build_management_features(derived, ticker, as_of_date):
    """Management quality proxies: capital allocation and shareholder friendliness."""
    mask = (derived["Ticker"] == ticker) & (derived["Publish Date"] <= as_of_date)
    rows = derived[mask].sort_values("Fiscal Year")
    if len(rows) < 2:
        return {}

    features = {}

    # Share buyback rate (negative = buyback, positive = dilution)
    # Use Equity Per Share as proxy — rising equity/share with stable earnings = good allocation
    eps_series = rows["Equity Per Share"].dropna()
    if len(eps_series) >= 2:
        equity_growth = _safe_pct_change(eps_series).dropna()
        if len(equity_growth) >= 1:
            features["mgmt_equity_per_share_growth"] = equity_growth.tail(3).mean()

    # Dividend payout ratio (moderate = disciplined, too high = unsustainable)
    dpr = rows["Dividend Payout Ratio"].dropna()
    if len(dpr) >= 1:
        features["mgmt_payout_ratio"] = dpr.iloc[-1]
        if len(dpr) >= 3:
            features["mgmt_payout_stability"] = dpr.tail(3).std()

    # FCF conversion consistency (cash flow > earnings = honest accounting)
    fcf_ni = rows["Free Cash Flow to Net Income"].dropna()
    if len(fcf_ni) >= 3:
        features["mgmt_fcf_conversion_mean"] = fcf_ni.tail(5).mean()
        features["mgmt_fcf_conversion_min"] = fcf_ni.tail(5).min()

    # Debt management: is leverage trending up or down?
    debt = rows["Debt Ratio"].dropna()
    if len(debt) >= 3:
        features["mgmt_debt_trend"] = _linear_slope(debt.tail(5))
        features["mgmt_debt_change_3y"] = debt.iloc[-1] - debt.iloc[-min(3, len(debt))]

    # ROIC improvement over time (management getting better at allocating capital?)
    roic = rows["Return On Invested Capital"].dropna()
    if len(roic) >= 3:
        features["mgmt_roic_trend"] = _linear_slope(roic.tail(5))

    return features


def build_valuation_features(prices_df, ticker, as_of_date):
    """Valuation metrics from price data (point-in-time)."""
    mask = (prices_df["Ticker"] == ticker) & (prices_df["Date"] <= as_of_date)
    rows = prices_df[mask].sort_values("Date")
    if rows.empty:
        return {}

    latest = rows.iloc[-1]
    features = {
        "val_pe_ttm": latest.get("Price to Earnings Ratio (ttm)"),
        "val_ps_ttm": latest.get("Price to Sales Ratio (ttm)"),
        "val_pb": latest.get("Price to Book Value"),
        "val_pfcf_ttm": latest.get("Price to Free Cash Flow (ttm)"),
        "val_ev_ebitda": latest.get("EV/EBITDA"),
        "val_ev_sales": latest.get("EV/Sales"),
        "val_book_to_market": latest.get("Book to Market Value"),
        "val_op_income_ev": latest.get("Operating Income/EV"),
        "val_dividend_yield": latest.get("Dividend Yield"),
    }

    # Earnings yield (inverse P/E, better for comparison)
    pe = features.get("val_pe_ttm")
    if pe and pe > 0:
        features["val_earnings_yield"] = 1.0 / pe
    else:
        features["val_earnings_yield"] = None

    return features


def build_momentum_features(prices_df, ticker, as_of_date):
    """Price momentum and relative strength features."""
    mask = (prices_df["Ticker"] == ticker) & (prices_df["Date"] <= as_of_date)
    rows = prices_df[mask].sort_values("Date")
    if len(rows) < 60:  # need at least ~3 months of data
        return {}

    close = rows.set_index("Date")["Adj. Close"].dropna()
    if len(close) < 60:
        return {}

    latest_price = close.iloc[-1]
    features = {}

    # Absolute momentum
    for months, label in [(1, "1m"), (3, "3m"), (6, "6m"), (12, "12m")]:
        days = months * 21
        if len(close) > days:
            features[f"mom_{label}"] = (latest_price / close.iloc[-days]) - 1

    # Momentum excluding last month (classic 12-1 momentum factor)
    if len(close) > 252:
        features["mom_12m_1m"] = (close.iloc[-21] / close.iloc[-252]) - 1

    # Volatility
    daily_ret = close.pct_change().dropna()
    if len(daily_ret) > 60:
        features["mom_vol_3m"] = daily_ret.tail(63).std() * np.sqrt(252)
        features["mom_vol_12m"] = daily_ret.tail(252).std() * np.sqrt(252) if len(daily_ret) > 252 else None

    # Drawdown from 52-week high
    if len(close) > 252:
        high_52w = close.tail(252).max()
        features["mom_drawdown_52w"] = (latest_price / high_52w) - 1

    # Price vs moving averages
    if len(close) > 200:
        features["mom_above_sma50"] = float(latest_price > close.tail(50).mean())
        features["mom_above_sma200"] = float(latest_price > close.tail(200).mean())
        features["mom_sma50_200_ratio"] = close.tail(50).mean() / close.tail(200).mean()

    return features


def build_sector_features(prices_df, derived, ticker, as_of_date):
    """Relative performance vs sector."""
    companies = load_companies()
    industries = load_industries()

    co = companies[companies["Ticker"] == ticker]
    if co.empty:
        return {}

    ind_id = co.iloc[0].get("IndustryId")
    if pd.isna(ind_id):
        return {}

    ind = industries[industries["IndustryId"] == ind_id]
    sector = ind.iloc[0].get("Sector", "") if not ind.empty else ""

    # Get all tickers in same sector
    sector_ids = industries[industries["Sector"] == sector]["IndustryId"].tolist()
    sector_tickers = companies[companies["IndustryId"].isin(sector_ids)]["Ticker"].tolist()

    features = {"sector_name": sector}

    # Relative ROIC vs sector median
    mask = (derived["Ticker"].isin(sector_tickers)) & (derived["Publish Date"] <= as_of_date)
    sector_data = derived[mask].groupby("Ticker").last()
    if not sector_data.empty and "Return On Invested Capital" in sector_data.columns:
        sector_roic = sector_data["Return On Invested Capital"].dropna()
        ticker_data = derived[(derived["Ticker"] == ticker) & (derived["Publish Date"] <= as_of_date)]
        if not ticker_data.empty:
            ticker_roic = ticker_data.iloc[-1].get("Return On Invested Capital")
            if pd.notna(ticker_roic) and len(sector_roic) > 0:
                features["sector_roic_rank_pct"] = (sector_roic < ticker_roic).mean()
                features["sector_roic_vs_median"] = ticker_roic - sector_roic.median()

    return features


def _linear_slope(series):
    """Simple linear regression slope, normalized."""
    if len(series) < 2:
        return 0.0
    y = series.values.astype(float)
    x = np.arange(len(y), dtype=float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return 0.0
    y, x = y[mask], x[mask]
    slope = np.polyfit(x, y, 1)[0]
    mean = np.abs(y).mean()
    if mean == 0:
        return 0.0
    return slope / mean  # normalized slope


def build_all_features(ticker, as_of_date, derived=None, prices=None):
    """Build all feature groups for one ticker at one date."""
    if derived is None:
        derived = load_derived_annual()
    if prices is None:
        prices = load_prices()

    features = {"ticker": ticker, "date": as_of_date}

    features.update(build_quality_features(derived, ticker, as_of_date))
    features.update(build_moat_features(derived, ticker, as_of_date))
    features.update(build_management_features(derived, ticker, as_of_date))
    features.update(build_valuation_features(prices, ticker, as_of_date))
    features.update(build_momentum_features(prices, ticker, as_of_date))
    features.update(build_sector_features(prices, derived, ticker, as_of_date))

    return features


def build_training_dataset(
    start_year=2006,
    end_year=2024,
    rebalance_freq="QS",
    min_market_cap=1e8,
    forward_months=3,
):
    """Build the full training dataset with features and forward returns.

    For each rebalance date:
    1. Get all tradeable tickers with fundamentals
    2. Build features (point-in-time, no look-ahead)
    3. Compute 3-month forward return as the target

    Returns a DataFrame ready for ML training.
    """
    derived = load_derived_annual()
    prices = load_prices()

    # Build price matrix for fast return lookups
    print("Building price matrix...")
    price_matrix = prices.pivot_table(
        index="Date", columns="Ticker", values="Adj. Close", aggfunc="first"
    )
    price_matrix = price_matrix.ffill(limit=5)

    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq=rebalance_freq)

    all_rows = []
    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        print(f"\n{date_str}:", end=" ", flush=True)

        # Get tickers with fundamentals available at this date
        mask = derived["Publish Date"] <= date
        available = derived[mask].groupby("Ticker").last().reset_index()
        tickers = available["Ticker"].tolist()

        # Filter to tickers with price data around this date
        price_window = price_matrix.loc[
            (price_matrix.index >= date - pd.Timedelta(days=10)) &
            (price_matrix.index <= date + pd.Timedelta(days=5))
        ]
        tradeable = [t for t in tickers if t in price_window.columns and price_window[t].notna().any()]

        # Compute forward returns (the target)
        forward_date = date + pd.DateOffset(months=forward_months)
        price_at_date = price_matrix.loc[price_matrix.index >= date].head(1)
        price_at_forward = price_matrix.loc[price_matrix.index >= forward_date].head(1)

        if price_at_date.empty or price_at_forward.empty:
            print(f"no forward data")
            continue

        forward_returns = {}
        for t in tradeable:
            p0 = price_at_date[t].values[0] if t in price_at_date.columns else np.nan
            p1 = price_at_forward[t].values[0] if t in price_at_forward.columns else np.nan
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                forward_returns[t] = (p1 / p0) - 1

        # Build features for each ticker
        count = 0
        for ticker in tradeable:
            if ticker not in forward_returns:
                continue
            fwd_ret = forward_returns[ticker]
            if abs(fwd_ret) > 5.0:  # filter extreme returns (data errors)
                continue

            features = build_all_features(ticker, date, derived=derived, prices=prices)
            features["forward_return_3m"] = fwd_ret
            all_rows.append(features)
            count += 1

        print(f"{count} samples", end="", flush=True)

    print(f"\n\nTotal samples: {len(all_rows)}")
    df = pd.DataFrame(all_rows)

    # Add target quintile labels
    if "forward_return_3m" in df.columns:
        df["target_quintile"] = df.groupby("date")["forward_return_3m"].transform(
            lambda x: pd.qcut(x, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
        )
        df["target_buy"] = (df["target_quintile"] == 5).astype(int)
        df["target_avoid"] = (df["target_quintile"] == 1).astype(int)

    return df
