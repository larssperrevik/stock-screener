"""Screening criteria inspired by value investing principles.

Works with SimFin derived-annual data columns.
"""

import pandas as pd
from dataclasses import dataclass


@dataclass
class ScreenCriteria:
    min_market_cap: float = 2e9
    max_pe_ratio: float = 25.0
    max_pb_ratio: float = 5.0
    min_roe: float = 0.15
    min_roa: float = 0.07
    max_debt_ratio: float = 0.60           # Total liabilities / Total assets
    min_current_ratio: float = 1.2
    min_operating_margin: float = 0.15
    min_gross_margin: float = 0.30
    min_fcf_to_net_income: float = 0.8     # Cash conversion
    min_piotroski: int = 5                 # Pre-computed in SimFin
    min_roic: float = 0.10
    # Earnings momentum + trend filter (opt-in; default off)
    min_eps_yoy_growth: float = None  # require EPS YoY growth >= this (e.g., 0.0)
    require_eps_acceleration: bool = False  # require this Q YoY growth >= prior Q YoY growth
    require_above_200dma: bool = True  # require current price >= 200-day SMA (validated +4.71pp OOS)


def apply_screen(fundamentals_df, criteria=None):
    """Apply screening criteria to a SimFin derived-annual DataFrame.

    Input: DataFrame from get_all_fundamentals_at_date() with columns like
    'Gross Profit Margin', 'Operating Margin', 'Return on Equity', etc.

    Returns filtered and ranked DataFrame.
    """
    if criteria is None:
        criteria = ScreenCriteria()

    df = fundamentals_df.copy()

    # Apply hard filters
    if "Gross Profit Margin" in df.columns:
        df = df[df["Gross Profit Margin"].isna() | (df["Gross Profit Margin"] >= criteria.min_gross_margin)]
    if "Operating Margin" in df.columns:
        df = df[df["Operating Margin"].isna() | (df["Operating Margin"] >= criteria.min_operating_margin)]
    if "Return on Equity" in df.columns:
        df = df[df["Return on Equity"].isna() | (df["Return on Equity"] >= criteria.min_roe)]
    if "Return on Assets" in df.columns:
        df = df[df["Return on Assets"].isna() | (df["Return on Assets"] >= criteria.min_roa)]
    if "Current Ratio" in df.columns:
        df = df[df["Current Ratio"].isna() | (df["Current Ratio"] >= criteria.min_current_ratio)]
    if "Debt Ratio" in df.columns:
        df = df[df["Debt Ratio"].isna() | (df["Debt Ratio"] <= criteria.max_debt_ratio)]
    if "Piotroski F-Score" in df.columns:
        df = df[df["Piotroski F-Score"].isna() | (df["Piotroski F-Score"] >= criteria.min_piotroski)]
    if "Return On Invested Capital" in df.columns:
        df = df[df["Return On Invested Capital"].isna() | (df["Return On Invested Capital"] >= criteria.min_roic)]
    if "Free Cash Flow to Net Income" in df.columns:
        df = df[df["Free Cash Flow to Net Income"].isna() | (df["Free Cash Flow to Net Income"] >= criteria.min_fcf_to_net_income)]

    if df.empty:
        return df

    # Composite ranking
    rank_cols = []
    for col, ascending in [
        ("Return On Invested Capital", False),
        ("Return on Equity", False),
        ("Piotroski F-Score", False),
        ("Gross Profit Margin", False),
        ("Operating Margin", False),
        ("Debt Ratio", True),
        ("Free Cash Flow to Net Income", False),
    ]:
        if col in df.columns and df[col].notna().any():
            df[f"_rank_{col}"] = df[col].rank(ascending=ascending, na_option="bottom")
            rank_cols.append(f"_rank_{col}")

    if rank_cols:
        df["composite_rank"] = df[rank_cols].mean(axis=1)
        df = df.sort_values("composite_rank")

    return df
