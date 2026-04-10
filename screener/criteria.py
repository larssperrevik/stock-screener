"""Screening criteria inspired by value investing principles."""

import pandas as pd
from dataclasses import dataclass


@dataclass
class ScreenCriteria:
    min_market_cap: float = 2e9
    max_pe_ratio: float = 25.0
    max_pb_ratio: float = 5.0
    min_roe: float = 0.15
    min_roa: float = 0.07
    max_debt_to_equity: float = 1.5       # FMP returns ratio not percentage
    min_current_ratio: float = 1.2
    min_operating_margin: float = 0.15
    min_fcf_yield: float = 0.04
    min_gross_margin: float = 0.30


def piotroski_score(f, prior=None):
    """Compute Piotroski F-Score (0-9). Higher = healthier."""
    score = 0
    if f.get("roa") and f["roa"] > 0:
        score += 1
    if f.get("operating_cash_flow") and f["operating_cash_flow"] > 0:
        score += 1
    # Quality of earnings: cash flow > net income (using income_quality > 1)
    if f.get("income_quality") and f["income_quality"] > 1:
        score += 1
    if f.get("current_ratio") and f["current_ratio"] > 1:
        score += 1
    if f.get("profit_margins") and f["profit_margins"] > 0:
        score += 1
    if prior:
        if f.get("roa") and prior.get("roa") and f["roa"] > prior["roa"]:
            score += 1
        if (f.get("debt_to_equity") and prior.get("debt_to_equity")
                and f["debt_to_equity"] < prior["debt_to_equity"]):
            score += 1
        if (f.get("gross_margins") and prior.get("gross_margins")
                and f["gross_margins"] > prior["gross_margins"]):
            score += 1
        if (f.get("operating_margins") and prior.get("operating_margins")
                and f["operating_margins"] > prior["operating_margins"]):
            score += 1
    return score


def apply_screen(fundamentals_list, criteria):
    """Apply screening criteria. Returns DataFrame ranked by composite score."""
    rows = []
    for f in fundamentals_list:
        if not f.get("market_cap") or f["market_cap"] < criteria.min_market_cap:
            continue
        if f.get("pe_ratio") is not None and (f["pe_ratio"] > criteria.max_pe_ratio or f["pe_ratio"] < 0):
            continue
        if f.get("pb_ratio") is not None and f["pb_ratio"] > criteria.max_pb_ratio:
            continue
        if f.get("roe") is not None and f["roe"] < criteria.min_roe:
            continue
        if f.get("roa") is not None and f["roa"] < criteria.min_roa:
            continue
        if f.get("debt_to_equity") is not None and f["debt_to_equity"] > criteria.max_debt_to_equity:
            continue
        if f.get("current_ratio") is not None and f["current_ratio"] < criteria.min_current_ratio:
            continue
        if f.get("operating_margins") is not None and f["operating_margins"] < criteria.min_operating_margin:
            continue
        if f.get("fcf_yield") is not None and f["fcf_yield"] < criteria.min_fcf_yield:
            continue
        if f.get("gross_margins") is not None and f["gross_margins"] < criteria.min_gross_margin:
            continue

        f["piotroski"] = piotroski_score(f)
        rows.append(f)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    rank_cols = []
    for col, ascending in [("fcf_yield", False), ("roe", False),
                           ("earnings_yield", False), ("piotroski", False),
                           ("debt_to_equity", True), ("roic", False)]:
        if col in df.columns and df[col].notna().any():
            df[f"{col}_rank"] = df[col].rank(ascending=ascending, na_option="bottom")
            rank_cols.append(f"{col}_rank")

    if rank_cols:
        df["composite_rank"] = df[rank_cols].mean(axis=1)
        df = df.sort_values("composite_rank")

    return df
