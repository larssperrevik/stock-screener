"""Portfolio performance metrics beyond simple returns."""

import numpy as np
import pandas as pd


def cagr(returns):
    total = (1 + returns).prod()
    years = len(returns) / 252
    if years <= 0 or total <= 0:
        return 0.0
    return total ** (1 / years) - 1


def sharpe_ratio(returns, risk_free_rate=0.04):
    if returns.std() == 0:
        return 0.0
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    excess = returns - daily_rf
    return np.sqrt(252) * excess.mean() / excess.std()


def sortino_ratio(returns, risk_free_rate=0.04):
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return np.sqrt(252) * excess.mean() / downside.std()


def max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def ulcer_index(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown_pct = 100 * (cumulative - peak) / peak
    return np.sqrt((drawdown_pct ** 2).mean())


def rolling_returns(returns, window=252):
    return (1 + returns).rolling(window).apply(
        lambda x: x.prod() ** (252 / window) - 1, raw=True
    )


def hit_rate(trade_returns):
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    rate = len(wins) / len(trade_returns) if len(trade_returns) > 0 else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    payoff = avg_win / avg_loss if avg_loss > 0 else float("inf")
    return {"hit_rate": rate, "avg_win": avg_win, "avg_loss": avg_loss, "payoff_ratio": payoff}


def full_report(portfolio_returns, benchmark_returns=None):
    report = {
        "cagr": cagr(portfolio_returns),
        "sharpe": sharpe_ratio(portfolio_returns),
        "sortino": sortino_ratio(portfolio_returns),
        "max_drawdown": max_drawdown(portfolio_returns),
        "ulcer_index": ulcer_index(portfolio_returns),
        "volatility": portfolio_returns.std() * np.sqrt(252),
        "total_return": (1 + portfolio_returns).prod() - 1,
    }
    if benchmark_returns is not None:
        bench_cagr = cagr(benchmark_returns)
        report["benchmark_cagr"] = bench_cagr
        report["excess_return"] = report["cagr"] - bench_cagr
        report["benchmark_max_dd"] = max_drawdown(benchmark_returns)
    return report


def print_report(report):
    print("\n" + "=" * 50)
    print("PERFORMANCE REPORT")
    print("=" * 50)
    print(f"  CAGR:              {report['cagr']:.2%}")
    print(f"  Total Return:      {report['total_return']:.2%}")
    print(f"  Volatility:        {report['volatility']:.2%}")
    print(f"  Sharpe Ratio:      {report['sharpe']:.2f}")
    print(f"  Sortino Ratio:     {report['sortino']:.2f}")
    print(f"  Max Drawdown:      {report['max_drawdown']:.2%}")
    print(f"  Ulcer Index:       {report['ulcer_index']:.2f}")
    if "benchmark_cagr" in report:
        print(f"\n  Benchmark CAGR:    {report['benchmark_cagr']:.2%}")
        print(f"  Excess Return:     {report['excess_return']:.2%}")
        print(f"  Benchmark Max DD:  {report['benchmark_max_dd']:.2%}")
    print("=" * 50)
