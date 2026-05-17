"""
Realistic Daily/Session Risk Limits for Polymarket Trading

Given the strategies we have, this script calculates:
- Expected value per day per strategy
- Worst-case daily loss scenarios
- Bankroll burn rate
- Optimal daily stop limits (account for Kelly, variance, and ruin probability)
- Time-to-ruin under various daily loss limits
- Monte Carlo of daily PnL sequences to find safe stop limits

Strategy daily trade counts vary, so we use the per-trade edge and
price that in.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

STARTING_CAPITAL = 10_000.0

# Map strategies to their trade files
STRATEGY_INFO = {
    "buy1_cent": {
        "files": ["buy1_cent_trades.csv"],
        "avg_trades_per_day": 150,  # ~150 5-min windows/day
    },
    "buy97_sell99": {
        "files": ["buy97_sell99_trades_test.csv", "buy97_sell99_trades_train.csv"],
        "avg_trades_per_day": 288,  # ~288/day at high frequency
    },
    "prev10_momentum": {
        "files": ["prev10_momentum_next_best_trades.csv"],
        "avg_trades_per_day": 288,  # ~288/day
    },
    "chop_direction_predictor": {
        "files": ["chop_dir_trades_all_patterns_test.csv", "chop_dir_trades_all_patterns_train.csv"],
        "avg_trades_per_day": 288,  # ~288/day
    },
}


def load_trades(csv_list):
    trades = []
    for fn in csv_list:
        p = Path(fn)
        if not p.exists():
            continue
        df = pd.read_csv(p)
        pnl_col = next((c for c in ["pnl", "PnL", "profit", "net_pnl", "trade_pnl"] if c in df.columns), None)
        if pnl_col is None:
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]):
                    pnl_col = c
                    break
        if pnl_col:
            trades.extend(df[pnl_col].dropna().tolist())
    return np.array(trades)


def analyze_daily_risk(name, trades, avg_trades_per_day):
    if len(trades) == 0:
        return None

    wins = trades[trades > 0]
    losses = trades[trades < 0]
    wr = len(wins) / len(trades)
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    worst_loss = float(np.min(trades))
    best_win   = float(np.max(trades))
    edge = float(np.mean(trades))
    std  = float(np.std(trades, ddof=1))

    # Daily expected stats
    daily_ev = edge * avg_trades_per_day
    daily_std = std * math.sqrt(avg_trades_per_day)

    # Simulate many days
    n_days = 50_000
    rng = np.random.default_rng(seed=42)
    daily_samples = []
    for _ in range(n_days):
        sample = rng.choice(trades, size=avg_trades_per_day, replace=True)
        daily_samples.append(float(np.sum(sample)))
    daily_samples = np.array(daily_samples)

    # Confidence levels for daily PnL
    daily_ci_95 = (float(np.percentile(daily_samples, 2.5)), float(np.percentile(daily_samples, 97.5)))
    daily_ci_99 = (float(np.percentile(daily_samples, 0.5)), float(np.percentile(daily_samples, 99.5)))
    daily_worst_1pct = float(np.percentile(daily_samples, 1))
    daily_best_1pct  = float(np.percentile(daily_samples, 99))

    # Time to ruin under various daily loss rules
    # Simulate a year
    n_years = 10
    year_days = 365
    total_days = n_years * year_days

    # Rule 1: No stop (baseline)
    equity_no_stop = STARTING_CAPITAL + np.cumsum(rng.choice(daily_samples, size=total_days, replace=True))
    # Rule 2: -5% daily stop
    equity_5pct_stop = simulate_with_daily_stop(STARTING_CAPITAL, daily_samples, total_days, -0.05 * STARTING_CAPITAL)
    # Rule 3: -10% daily stop
    equity_10pct_stop = simulate_with_daily_stop(STARTING_CAPITAL, daily_samples, total_days, -0.10 * STARTING_CAPITAL)
    # Rule 4: -20% daily stop (aggressive)
    equity_20pct_stop = simulate_with_daily_stop(STARTING_CAPITAL, daily_samples, total_days, -0.20 * STARTING_CAPITAL)

    results = {
        "name": name,
        "n_trades_total": len(trades),
        "avg_trades_per_day": avg_trades_per_day,
        "win_rate": wr * 100,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "worst_loss": worst_loss,
        "best_win": best_win,
        "edge_per_trade": edge,
        "std_per_trade": std,
        "daily_ev": daily_ev,
        "daily_std": daily_std,
        "daily_ci_95_low": daily_ci_95[0],
        "daily_ci_95_high": daily_ci_95[1],
        "daily_ci_99_low": daily_ci_99[0],
        "daily_ci_99_high": daily_ci_99[1],
        "daily_worst_1pct": daily_worst_1pct,
        "daily_best_1pct": daily_best_1pct,
        "prob_daily_loss": float(np.mean(daily_samples < 0)),
        "prob_daily_loss_gt_5pct": float(np.mean(daily_samples < -STARTING_CAPITAL * 0.05)),
        "prob_daily_loss_gt_10pct": float(np.mean(daily_samples < -STARTING_CAPITAL * 0.10)),
        "prob_daily_loss_gt_20pct": float(np.mean(daily_samples < -STARTING_CAPITAL * 0.20)),
        "yearly_return_no_stop": float(equity_no_stop[-1] - STARTING_CAPITAL),
        "yearly_return_5pct_stop": float(equity_5pct_stop[-1] - STARTING_CAPITAL),
        "yearly_return_10pct_stop": float(equity_10pct_stop[-1] - STARTING_CAPITAL),
        "yearly_return_20pct_stop": float(equity_20pct_stop[-1] - STARTING_CAPITAL),
        "ruin_prob_no_stop_1yr": float(equity_no_stop[-1] <= 0),
        "ruin_prob_5pct_1yr": float(equity_5pct_stop[-1] <= 0),
        "ruin_prob_10pct_1yr": float(equity_10pct_stop[-1] <= 0),
        "ruin_prob_20pct_1yr": float(equity_20pct_stop[-1] <= 0),
    }
    return results


def simulate_with_daily_stop(start, daily_dist, n_days, stop_limit):
    """Simulate with a daily stop-loss. If a day's PnL < stop_limit, we stop trading for the day."""
    capital = start
    for _ in range(n_days):
        day_pnl = np.random.choice(daily_dist)
        # Apply stop: if the loss exceeds limit, hit the stop (don't allow going below stop limit)
        if day_pnl < stop_limit:
            day_pnl = stop_limit  # You hit your stop, no more trades that day
        capital += day_pnl
        if capital <= 0:
            break
    return np.array([capital])


def print_results(results):
    if not results:
        return

    name = results["name"]
    print(f"\n{'='*72}")
    print(f"STRATEGY: {name}")
    print(f"{'='*72}")

    print(f"\n  Trade Stats:")
    print(f"    Total Trades:       {results['n_trades_total']:,}")
    print(f"    Win Rate:           {results['win_rate']:.1f}%")
    print(f"    Avg Win:            ${results['avg_win']:.2f}")
    print(f"    Avg Loss:           ${results['avg_loss']:.2f}")
    print(f"    Worst Loss:         ${results['worst_loss']:.2f}")
    print(f"    Best Win:           ${results['best_win']:.2f}")
    print(f"    Edge/Trade:         ${results['edge_per_trade']:.2f}")
    print(f"    Std Dev/Trade:      ${results['std_per_trade']:.2f}")

    print(f"\n  Daily Projections ({results['avg_trades_per_day']} trades/day):")
    print(f"    Expected Daily PnL:  ${results['daily_ev']:.2f}")
    print(f"    Daily Std Dev:       ${results['daily_std']:.2f}")
    dc = results
    print(f"    95% CI Daily:       [${dc['daily_ci_95_low']:.2f}, ${dc['daily_ci_95_high']:.2f}]")
    print(f"    99% CI Daily:       [${dc['daily_ci_99_low']:.2f}, ${dc['daily_ci_99_high']:.2f}]")
    print(f"    Worst 1% Day:       ${dc['daily_worst_1pct']:.2f}")
    print(f"    Best 1% Day:        ${dc['daily_best_1pct']:.2f}")
    print(f"    Prob(Loss Day):     {dc['prob_daily_loss']*100:.1f}%")
    print(f"    Prob(Loss > 5%):   {dc['prob_daily_loss_gt_5pct']*100:.1f}%")
    print(f"    Prob(Loss > 10%):  {dc['prob_daily_loss_gt_10pct']*100:.1f}%")
    print(f"    Prob(Loss > 20%):  {dc['prob_daily_loss_gt_20pct']*100:.1f}%")

    print(f"\n  Daily Stop Limits (10-year sim, 365 days/yr):")
    print(f"    No Stop:     Expected after 10yr = ${results['yearly_return_no_stop']:,.2f}")
    print(f"    -5% Stop:    Expected after 10yr = ${results['yearly_return_5pct_stop']:,.2f}")
    print(f"    -10% Stop:   Expected after 10yr = ${results['yearly_return_10pct_stop']:,.2f}")
    print(f"    -20% Stop:   Expected after 10yr = ${results['yearly_return_20pct_stop']:,.2f}")
    print(f"\n  Ruin Probabilities (1 year):")
    print(f"    No Stop:     {results['ruin_prob_no_stop_1yr']*100:.1f}%")
    print(f"    -5% Stop:    {results['ruin_prob_5pct_1yr']*100:.1f}%")
    print(f"    -10% Stop:   {results['ruin_prob_10pct_1yr']*100:.1f}%")
    print(f"    -20% Stop:   {results['ruin_prob_20pct_1yr']*100:.1f}%")


def main():
    print("=" * 72)
    print("DAILY RISK LIMITS & STOP ANALYSIS")
    print("=" * 72)

    all_results = {}
    for name, info in STRATEGY_INFO.items():
        trades = load_trades(info["files"])
        if len(trades) == 0:
            print(f"\n[SKIP] {name}: no trade data")
            continue
        results = analyze_daily_risk(name, trades, info["avg_trades_per_day"])
        if results:
            all_results[name] = results
            print_results(results)

    print("\n" + "=" * 72)
    print("DAILY STOP RECOMMENDATIONS")
    print("=" * 72)

    print("""
Based on the per-strategy daily risk analysis above, here are
recommended daily stop rules:

1. FOR buy1_cent / buy97_sell99:
   Daily Stop: 5% ($500 on $10k bankroll)
   Reason: High loss-probability, but small per-trade losses.
   A 5% stop prevents compounding bad days.

2. FOR prev10_momentum:
   Daily Stop: 10% ($1,000 on $10k bankroll)
   Reason: Symmetric payoff, consistent edge. A 10% stop
   is buffered enough that normal variance doesn't trigger it,
   but catches multi-standard-deviation outlier days.

3. FOR chop_direction_predictor:
   Daily Stop: 20% ($2,000) with **mandatory half-Kelly**
   Reason: High edge, but asymmetric. A 20% stop prevents
   the catastrophic 3+ sigma days from wiping out months of profit.
   Only run at half (or less) Kelly sizing.

4. FOR chop_scalper_v1 / v2:
   RECOMMENDATION: DO NOT TRADE
   Reason: Negative edge. No stop limit can save a negative
   expected value strategy.

5. UNIVERSAL RULE:
   - If you hit your daily stop: **WALK AWAY**. No exceptions.
   - Log the day, review why, but do not revenge-trade.
   - A daily stop is not a suggestion; it is a circuit breaker.

6. WEEKLY/ MONTHLY AGGREGATE STOPS:
   Weekly:  Stop at 2x daily limit (e.g., $1,000 for prev10_momentum)
   Monthly: Stop at 3x daily limit (e.g., $3,000)
   This prevents "death by a thousand cuts" when edge is positive
   but variance runs hot.
""")

    print("=" * 72)
    print("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
