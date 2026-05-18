"""
Kelly Criterion Sizing + Consecutive Loss Streak Analysis
for all strategies.

Computes:
1. Kelly Criterion (full & half-Kelly)
2. Optimal bet sizing in $ and units
3. Probability of N consecutive losses
4. Expected time to ruin at various bet sizes
5. Streak frequency analysis (how often does a bad streak occur)

Usage: python kelly_and_streak_analysis.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

STARTING_CAPITAL = 10_000.0
STRATEGY_FILES = {
    "chop_direction_predictor": ["chop_dir_trades_all_patterns_test.csv", "chop_dir_trades_all_patterns_train.csv"],
    "buy1_cent": ["buy1_cent_trades.csv"],
    "chop_scalper_v1": ["chop_scalper_trades_test.csv", "chop_scalper_trades_train.csv"],
    "chop_scalper_v2": ["chop_v2_trades_chop_test.csv", "chop_v2_trades_chop_train.csv"],
    "buy97_sell99": ["buy97_sell99_trades_test.csv", "buy97_sell99_trades_train.csv"],
    "prev10_momentum": ["prev10_momentum_next_best_trades.csv"],
    "live_guarded_v1": ["live_guarded_metrics_slip_0.00c.csv"],
}


def load_trades(csv_path: str) -> list[float]:
    """Load trade PnLs from a CSV."""
    df = pd.read_csv(csv_path)
    pnl_col = None
    for col in ["pnl", "PnL", "profit", "net_pnl", "trade_pnl"]:
        if col in df.columns:
            pnl_col = col
            break
    if pnl_col is None:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                pnl_col = col
                break
    if pnl_col is None:
        return []
    return [float(v) for v in df[pnl_col] if pd.notnull(v)]


def find_all_trades(strategy_name: str) -> list[float]:
    all_trades = []
    for fn in STRATEGY_FILES.get(strategy_name, []):
        p = Path(fn)
        if p.exists():
            all_trades.extend(load_trades(str(p)))
    return all_trades


def kelly_criterion_wins(trades: list[float]) -> tuple[float, float, float, float, float]:
    """
    Compute Kelly criterion for a binary-ish win/loss trading strategy.
    Returns (fractional_kelly, win_rate, avg_win, avg_loss, edge_per_trade).
    """
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    if not wins or not losses:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    w = len(wins) / len(trades)
    avg_w = np.mean(wins)
    avg_l = abs(np.mean(losses))
    b = avg_w / avg_l  # avg win / avg loss ratio

    # Full Kelly: f* = (b * w - (1 - w)) / b
    f_star = (b * w - (1 - w)) / b if b > 0 else 0.0
    # Edge per trade (as fraction of bankroll per full unit trade)
    edge = w * avg_w - (1 - w) * avg_l
    return max(f_star, 0.0), w, avg_w, avg_l, edge


def prob_consecutive_losses(loss_rate: float, n: int) -> float:
    """Probability of N consecutive losses."""
    return loss_rate ** n


def expected_streaks(n_trials: int, loss_rate: float, streak_len: int) -> float:
    """Expected number of streaks of `streak_len` consecutive losses in `n_trials`."""
    return n_trials * prob_consecutive_losses(loss_rate, streak_len)


def time_to_ruin(trades: list[float], bet_fraction: float, bankroll: float) -> int | None:
    """
    Simulate until ruin. Returns number of trades to go bust,
    or None if doesn't ruin within a reasonable horizon.
    """
    capital = bankroll
    np.random.seed(42)
    sample = np.random.choice(trades, size=1_000_000, replace=True)
    for i, pnl in enumerate(sample):
        # bet size = bet_fraction * current bankroll, but not more than trade PnL allows
        # simplified: treat each trade as unit bet scaled by fraction
        scaled_pnl = pnl * bet_fraction
        capital += scaled_pnl
        if capital <= 0:
            return i + 1
        if capital > bankroll * 10:  # stop early if we 10x
            return None
    return None


def analyze_strategy(name: str, trades: list[float]) -> dict:
    if not trades:
        return {"error": "No trades"}

    kelly, wr, avg_w, avg_l, edge = kelly_criterion_wins(trades)
    loss_rate = 1 - wr

    # Streak probabilities
    streaks = {}
    for streak_len in [2, 3, 5, 7, 10, 15]:
        p = prob_consecutive_losses(loss_rate, streak_len)
        expected_per_1000 = expected_streaks(1000, loss_rate, streak_len)
        expected_per_year = expected_streaks(365 * 24 * 12, loss_rate, streak_len)  # ~1yr at 5min markets
        streaks[streak_len] = {
            "prob": p,
            "expected_per_1000": expected_per_1000,
            "expected_per_year": expected_per_year,
        }

    # Bet sizing
    results = {
        "n_trades": len(trades),
        "win_rate": wr * 100,
        "avg_win": avg_w,
        "avg_loss": -avg_l,
        "payoff_ratio": avg_w / avg_l if avg_l > 0 else float("inf"),
        "kelly_fraction": kelly,
        "half_kelly": kelly / 2,
        "quarter_kelly": kelly / 4,
        "edge_per_trade_dollars": edge,
        "optimal_bet_dollars": edge,  # For unit-sized trades, optimal $ bet is the edge
        "streak_probabilities": streaks,
    }

    # Simulate time to ruin at various bet sizes
    ruin_results = {}
    for label, frac in [("full_kelly", kelly), ("half_kelly", kelly / 2), ("quarter_kelly", kelly / 4), ("fixed_1unit", 1.0 / abs(avg_l) if avg_l != 0 else 0)]:
        if frac <= 0 or frac > 1:
            ruin_results[label] = None
            continue
        n_to_ruin = time_to_ruin(trades, frac, STARTING_CAPITAL)
        ruin_results[label] = n_to_ruin
    results["ruin_at_bet_size"] = ruin_results

    return results


def print_results(results: dict, name: str):
    print(f"\n{'='*72}")
    print(f"STRATEGY: {name}")
    print(f"{'='*72}")

    if "error" in results:
        print(f"  ERROR: {results['error']}")
        return

    print(f"\n  Trades:             {results['n_trades']:,}")
    print(f"  Win Rate:           {results['win_rate']:.1f}%")
    print(f"  Avg Win:            ${results['avg_win']:.2f}")
    print(f"  Avg Loss:           ${results['avg_loss']:.2f}")
    print(f"  Payoff Ratio:       {results['payoff_ratio']:.2f}x")

    print(f"\n  --- Kelly Criterion ---")
    print(f"  Full Kelly:         {results['kelly_fraction']*100:.2f}% of bankroll per trade")
    print(f"  Half Kelly:         {results['half_kelly']*100:.2f}%")
    print(f"  Quarter Kelly:      {results['quarter_kelly']*100:.2f}%")
    print(f"  Edge per Trade:     ${results['edge_per_trade_dollars']:.2f}")

    # Optimal $ bet for $10k starting capital
    bankroll = STARTING_CAPITAL
    for label, frac in [("Full", results['kelly_fraction']), ("Half", results['half_kelly']), ("Quarter", results['quarter_kelly'])]:
        if frac > 0:
            bet = bankroll * frac
            print(f"  Optimal ${label} Bet:  ${bet:.2f} per trade")
        else:
            print(f"  Optimal ${label} Bet:  N/A (negative edge)")

    print(f"\n  --- Consecutive Loss Streaks ---")
    streaks = results.get("streak_probabilities", {})
    print(f"  {'Streak':>8} {'Prob':>12} {'Per 1k Trades':>15} {'Per Year (est)':>15}")
    for streak_len in sorted(streaks.keys()):
        s = streaks[streak_len]
        print(f"  {streak_len:>8} {s['prob']*100:>11.2f}% {s['expected_per_1000']:>15.2f} {s['expected_per_year']:>15.2f}")

    print(f"\n  --- Time to Ruin ---")
    ruin = results.get("ruin_at_bet_size", {})
    for label, n in ruin.items():
        if n is None:
            print(f"  {label:15s}: Survives >1,000,000 trades")
        else:
            print(f"  {label:15s}: ~{n:,} trades to ruin")


def main():
    print("=" * 72)
    print("KELLY CRITERION & STREAK ANALYSIS")
    print("=" * 72)

    for name in STRATEGY_FILES:
        trades = find_all_trades(name)
        if not trades:
            print(f"\n[SKIP] {name}: no trade data")
            continue
        results = analyze_strategy(name, trades)
        print_results(results, name)

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
