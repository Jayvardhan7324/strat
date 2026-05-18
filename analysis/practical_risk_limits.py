"""
Practical Risk Limits for Polymarket Trading

STOP-LOSS FRAMEWORK:
1. Per-Trade Markdown (how far can individual trades go against you)
2. Session Cash Burn (how much of bankroll can you lose in one session)
3. Consecutive Loss Kill Switch
4. Overall Portfolio Circuit Breaker

Given the strategies, the game is not 'per day' but per market / per session.
Polymarket has 5-min markets; you may take 1-20 trades per session.

This script calculates practical limits.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

STARTING_CAPITAL = 10_000.0
STRATEGY_FILES = {
    "buy1_cent": ["buy1_cent_trades.csv"],
    "buy97_sell99": ["buy97_sell99_trades_test.csv", "buy97_sell99_trades_train.csv"],
    "prev10_momentum": ["prev10_momentum_next_best_trades.csv"],
    "chop_direction_predictor": ["chop_dir_trades_all_patterns_test.csv", "chop_dir_trades_all_patterns_train.csv"],
    "chop_scalper_v1": ["chop_scalper_trades_test.csv", "chop_scalper_trades_train.csv"],
    "chop_scalper_v2": ["chop_v2_trades_chop_test.csv", "chop_v2_trades_chop_train.csv"],
    "live_guarded_v1": ["live_guarded_metrics_slip_0.00c.csv"],
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


def analyze_risk(name, trades):
    if len(trades) == 0:
        return None

    wins = trades[trades > 0]
    losses = trades[trades < 0]
    wr = len(wins) / len(trades)
    loss_rate = 1 - wr
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    median_win = float(np.median(wins)) if len(wins) > 0 else 0.0
    median_loss = float(np.median(losses)) if len(losses) > 0 else 0.0
    worst_loss = float(np.min(trades))
    best_win = float(np.max(trades))
    std = float(np.std(trades, ddof=1))
    edge = float(np.mean(trades))

    # Percentiles
    p01 = float(np.percentile(trades, 1))
    p05 = float(np.percentile(trades, 5))
    p10 = float(np.percentile(trades, 10))
    p90 = float(np.percentile(trades, 90))
    p95 = float(np.percentile(trades, 95))
    p99 = float(np.percentile(trades, 99))

    # Streak analysis: What is P(2 consecutive losses)?
    consecutive = 0
    max_streak = 0
    for t in trades:
        if t < 0:
            consecutive += 1
            max_streak = max(max_streak, consecutive)
        else:
            consecutive = 0

    # Consecutive loss probability
    p_2_losses = loss_rate ** 2
    p_3_losses = loss_rate ** 3
    p_5_losses = loss_rate ** 5
    p_7_losses = loss_rate ** 7

    return {
        "name": name,
        "n_trades": len(trades),
        "win_rate": wr * 100,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "median_win": median_win,
        "median_loss": median_loss,
        "worst_loss": worst_loss,
        "best_win": best_win,
        "std_per_trade": std,
        "edge_per_trade": edge,
        "percentile_01": p01,
        "percentile_05": p05,
        "percentile_10": p10,
        "percentile_90": p90,
        "percentile_95": p95,
        "percentile_99": p99,
        "max_loss_streak": max_streak,
        "prob_2_losses": p_2_losses,
        "prob_3_losses": p_3_losses,
        "prob_5_losses": p_5_losses,
        "prob_7_losses": p_7_losses,
    }


def print_risk_limits(result):
    if not result:
        return

    name = result["name"]
    print(f"\n{'='*76}")
    print(f"  STRATEGY: {name}")
    print(f"{'='*76}")

    print(f"\n  Trade Stats:")
    print(f"    Trades:           {result['n_trades']:,}")
    print(f"    Win Rate:         {result['win_rate']:.1f}%")
    print(f"    Avg Win:          ${result['avg_win']:.2f}")
    print(f"    Avg Loss:         ${result['avg_loss']:.2f}")
    print(f"    Median Win:       ${result['median_win']:.2f}")
    print(f"    Median Loss:      ${result['median_loss']:.2f}")
    print(f"    Worst Loss:       ${result['worst_loss']:.2f}")
    print(f"    Best Win:         ${result['best_win']:.2f}")
    print(f"    Std Dev:          ${result['std_per_trade']:.2f}")
    print(f"    Edge/Trade:       ${result['edge_per_trade']:.2f}")

    print(f"\n  Percentiles:")
    print(f"    1% worst:         ${result['percentile_01']:.2f}")
    print(f"    5% worst:         ${result['percentile_05']:.2f}")
    print(f"    10% worst:        ${result['percentile_10']:.2f}")
    print(f"    90% best:         ${result['percentile_90']:.2f}")
    print(f"    95% best:         ${result['percentile_95']:.2f}")
    print(f"    99% best:         ${result['percentile_99']:.2f}")

    print(f"\n  Loss Streaks (observed in data):")
    print(f"    Max Consecutive:  {result['max_loss_streak']}")
    print(f"    Prob(2 straight): {result['prob_2_losses']*100:.1f}%")
    print(f"    Prob(3 straight): {result['prob_3_losses']*100:.2f}%")
    print(f"    Prob(5 straight): {result['prob_5_losses']:.4f}%")
    print(f"    Prob(7 straight): {result['prob_7_losses']:.6f}%")

    # Recommendations
    print(f"\n  --- PRACTICAL RISK LIMITS ---")

    # Per-Trade Stop
    worst_1pct = result['percentile_01']
    per_trade_stop = max(abs(worst_1pct) * 1.2, abs(result['worst_loss']))
    print(f"    Per-Trade Stop:   ${per_trade_stop:.2f} (below 1% worst case * 1.2)")

    # Session Burn (stops after N consecutive losses or X% capital)
    session_loss_limit = abs(result['worst_loss']) * 3  # e.g., 3 worst-case losses in a row
    session_pct = (session_loss_limit / STARTING_CAPITAL) * 100
    print(f"    Session Stop:     ${session_loss_limit:.2f}  ({session_pct:.1f}% of $10k capital)")
    print(f"    Hard Stop:        10% of bankroll = $1,000 (recommended max session loss)")
    print(f"    Nuclear Stop:     20% of bankroll = $2,000 (absolutely never exceed)")

    # Consecutive loss kill switch
    for streak in [2, 3, 5]:
        prob = result[f"prob_{streak}_losses"]
        print(f"    Stop after {streak} consecutive losses?  Prob: {prob*100:.1f}%")

    # Adverse move size
    print(f"\n  --- ADVERSE MOVE SCENARIOS ---")
    print(f"    Worst single trade = ${result['worst_loss']:.2f}")
    print(f"    Worst 2 in a row  = ${result['worst_loss']*2:.2f}")
    print(f"    Worst 3 in a row  = ${result['worst_loss']*3:.2f}")
    print(f"    Worst 5 in a row  = ${result['worst_loss']*5:.2f}")


def main():
    print("=" * 76)
    print(" PRACTICAL RISK LIMITS & SESSION CIRCUIT BREAKERS")
    print("=" * 76)

    for name, files in STRATEGY_FILES.items():
        trades = load_trades(files)
        if len(trades) == 0:
            print(f"\n[SKIP] {name}: no trade data")
            continue
        results = analyze_risk(name, trades)
        print_risk_limits(results)

    print("\n" + "=" * 76)
    print(" SUMMARY RECOMMENDATIONS")
    print("=" * 76)
    print("""
Per-Trade Hard Stop:
- Never let any single trade lose more than 1% of your bankroll.
  For a $10k account, that is $100 per trade. Scale down position size
  to achieve this (use Kelly fraction * 0.5).

Session Soft Stop:
- If you are down 5% of bankroll ($500), STOP trading for the day.
  Walk away, review what went wrong. Do not revenge-trade.

Session Hard Stop:
- If you are down 10% of bankroll ($1,000), the session is OVER.
  No exceptions. Log the day. Come back tomorrow.

Nuclear Stop:
- If you EVER hit 20% drawdown ($2,000), you STOP the strategy.
  Not the day. The strategy. Full stop, review, paper-trade until fixed.

Kill Switch (Consecutive Losses):
- If you lose 3 trades in a row, PAUSE. This is a 5-10 sigma signal
  that either (a) the edge is gone, or (b) you are on tilt.
  Review before taking the 4th trade.

Portfolio Rule:
- Never risk more than 20% of total bankroll in any single strategy.
  Diversify across strategies, or bet smaller per strategy.
    """)
    print("=" * 76)


if __name__ == "__main__":
    main()
