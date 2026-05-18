"""
Buy97 Dual-Exit Strategy — Proper Analysis

Strategy: Buy at ~50c (late entry, high confidence), set two exits:
1. TP at 99c (max profit on winners)
2. SL at 80c (cut loss on losers)

In deterministic binary models, 80c always fires before 99c for winners,
which hurts performance. In real markets with microstructure noise,
this might be different. We test both.

Usage: python buy97_dual_exit_proper.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backtests.polymarket_updown_backtest import (
    STARTING_CAPITAL,
    TAKER_FEE_RATE,
    Window,
    build_windows,
    load_spot_prices,
    max_drawdown,
    split_windows_chronologically,
)


def side_prices(w, side):
    if side == "UP":
        return w.up_bid.copy(), w.up_ask.copy()
    return w.down_bid.copy(), w.down_ask.copy()


def outcome_value(w, side):
    return float(w.outcome_up) if side == "UP" else float(1 - w.outcome_up)


def simulate_trade(w, entry_seconds_left, tp, sl, stake_usd, noise_bps=0):
    """
    Simulate a single trade with TP and SL levels.
    Returns: (exit_type, pnl, exit_price, exit_idx)
    """
    idx = 300 - entry_seconds_left
    leader = "UP" if w.prices[idx] >= w.open_price else "DOWN"
    bid, ask = side_prices(w, leader)
    entry_price = float(ask[idx])

    if entry_price > 0.97:
        return None

    # Add noise if requested
    if noise_bps > 0:
        bid += np.random.randn(len(bid)) * (noise_bps / 10000.0)
        bid = np.clip(bid, 0.001, 0.999)

    shares = stake_usd / entry_price
    entry_fee = stake_usd * TAKER_FEE_RATE

    # Find first fill: TP or SL
    future_bids = bid[idx + 1:]
    hit_tp = np.flatnonzero(future_bids >= tp)
    hit_sl = np.flatnonzero(future_bids <= sl)

    idx_tp = hit_tp[0] + idx + 1 if len(hit_tp) > 0 else float('inf')
    idx_sl = hit_sl[0] + idx + 1 if len(hit_sl) > 0 else float('inf')

    if idx_tp < idx_sl:
        exit_price = tp
        exit_idx = int(idx_tp)
        exit_value = exit_price * shares
        exit_fee = exit_value * TAKER_FEE_RATE
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return "tp", float(pnl), exit_price, exit_idx

    if idx_sl < idx_tp:
        exit_price = sl
        exit_idx = int(idx_sl)
        exit_value = exit_price * shares
        exit_fee = exit_value * TAKER_FEE_RATE
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return "sl", float(pnl), exit_price, exit_idx

    # Neither hit (rare with deterministic paths)
    final = outcome_value(w, leader)
    pnl = final * shares - stake_usd - entry_fee
    return "settlement", float(pnl), final, None


def run_strategy(windows, name, trade_fn):
    trades = []
    for w in windows:
        result = trade_fn(w)
        if result is not None and result[0] is not None:
            exit_type, pnl, exit_price, exit_idx = result
            trades.append({
                "strategy": name, "exit_type": exit_type, "pnl": pnl,
                "exit_price": exit_price, "exit_idx": exit_idx,
            })

    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    equity = STARTING_CAPITAL + np.cumsum(df["pnl"].values)
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
    raw_sharpe = float(returns.mean() / returns.std(ddof=1)) if len(returns) > 1 and returns.std(ddof=1) > 0 else 0.0
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] < 0]

    metrics = {
        "strategy": name, "trades": len(trades),
        "wins": len(wins), "losses": len(losses),
        "win_rate": len(wins) / len(trades),
        "total_pnl": float(equity[-1] - STARTING_CAPITAL),
        "avg_pnl": float(df["pnl"].mean()),
        "max_drawdown": float(max_drawdown(equity)),
        "raw_5m_sharpe": raw_sharpe,
        "exit_tp": len(df[df["exit_type"] == "tp"]),
        "exit_sl": len(df[df["exit_type"] == "sl"]),
        "exit_settlement": len(df[df["exit_type"] == "settlement"]),
    }
    return pd.DataFrame([metrics])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-windows", type=int, default=500)
    parser.add_argument("--noise-bps", type=float, default=0)
    args = parser.parse_args()

    print("=" * 76)
    print("BUY97 DUAL-EXIT PROPER ANALYSIS")
    print("=" * 76)
    print("Strategy: Buy late-leading side, TP at 99c, SL at 80c")
    print(f"Noise: {args.noise_bps}bps")
    print()

    price_df = load_spot_prices("btcusdt", "binance", dataset_source="aliplayer_spot")
    windows = build_windows(price_df, args.max_windows)
    train, test = split_windows_chronologically(windows, 0.70)

    # Strategies to compare
    strategies = {
        "no_exit": lambda w: simulate_trade(w, 30, 1.0, -0.01, 10.0, args.noise_bps),  # Never fires (hold to settlement)
        "tp_only": lambda w: simulate_trade(w, 30, 0.99, -0.01, 10.0, args.noise_bps),  # Only TP at 99c
        "sl_only": lambda w: simulate_trade(w, 30, 1.0, 0.80, 10.0, args.noise_bps),  # Only SL at 80c
        "dual_99_80": lambda w: simulate_trade(w, 30, 0.99, 0.80, 10.0, args.noise_bps),  # Dual
        "dual_99_50": lambda w: simulate_trade(w, 30, 0.99, 0.50, 10.0, args.noise_bps),  # SL at 50c (deeper)
    }

    all_results = []
    for sample_name, sample_windows in [("TRAIN", train), ("TEST", test)]:
        print(f"\n{'='*76}")
        print(f"{sample_name} Results")
        print(f"{'='*76}\n")

        results = []
        for name, fn in strategies.items():
            m = run_strategy(sample_windows, name, fn)
            if not m.empty:
                m.insert(0, "sample", sample_name)
                results.append(m)

        df = pd.concat(results, ignore_index=True)
        all_results.append(df)

        # Print
        print(f"{'Strategy':<18} {'Trades':>8} {'Wins':>6} {'Losses':>8} {'WR':>8} {'PnL':>12} {'Avg PnL':>10} {'Sharpe':>8} {'DD':>8}")
        print("-" * 90)
        for _, row in df.iterrows():
            print(f"{row['strategy']:<18} {row['trades']:>8} {row['wins']:>6} {row['losses']:>8} {row['win_rate']*100:>6.1f}% ${row['total_pnl']:>9,.2f} ${row['avg_pnl']:>8.2f} {row['raw_5m_sharpe']:>6.3f} {row['max_drawdown']*100:>6.2f}%")

        # Exit breakdown
        print(f"\nExit Breakdown:")
        # Detailed breakdown per strategy
        for name, fn in strategies.items():
            result_list = []
            for w in sample_windows:
                result = fn(w)
                if result and result[0]:
                    result_list.append(result)
            tp_count = sum(1 for r in result_list if r[0] == "tp")
            sl_count = sum(1 for r in result_list if r[0] == "sl")
            st_count = sum(1 for r in result_list if r[0] == "settlement")
            print(f"  {name:<18}: TP={tp_count}, SL={sl_count}, Settlement={st_count}")

    final = pd.concat(all_results, ignore_index=True)
    final.to_csv("buy97_dual_exit_proper_metrics.csv", index=False)
    print(f"\n{'='*76}")
    print("Analysis complete. Saved to buy97_dual_exit_proper_metrics.csv")


if __name__ == "__main__":
    main()
