"""
Buy97 Dual-Exit Strategy

Core idea: Buy at 0.97 (or better), then place TWO limit sell orders:
1. Aggressive target at 0.99 (high profit if it fills)
2. Conservative fallback at 0.80 (guaranteed profit if early move fizzles)

If 0.99 hits first → take full profit
If 0.80 hits first → small profit but protected from settlement risk
If neither hits → settle

Usage:
    python buy97_dual_exit.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from polymarket_updown_backtest import (
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
        return w.up_bid, w.up_ask
    if side == "DOWN":
        return w.down_bid, w.down_ask


def outcome_value(w, side):
    return float(w.outcome_up) if side == "UP" else float(1 - w.outcome_up)


# ========================================================================
# Dual Exit: Sell at 99c OR 80c (first fill wins)
# ========================================================================

def dual_exit(w, stake_usd=10.0, entry_seconds_left=30):
    idx = 300 - entry_seconds_left
    leader = "UP" if w.prices[idx] >= w.open_price else "DOWN"
    bid, ask = side_prices(w, leader)
    entry_price = float(ask[idx])

    if entry_price > 0.97:
        return None, None

    shares = stake_usd / entry_price
    entry_fee = stake_usd * TAKER_FEE_RATE

    # Find first fill: 99c or 80c
    hit_99 = np.flatnonzero(bid[idx + 1:] >= 0.99)
    hit_80 = np.flatnonzero(bid[idx + 1:] >= 0.80)

    idx_99 = hit_99[0] + idx + 1 if len(hit_99) > 0 else float('inf')
    idx_80 = hit_80[0] + idx + 1 if len(hit_80) > 0 else float('inf')

    if idx_99 < idx_80:
        exit_value = 0.99 * shares
        exit_fee = exit_value * TAKER_FEE_RATE
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return "sold_99", {
            "pnl": float(pnl), "entry_idx": idx, "exit_idx": int(idx_99),
            "entry_price": entry_price, "exit_price": 0.99
        }
    elif idx_80 < idx_99:
        exit_value = 0.80 * shares
        exit_fee = exit_value * TAKER_FEE_RATE
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return "sold_80", {
            "pnl": float(pnl), "entry_idx": idx, "exit_idx": int(idx_80),
            "entry_price": entry_price, "exit_price": 0.80
        }

    # Settle
    final = outcome_value(w, leader) * shares
    pnl = final - stake_usd - entry_fee
    return "settlement", {
        "pnl": float(pnl), "entry_idx": idx, "exit_idx": None,
        "entry_price": entry_price, "exit_price": outcome_value(w, leader)
    }


# Base comparison

def base_9799(w, stake_usd=10.0, entry_seconds_left=30):
    idx = 300 - entry_seconds_left
    leader = "UP" if w.prices[idx] >= w.open_price else "DOWN"
    bid, ask = side_prices(w, leader)
    entry_price = float(ask[idx])

    if entry_price > 0.97:
        return None, None

    shares = stake_usd / entry_price
    entry_fee = stake_usd * TAKER_FEE_RATE
    future_hit = np.flatnonzero(bid[idx+1:] >= 0.99)
    if len(future_hit) > 0:
        exit_idx = int(idx + 1 + future_hit[0])
        exit_value = 0.99 * shares
        exit_fee = exit_value * TAKER_FEE_RATE
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return "sold_99", {
            "pnl": float(pnl), "entry_idx": idx, "exit_idx": exit_idx,
            "entry_price": entry_price, "exit_price": 0.99
        }

    final = outcome_value(w, leader) * shares
    pnl = final - stake_usd - entry_fee
    return "settlement", {
        "pnl": float(pnl), "entry_idx": idx, "exit_idx": None,
        "entry_price": entry_price, "exit_price": outcome_value(w, leader)
    }


# ========================================================================
# Runner
# ========================================================================

def run_strategy(windows, name, fn):
    pnls = []
    trades = []
    for w in windows:
        exit_type, data = fn(w)
        if data is None:
            continue
        pnls.append(data["pnl"])
        trades.append({"strategy": name, "exit_type": exit_type, **data})

    equity = STARTING_CAPITAL + np.cumsum(pnls)
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
    raw_sharpe = float(returns.mean() / returns.std(ddof=1) if len(returns) > 1 and returns.std(ddof=1) > 0 else 0.0)
    wins = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]

    metrics = {
        "strategy": name,
        "total_pnl": float(equity[-1] - STARTING_CAPITAL),
        "trades": len(trades),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "profit_factor": abs(sum(wins) / sum(losses)) if losses else float("inf"),
        "avg_pnl": float(np.mean(pnls)) if pnls else 0.0,
        "max_drawdown": float(max_drawdown(equity)),
        "raw_5m_sharpe": raw_sharpe,
    }

    df = pd.DataFrame([metrics])
    trade_df = pd.DataFrame(trades)
    return df, trade_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-windows", type=int, default=500)
    parser.add_argument("--train-frac", type=float, default=0.70)
    args = parser.parse_args()

    print("=" * 72)
    print("DUAL EXIT STRATEGY: Buy at 97c, sell at 99c OR 80c (first fill wins)")
    print("=" * 72)
    print("Loading data...\n")

    price_df = load_spot_prices("btcusdt", "binance", dataset_source="aliplayer_spot")
    windows = build_windows(price_df, args.max_windows)
    train, test = split_windows_chronologically(windows, args.train_frac)

    all_metrics = []
    for sample_name, sample_windows in [("TRAIN", train), ("TEST", test)]:
        print(f"\n{'='*72}")
        print(f"{sample_name} Results")
        print(f"{'='*72}")

        m_base, _ = run_strategy(sample_windows, "base", base_9799)
        m_dual, t_dual = run_strategy(sample_windows, "dual_exit", dual_exit)

        m_base.insert(0, "sample", sample_name)
        m_dual.insert(0, "sample", sample_name)

        all_metrics.extend([m_base, m_dual])

        # Print comparison
        print(f"\n  Base (sell at 99c only):")
        for c in m_base.columns:
            if c in ["total_pnl", "avg_pnl"]:
                print(f"    {c}: ${m_base[c].values[0]:,.2f}")
            elif c in ["win_rate", "max_drawdown"]:
                print(f"    {c}: {m_base[c].values[0]*100:.2f}%")
            elif c == "raw_5m_sharpe":
                print(f"    {c}: {m_base[c].values[0]:.3f}")

        print(f"\n  Dual Exit (sell at 99c or 80c):")
        for c in m_dual.columns:
            if c in ["total_pnl", "avg_pnl"]:
                print(f"    {c}: ${m_dual[c].values[0]:,.2f}")
            elif c in ["win_rate", "max_drawdown"]:
                print(f"    {c}: {m_dual[c].values[0]*100:.2f}%")
            elif c == "raw_5m_sharpe":
                print(f"    {c}: {m_dual[c].values[0]:.3f}")

        # Exit breakdown
        sold_99 = t_dual[t_dual["exit_type"] == "sold_99"]
        sold_80 = t_dual[t_dual["exit_type"] == "sold_80"]
        settled = t_dual[t_dual["exit_type"] == "settlement"]
        print(f"\n  Exit Breakdown (Dual Exit):")
        print(f"    Sold at 99c:   {len(sold_99)} trades, avg PnL: ${sold_99['pnl'].mean():.2f}" if len(sold_99) else "    Sold at 99c:   0 trades")
        print(f"    Sold at 80c:   {len(sold_80)} trades, avg PnL: ${sold_80['pnl'].mean():.2f}" if len(sold_80) else "    Sold at 80c:   0 trades")
        print(f"    Settled:       {len(settled)} trades, avg PnL: ${settled['pnl'].mean():.2f}" if len(settled) else "    Settled:       0 trades")

        # Save
        t_dual.to_csv(f"dual_exit_trades_{sample_name.lower()}.csv", index=False)

    final = pd.concat(all_metrics, ignore_index=True)
    final.to_csv("dual_exit_metrics.csv", index=False)
    print(f"\n{'='*72}")
    print("Analysis complete.")


if __name__ == "__main__":
    main()
