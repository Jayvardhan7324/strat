"""
Buy-97 Optimization Suite

Tests multiple exit targets, entry timings, and hedging combinations
to find the best way to juice the buy97 strategy.

Strategy matrix:
- Entry: 0-60 seconds before settlement, 5-30 second steps
- Targets: 99c, 98c, 97c, 96c, 95c (sell at progressively lower prices)
- Hedge: Buy both sides, sell one when profitable, hold other to settlement
- Stagger: Enter multiple times at different prices within a window

Usage:
    python buy97_optimization_suite.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

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


@dataclass
class Trade:
    strategy: str
    entry_idx: int
    exit_idx: int | None
    side: str
    entry_price: float
    exit_price: float
    exit_type: str
    pnl: float


def side_prices(w: Window, side: str):
    if side == "UP":
        return w.up_bid, w.up_ask
    if side == "DOWN":
        return w.down_bid, w.down_ask
    raise ValueError(side)


def outcome_value(w: Window, side: str) -> float:
    return float(w.outcome_up) if side == "UP" else float(1 - w.outcome_up)


# ========================================================================
# A: Base (entry at fixed point, sell at 99c or settle)
# ========================================================================

def base_strategy(w: Window, stake_usd: float = 10.0, entry_seconds_left: int = 30, target: float = 0.99) -> Trade | None:
    idx = 300 - entry_seconds_left
    leader = "UP" if w.prices[idx] >= w.open_price else "DOWN"
    bid, ask = side_prices(w, leader)
    entry_price = float(ask[idx])
    if entry_price > 0.97:
        return None

    shares = stake_usd / entry_price
    entry_fee = stake_usd * TAKER_FEE_RATE
    future_hit = np.flatnonzero(bid[idx + 1 :] >= target)
    if len(future_hit) > 0:
        exit_idx = int(idx + 1 + future_hit[0])
        exit_value = target * shares
        exit_fee = exit_value * TAKER_FEE_RATE
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return Trade(f"base_target_{target:.2f}", idx, exit_idx, leader, entry_price, target, f"sold_{int(target*100)}", float(pnl))

    final = outcome_value(w, leader) * shares
    pnl = final - stake_usd - entry_fee
    return Trade(f"base_target_{target:.2f}", idx, None, leader, entry_price, outcome_value(w, leader), "settlement", float(pnl))


# ========================================================================
# B: Multi-target (try 99c, then 98c, then settle)
# ========================================================================

def multi_target_strategy(w: Window, stake_usd: float = 10.0, entry_seconds_left: int = 30) -> Trade | None:
    idx = 300 - entry_seconds_left
    leader = "UP" if w.prices[idx] >= w.open_price else "DOWN"
    bid, ask = side_prices(w, leader)
    entry_price = float(ask[idx])
    if entry_price > 0.97:
        return None

    shares = stake_usd / entry_price
    entry_fee = stake_usd * TAKER_FEE_RATE

    for target in [0.99, 0.98, 0.97, 0.96, 0.95]:
        future_hit = np.flatnonzero(bid[idx + 1 :] >= target)
        if len(future_hit) > 0:
            exit_idx = int(idx + 1 + future_hit[0])
            exit_value = target * shares
            exit_fee = exit_value * TAKER_FEE_RATE
            pnl = exit_value - stake_usd - entry_fee - exit_fee
            return Trade(f"multi_target", idx, exit_idx, leader, entry_price, target, f"sold_{int(target*100)}", float(pnl))

    final = outcome_value(w, leader) * shares
    pnl = final - stake_usd - entry_fee
    return Trade("multi_target", idx, None, leader, entry_price, outcome_value(w, leader), "settlement", float(pnl))


# ========================================================================
# C: Entry Sweep (try multiple entry points, take best)
# ========================================================================

def entry_sweep_strategy(w: Window, stake_usd: float = 10.0) -> Trade | None:
    best = None
    for entry_seconds_left in [5, 10, 15, 20, 25, 30, 40, 50, 60]:
        idx = 300 - entry_seconds_left
        if idx >= len(w.prices) or idx < 0:
            continue
        leader = "UP" if w.prices[idx] >= w.open_price else "DOWN"
        bid, ask = side_prices(w, leader)
        entry_price = float(ask[idx])
        if entry_price > 0.97:
            continue

        shares = stake_usd / entry_price
        entry_fee = stake_usd * TAKER_FEE_RATE
        future_hit = np.flatnonzero(bid[idx + 1 :] >= 0.99)
        if len(future_hit) > 0:
            exit_idx = int(idx + 1 + future_hit[0])
            exit_value = 0.99 * shares
            exit_fee = exit_value * TAKER_FEE_RATE
            pnl = exit_value - stake_usd - entry_fee - exit_fee
            t = Trade("entry_sweep", idx, exit_idx, leader, entry_price, 0.99, "sold_99", float(pnl))
        else:
            final = outcome_value(w, leader) * shares
            pnl = final - stake_usd - entry_fee
            t = Trade("entry_sweep", idx, None, leader, entry_price, outcome_value(w, leader), "settlement", float(pnl))

        if best is None or t.pnl > best.pnl:
            best = t

    return best


# ========================================================================
# D: Hedge (buy both sides, sell winner, hold loser to settlement)
# ========================================================================

def hedge_strategy(w: Window, stake_usd: float = 10.0, entry_seconds_left: int = 30) -> list[Trade]:
    idx = 300 - entry_seconds_left
    up_bid, up_ask = side_prices(w, "UP")
    down_bid, down_ask = side_prices(w, "DOWN")

    up_entry = float(up_ask[idx])
    down_entry = float(down_ask[idx])

    if up_entry > 0.97 or down_entry > 0.97:
        return []

    trades = []
    for side, entry_price in [("UP", up_entry), ("DOWN", down_entry)]:
        shares = stake_usd / entry_price
        entry_fee = stake_usd * TAKER_FEE_RATE
        bid, _ = side_prices(w, side)

        future_hit = np.flatnonzero(bid[idx + 1 :] >= 0.99)
        if len(future_hit) > 0:
            exit_idx = int(idx + 1 + future_hit[0])
            exit_value = 0.99 * shares
            exit_fee = exit_value * TAKER_FEE_RATE
            pnl = exit_value - stake_usd - entry_fee - exit_fee
            trades.append(Trade("hedge", idx, exit_idx, side, entry_price, 0.99, "sold_99", float(pnl)))
        else:
            final = outcome_value(w, side) * shares
            pnl = final - stake_usd - entry_fee
            trades.append(Trade("hedge", idx, None, side, entry_price, outcome_value(w, side), "settlement", float(pnl)))

    return trades


# ========================================================================
# E: Stagger (buy multiple times in one window, average down)
# ========================================================================

def stagger_strategy(w: Window, stake_usd: float = 10.0, num_entries: int = 3, entry_seconds_left: int = 30) -> list[Trade]:
    entries = []
    for i in range(num_entries):
        seconds_left = entry_seconds_left + i * 5
        idx = 300 - seconds_left
        if idx >= len(w.prices) or idx < 0:
            continue
        leader = "UP" if w.prices[idx] >= w.open_price else "DOWN"
        bid, ask = side_prices(w, leader)
        entry_price = float(ask[idx])
        if entry_price > 0.97:
            continue

        shares = stake_usd / entry_price
        entry_fee = stake_usd * TAKER_FEE_RATE
        future_hit = np.flatnonzero(bid[idx + 1 :] >= 0.99)
        if len(future_hit) > 0:
            exit_idx = int(idx + 1 + future_hit[0])
            exit_value = 0.99 * shares
            exit_fee = exit_value * TAKER_FEE_RATE
            pnl = exit_value - stake_usd - entry_fee - exit_fee
            entries.append(Trade("stagger", idx, exit_idx, leader, entry_price, 0.99, "sold_99", float(pnl)))
        else:
            final = outcome_value(w, leader) * shares
            pnl = final - stake_usd - entry_fee
            entries.append(Trade("stagger", idx, None, leader, entry_price, outcome_value(w, leader), "settlement", float(pnl)))
    return entries


# ========================================================================
# Runner
# ========================================================================

def run_and_metrics(windows, name, trade_fn, multi=False):
    pnls = np.zeros(len(windows), dtype=float)
    trades = []
    for i, w in enumerate(windows):
        result = trade_fn(w)
        if multi:
            for t in result:
                pnls[i] += t.pnl
                trades.append(t)
        else:
            if result is not None:
                pnls[i] = result.pnl
                trades.append(result)

    equity = STARTING_CAPITAL + np.cumsum(pnls)
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
    raw_sharpe = float(returns.mean() / returns.std(ddof=1) if len(returns) > 1 and returns.std(ddof=1) > 0 else 0.0)
    trade_pnls = [t.pnl for t in trades]
    wins = [x for x in trade_pnls if x > 0]
    losses = [x for x in trade_pnls if x < 0]
    gross_profit = float(sum(wins))
    gross_loss = float(-sum(losses))

    metrics = pd.DataFrame([
        {
            "strategy": name,
            "total_pnl": float(equity[-1] - STARTING_CAPITAL),
            "trades": len(trades),
            "win_rate": len(wins) / len(trades) if trades else 0.0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
            "avg_pnl": float(np.mean(trade_pnls)) if trade_pnls else 0.0,
            "max_drawdown": max_drawdown(equity),
            "raw_5m_sharpe": raw_sharpe,
        }
    ])
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-windows", type=int, default=2000)
    parser.add_argument("--train-frac", type=float, default=0.70)
    args = parser.parse_args()

    print("Loading data...")
    price_df = load_spot_prices("btcusdt", "binance", dataset_source="aliplayer_spot")
    windows = build_windows(price_df, args.max_windows)
    train, test = split_windows_chronologically(windows, args.train_frac)

    strategies = {
        "base_30s": lambda w: base_strategy(w, 10.0, 30, 0.99),
        "base_15s": lambda w: base_strategy(w, 10.0, 15, 0.99),
        "base_5s": lambda w: base_strategy(w, 10.0, 5, 0.99),
        "multi_target": multi_target_strategy,
        "entry_sweep": entry_sweep_strategy,
        "hedge": lambda w, fn=hedge_strategy: fn(w, 10.0, 30),
        "stagger_3x": lambda w, fn=stagger_strategy: fn(w, 10.0, 3, 30),
        "stagger_5x": lambda w, fn=stagger_strategy: fn(w, 10.0, 5, 20),
    }

    all_metrics = []
    for sample_name, sample_windows in [("TRAIN", train), ("TEST", test)]:
        sample_metrics = []
        for strategy_name, strategy_fn in strategies.items():
            print(f"Running {strategy_name} on {sample_name}...")
            is_multi = strategy_name in ("hedge", "stagger_3x", "stagger_5x")
            m = run_and_metrics(sample_windows, strategy_name, strategy_fn, multi=is_multi)
            m.insert(0, "sample", sample_name)
            sample_metrics.append(m)

        combined = pd.concat(sample_metrics, ignore_index=True)
        all_metrics.append(combined)
        print(f"\n{sample_name} Results:")
        print(combined.to_string(index=False))

    final = pd.concat(all_metrics, ignore_index=True)
    final.to_csv("buy97_optimization_metrics.csv", index=False)
    print("\nSaved to buy97_optimization_metrics.csv")


if __name__ == "__main__":
    main()
