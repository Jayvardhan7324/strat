"""
Buy97 Realistic Enhancement

Core idea: In synthetic data, the bid never hits 99c before settlement.
In real markets, it might. But instead of relying on that, let's focus on
what we CAN control: position sizing, time-based staggering, and
early-exit heuristics.

Strategies tested:
1. Base: Hold to settlement
2. Early-exit at midpoint: If the bid goes from entry to entry + delta, take profit
3. Staggered entries: Buy multiple positions at different times, average down
4. Dynamic sizing: Size position inversely to distance from settlement
5. Time-weighted exit: Exit early if time running out and not in profit

Usage:
    python buy97_realistic_enhancement.py
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


def base(w, stake=10.0, entry_sl=30):
    """Hold to settlement."""
    idx = 300 - entry_sl
    side = "UP" if w.prices[idx] >= w.open_price else "DOWN"
    bid, ask = (w.up_bid[idx], w.up_ask[idx]) if side == "UP" else (w.down_bid[idx], w.down_ask[idx])
    entry_price = float(ask)
    if entry_price > 0.97:
        return None, None
    shares = stake / entry_price
    entry_fee = stake * TAKER_FEE_RATE
    final = float(w.outcome_up) if side == "UP" else float(1 - w.outcome_up)
    pnl = final * shares - stake - entry_fee
    return "settlement", {"pnl": float(pnl), "entry_price": entry_price, "exit_price": final}


def early_exit_60pct(w, stake=10.0, entry_sl=30):
    """Exit when bid reaches entry_price + 0.10 (10c gain = ~20% return on 50c entry)."""
    idx = 300 - entry_sl
    side = "UP" if w.prices[idx] >= w.open_price else "DOWN"
    bid, ask = (w.up_bid[idx], w.up_ask[idx]) if side == "UP" else (w.down_bid[idx], w.down_ask[idx])
    entry_price = float(ask)
    if entry_price > 0.97:
        return None, None
    shares = stake / entry_price
    entry_fee = stake * TAKER_FEE_RATE

    future_bids = w.up_bid[idx+1:] if side == "UP" else w.down_bid[idx+1:]
    target = entry_price + 0.10
    hit = np.flatnonzero(future_bids >= target)
    if len(hit) > 0:
        exit_price = target
        exit_value = exit_price * shares
        exit_fee = exit_value * TAKER_FEE_RATE
        pnl = exit_value - stake - entry_fee - exit_fee
        return "early_60", {"pnl": float(pnl), "entry_price": entry_price, "exit_price": exit_price}

    final = float(w.outcome_up) if side == "UP" else float(1 - w.outcome_up)
    pnl = final * shares - stake - entry_fee
    return "settlement", {"pnl": float(pnl), "entry_price": entry_price, "exit_price": final}


def stagger_3x(w, stake=10.0, entry_sl=30):
    """Enter 3 times at different times, each with 1/3 stake."""
    total_pnl = 0
    for i in range(3):
        seconds = entry_sl + i * 10
        idx = 300 - seconds
        if idx < 0 or idx >= len(w.prices):
            continue
        side = "UP" if w.prices[idx] >= w.open_price else "DOWN"
        bid, ask = (w.up_bid[idx], w.up_ask[idx]) if side == "UP" else (w.down_bid[idx], w.down_ask[idx])
        entry_price = float(ask)
        if entry_price > 0.97:
            continue
        shares = (stake / 3) / entry_price
        entry_fee = (stake / 3) * TAKER_FEE_RATE
        final = float(w.outcome_up) if side == "UP" else float(1 - w.outcome_up)
        pnl = final * shares - (stake / 3) - entry_fee
        total_pnl += float(pnl)
    return "stagger_3x", {"pnl": total_pnl, "entry_price": entry_price, "exit_price": final}


def run(windows, name, fn):
    trades = []
    for w in windows:
        result = fn(w)
        if result and result[0]:
            trades.append({"strategy": name, "exit_type": result[0], "pnl": result[1]["pnl"],
                          "entry_price": result[1]["entry_price"], "exit_price": result[1]["exit_price"]})

    if not trades:
        return pd.DataFrame(), pd.DataFrame()

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
    }
    return pd.DataFrame([metrics]), df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-windows", type=int, default=500)
    args = parser.parse_args()

    print("Loading data...")
    price_df = load_spot_prices("btcusdt", "binance", dataset_source="aliplayer_spot")
    windows = build_windows(price_df, args.max_windows)

    strategies = {
        "base": base,
        "early_exit_60": early_exit_60pct,
        "stagger_3x": stagger_3x,
    }

    all_metrics = []
    for name, fn in strategies.items():
        print(f"Running {name}...")
        m, _ = run(windows, name, fn)
        if not m.empty:
            all_metrics.append(m)

    if all_metrics:
        final = pd.concat(all_metrics, ignore_index=True)
        print(final.to_string(index=False))
        final.to_csv("buy97_realistic_enhancement_metrics.csv", index=False)
        print("\nSaved to buy97_realistic_enhancement_metrics.csv")


if __name__ == "__main__":
    main()
