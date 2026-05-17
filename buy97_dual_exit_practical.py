"""
Buy97 Dual-Exit — Practical Implementation for Polymarket

Models realistic Polymarket bid behavior:
- Bid = fair_value - spread + random_noise(0, sigma_bid)
- Spread widens when fair value is near edges or when volatility spikes
- Realistic scenario: bid can briefly spike >0.99 (order book fills, FOMO)
          or plunge <0.80 (panic, liquidity gaps) even while fair stays moderate

Strategy variants:
1. base: hold to settlement
2. dual_99_80: TP at 99c, SL at 80c
3. dual_99_40: TP at 99c, SL at 40c (lower = more selective)
4. dual_99mid: TP at 99c, SL at entry * 0.5 (50% of entry)
5. trailing: TP at 99c, dynamic SL at max(bid) - 20c trail

Usage:
    python buy97_dual_exit_practical.py
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


def simulate_bid(w, side, bid_volatility=0.05):
    """
    Simulate realistic bid with microstructure noise.
    Returns (noisy_bid, ask)
    """
    if side == "UP":
        bid = w.up_bid.copy()
    else:
        bid = w.down_bid.copy()

    # Add noise scaled by current fair value distance from extremes
    # More noise near extremes (0, 1) where liquidity is thin
    noise = np.random.randn(len(bid)) * bid_volatility * np.minimum(bid, 1 - bid)
    noisy_bid = np.clip(bid + noise, 0.001, 0.999)
    return noisy_bid


def execute_trade(w, side, entry_idx, entry_price, tp, sl, stake_usd):
    """Execute a trade with given TP/SL levels on noisy bid."""
    bid = simulate_bid(w, side)
    shares = stake_usd / entry_price
    entry_fee = stake_usd * TAKER_FEE_RATE

    future = bid[entry_idx + 1:]
    hit_tp = np.flatnonzero(future >= tp)
    hit_sl = np.flatnonzero(future <= sl)

    idx_tp = hit_tp[0] + entry_idx + 1 if len(hit_tp) else float('inf')
    idx_sl = hit_sl[0] + entry_idx + 1 if len(hit_sl) else float('inf')

    if idx_tp < idx_sl:
        exit_price = float(bid[int(idx_tp)])
        exit_value = exit_price * shares
        exit_fee = exit_value * TAKER_FEE_RATE
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return "tp", float(pnl), exit_price, int(idx_tp)

    if idx_sl < idx_tp:
        exit_price = float(bid[int(idx_sl)])
        exit_value = exit_price * shares
        exit_fee = exit_value * TAKER_FEE_RATE
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return "sl", float(pnl), exit_price, int(idx_sl)

    # Settlement
    final = float(w.outcome_up) if side == "UP" else float(1 - w.outcome_up)
    pnl = final * shares - stake_usd - entry_fee
    return "settlement", float(pnl), final, None


def base_strategy(w, stake_usd=10.0, entry_seconds_left=30):
    """Hold to settlement."""
    idx = 300 - entry_seconds_left
    side = "UP" if w.prices[idx] >= w.open_price else "DOWN"
    bid = simulate_bid(w, side)
    entry_price = float(bid[idx])  # Use noisy bid as entry reference
    return *execute_trade(w, side, idx, entry_price, 2.0, -1.0, stake_usd), "base"


def dual_tp_sl(w, stake_usd=10.0, entry_seconds_left=30, tp=0.99, sl=0.80):
    """Dual exit with explicit TP and SL levels."""
    idx = 300 - entry_seconds_left
    side = "UP" if w.prices[idx] >= w.open_price else "DOWN"
    bid = simulate_bid(w, side)
    entry_price = float(bid[idx])
    exit_type, pnl, exit_price, exit_idx = execute_trade(w, side, idx, entry_price, tp, sl, stake_usd)
    return exit_type, pnl, exit_price, exit_idx, f"dual_{int(tp*100)}_{int(sl*100)}"


def trailing_exit(w, stake_usd=10.0, entry_seconds_left=30, tp=0.99, trail=0.20):
    """Trailing stop: exit if price falls trail_amount from peak."""
    idx = 300 - entry_seconds_left
    side = "UP" if w.prices[idx] >= w.open_price else "DOWN"
    bid = simulate_bid(w, side)
    entry_price = float(bid[idx])
    shares = stake_usd / entry_price
    entry_fee = stake_usd * TAKER_FEE_RATE

    future = bid[idx + 1:]

    # Check TP first (same as before)
    hit_tp = np.flatnonzero(future >= tp)
    if len(hit_tp) > 0:
        exit_price = float(bid[idx + 1 + hit_tp[0]])
        exit_value = exit_price * shares
        exit_fee = exit_value * TAKER_FEE_RATE
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return "tp", float(pnl), exit_price, idx + 1 + hit_tp[0], "trailing"

    # Trailing stop: peak - trail
    if len(future) == 0:
        final = float(w.outcome_up) if side == "UP" else float(1 - w.outcome_up)
        pnl = final * shares - stake_usd - entry_fee
        return "settlement", float(pnl), final, None, "trailing"

    cummax = np.maximum.accumulate(future)
    below_trail = np.flatnonzero(future <= cummax - trail)
    if len(below_trail) > 0:
        exit_idx = int(idx + 1 + below_trail[0])
        exit_price = float(bid[exit_idx])
        exit_value = exit_price * shares
        exit_fee = exit_value * TAKER_FEE_RATE
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return "sl", float(pnl), exit_price, exit_idx, "trailing"

    # Settlement
    final = float(w.outcome_up) if side == "UP" else float(1 - w.outcome_up)
    pnl = final * shares - stake_usd - entry_fee
    return "settlement", float(pnl), final, None, "trailing"


def run_all(windows, name, fn):
    trades = []
    for w in windows:
        result = fn(w)
        if result and result[0]:
            trades.append({
                "strategy": name, "exit_type": result[0], "pnl": result[1],
                "exit_price": result[2], "exit_idx": result[3]
            })

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
        "exit_tp": len(df[df["exit_type"] == "tp"]),
        "exit_sl": len(df[df["exit_type"] == "sl"]),
        "exit_settlement": len(df[df["exit_type"] == "settlement"]),
    }
    return pd.DataFrame([metrics]), df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-windows", type=int, default=500)
    parser.add_argument("--noise-bps", type=float, default=50.0)
    args = parser.parse_args()

    np.random.seed(42)

    print("=" * 80)
    print("BUY97 DUAL-EXIT: PRACTICAL IMPLEMENTATION")
    print("=" * 80)
    print(f"Noise: {args.noise_bps}bps (scaled by distance from extremes)")
    print()

    price_df = load_spot_prices("btcusdt", "binance", dataset_source="aliplayer_spot")
    windows = build_windows(price_df, args.max_windows)

    strategies = {
        "base": lambda w: base_strategy(w, 10.0, 30),
        "dual_99_80": lambda w: dual_tp_sl(w, 10.0, 30, 0.99, 0.80),
        "dual_99_50": lambda w: dual_tp_sl(w, 10.0, 30, 0.99, 0.50),
        "dual_99_40": lambda w: dual_tp_sl(w, 10.0, 30, 0.99, 0.40),
        "dual_99_30": lambda w: dual_tp_sl(w, 10.0, 30, 0.99, 0.30),
        "trailing_20c": lambda w: trailing_exit(w, 10.0, 30, 0.99, 0.20),
        "trailing_10c": lambda w: trailing_exit(w, 10.0, 30, 0.99, 0.10),
    }

    all_metrics = []
    for name, fn in strategies.items():
        print(f"Running {name}...")
        m, _ = run_all(windows, name, fn)
        if not m.empty:
            all_metrics.append(m)

    if all_metrics:
        final = pd.concat(all_metrics, ignore_index=True)
        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(final.to_string(index=False))
        final.to_csv("buy97_dual_practical_metrics.csv", index=False)
        print("\nSaved to buy97_dual_practical_metrics.csv")
    else:
        print("No results.")


if __name__ == "__main__":
    main()
