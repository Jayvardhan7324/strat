"""
Buy97 Dual-Exit Strategy with Realistic Microstructure Noise

Models real Polymarket behavior by adding noise to the synthetic bid/ask:
- Bid spikes above 0.99 or 0.80 with small probability (momentum, panic)
- Stops fire as price drops below threshold (stop-loss concept)
- Dual-bracket: TP at 99c, SL at 80c (or other levels)

Usage:
    python buy97_dual_exit_realistic.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm

from backtests.polymarket_updown_backtest import (
    STARTING_CAPITAL,
    TAKER_FEE_RATE,
    Window,
    build_windows,
    load_spot_prices,
    max_drawdown,
    split_windows_chronologically,
    SIGMA,
    fair_prices_for_window,
    clip_contract_price,
)


def side_prices(w, side):
    if side == "UP":
        return w.up_bid, w.up_ask
    if side == "DOWN":
        return w.down_bid, w.down_ask


def outcome_value(w, side):
    return float(w.outcome_up) if side == "UP" else float(1 - w.outcome_up)


# ========================================================================
# Model realistic noisy bids by adding stochastic jitter to synthetic prices
# ========================================================================

def add_microstructure_noise(w: Window, side: str, volatility_bps: float = 200.0):
    """Add realistic noise to simulate order book volatility."""
    bid, ask = side_prices(w, side)
    # Add Gaussian noise to bid, scaled by volatility_bps
    noise = np.random.randn(len(bid)) * (volatility_bps / 10000.0)
    noisy_bid = np.clip(bid + noise, 0.001, 0.999)
    return noisy_bid, ask


def dual_exit_realistic(w: Window, stake_usd=10.0, entry_seconds_left=30, tp=0.99, sl=0.80, noise_bps=200.0):
    """Buy at 97c, place TP at 99c, SL at 80c with realistic noise."""
    idx = 300 - entry_seconds_left
    leader = "UP" if w.prices[idx] >= w.open_price else "DOWN"
    bid, ask = add_microstructure_noise(w, leader, noise_bps)
    entry_price = float(ask[idx])

    if entry_price > 0.97:
        return None, None

    shares = stake_usd / entry_price
    entry_fee = stake_usd * TAKER_FEE_RATE

    # Check TP first
    hit_tp = np.flatnonzero(bid[idx + 1:] >= tp)
    # Check SL (when bid drops to or below SL)
    hit_sl = np.flatnonzero(bid[idx + 1:] <= sl)

    idx_tp = hit_tp[0] + idx + 1 if len(hit_tp) > 0 else float('inf')
    idx_sl = hit_sl[0] + idx + 1 if len(hit_sl) > 0 else float('inf')

    if idx_tp < idx_sl:
        exit_price = float(bid[idx_tp])
        exit_value = exit_price * shares
        exit_fee = exit_value * TAKER_FEE_RATE
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return "sold_tp", {
            "pnl": float(pnl), "entry_idx": idx, "exit_idx": int(idx_tp),
            "entry_price": entry_price, "exit_price": exit_price
        }
    elif idx_sl < idx_tp:
        exit_price = float(bid[idx_sl])
        exit_value = exit_price * shares
        exit_fee = exit_value * TAKER_FEE_RATE
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return "stopped", {
            "pnl": float(pnl), "entry_idx": idx, "exit_idx": int(idx_sl),
            "entry_price": entry_price, "exit_price": exit_price
        }

    # Neither hit
    final = outcome_value(w, leader) * shares
    pnl = final - stake_usd - entry_fee
    return "settlement", {
        "pnl": float(pnl), "entry_idx": idx, "exit_idx": None,
        "entry_price": entry_price, "exit_price": outcome_value(w, leader)
    }


# Single target strategies for comparison

def base_9799(w: Window, stake_usd=10.0, entry_seconds_left=30, noise_bps=200.0):
    """Original: sell at 99c only (with noise)."""
    idx = 300 - entry_seconds_left
    leader = "UP" if w.prices[idx] >= w.open_price else "DOWN"
    bid, ask = add_microstructure_noise(w, leader, noise_bps)
    entry_price = float(ask[idx])

    if entry_price > 0.97:
        return None, None

    shares = stake_usd / entry_price
    entry_fee = stake_usd * TAKER_FEE_RATE
    future_hit = np.flatnonzero(bid[idx+1:] >= 0.99)
    if len(future_hit) > 0:
        exit_idx = int(idx + 1 + future_hit[0])
        exit_price = float(bid[exit_idx])
        exit_value = exit_price * shares
        exit_fee = exit_value * TAKER_FEE_RATE
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return "sold_99", {
            "pnl": float(pnl), "entry_idx": idx, "exit_idx": exit_idx,
            "entry_price": entry_price, "exit_price": exit_price
        }

    final = outcome_value(w, leader) * shares
    pnl = final - stake_usd - entry_fee
    return "settlement", {
        "pnl": float(pnl), "entry_idx": idx, "exit_idx": None,
        "entry_price": entry_price, "exit_price": outcome_value(w, leader)
    }


def base_9780(w: Window, stake_usd=10.0, entry_seconds_left=30, noise_bps=200.0):
    """Sell at 80c only (with noise)."""
    idx = 300 - entry_seconds_left
    leader = "UP" if w.prices[idx] >= w.open_price else "DOWN"
    bid, ask = add_microstructure_noise(w, leader, noise_bps)
    entry_price = float(ask[idx])

    if entry_price > 0.97:
        return None, None

    shares = stake_usd / entry_price
    entry_fee = stake_usd * TAKER_FEE_RATE
    future_hit = np.flatnonzero(bid[idx+1:] >= 0.80)
    if len(future_hit) > 0:
        exit_idx = int(idx + 1 + future_hit[0])
        exit_price = float(bid[exit_idx])
        exit_value = exit_price * shares
        exit_fee = exit_value * TAKER_FEE_RATE
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return "sold_80", {
            "pnl": float(pnl), "entry_idx": idx, "exit_idx": exit_idx,
            "entry_price": entry_price, "exit_price": exit_price
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
    parser.add_argument("--noise-bps", type=float, default=200.0)
    args = parser.parse_args()

    print("=" * 72)
    print("DUAL-EXIT WITH REALISTIC MICROSTRUCTURE NOISE")
    print("=" * 72)
    print("Testing: Buy at 97c, sell at 99c OR 80c (first fill wins)")
    print(f"Bid noise: {args.noise_bps}bps (~{args.noise_bps/10000:.4f} per tick)")
    print("Loading data...\n")

    price_df = load_spot_prices("btcusdt", "binance", dataset_source="aliplayer_spot")
    windows = build_windows(price_df, args.max_windows)
    train, test = split_windows_chronologically(windows, args.train_frac)

    all_metrics = []
    for sample_name, sample_windows in [("TRAIN", train), ("TEST", test)]:
        print(f"\n{'='*72}")
        print(f"{sample_name} Results (noise={args.noise_bps}bps)")
        print(f"{'='*72}")

        m_base99, t_base99 = run_strategy(sample_windows, "base_99", base_9799)
        m_base80, t_base80 = run_strategy(sample_windows, "base_80", base_9780)
        m_dual, t_dual = run_strategy(sample_windows, "dual_99_80", lambda w: dual_exit_realistic(w, 10.0, 30, 0.99, 0.80, args.noise_bps))

        all_metrics.extend([m_base99, m_base80, m_dual])

        # Print comparison
        df = pd.concat([m_base99, m_base80, m_dual]).reset_index(drop=True)
        print(f"\n  {'Strategy':<18} {'PnL':>12} {'WR':>8} {'Avg PnL':>10} {'Sharpe':>8} {'DD':>8}")
        print(f"  {'-'*72}")
        for _, row in df.iterrows():
            print(f"  {row['strategy']:<18} ${row['total_pnl']:>10,.2f} {row['win_rate']*100:>6.1f}% ${row['avg_pnl']:>8.2f} {row['raw_5m_sharpe']:>6.3f} {row['max_drawdown']*100:>6.2f}%")

        # Exit breakdown
        for t_name, t_df in [("base_99", t_base99), ("base_80", t_base80), ("dual_99_80", t_dual)]:
            print(f"\n  Exit Breakdown: {t_name}")
            for exit_type in t_df["exit_type"].unique():
                count = len(t_df[t_df["exit_type"] == exit_type])
                avg_pnl = t_df[t_df["exit_type"] == exit_type]["pnl"].mean()
                print(f"    {exit_type:15s}: {count:>5} trades, avg PnL: ${avg_pnl:.2f}")

        # Save
        t_dual.to_csv(f"dual_exit_realistic_trades_{sample_name.lower()}.csv", index=False)

    final = pd.concat(all_metrics, ignore_index=True)
    final.to_csv("dual_exit_realistic_metrics.csv", index=False)
    print(f"\n{'='*72}")
    print("Analysis complete.")


if __name__ == "__main__":
    main()
