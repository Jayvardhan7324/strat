"""
Enhanced Buy-97 / Sell-99 Strategy Suite

Original: Buy at 97c, sell at 99c, settle if not hit.

Enhancements tested:
1.  Second sell order (bracket exit) - sell at 98c if 99c doesn't hit
2.  Time-decay entry (only enter if X seconds left)
3.  Time-decay stop (auto-exit at Y seconds left if not in profit)
4.  Trailing stop (exit if price drops Z% from entry)
5.  Combination strategies (pick the best enhancement)

Usage:
    python enhanced_buy97_strategies.py
    python enhanced_buy97_strategies.py --plot
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

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


def side_prices(w: Window, side: str) -> tuple[np.ndarray, np.ndarray]:
    if side == "UP":
        return w.up_bid, w.up_ask
    if side == "DOWN":
        return w.down_bid, w.down_ask
    raise ValueError(side)


def outcome_value(w: Window, side: str) -> float:
    return float(w.outcome_up) if side == "UP" else float(1 - w.outcome_up)


# ========================================================================
# Core strategy: buy at 97c, try to sell at 99c, settle otherwise
# ========================================================================

def base_9799(
    w: Window,
    stake_usd: float,
    entry_seconds_left: int = 30,
    fee_rate: float = TAKER_FEE_RATE,
) -> Trade | None:
    """Original: buy late leader at ~97c, try sell at 99c."""
    idx = 300 - entry_seconds_left
    leader = "UP" if w.prices[idx] >= w.open_price else "DOWN"
    bid, ask = side_prices(w, leader)
    entry_price = float(ask[idx])

    if entry_price > 0.97:
        return None

    shares = stake_usd / entry_price
    entry_fee = stake_usd * fee_rate

    # Try to sell at 99c
    future_hit = np.flatnonzero(bid[idx + 1 :] >= 0.99)
    if len(future_hit) > 0:
        exit_idx = int(idx + 1 + future_hit[0])
        exit_value = 0.99 * shares
        exit_fee = exit_value * fee_rate
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return Trade("base_9799", idx, exit_idx, leader, entry_price, 0.99, "sold_99", float(pnl))

    # Settle
    final = outcome_value(w, leader) * shares
    pnl = final - stake_usd - entry_fee
    return Trade("base_9799", idx, None, leader, entry_price, outcome_value(w, leader), "settlement", float(pnl))


# ========================================================================
# Enhancement 1: Bracket exit (sell at 98c if 99c doesn't hit)
# ========================================================================

def bracket_9798_99(
    w: Window,
    stake_usd: float,
    entry_seconds_left: int = 30,
    fee_rate: float = TAKER_FEE_RATE,
) -> Trade | None:
    """Sell at 99c if hit, else sell at 98c if hit, else settle."""
    idx = 300 - entry_seconds_left
    leader = "UP" if w.prices[idx] >= w.open_price else "DOWN"
    bid, ask = side_prices(w, leader)
    entry_price = float(ask[idx])

    if entry_price > 0.97:
        return None

    shares = stake_usd / entry_price
    entry_fee = stake_usd * fee_rate

    # Try 99c first
    hit_99 = np.flatnonzero(bid[idx + 1 :] >= 0.99)
    if len(hit_99) > 0:
        exit_idx = int(idx + 1 + hit_99[0])
        exit_value = 0.99 * shares
        exit_fee = exit_value * fee_rate
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return Trade("bracket_9798_99", idx, exit_idx, leader, entry_price, 0.99, "sold_99", float(pnl))

    # Try 98c second
    hit_98 = np.flatnonzero(bid[idx + 1:] >= 0.98)
    if len(hit_98) > 0:
        exit_idx = int(idx + 1 + hit_98[0])
        exit_value = 0.98 * shares
        exit_fee = exit_value * fee_rate
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return Trade("bracket_9798_99", idx, exit_idx, leader, entry_price, 0.98, "sold_98", float(pnl))

    # Settle
    final = outcome_value(w, leader) * shares
    pnl = final - stake_usd - entry_fee
    return Trade("bracket_9798_99", idx, None, leader, entry_price, outcome_value(w, leader), "settlement", float(pnl))


# ========================================================================
# Enhancement 2: Time-based stop (exit if not profitable by cutoff)
# ========================================================================

def timed_exit_9799(
    w: Window,
    stake_usd: float,
    entry_seconds_left: int = 30,
    min_seconds_left_to_exit: int = 5,
    fee_rate: float = TAKER_FEE_RATE,
) -> Trade | None:
    """If by `min_seconds_left_to_exit` the bid >= entry, hold; else settle early."""
    idx = 300 - entry_seconds_left
    leader = "UP" if w.prices[idx] >= w.open_price else "DOWN"
    bid, ask = side_prices(w, leader)
    entry_price = float(ask[idx])

    if entry_price > 0.97:
        return None

    shares = stake_usd / entry_price
    entry_fee = stake_usd * fee_rate

    # Try 99c
    hit_99 = np.flatnonzero(bid[idx + 1 :] >= 0.99)
    if len(hit_99) > 0:
        exit_idx = int(idx + 1 + hit_99[0])
        exit_value = 0.99 * shares
        exit_fee = exit_value * fee_rate
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return Trade("timed_exit_9799", idx, exit_idx, leader, entry_price, 0.99, "sold_99", float(pnl))

    # Time-based check
    exit_cutoff_idx = 300 - min_seconds_left_to_exit
    if exit_cutoff_idx > idx:
        bid_at_cutoff = bid[exit_cutoff_idx] if exit_cutoff_idx < len(bid) else bid[-1]
        # If bid is below entry, bail to cut loss
        if bid_at_cutoff < entry_price:
            exit_value = bid_at_cutoff * shares
            exit_fee = exit_value * fee_rate
            pnl = exit_value - stake_usd - entry_fee - exit_fee
            return Trade("timed_exit_9799", idx, exit_cutoff_idx, leader, entry_price, float(bid_at_cutoff), "timed_stop", float(pnl))

    # Settle
    final = outcome_value(w, leader) * shares
    pnl = final - stake_usd - entry_fee
    return Trade("timed_exit_9799", idx, None, leader, entry_price, outcome_value(w, leader), "settlement", float(pnl))


# ========================================================================
# Enhancement 3: Trailing stop (exit if price drops X% from best after entry)
# ========================================================================

def trailing_stop_9799(
    w: Window,
    stake_usd: float,
    entry_seconds_left: int = 30,
    trail_bps: float = 50.0,  # exit if bid drops 50bps below the max bid seen after entry
    fee_rate: float = TAKER_FEE_RATE,
) -> Trade | None:
    """Sell at 99c if hit; otherwise trail-stop at `trail_bps` below max bid."""
    idx = 300 - entry_seconds_left
    leader = "UP" if w.prices[idx] >= w.open_price else "DOWN"
    bid, ask = side_prices(w, leader)
    entry_price = float(ask[idx])

    if entry_price > 0.97:
        return None

    shares = stake_usd / entry_price
    entry_fee = stake_usd * fee_rate

    # Try 99c
    hit_99 = np.flatnonzero(bid[idx + 1 :] >= 0.99)
    if len(hit_99) > 0:
        exit_idx = int(idx + 1 + hit_99[0])
        exit_value = 0.99 * shares
        exit_fee = exit_value * fee_rate
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return Trade("trailing_9799", idx, exit_idx, leader, entry_price, 0.99, "sold_99", float(pnl))

    # Trailing stop
    max_bid = float(np.max(bid[idx:]))
    trail_threshold = max_bid - (trail_bps / 10000.0)
    # Find earliest point where bid drops below threshold
    trail_hits = np.flatnonzero(bid[idx + 1:] < trail_threshold)
    if len(trail_hits) > 0:
        exit_idx = int(idx + 1 + trail_hits[0])
        exit_price = float(bid[exit_idx])
        exit_value = exit_price * shares
        exit_fee = exit_value * fee_rate
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return Trade("trailing_9799", idx, exit_idx, leader, entry_price, exit_price, "trailing_stop", float(pnl))

    # Settle
    final = outcome_value(w, leader) * shares
    pnl = final - stake_usd - entry_fee
    return Trade("trailing_9799", idx, None, leader, entry_price, outcome_value(w, leader), "settlement", float(pnl))


# ========================================================================
# Enhancement 4: Combo (bracket + timed exit)
# ========================================================================

def combo_bracket_timed(
    w: Window,
    stake_usd: float,
    entry_seconds_left: int = 30,
    min_seconds_left_to_exit: int = 5,
    fee_rate: float = TAKER_FEE_RATE,
) -> Trade | None:
    """Bracked exit (99c then 98c) + timed stop."""
    idx = 300 - entry_seconds_left
    leader = "UP" if w.prices[idx] >= w.open_price else "DOWN"
    bid, ask = side_prices(w, leader)
    entry_price = float(ask[idx])

    if entry_price > 0.97:
        return None

    shares = stake_usd / entry_price
    entry_fee = stake_usd * fee_rate

    # Try 99c
    hit_99 = np.flatnonzero(bid[idx + 1 :] >= 0.99)
    if len(hit_99) > 0:
        exit_idx = int(idx + 1 + hit_99[0])
        exit_value = 0.99 * shares
        exit_fee = exit_value * fee_rate
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return Trade("combo_bracket_timed", idx, exit_idx, leader, entry_price, 0.99, "sold_99", float(pnl))

    # Try 98c
    hit_98 = np.flatnonzero(bid[idx + 1:] >= 0.98)
    if len(hit_98) > 0:
        exit_idx = int(idx + 1 + hit_98[0])
        exit_value = 0.98 * shares
        exit_fee = exit_value * fee_rate
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return Trade("combo_bracket_timed", idx, exit_idx, leader, entry_price, 0.98, "sold_98", float(pnl))

    # Time-based stop
    exit_cutoff_idx = 300 - min_seconds_left_to_exit
    if exit_cutoff_idx > idx:
        bid_at_cutoff = bid[exit_cutoff_idx] if exit_cutoff_idx < len(bid) else bid[-1]
        if bid_at_cutoff < entry_price:
            exit_value = bid_at_cutoff * shares
            exit_fee = exit_value * fee_rate
            pnl = exit_value - stake_usd - entry_fee - exit_fee
            return Trade("combo_bracket_timed", idx, exit_cutoff_idx, leader, entry_price, float(bid_at_cutoff), "timed_stop", float(pnl))

    # Settle
    final = outcome_value(w, leader) * shares
    pnl = final - stake_usd - entry_fee
    return Trade("combo_bracket_timed", idx, None, leader, entry_price, outcome_value(w, leader), "settlement", float(pnl))


# ========================================================================
# Enhancement 5: Combo (trailing + bracket)
# ========================================================================

# ========================================================================
# Runner
# ========================================================================

def run_strategy(windows: list[Window], name: str, strategy_fn) -> tuple[pd.DataFrame, pd.DataFrame]:
    pnl_by_window = np.zeros(len(windows), dtype=float)
    trades: list[Trade] = []
    for i, w in enumerate(windows):
        trade = strategy_fn(w)
        if trade is None:
            continue
        pnl_by_window[i] = trade.pnl
        trades.append(trade)

    equity = STARTING_CAPITAL + np.cumsum(pnl_by_window)
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
            "sold_99": sum(1 for t in trades if t.exit_type == "sold_99"),
            "sold_98": sum(1 for t in trades if t.exit_type == "sold_98"),
            "settled": sum(1 for t in trades if t.exit_type == "settlement"),
            "timed_stop": sum(1 for t in trades if t.exit_type == "timed_stop"),
            "trailing_stop": sum(1 for t in trades if t.exit_type == "trailing_stop"),
            "avg_entry_price": float(np.mean([t.entry_price for t in trades])) if trades else 0.0,
        }
    ])
    trade_df = pd.DataFrame([t.__dict__ for t in trades])
    return metrics, trade_df


def print_table(title: str, df: pd.DataFrame) -> None:
    print("\n" + title)
    out = df.copy()
    for col in ["total_pnl", "avg_pnl"]:
        out[col] = out[col].map(lambda x: f"${x:,.2f}")
    for col in ["win_rate", "max_drawdown"]:
        out[col] = out[col].map(lambda x: f"{100*x:.2f}%")
    out["profit_factor"] = out["profit_factor"].map(lambda x: "inf" if np.isinf(x) else f"{x:.2f}")
    out["raw_5m_sharpe"] = out["raw_5m_sharpe"].map(lambda x: f"{x:.3f}")
    out["avg_entry_price"] = out["avg_entry_price"].map(lambda x: f"{x:.3f}")
    print(out.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument("--source", default="binance")
    parser.add_argument("--dataset-source", default="aliplayer_spot")
    parser.add_argument("--stake-usd", type=float, default=10.0)
    parser.add_argument("--entry-seconds-left", type=int, default=30)
    parser.add_argument("--min-seconds-left", type=int, default=5)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--trail-bps", type=float, default=50.0)
    args = parser.parse_args()

    price_df = load_spot_prices(args.symbol, None if args.source.lower() in {"none", "all"} else args.source, dataset_source=args.dataset_source)
    windows = build_windows(price_df, args.max_windows)
    train, test = split_windows_chronologically(windows, args.train_frac)

    strategies = {
        "base_9799": lambda w: base_9799(w, args.stake_usd, args.entry_seconds_left, TAKER_FEE_RATE),
        "bracket_9798_99": lambda w: bracket_9798_99(w, args.stake_usd, args.entry_seconds_left, TAKER_FEE_RATE),
        "timed_exit_9799": lambda w: timed_exit_9799(w, args.stake_usd, args.entry_seconds_left, args.min_seconds_left, TAKER_FEE_RATE),
        "trailing_9799": lambda w: trailing_stop_9799(w, args.stake_usd, args.entry_seconds_left, args.trail_bps, TAKER_FEE_RATE),
        "combo_bracket_timed": lambda w: combo_bracket_timed(w, args.stake_usd, args.entry_seconds_left, args.min_seconds_left, TAKER_FEE_RATE),
    }

    all_metrics = []

    for sample_name, sample_windows in [("TRAIN", train), ("TEST", test)]:
        sample_metrics = []
        for strategy_name, strategy_fn in strategies.items():
            m, t = run_strategy(sample_windows, strategy_name, strategy_fn)
            m.insert(0, "sample", sample_name)
            t.insert(0, "sample", sample_name)
            sample_metrics.append(m)
            # Save trades for analysis
            t.to_csv(f"enhanced_buy97_{strategy_name}_{sample_name.lower()}_trades.csv", index=False)

        combined = pd.concat(sample_metrics, ignore_index=True)
        all_metrics.append(combined)
        print_table(f"{sample_name} Metrics", combined)

    final_metrics = pd.concat(all_metrics, ignore_index=True)
    final_metrics.to_csv("enhanced_buy97_strategies_metrics.csv", index=False)
    print("\nSaved enhanced_buy97_strategies_metrics.csv")


if __name__ == "__main__":
    main()
