"""
Opening Chop Scalper - Backtest

Exploits price oscillations in the first 2 minutes of each 5-minute BTC market.

Strategy:
1. First 120s: measure volatility, establish high/low range and midpoint
2. When price crosses above mid and starts reversing down -> scalp DOWN
3. When price crosses below mid and starts reversing up -> scalp UP
4. Each scalp targets ~3-5 cents on contract price
5. Multiple scalps per window possible

Usage:
    python backtest_opening_chop_scalper.py
    python backtest_opening_chop_scalper.py --max-windows 5000
    python backtest_opening_chop_scalper.py --scalp-target 0.03
    python backtest_opening_chop_scalper.py --min-volatility 0.0005
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from polymarket_updown_backtest import (
    STARTING_CAPITAL,
    TAKER_FEE_RATE,
    MAKER_REBATE_RATE,
    WINDOW_SECONDS,
    Window,
    build_windows,
    load_spot_prices,
    max_drawdown,
    split_windows_chronologically,
    side_arrays,
    settlement_value,
    clip_contract_price,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

@dataclass
class ScalpTrade:
    window_start: pd.Timestamp
    scalp_num: int
    side: str  # "UP" or "DOWN"
    entry_idx: int
    entry_price: float
    exit_idx: int | None
    exit_price: float
    exit_type: str  # "target", "stop", "settlement", "timeout"
    pnl: float
    seconds_held: int
    entry_reason: str  # "cross_up_revert", "cross_down_revert"


# ---------------------------------------------------------------------------
# Core strategy
# ---------------------------------------------------------------------------

def opening_chop_scalper(
    w: Window,
    scalp_target: float = 0.04,
    stop_loss: float = 0.02,
    min_volatility: float = 0.0003,
    measure_seconds: int = 120,
    reversal_confirm_seconds: int = 5,
    max_scalps_per_window: int = 5,
    cooldown_seconds: int = 15,
    stake_qty: float = 1000.0,
) -> list[ScalpTrade]:
    """Scalp oscillations in the opening chop zone.

    Args:
        w: The 5-minute window.
        scalp_target: Contract price move to take profit (e.g. 0.04 = 4 cents).
        stop_loss: Contract price move to cut loss.
        min_volatility: Minimum BTC price range % in measure period to trade.
        measure_seconds: How many seconds to measure the opening range.
        reversal_confirm_seconds: Seconds of reversal confirmation before entry.
        max_scalps_per_window: Max scalps per window.
        cooldown_seconds: Seconds to wait after exiting before next entry.
    """
    p = w.prices
    trades: list[ScalpTrade] = []

    # --- Phase 1: Measure opening range (first N seconds) ---
    open_price = p[0]
    high_120 = float(p[:measure_seconds].max())
    low_120 = float(p[:measure_seconds].min())
    mid_120 = (high_120 + low_120) / 2.0
    range_pct = (high_120 - low_120) / open_price

    if range_pct < min_volatility:
        return trades  # Not enough chop, skip

    # --- Phase 2: Scalp reversals after the measure period ---
    last_exit_idx = measure_seconds
    scalp_num = 0

    for idx in range(measure_seconds + reversal_confirm_seconds, WINDOW_SECONDS - 10):
        if scalp_num >= max_scalps_per_window:
            break
        if idx < last_exit_idx + cooldown_seconds:
            continue

        # Check if price crossed above mid and is now reversing down
        # Look back: was price above mid recently, now below?
        recent_high = float(p[idx - reversal_confirm_seconds:idx].max())
        current_price = float(p[idx])

        # Cross above mid then revert down
        crossed_above = recent_high > mid_120
        reverting_down = current_price < mid_120

        # Cross below mid then revert up
        recent_low = float(p[idx - reversal_confirm_seconds:idx].min())
        crossed_below = recent_low < mid_120
        reverting_up = current_price > mid_120

        side = None
        entry_reason = None

        if crossed_above and reverting_down:
            side = "DOWN"
            entry_reason = "cross_above_revert_down"
        elif crossed_below and reverting_up:
            side = "UP"
            entry_reason = "cross_below_revert_up"

        if side is None:
            continue

        # Get entry price (ask for the side we're buying)
        _, ask = side_arrays(w, side.lower())
        entry_price = float(ask[idx])

        # --- Phase 3: Manage the trade ---
        bid, _ = side_arrays(w, side.lower())
        entry_contract_price = entry_price

        target_price = entry_contract_price + scalp_target if side == "UP" else entry_contract_price - scalp_target
        stop_price = entry_contract_price - stop_loss if side == "UP" else entry_contract_price + stop_loss

        exited = False
        exit_idx = None
        exit_price = None
        exit_type = None

        for exit_idx_candidate in range(idx + 1, WINDOW_SECONDS - 5):
            current_bid = float(bid[exit_idx_candidate])

            # Check target
            if side == "UP" and current_bid >= target_price:
                exit_price = current_bid
                exit_type = "target"
                exited = True
                break
            if side == "DOWN" and current_bid <= target_price:
                exit_price = current_bid
                exit_type = "target"
                exited = True
                break

            # Check stop
            if side == "UP" and current_bid <= stop_price:
                exit_price = current_bid
                exit_type = "stop"
                exited = True
                break
            if side == "DOWN" and current_bid >= stop_price:
                exit_price = current_bid
                exit_type = "stop"
                exited = True
                break

        if not exited:
            # Hold to settlement
            exit_idx_candidate = WINDOW_SECONDS - 1
            exit_price = float(bid[exit_idx_candidate])
            exit_type = "settlement"
            exited = True

        # Compute PnL
        shares = stake_qty / entry_price
        entry_fee = entry_price * shares * TAKER_FEE_RATE
        exit_fee = exit_price * shares * TAKER_FEE_RATE
        pnl = (exit_price - entry_price) * shares - entry_fee - exit_fee

        scalp_num += 1
        trades.append(ScalpTrade(
            window_start=w.start,
            scalp_num=scalp_num,
            side=side,
            entry_idx=idx,
            entry_price=entry_price,
            exit_idx=exit_idx_candidate,
            exit_price=exit_price,
            exit_type=exit_type,
            pnl=pnl,
            seconds_held=exit_idx_candidate - idx,
            entry_reason=entry_reason,
        ))

        last_exit_idx = exit_idx_candidate

    return trades


# ---------------------------------------------------------------------------
# Alternative: Multi-level scalp (place orders at multiple levels)
# ---------------------------------------------------------------------------

def multi_level_chop_scalper(
    w: Window,
    scalp_target: float = 0.03,
    stop_loss: float = 0.015,
    min_volatility: float = 0.0003,
    measure_seconds: int = 120,
    num_levels: int = 3,
    level_spacing: float = 0.0002,
    stake_qty: float = 1000.0,
) -> list[ScalpTrade]:
    """Place multiple scalp orders at different levels within the chop range.

    Instead of waiting for reversals, this places limit orders at multiple
    levels within the measured range and scalps each bounce.
    """
    p = w.prices
    trades: list[ScalpTrade] = []

    open_price = p[0]
    high_120 = float(p[:measure_seconds].max())
    low_120 = float(p[:measure_seconds].min())
    mid_120 = (high_120 + low_120) / 2.0
    range_pct = (high_120 - low_120) / open_price

    if range_pct < min_volatility:
        return trades

    # Create levels within the range
    levels = []
    for i in range(num_levels):
        frac = (i + 1) / (num_levels + 1)
        level_price = low_120 + frac * (high_120 - low_120)
        levels.append(level_price)

    # Track which levels have been triggered
    level_triggered = [False] * num_levels
    last_exit_idx = measure_seconds

    for idx in range(measure_seconds, WINDOW_SECONDS - 10):
        current_price = float(p[idx])

        for i, level_price in enumerate(levels):
            if level_triggered[i]:
                continue

            # Price crossed this level
            if idx > 0:
                prev_price = float(p[idx - 1])
                crossed = (prev_price < level_price <= current_price) or (prev_price > level_price >= current_price)

                if crossed:
                    # Determine direction: if crossing upward, scalp DOWN (fade)
                    # If crossing downward, scalp UP (fade)
                    side = "DOWN" if current_price > prev_price else "UP"
                    entry_reason = f"level_{i}_fade"

                    _, ask = side_arrays(w, side.lower())
                    entry_price = float(ask[idx])

                    # Manage trade
                    bid, _ = side_arrays(w, side.lower())
                    target_price = entry_price + scalp_target if side == "UP" else entry_price - scalp_target
                    stop_price = entry_price - stop_loss if side == "UP" else entry_price + stop_loss

                    exited = False
                    exit_idx_candidate = None
                    exit_price = None
                    exit_type = None

                    for exit_idx_candidate in range(idx + 1, WINDOW_SECONDS - 5):
                        current_bid = float(bid[exit_idx_candidate])

                        if side == "UP" and current_bid >= target_price:
                            exit_price = current_bid
                            exit_type = "target"
                            exited = True
                            break
                        if side == "DOWN" and current_bid <= target_price:
                            exit_price = current_bid
                            exit_type = "target"
                            exited = True
                            break
                        if side == "UP" and current_bid <= stop_price:
                            exit_price = current_bid
                            exit_type = "stop"
                            exited = True
                            break
                        if side == "DOWN" and current_bid >= stop_price:
                            exit_price = current_bid
                            exit_type = "stop"
                            exited = True
                            break

                    if not exited:
                        exit_idx_candidate = WINDOW_SECONDS - 1
                        exit_price = float(bid[exit_idx_candidate])
                        exit_type = "settlement"

                    shares = stake_qty / entry_price
                    entry_fee = entry_price * shares * TAKER_FEE_RATE
                    exit_fee = exit_price * shares * TAKER_FEE_RATE
                    pnl = (exit_price - entry_price) * shares - entry_fee - exit_fee

                    level_triggered[i] = True
                    trades.append(ScalpTrade(
                        window_start=w.start,
                        scalp_num=i + 1,
                        side=side,
                        entry_idx=idx,
                        entry_price=entry_price,
                        exit_idx=exit_idx_candidate,
                        exit_price=exit_price,
                        exit_type=exit_type,
                        pnl=pnl,
                        seconds_held=exit_idx_candidate - idx,
                        entry_reason=entry_reason,
                    ))

                    last_exit_idx = exit_idx_candidate

    return trades


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

DEFAULT_TRADE_QTY = 1000.0


@dataclass
class ScalpMetrics:
    strategy: str
    sample: str
    total_pnl: float
    total_trades: int
    windows_traded: int
    windows_total: int
    win_rate: float
    avg_pnl: float
    profit_factor: float
    max_drawdown: float
    sharpe: float
    avg_hold_seconds: float
    avg_entry_price: float
    target_hits: int
    stop_hits: int
    settlement_exits: int
    avg_scalps_per_window: float
    avg_range_pct: float


def compute_scalp_metrics(
    all_trades: list[ScalpTrade],
    windows: list[Window],
    pnl_by_window: np.ndarray,
    sample: str,
    strategy: str,
) -> ScalpMetrics:
    equity = STARTING_CAPITAL + np.cumsum(pnl_by_window)
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-12)

    sharpe = 0.0
    if len(returns) > 1 and returns.std(ddof=1) > 0:
        sharpe = float((returns.mean() / returns.std(ddof=1)) * np.sqrt(365 * 24 * 12))

    pnls = [t.pnl for t in all_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    gross_profit = sum(wins) if wins else 0
    gross_loss = -sum(losses) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    target_hits = sum(1 for t in all_trades if t.exit_type == "target")
    stop_hits = sum(1 for t in all_trades if t.exit_type == "stop")
    settlement_exits = sum(1 for t in all_trades if t.exit_type == "settlement")

    windows_traded = len(set(t.window_start for t in all_trades))

    range_pcts = []
    for w in windows:
        high = float(w.prices[:120].max())
        low = float(w.prices[:120].min())
        range_pcts.append((high - low) / w.open_price)

    return ScalpMetrics(
        strategy=strategy,
        sample=sample,
        total_pnl=float(equity[-1] - STARTING_CAPITAL),
        total_trades=len(all_trades),
        windows_traded=windows_traded,
        windows_total=len(windows),
        win_rate=len(wins) / len(all_trades) if all_trades else 0.0,
        avg_pnl=float(np.mean(pnls)) if pnls else 0.0,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown(equity),
        sharpe=sharpe,
        avg_hold_seconds=float(np.mean([t.seconds_held for t in all_trades])) if all_trades else 0.0,
        avg_entry_price=float(np.mean([t.entry_price for t in all_trades])) if all_trades else 0.0,
        target_hits=target_hits,
        stop_hits=stop_hits,
        settlement_exits=settlement_exits,
        avg_scalps_per_window=len(all_trades) / max(windows_traded, 1),
        avg_range_pct=float(np.mean(range_pcts)) if range_pcts else 0.0,
    )


def print_metrics_table(metrics_list: list[ScalpMetrics]) -> None:
    df = pd.DataFrame([m.__dict__ for m in metrics_list])
    if df.empty:
        print("No trades generated.")
        return

    out = df.copy()
    for col in ["total_pnl", "avg_pnl"]:
        out[col] = out[col].map(lambda x: f"${x:,.2f}")
    for col in ["win_rate", "max_drawdown"]:
        out[col] = out[col].map(lambda x: f"{100*x:.1f}%")
    out["profit_factor"] = out["profit_factor"].map(lambda x: "inf" if np.isinf(x) else f"{x:.2f}")
    out["sharpe"] = out["sharpe"].map(lambda x: f"{x:.2f}")
    out["avg_hold_seconds"] = out["avg_hold_seconds"].map(lambda x: f"{x:.1f}s")
    out["avg_entry_price"] = out["avg_entry_price"].map(lambda x: f"{x:.4f}")
    out["avg_range_pct"] = out["avg_range_pct"].map(lambda x: f"{100*x:.3f}%")
    out["avg_scalps_per_window"] = out["avg_scalps_per_window"].map(lambda x: f"{x:.2f}")

    cols = [
        "sample", "strategy", "total_pnl", "total_trades", "windows_traded",
        "win_rate", "profit_factor", "avg_pnl", "sharpe", "max_drawdown",
        "avg_hold_seconds", "target_hits", "stop_hits", "settlement_exits",
        "avg_scalps_per_window", "avg_range_pct",
    ]
    print(out[cols].to_string(index=False))


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def param_sweep(
    windows: list[Window],
    sample: str,
) -> pd.DataFrame:
    """Sweep key parameters to find optimal settings."""
    results = []

    scalp_targets = [0.02, 0.03, 0.04, 0.05, 0.06]
    min_vols = [0.0002, 0.0003, 0.0005, 0.0008]
    stop_losses = [0.01, 0.015, 0.02, 0.025]

    for target in scalp_targets:
        for min_vol in min_vols:
            for stop in stop_losses:
                all_trades = []
                pnl_by_window = np.zeros(len(windows), dtype=float)

                for i, w in enumerate(windows):
                    trades = opening_chop_scalper(
                        w,
                        scalp_target=target,
                        stop_loss=stop,
                        min_volatility=min_vol,
                        stake_qty=args.stake_usd,
                    )
                    pnl_by_window[i] = sum(t.pnl for t in trades)
                    all_trades.extend(trades)

                if not all_trades:
                    continue

                metrics = compute_scalp_metrics(
                    all_trades, windows, pnl_by_window, sample,
                    f"target={target:.2f}_vol={min_vol:.4f}_stop={stop:.2f}"
                )
                results.append({
                    "target": target,
                    "min_vol": min_vol,
                    "stop": stop,
                    "total_pnl": metrics.total_pnl,
                    "trades": metrics.total_trades,
                    "win_rate": metrics.win_rate,
                    "profit_factor": metrics.profit_factor if not np.isinf(metrics.profit_factor) else 999,
                    "sharpe": metrics.sharpe,
                    "max_drawdown": metrics.max_drawdown,
                })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Opening Chop Scalper Backtest")
    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument("--source", default="binance")
    parser.add_argument("--dataset-source", default="aliplayer_spot")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--stake-usd", type=float, default=DEFAULT_TRADE_QTY)
    parser.add_argument("--scalp-target", type=float, default=0.04, help="Target profit in contract price")
    parser.add_argument("--stop-loss", type=float, default=0.02, help="Stop loss in contract price")
    parser.add_argument("--min-volatility", type=float, default=0.0003, help="Min BTC price range pct to trade")
    parser.add_argument("--measure-seconds", type=int, default=120, help="Opening range measurement period")
    parser.add_argument("--max-scalps", type=int, default=5, help="Max scalps per window")
    parser.add_argument("--cooldown", type=int, default=15, help="Seconds between scalps")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--multi-level", action="store_true", help="Use multi-level scalp mode")
    parser.add_argument("--num-levels", type=int, default=3, help="Number of levels for multi-level mode")
    args = parser.parse_args()

    price_df = load_spot_prices(args.symbol, None if args.source.lower() in {"none", "all"} else args.source, dataset_source=args.dataset_source)
    windows = build_windows(price_df, args.max_windows)

    if args.sweep:
        print("\n" + "=" * 80)
        print("PARAMETER SWEEP")
        print("=" * 80)

        train_windows, test_windows = split_windows_chronologically(windows, args.train_frac)

        print("\nTRAIN sweep...")
        train_results = param_sweep(train_windows, "TRAIN")
        if not train_results.empty:
            train_results = train_results.sort_values("total_pnl", ascending=False)
            print(f"\nTop 10 configs (by TRAIN PnL):")
            print(train_results.head(10).to_string(index=False))
            train_results.to_csv("chop_scalper_sweep_train.csv", index=False)

        print("\nTEST sweep (top 5 train configs)...")
        if not train_results.empty:
            top_configs = train_results.head(5)
            test_results = []
            for _, row in top_configs.iterrows():
                all_trades = []
                pnl_by_window = np.zeros(len(test_windows), dtype=float)
                for i, w in enumerate(test_windows):
                    trades = opening_chop_scalper(
                        w,
                        scalp_target=row["target"],
                        stop_loss=row["stop"],
                        min_volatility=row["min_vol"],
                        stake_qty=args.stake_usd,
                    )
                    pnl_by_window[i] = sum(t.pnl for t in trades)
                    all_trades.extend(trades)

                if all_trades:
                    metrics = compute_scalp_metrics(all_trades, test_windows, pnl_by_window, "TEST",
                        f"target={row['target']:.2f}_vol={row['min_vol']:.4f}_stop={row['stop']:.2f}")
                    test_results.append({
                        "target": row["target"],
                        "min_vol": row["min_vol"],
                        "stop": row["stop"],
                        "total_pnl": metrics.total_pnl,
                        "trades": metrics.total_trades,
                        "win_rate": metrics.win_rate,
                        "profit_factor": metrics.profit_factor if not np.isinf(metrics.profit_factor) else 999,
                        "sharpe": metrics.sharpe,
                        "max_drawdown": metrics.max_drawdown,
                    })

            if test_results:
                test_df = pd.DataFrame(test_results).sort_values("total_pnl", ascending=False)
                print(f"\nTEST results for top TRAIN configs:")
                print(test_df.to_string(index=False))
                test_df.to_csv("chop_scalper_sweep_test.csv", index=False)

        return

    # --- Single config backtest ---
    print("\n" + "=" * 80)
    print("OPENING CHOP SCALPER BACKTEST")
    print("=" * 80)
    print(f"Scalp target: {args.scalp_target:.2f} (contract price)")
    print(f"Stop loss: {args.stop_loss:.2f}")
    print(f"Min volatility: {args.min_volatility:.4f}")
    print(f"Measure period: {args.measure_seconds}s")
    print(f"Max scalps/window: {args.max_scalps}")
    print(f"Cooldown: {args.cooldown}s")
    print(f"Stake: ${args.stake_usd:,.0f}")
    print()

    train_windows, test_windows = split_windows_chronologically(windows, args.train_frac)

    all_metrics = []

    for sample, sample_windows in [("TRAIN", train_windows), ("TEST", test_windows)]:
        print(f"\n--- {sample} SET ({len(sample_windows)} windows) ---")

        all_trades = []
        pnl_by_window = np.zeros(len(sample_windows), dtype=float)

        for i, w in enumerate(sample_windows):
            if args.multi_level:
                trades = multi_level_chop_scalper(
                    w,
                    scalp_target=args.scalp_target,
                    stop_loss=args.stop_loss,
                    min_volatility=args.min_volatility,
                    measure_seconds=args.measure_seconds,
                    num_levels=args.num_levels,
                    stake_qty=args.stake_usd,
                )
            else:
                trades = opening_chop_scalper(
                    w,
                    scalp_target=args.scalp_target,
                    stop_loss=args.stop_loss,
                    min_volatility=args.min_volatility,
                    measure_seconds=args.measure_seconds,
                    max_scalps_per_window=args.max_scalps,
                    cooldown_seconds=args.cooldown,
                    stake_qty=args.stake_usd,
                )

            pnl_by_window[i] = sum(t.pnl for t in trades)
            all_trades.extend(trades)

        strategy_name = "multi_level_chop" if args.multi_level else "opening_chop_scalper"
        metrics = compute_scalp_metrics(all_trades, sample_windows, pnl_by_window, sample, strategy_name)
        all_metrics.append(metrics)

        print(f"  Trades: {metrics.total_trades}")
        print(f"  Windows traded: {metrics.windows_traded}/{metrics.windows_total}")
        print(f"  Total PnL: ${metrics.total_pnl:,.2f}")
        print(f"  Win rate: {100*metrics.win_rate:.1f}%")
        print(f"  Profit factor: {metrics.profit_factor:.2f}" if not np.isinf(metrics.profit_factor) else "  Profit factor: inf")
        print(f"  Sharpe: {metrics.sharpe:.2f}")
        print(f"  Max DD: {100*metrics.max_drawdown:.1f}%")
        print(f"  Avg hold: {metrics.avg_hold_seconds:.1f}s")
        print(f"  Target hits: {metrics.target_hits}, Stops: {metrics.stop_hits}, Settlement: {metrics.settlement_exits}")

        # Save trades
        if all_trades:
            trades_df = pd.DataFrame([t.__dict__ for t in all_trades])
            trades_path = f"chop_scalper_trades_{sample.lower()}.csv"
            trades_df.to_csv(trades_path, index=False)
            print(f"  Saved {trades_path}")

    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)
    print_metrics_table(all_metrics)

    metrics_df = pd.DataFrame([m.__dict__ for m in all_metrics])
    metrics_df.to_csv("chop_scalper_metrics.csv", index=False)
    print(f"\nSaved chop_scalper_metrics.csv")


if __name__ == "__main__":
    main()
