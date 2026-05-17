"""
Opening Chop Scalper V2 - Backtest

Exploits price oscillations in the first 2 minutes of each 5-minute BTC market.

Key insight: When BTC price chops (goes up then down, or down then up) in the
opening minutes, the contract fair value wobbles. We scalp these micro-reversions
with tight targets (0.5-2 cents on contract price).

Strategy modes:
1. REVERSAL: Fade the first significant move after the opening range
2. CHOP: Trade multiple cross-backs of the opening range midpoint
3. BREAKOUT_FADE: Fade false breakouts of the opening range

Usage:
    python backtest_opening_chop_scalper_v2.py
    python backtest_opening_chop_scalper_v2.py --mode chop --scalp-target 0.008
    python backtest_opening_chop_scalper_v2.py --sweep
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
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
    entry_reason: str


# ---------------------------------------------------------------------------
# Strategy Mode 1: REVERSAL
# Fade the first significant move after opening range
# ---------------------------------------------------------------------------

def strat_reversal(
    w: Window,
    scalp_target: float = 0.008,
    stop_loss: float = 0.006,
    min_volatility: float = 0.0004,
    measure_seconds: int = 120,
    stake_qty: float = 1000.0,
) -> list[ScalpTrade]:
    """Fade the first significant directional move after opening range."""
    p = w.prices
    trades: list[ScalpTrade] = []

    open_price = p[0]
    high_m = float(p[:measure_seconds].max())
    low_m = float(p[:measure_seconds].min())
    range_pct = (high_m - low_m) / open_price

    if range_pct < min_volatility:
        return trades

    mid = (high_m + low_m) / 2.0

    # After measure period, look for price to extend beyond range then revert
    for idx in range(measure_seconds, WINDOW_SECONDS - 30):
        current = float(p[idx])

        # Price broke above range high and is now coming back
        if current > high_m:
            # Wait for it to cross back below high
            if idx > measure_seconds and float(p[idx - 1]) >= high_m and current < high_m:
                side = "DOWN"
                entry_reason = "fade_breakout_above"
                _, ask = side_arrays(w, side.lower())
                entry_price = float(ask[idx])

                bid, ask_arr = side_arrays(w, side.lower())
                exit_idx, exit_price, exit_type = _manage_trade(w, bid, ask_arr, idx + 1, entry_price, scalp_target, stop_loss, side)
                pnl = _calc_pnl(entry_price, exit_price, stake_qty)

                trades.append(ScalpTrade(w.start, 1, side, idx, entry_price, exit_idx, exit_price, exit_type, pnl, exit_idx - idx, entry_reason))
                break

        # Price broke below range low and is now coming back
        elif current < low_m:
            if idx > measure_seconds and float(p[idx - 1]) <= low_m and current > low_m:
                side = "UP"
                entry_reason = "fade_breakout_below"
                _, ask = side_arrays(w, side.lower())
                entry_price = float(ask[idx])

                bid, ask_arr = side_arrays(w, side.lower())
                exit_idx, exit_price, exit_type = _manage_trade(w, bid, ask_arr, idx + 1, entry_price, scalp_target, stop_loss, side)
                pnl = _calc_pnl(entry_price, exit_price, stake_qty)

                trades.append(ScalpTrade(w.start, 1, side, idx, entry_price, exit_idx, exit_price, exit_type, pnl, exit_idx - idx, entry_reason))
                break

    return trades


# ---------------------------------------------------------------------------
# Strategy Mode 2: CHOP
# Trade multiple cross-backs of the opening range midpoint
# ---------------------------------------------------------------------------

def strat_chop(
    w: Window,
    scalp_target: float = 0.006,
    stop_loss: float = 0.004,
    min_volatility: float = 0.0004,
    measure_seconds: int = 120,
    max_scalps: int = 3,
    cooldown: int = 20,
    stake_qty: float = 1000.0,
) -> list[ScalpTrade]:
    """Trade each cross-back of the midpoint as a mean-reversion scalp."""
    p = w.prices
    trades: list[ScalpTrade] = []

    open_price = p[0]
    high_m = float(p[:measure_seconds].max())
    low_m = float(p[:measure_seconds].min())
    range_pct = (high_m - low_m) / open_price

    if range_pct < min_volatility:
        return trades

    mid = (high_m + low_m) / 2.0
    scalp_num = 0
    last_exit = measure_seconds

    # Track which side of mid we're on
    above_mid = float(p[measure_seconds - 1]) > mid

    for idx in range(measure_seconds, WINDOW_SECONDS - 30):
        if scalp_num >= max_scalps:
            break
        if idx < last_exit + cooldown:
            continue

        current = float(p[idx])
        now_above = current > mid

        # Crossed from above to below -> buy UP (expect return to mid)
        if above_mid and not now_above:
            side = "UP"
            entry_reason = "cross_below_mid"
            _, ask = side_arrays(w, side.lower())
            entry_price = float(ask[idx])

            bid, ask_arr = side_arrays(w, side.lower())
            exit_idx, exit_price, exit_type = _manage_trade(w, bid, ask_arr, idx + 1, entry_price, scalp_target, stop_loss, side)
            pnl = _calc_pnl(entry_price, exit_price, stake_qty)

            scalp_num += 1
            trades.append(ScalpTrade(w.start, scalp_num, side, idx, entry_price, exit_idx, exit_price, exit_type, pnl, exit_idx - idx, entry_reason))
            last_exit = exit_idx

        # Crossed from below to above -> buy DOWN (expect return to mid)
        elif not above_mid and now_above:
            side = "DOWN"
            entry_reason = "cross_above_mid"
            _, ask = side_arrays(w, side.lower())
            entry_price = float(ask[idx])

            bid, ask_arr = side_arrays(w, side.lower())
            exit_idx, exit_price, exit_type = _manage_trade(w, bid, ask_arr, idx + 1, entry_price, scalp_target, stop_loss, side)
            pnl = _calc_pnl(entry_price, exit_price, stake_qty)

            scalp_num += 1
            trades.append(ScalpTrade(w.start, scalp_num, side, idx, entry_price, exit_idx, exit_price, exit_type, pnl, exit_idx - idx, entry_reason))
            last_exit = exit_idx

        above_mid = now_above

    return trades


# ---------------------------------------------------------------------------
# Strategy Mode 3: BREAKOUT_FADE
# Fade false breakouts of opening range
# ---------------------------------------------------------------------------

def strat_breakout_fade(
    w: Window,
    scalp_target: float = 0.008,
    stop_loss: float = 0.005,
    min_volatility: float = 0.0003,
    measure_seconds: int = 120,
    breakout_extend: float = 0.0002,
    stake_qty: float = 1000.0,
) -> list[ScalpTrade]:
    """Fade breakouts that don't hold - price breaks range then falls back in."""
    p = w.prices
    trades: list[ScalpTrade] = []

    open_price = p[0]
    high_m = float(p[:measure_seconds].max())
    low_m = float(p[:measure_seconds].min())
    range_pct = (high_m - low_m) / open_price

    if range_pct < min_volatility:
        return trades

    breakout_high = high_m * (1 + breakout_extend)
    breakout_low = low_m * (1 - breakout_extend)

    # Track if we've seen a breakout
    broke_out_up = False
    broke_out_down = False
    breakout_idx = None

    for idx in range(measure_seconds, WINDOW_SECONDS - 30):
        current = float(p[idx])

        # Detect breakout
        if not broke_out_up and not broke_out_down:
            if current > breakout_high:
                broke_out_up = True
                breakout_idx = idx
            elif current < breakout_low:
                broke_out_down = True
                breakout_idx = idx
            continue

        # Fade: if broke out up, wait for price to fall back below high
        if broke_out_up and current < high_m:
            side = "DOWN"
            entry_reason = "fade_false_breakout_up"
            _, ask = side_arrays(w, side.lower())
            entry_price = float(ask[idx])

            bid, ask_arr = side_arrays(w, side.lower())
            exit_idx, exit_price, exit_type = _manage_trade(w, bid, ask_arr, idx + 1, entry_price, scalp_target, stop_loss, side)
            pnl = _calc_pnl(entry_price, exit_price, stake_qty)

            trades.append(ScalpTrade(w.start, 1, side, idx, entry_price, exit_idx, exit_price, exit_type, pnl, exit_idx - idx, entry_reason))
            break

        # Fade: if broke out down, wait for price to rise back above low
        if broke_out_down and current > low_m:
            side = "UP"
            entry_reason = "fade_false_breakout_down"
            _, ask = side_arrays(w, side.lower())
            entry_price = float(ask[idx])

            bid, ask_arr = side_arrays(w, side.lower())
            exit_idx, exit_price, exit_type = _manage_trade(w, bid, ask_arr, idx + 1, entry_price, scalp_target, stop_loss, side)
            pnl = _calc_pnl(entry_price, exit_price, stake_qty)

            trades.append(ScalpTrade(w.start, 1, side, idx, entry_price, exit_idx, exit_price, exit_type, pnl, exit_idx - idx, entry_reason))
            break

    return trades


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _manage_trade(
    w: Window,
    bid_array: np.ndarray,
    ask_array: np.ndarray,
    start_idx: int,
    entry_price: float,
    scalp_target: float,
    stop_loss: float,
    side: str,
) -> tuple[int, float, str]:
    """Manage a trade with target and stop based on PnL from entry.

    We buy at entry_price (ask). We can sell at bid.
    Target: bid has moved enough that (bid - entry_price) >= scalp_target
    Stop: bid has moved against us such that (bid - entry_price) <= -stop_loss
    """
    for idx in range(start_idx, WINDOW_SECONDS - 5):
        current_bid = float(bid_array[idx])
        pnl_per_share = current_bid - entry_price

        if pnl_per_share >= scalp_target:
            return idx, current_bid, "target"
        if pnl_per_share <= -stop_loss:
            return idx, current_bid, "stop"

    # Hold to settlement
    final_idx = WINDOW_SECONDS - 1
    final_price = float(bid_array[final_idx])
    return final_idx, final_price, "settlement"


def _calc_pnl(entry_price: float, exit_price: float, stake_qty: float) -> float:
    """Compute PnL for a trade."""
    shares = stake_qty / entry_price
    entry_fee = entry_price * shares * TAKER_FEE_RATE
    exit_fee = exit_price * shares * TAKER_FEE_RATE
    return (exit_price - entry_price) * shares - entry_fee - exit_fee


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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
    out["avg_scalps_per_window"] = out["avg_scalps_per_window"].map(lambda x: f"{x:.2f}")

    cols = [
        "sample", "strategy", "total_pnl", "total_trades", "windows_traded",
        "win_rate", "profit_factor", "avg_pnl", "sharpe", "max_drawdown",
        "avg_hold_seconds", "target_hits", "stop_hits", "settlement_exits",
        "avg_scalps_per_window",
    ]
    print(out[cols].to_string(index=False))


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

STRATEGY_FNS = {
    "reversal": strat_reversal,
    "chop": strat_chop,
    "breakout_fade": strat_breakout_fade,
}


def run_strategy_on_windows(
    windows: list[Window],
    mode: str,
    scalp_target: float,
    stop_loss: float,
    min_volatility: float,
    measure_seconds: int,
    stake_qty: float,
    max_scalps: int = 3,
    cooldown: int = 20,
    breakout_extend: float = 0.0002,
) -> tuple[list[ScalpTrade], np.ndarray]:
    """Run a strategy on a set of windows."""
    all_trades = []
    pnl_by_window = np.zeros(len(windows), dtype=float)

    for i, w in enumerate(windows):
        fn = STRATEGY_FNS[mode]

        if mode == "chop":
            trades = fn(
                w, scalp_target=scalp_target, stop_loss=stop_loss,
                min_volatility=min_volatility, measure_seconds=measure_seconds,
                max_scalps=max_scalps, cooldown=cooldown, stake_qty=stake_qty,
            )
        elif mode == "breakout_fade":
            trades = fn(
                w, scalp_target=scalp_target, stop_loss=stop_loss,
                min_volatility=min_volatility, measure_seconds=measure_seconds,
                breakout_extend=breakout_extend, stake_qty=stake_qty,
            )
        else:
            trades = fn(
                w, scalp_target=scalp_target, stop_loss=stop_loss,
                min_volatility=min_volatility, measure_seconds=measure_seconds,
                stake_qty=stake_qty,
            )

        pnl_by_window[i] = sum(t.pnl for t in trades)
        all_trades.extend(trades)

    return all_trades, pnl_by_window


def param_sweep(
    windows: list[Window],
    sample: str,
    stake_qty: float,
) -> pd.DataFrame:
    """Sweep key parameters across all modes."""
    results = []

    for mode in STRATEGY_FNS:
        targets = [0.004, 0.006, 0.008, 0.010, 0.012]
        stops = [0.003, 0.004, 0.005, 0.006]
        min_vols = [0.0003, 0.0004, 0.0005, 0.0006]

        for target in targets:
            for stop in stops:
                for min_vol in min_vols:
                    all_trades, pnl_by_window = run_strategy_on_windows(
                        windows, mode, target, stop, min_vol, 120, stake_qty,
                    )

                    if not all_trades:
                        continue

                    metrics = compute_scalp_metrics(
                        all_trades, windows, pnl_by_window, sample,
                        f"{mode}_t={target:.3f}_s={stop:.3f}_v={min_vol:.4f}"
                    )
                    results.append({
                        "mode": mode,
                        "target": target,
                        "stop": stop,
                        "min_vol": min_vol,
                        "total_pnl": metrics.total_pnl,
                        "trades": metrics.total_trades,
                        "win_rate": metrics.win_rate,
                        "profit_factor": metrics.profit_factor if not np.isinf(metrics.profit_factor) else 999,
                        "sharpe": metrics.sharpe,
                        "max_drawdown": metrics.max_drawdown,
                        "settlement_pct": metrics.settlement_exits / max(metrics.total_trades, 1),
                    })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Opening Chop Scalper V2 Backtest")
    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument("--source", default="binance")
    parser.add_argument("--dataset-source", default="aliplayer_spot")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--stake-usd", type=float, default=1000.0)
    parser.add_argument("--mode", default="chop", choices=["reversal", "chop", "breakout_fade"])
    parser.add_argument("--scalp-target", type=float, default=0.006, help="Target profit in contract price")
    parser.add_argument("--stop-loss", type=float, default=0.004, help="Stop loss in contract price")
    parser.add_argument("--min-volatility", type=float, default=0.0004, help="Min BTC price range pct to trade")
    parser.add_argument("--measure-seconds", type=int, default=120)
    parser.add_argument("--max-scalps", type=int, default=3)
    parser.add_argument("--cooldown", type=int, default=20)
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    args = parser.parse_args()

    price_df = load_spot_prices(args.symbol, None if args.source.lower() in {"none", "all"} else args.source, dataset_source=args.dataset_source)
    windows = build_windows(price_df, args.max_windows)

    if args.sweep:
        print("\n" + "=" * 80)
        print("PARAMETER SWEEP - ALL MODES")
        print("=" * 80)

        train_windows, test_windows = split_windows_chronologically(windows, args.train_frac)

        print("\nTRAIN sweep...")
        train_results = param_sweep(train_windows, "TRAIN", args.stake_usd)
        if not train_results.empty:
            train_results = train_results.sort_values("total_pnl", ascending=False)
            print(f"\nTop 15 configs (by TRAIN PnL):")
            print(train_results.head(15).to_string(index=False))
            train_results.to_csv("chop_scalper_v2_sweep_train.csv", index=False)

            # Test top configs
            print("\nTEST sweep (top 10 train configs)...")
            top_configs = train_results.head(10)
            test_results = []
            for _, row in top_configs.iterrows():
                all_trades, pnl_by_window = run_strategy_on_windows(
                    test_windows, row["mode"], row["target"], row["stop"],
                    row["min_vol"], 120, args.stake_usd,
                )
                if all_trades:
                    metrics = compute_scalp_metrics(all_trades, test_windows, pnl_by_window, "TEST",
                        f"{row['mode']}_t={row['target']:.3f}_s={row['stop']:.3f}_v={row['min_vol']:.4f}")
                    test_results.append({
                        "mode": row["mode"],
                        "target": row["target"],
                        "stop": row["stop"],
                        "min_vol": row["min_vol"],
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
                test_df.to_csv("chop_scalper_v2_sweep_test.csv", index=False)

        return

    # --- Single config backtest ---
    print("\n" + "=" * 80)
    print(f"OPENING CHOP SCALPER V2 - Mode: {args.mode.upper()}")
    print("=" * 80)
    print(f"Scalp target: {args.scalp_target:.3f}")
    print(f"Stop loss: {args.stop_loss:.3f}")
    print(f"Min volatility: {args.min_volatility:.4f}")
    print(f"Measure period: {args.measure_seconds}s")
    print(f"Stake: ${args.stake_usd:,.0f}")
    print()

    train_windows, test_windows = split_windows_chronologically(windows, args.train_frac)
    all_metrics = []

    for sample, sample_windows in [("TRAIN", train_windows), ("TEST", test_windows)]:
        print(f"\n--- {sample} SET ({len(sample_windows)} windows) ---")

        all_trades, pnl_by_window = run_strategy_on_windows(
            sample_windows, args.mode, args.scalp_target, args.stop_loss,
            args.min_volatility, args.measure_seconds, args.stake_usd,
            max_scalps=args.max_scalps, cooldown=args.cooldown,
        )

        metrics = compute_scalp_metrics(all_trades, sample_windows, pnl_by_window, sample, f"chop_v2_{args.mode}")
        all_metrics.append(metrics)

        print(f"  Trades: {metrics.total_trades}")
        print(f"  Windows traded: {metrics.windows_traded}/{metrics.windows_total}")
        print(f"  Total PnL: ${metrics.total_pnl:,.2f}")
        print(f"  Win rate: {100*metrics.win_rate:.1f}%")
        pf_str = f"{metrics.profit_factor:.2f}" if not np.isinf(metrics.profit_factor) else "inf"
        print(f"  Profit factor: {pf_str}")
        print(f"  Sharpe: {metrics.sharpe:.2f}")
        print(f"  Max DD: {100*metrics.max_drawdown:.1f}%")
        print(f"  Avg hold: {metrics.avg_hold_seconds:.1f}s")
        print(f"  Target hits: {metrics.target_hits}, Stops: {metrics.stop_hits}, Settlement: {metrics.settlement_exits}")

        if all_trades:
            trades_df = pd.DataFrame([t.__dict__ for t in all_trades])
            trades_path = f"chop_v2_trades_{args.mode}_{sample.lower()}.csv"
            trades_df.to_csv(trades_path, index=False)
            print(f"  Saved {trades_path}")

    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)
    print_metrics_table(all_metrics)

    metrics_df = pd.DataFrame([m.__dict__ for m in all_metrics])
    metrics_df.to_csv("chop_scalper_v2_metrics.csv", index=False)
    print(f"\nSaved chop_scalper_v2_metrics.csv")


if __name__ == "__main__":
    main()
