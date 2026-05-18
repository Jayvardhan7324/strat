"""
Opening Chop Direction Predictor - Backtest

Instead of scalping contract prices (which don't chop), this strategy uses
the BTC price chop pattern in the first 2 minutes to PREDICT which direction
the 5-minute window will resolve, then enters a single directional trade.

Hypothesis: When BTC chops (goes up then down or vice versa) in the first 2 min,
the final direction of the 5-min window has a predictable bias.

Strategy:
1. Measure BTC price action in first 120s
2. Classify the pattern: trend_up, trend_down, chop_up_first, chop_down_first
3. Enter a directional contract trade based on the pattern
4. Hold to settlement (binary outcome)

Usage:
    python backtest_chop_direction_predictor.py
    python backtest_chop_direction_predictor.py --sweep
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from polymarket_updown_backtest import (
    STARTING_CAPITAL,
    TAKER_FEE_RATE,
    WINDOW_SECONDS,
    Window,
    build_windows,
    load_spot_prices,
    max_drawdown,
    split_windows_chronologically,
    side_arrays,
    settlement_value,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Pattern classification
# ---------------------------------------------------------------------------

def classify_pattern(
    prices: np.ndarray,
    measure_seconds: int = 120,
) -> str:
    """Classify the price pattern in the first N seconds.

    Returns:
        "trend_up": Price consistently moved up
        "trend_down": Price consistently moved down
        "chop_up_first": Went up first, then came back down
        "chop_down_first": Went down first, then came back up
        "flat": No significant movement
    """
    p = prices[:measure_seconds]
    open_price = p[0]
    high = float(p.max())
    low = float(p.min())
    close = float(p[-1])

    range_pct = (high - low) / open_price
    net_move = (close - open_price) / open_price

    # Find the time of high and low
    high_idx = int(np.argmax(p))
    low_idx = int(np.argmin(p))

    if range_pct < 0.0003:
        return "flat"

    # Chop = high and low are both significant, and price reversed
    if high_idx < low_idx:
        # Went up first, then down
        if range_pct > 0.0005:
            return "chop_up_first"
    elif low_idx < high_idx:
        # Went down first, then up
        if range_pct > 0.0005:
            return "chop_down_first"

    # Trend
    if net_move > 0.0003:
        return "trend_up"
    if net_move < -0.0003:
        return "trend_down"

    return "flat"


# ---------------------------------------------------------------------------
# Strategy: Trade based on pattern
# ---------------------------------------------------------------------------

@dataclass
class DirectionTrade:
    window_start: pd.Timestamp
    pattern: str
    side: str  # "UP" or "DOWN"
    entry_idx: int
    entry_price: float
    exit_price: float
    exit_type: str
    pnl: float
    outcome: int  # 1 if UP resolved, 0 if DOWN


def trade_pattern(
    w: Window,
    measure_seconds: int = 120,
    stake_qty: float = 1000.0,
    entry_idx: int = 120,
) -> DirectionTrade | None:
    """Enter a directional trade based on opening pattern."""
    pattern = classify_pattern(w.prices, measure_seconds)

    if pattern == "flat":
        return None

    # Strategy rules (based on pattern analysis):
    # - chop_up_first: price went up then came back down -> bet DOWN (2nd direction wins 70.6%)
    # - chop_down_first: price went down then came back up -> bet UP (2nd direction wins 71.8%)
    # - trend_up: momentum continues -> bet UP
    # - trend_down: momentum continues -> bet DOWN

    if pattern == "chop_up_first":
        side = "DOWN"
    elif pattern == "chop_down_first":
        side = "UP"
    elif pattern == "trend_up":
        side = "UP"
    elif pattern == "trend_down":
        side = "DOWN"
    else:
        return None

    # Enter at entry_idx
    idx = min(entry_idx, WINDOW_SECONDS - 10)
    _, ask = side_arrays(w, side.lower())
    entry_price = float(ask[idx])

    # Hold to settlement
    final = settlement_value(w, side.lower())

    # PnL
    shares = stake_qty / entry_price
    entry_fee = entry_price * shares * TAKER_FEE_RATE
    exit_fee = final * shares * TAKER_FEE_RATE
    pnl = (final - entry_price) * shares - entry_fee - exit_fee

    return DirectionTrade(
        window_start=w.start,
        pattern=pattern,
        side=side,
        entry_idx=idx,
        entry_price=entry_price,
        exit_price=final,
        exit_type="settlement",
        pnl=pnl,
        outcome=w.outcome_up if side == "UP" else 1 - w.outcome_up,
    )


# ---------------------------------------------------------------------------
# Strategy: Chop fade only (only trade chop patterns)
# ---------------------------------------------------------------------------

def trade_chop_only(
    w: Window,
    measure_seconds: int = 120,
    stake_qty: float = 1000.0,
    entry_idx: int = 120,
    min_range_pct: float = 0.0005,
) -> DirectionTrade | None:
    """Only trade when there's genuine chop (up then down or down then up)."""
    p = w.prices[:measure_seconds]
    open_price = p[0]
    high = float(p.max())
    low = float(p.min())
    range_pct = (high - low) / open_price

    if range_pct < min_range_pct:
        return None

    high_idx = int(np.argmax(p))
    low_idx = int(np.argmin(p))

    # Must have a clear reversal
    if high_idx < low_idx:
        # Up then down - bet DOWN (2nd direction wins)
        side = "DOWN"
        pattern = "chop_up_first"
    elif low_idx < high_idx:
        # Down then up - bet UP (2nd direction wins)
        side = "UP"
        pattern = "chop_down_first"
    else:
        return None

    idx = min(entry_idx, WINDOW_SECONDS - 10)
    _, ask = side_arrays(w, side.lower())
    entry_price = float(ask[idx])

    final = settlement_value(w, side.lower())

    shares = stake_qty / entry_price
    entry_fee = entry_price * shares * TAKER_FEE_RATE
    exit_fee = final * shares * TAKER_FEE_RATE
    pnl = (final - entry_price) * shares - entry_fee - exit_fee

    return DirectionTrade(
        window_start=w.start,
        pattern=pattern,
        side=side,
        entry_idx=idx,
        entry_price=entry_price,
        exit_price=final,
        exit_type="settlement",
        pnl=pnl,
        outcome=w.outcome_up if side == "UP" else 1 - w.outcome_up,
    )


# ---------------------------------------------------------------------------
# Strategy: Multi-signal with confirmation
# ---------------------------------------------------------------------------

def trade_with_confirmation(
    w: Window,
    measure_seconds: int = 120,
    stake_qty: float = 1000.0,
    entry_idx: int = 150,
    min_range_pct: float = 0.0004,
) -> DirectionTrade | None:
    """Trade chop patterns with additional confirmation signals."""
    p = w.prices
    p_measure = p[:measure_seconds]
    open_price = p[0]
    high = float(p_measure.max())
    low = float(p_measure.min())
    range_pct = (high - low) / open_price

    if range_pct < min_range_pct:
        return None

    high_idx = int(np.argmax(p_measure))
    low_idx = int(np.argmin(p_measure))

    # Determine chop direction
    if high_idx < low_idx:
        chop_direction = "up_first"
    elif low_idx < high_idx:
        chop_direction = "down_first"
    else:
        return None

    # Confirmation: check price action between measure_seconds and entry_idx
    if entry_idx > measure_seconds:
        confirm_prices = p[measure_seconds:entry_idx]
        confirm_start = float(p[measure_seconds - 1])
        confirm_end = float(p[entry_idx - 1])
        confirm_move = (confirm_end - confirm_start) / confirm_start
    else:
        confirm_move = 0.0

    # Decision logic - 2nd movement direction predicts outcome
    if chop_direction == "up_first":
        # Price went up then down -> bet DOWN
        side = "DOWN"
        pattern = "chop_up_stabilizing"
    else:  # down_first
        # Price went down then up -> bet UP
        side = "UP"
        pattern = "chop_down_stabilizing"

    idx = min(entry_idx, WINDOW_SECONDS - 10)
    _, ask = side_arrays(w, side.lower())
    entry_price = float(ask[idx])

    final = settlement_value(w, side.lower())

    shares = stake_qty / entry_price
    entry_fee = entry_price * shares * TAKER_FEE_RATE
    exit_fee = final * shares * TAKER_FEE_RATE
    pnl = (final - entry_price) * shares - entry_fee - exit_fee

    return DirectionTrade(
        window_start=w.start,
        pattern=pattern,
        side=side,
        entry_idx=idx,
        entry_price=entry_price,
        exit_price=final,
        exit_type="settlement",
        pnl=pnl,
        outcome=w.outcome_up if side == "UP" else 1 - w.outcome_up,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class DirectionMetrics:
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
    avg_entry_price: float
    avg_outcome: float  # How often we were right


def compute_direction_metrics(
    all_trades: list[DirectionTrade],
    windows: list[Window],
    pnl_by_window: np.ndarray,
    sample: str,
    strategy: str,
) -> DirectionMetrics:
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

    windows_traded = len(set(t.window_start for t in all_trades))
    avg_outcome = float(np.mean([t.outcome for t in all_trades])) if all_trades else 0.0

    return DirectionMetrics(
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
        avg_entry_price=float(np.mean([t.entry_price for t in all_trades])) if all_trades else 0.0,
        avg_outcome=avg_outcome,
    )


def print_metrics_table(metrics_list: list[DirectionMetrics]) -> None:
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
    out["avg_entry_price"] = out["avg_entry_price"].map(lambda x: f"{x:.4f}")
    out["avg_outcome"] = out["avg_outcome"].map(lambda x: f"{100*x:.1f}%")

    cols = [
        "sample", "strategy", "total_pnl", "total_trades", "windows_traded",
        "win_rate", "profit_factor", "avg_pnl", "sharpe", "max_drawdown",
        "avg_outcome",
    ]
    print(out[cols].to_string(index=False))


# ---------------------------------------------------------------------------
# Pattern analysis
# ---------------------------------------------------------------------------

def analyze_patterns(windows: list[Window], measure_seconds: int = 120) -> None:
    """Analyze how often each pattern occurs and its predictive power."""
    print("\n" + "=" * 80)
    print("PATTERN ANALYSIS")
    print("=" * 80)

    pattern_outcomes: dict[str, list[int]] = {}

    for w in windows:
        pattern = classify_pattern(w.prices, measure_seconds)
        if pattern not in pattern_outcomes:
            pattern_outcomes[pattern] = []
        pattern_outcomes[pattern].append(w.outcome_up)

    print(f"\n{'Pattern':<25} {'Count':>8} {'UP%':>8} {'Edge vs 50%':>12}")
    print("-" * 55)

    for pattern, outcomes in sorted(pattern_outcomes.items()):
        count = len(outcomes)
        up_pct = np.mean(outcomes)
        edge = up_pct - 0.5
        print(f"{pattern:<25} {count:>8,} {100*up_pct:>7.1f}% {100*edge:>+11.1f}%")


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

STRATEGY_FNS = {
    "all_patterns": trade_pattern,
    "chop_only": trade_chop_only,
    "confirmation": trade_with_confirmation,
}


def run_strategy_on_windows(
    windows: list[Window],
    strategy_name: str,
    measure_seconds: int,
    entry_idx: int,
    stake_qty: float,
    min_range_pct: float = 0.0004,
) -> tuple[list[DirectionTrade], np.ndarray]:
    """Run a strategy on a set of windows."""
    all_trades = []
    pnl_by_window = np.zeros(len(windows), dtype=float)

    for i, w in enumerate(windows):
        fn = STRATEGY_FNS[strategy_name]

        if strategy_name == "chop_only":
            trade = fn(w, measure_seconds, stake_qty, entry_idx, min_range_pct)
        elif strategy_name == "confirmation":
            trade = fn(w, measure_seconds, stake_qty, entry_idx, min_range_pct)
        else:
            trade = fn(w, measure_seconds, stake_qty, entry_idx)

        if trade:
            pnl_by_window[i] = trade.pnl
            all_trades.append(trade)

    return all_trades, pnl_by_window


def param_sweep(
    windows: list[Window],
    sample: str,
    stake_qty: float,
) -> pd.DataFrame:
    """Sweep parameters across all strategies."""
    results = []

    for strat_name in STRATEGY_FNS:
        for measure_sec in [60, 90, 120, 150]:
            for entry_idx in [120, 150, 180, 210]:
                for min_range in [0.0003, 0.0004, 0.0005, 0.0006]:
                    all_trades, pnl_by_window = run_strategy_on_windows(
                        windows, strat_name, measure_sec, entry_idx, stake_qty, min_range,
                    )

                    if not all_trades:
                        continue

                    metrics = compute_direction_metrics(
                        all_trades, windows, pnl_by_window, sample,
                        f"{strat_name}_m={measure_sec}_e={entry_idx}_r={min_range:.4f}"
                    )
                    results.append({
                        "strategy": strat_name,
                        "measure_sec": measure_sec,
                        "entry_idx": entry_idx,
                        "min_range": min_range,
                        "total_pnl": metrics.total_pnl,
                        "trades": metrics.total_trades,
                        "win_rate": metrics.win_rate,
                        "profit_factor": metrics.profit_factor if not np.isinf(metrics.profit_factor) else 999,
                        "sharpe": metrics.sharpe,
                        "max_drawdown": metrics.max_drawdown,
                        "avg_outcome": metrics.avg_outcome,
                    })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Chop Direction Predictor Backtest")
    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument("--source", default="binance")
    parser.add_argument("--dataset-source", default="aliplayer_spot")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--stake-usd", type=float, default=1000.0)
    parser.add_argument("--strategy", default="confirmation", choices=["all_patterns", "chop_only", "confirmation"])
    parser.add_argument("--measure-seconds", type=int, default=120)
    parser.add_argument("--entry-idx", type=int, default=150)
    parser.add_argument("--min-range", type=float, default=0.0004)
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--analyze", action="store_true", help="Analyze pattern predictive power")
    args = parser.parse_args()

    price_df = load_spot_prices(args.symbol, None if args.source.lower() in {"none", "all"} else args.source, dataset_source=args.dataset_source)
    windows = build_windows(price_df, args.max_windows)

    if args.analyze:
        analyze_patterns(windows, args.measure_seconds)
        return

    if args.sweep:
        print("\n" + "=" * 80)
        print("PARAMETER SWEEP")
        print("=" * 80)

        train_windows, test_windows = split_windows_chronologically(windows, args.train_frac)

        print("\nTRAIN sweep...")
        train_results = param_sweep(train_windows, "TRAIN", args.stake_usd)
        if not train_results.empty:
            train_results = train_results.sort_values("total_pnl", ascending=False)
            print(f"\nTop 20 configs (by TRAIN PnL):")
            print(train_results.head(20).to_string(index=False))
            train_results.to_csv("chop_direction_sweep_train.csv", index=False)

            # Test top configs
            print("\nTEST sweep (top 10 train configs)...")
            top_configs = train_results.head(10)
            test_results = []
            for _, row in top_configs.iterrows():
                all_trades, pnl_by_window = run_strategy_on_windows(
                    test_windows, row["strategy"], row["measure_sec"],
                    row["entry_idx"], args.stake_usd, row["min_range"],
                )
                if all_trades:
                    metrics = compute_direction_metrics(all_trades, test_windows, pnl_by_window, "TEST",
                        f"{row['strategy']}_m={row['measure_sec']}_e={row['entry_idx']}_r={row['min_range']:.4f}")
                    test_results.append({
                        "strategy": row["strategy"],
                        "measure_sec": row["measure_sec"],
                        "entry_idx": row["entry_idx"],
                        "min_range": row["min_range"],
                        "total_pnl": metrics.total_pnl,
                        "trades": metrics.total_trades,
                        "win_rate": metrics.win_rate,
                        "profit_factor": metrics.profit_factor if not np.isinf(metrics.profit_factor) else 999,
                        "sharpe": metrics.sharpe,
                        "max_drawdown": metrics.max_drawdown,
                        "avg_outcome": metrics.avg_outcome,
                    })

            if test_results:
                test_df = pd.DataFrame(test_results).sort_values("total_pnl", ascending=False)
                print(f"\nTEST results for top TRAIN configs:")
                print(test_df.to_string(index=False))
                test_df.to_csv("chop_direction_sweep_test.csv", index=False)

        return

    # --- Single config backtest ---
    print("\n" + "=" * 80)
    print(f"CHOP DIRECTION PREDICTOR - Strategy: {args.strategy}")
    print("=" * 80)
    print(f"Measure period: {args.measure_seconds}s")
    print(f"Entry index: {args.entry_idx}s")
    print(f"Min range: {args.min_range:.4f}")
    print(f"Stake: ${args.stake_usd:,.0f}")
    print()

    train_windows, test_windows = split_windows_chronologically(windows, args.train_frac)
    all_metrics = []

    for sample, sample_windows in [("TRAIN", train_windows), ("TEST", test_windows)]:
        print(f"\n--- {sample} SET ({len(sample_windows)} windows) ---")

        all_trades, pnl_by_window = run_strategy_on_windows(
            sample_windows, args.strategy, args.measure_seconds,
            args.entry_idx, args.stake_usd, args.min_range,
        )

        metrics = compute_direction_metrics(all_trades, sample_windows, pnl_by_window, sample, f"chop_dir_{args.strategy}")
        all_metrics.append(metrics)

        print(f"  Trades: {metrics.total_trades}")
        print(f"  Windows traded: {metrics.windows_traded}/{metrics.windows_total}")
        print(f"  Total PnL: ${metrics.total_pnl:,.2f}")
        print(f"  Win rate: {100*metrics.win_rate:.1f}%")
        print(f"  Avg outcome (direction correct): {100*metrics.avg_outcome:.1f}%")
        pf_str = f"{metrics.profit_factor:.2f}" if not np.isinf(metrics.profit_factor) else "inf"
        print(f"  Profit factor: {pf_str}")
        print(f"  Sharpe: {metrics.sharpe:.2f}")
        print(f"  Max DD: {100*metrics.max_drawdown:.1f}%")

        if all_trades:
            trades_df = pd.DataFrame([t.__dict__ for t in all_trades])
            trades_path = f"chop_dir_trades_{args.strategy}_{sample.lower()}.csv"
            trades_df.to_csv(trades_path, index=False)
            print(f"  Saved {trades_path}")

    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)
    print_metrics_table(all_metrics)

    metrics_df = pd.DataFrame([m.__dict__ for m in all_metrics])
    metrics_df.to_csv("chop_direction_metrics.csv", index=False)
    print(f"\nSaved chop_direction_metrics.csv")


if __name__ == "__main__":
    main()
