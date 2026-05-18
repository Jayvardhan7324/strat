"""
Live-guarded Polymarket strategy backtest.

This is intentionally harsher than `polymarket_updown_backtest.py`.
It tests the live-style version of the strategy stack:

- smaller fixed stakes
- max entry price caps
- minimum profit-if-win
- max "one loss wipes out N wins"
- one portfolio trade per market
- optional ask slippage stress

It still uses the synthetic contract prices from the spot-only dataset, so it
cannot prove a live Polymarket edge. The point is to reject strategies that
only worked because the first backtest allowed expensive, asymmetric entries.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from polymarket_updown_backtest import (
    STARTING_CAPITAL,
    Window,
    build_windows,
    load_spot_prices,
    max_drawdown,
    split_windows_chronologically,
)


@dataclass(frozen=True)
class Guard:
    stake_usd: float
    max_entry: float
    max_one_loss_wipes_out_wins: float
    min_profit_if_win_usd: float


@dataclass(frozen=True)
class Entry:
    name: str
    side: str
    idx: int
    price: float
    stake_usd: float
    shares: float
    profit_if_win: float
    loss_if_wrong: float
    one_loss_wipes_out_wins: float


GUARDS: dict[str, Guard] = {
    "LIVE_ORB": Guard(10, 0.72, 2.60, 3.00),
    "LIVE_VWM": Guard(10, 0.70, 2.35, 3.00),
    "LIVE_RENKO": Guard(10, 0.70, 2.35, 3.00),
    "CONSENSUS_2_OF_3": Guard(10, 0.70, 2.35, 3.00),
    "CHEAP_LEADER_PULLBACK": Guard(10, 0.60, 1.50, 5.00),
    "OPEN_FLIP_VALUE": Guard(10, 0.58, 1.38, 6.00),
}


def side_ask(w: Window, side: str, idx: int, slippage: float) -> float:
    ask = w.up_ask[idx] if side == "UP" else w.down_ask[idx]
    return float(min(0.999, ask + slippage))


def winner(w: Window) -> str:
    return "UP" if w.outcome_up else "DOWN"


def settlement_pnl(w: Window, entry: Entry) -> float:
    if entry.side == winner(w):
        return entry.profit_if_win
    return -entry.loss_if_wrong


def guarded_entry(
    w: Window,
    name: str,
    side: str,
    idx: int,
    slippage: float,
) -> Entry | None:
    guard = GUARDS[name]
    price = side_ask(w, side, idx, slippage)
    if price > guard.max_entry:
        return None
    shares = guard.stake_usd / price
    profit_if_win = shares - guard.stake_usd
    loss_if_wrong = guard.stake_usd
    one_loss = loss_if_wrong / profit_if_win if profit_if_win > 0 else float("inf")
    if profit_if_win < guard.min_profit_if_win_usd:
        return None
    if one_loss > guard.max_one_loss_wipes_out_wins:
        return None
    return Entry(
        name=name,
        side=side,
        idx=idx,
        price=price,
        stake_usd=guard.stake_usd,
        shares=shares,
        profit_if_win=profit_if_win,
        loss_if_wrong=loss_if_wrong,
        one_loss_wipes_out_wins=one_loss,
    )


def opening_range_signal(w: Window) -> tuple[str, int] | None:
    high = float(w.prices[:30].max())
    low = float(w.prices[:30].min())
    upper = high * 1.00005
    lower = low * 0.99995
    for idx in range(120, 296):
        if w.prices[idx] > upper and w.prices[idx - 1] > upper:
            return "UP", idx
        if w.prices[idx] < lower and w.prices[idx - 1] < lower:
            return "DOWN", idx
    return None


def volume_weighted_momentum_signal(w: Window) -> tuple[str, int] | None:
    idx = 120
    diffs = np.diff(w.prices[: idx + 1])
    up_vol = float(np.clip(diffs, 0, None).sum())
    down_vol = float((-np.clip(diffs, None, 0)).sum())
    total = up_vol + down_vol
    if total <= 0:
        return None
    ratio = (up_vol - down_vol) / total
    sma = float(w.prices[idx - 29 : idx + 1].mean())
    if ratio > 0.20 and w.prices[idx] > sma:
        return "UP", idx
    if ratio < -0.20 and w.prices[idx] < sma:
        return "DOWN", idx
    return None


def renko_signal(w: Window) -> tuple[str, int] | None:
    idx = 180
    brick = w.open_price * 0.0005
    ref = w.open_price
    up_bricks = 0
    down_bricks = 0
    for price in w.prices[: idx + 1]:
        while price >= ref + brick:
            up_bricks += 1
            ref += brick
        while price <= ref - brick:
            down_bricks += 1
            ref -= brick
    if up_bricks - down_bricks >= 2:
        return "UP", idx
    if down_bricks - up_bricks >= 2:
        return "DOWN", idx
    return None


def cheap_leader_pullback_signal(w: Window) -> tuple[str, int] | None:
    # Buy the current leader only if the contract is still cheap enough.
    # This is the opposite of the failed live behavior where we bought near 0.90+.
    for idx in range(150, 270):
        delta = w.prices[idx] / w.open_price - 1.0
        last10 = w.prices[idx] / w.prices[max(0, idx - 10)] - 1.0
        if delta > 0.00035 and last10 > -0.00015:
            return "UP", idx
        if delta < -0.00035 and last10 < 0.00015:
            return "DOWN", idx
    return None


def open_flip_value_signal(w: Window) -> tuple[str, int] | None:
    # If price crossed both sides of the open, only buy the side that re-breaks
    # with a still-cheap contract. This is a value filter, not a pure fade.
    crossed_up = False
    crossed_down = False
    for idx in range(60, 240):
        crossed_up = crossed_up or w.prices[idx] > w.open_price * 1.0004
        crossed_down = crossed_down or w.prices[idx] < w.open_price * 0.9996
        if not (crossed_up and crossed_down):
            continue
        if w.prices[idx] > w.open_price * 1.0002 and w.prices[idx] > w.prices[idx - 5]:
            return "UP", idx
        if w.prices[idx] < w.open_price * 0.9998 and w.prices[idx] < w.prices[idx - 5]:
            return "DOWN", idx
    return None


SIGNALS = {
    "LIVE_ORB": opening_range_signal,
    "LIVE_VWM": volume_weighted_momentum_signal,
    "LIVE_RENKO": renko_signal,
    "CHEAP_LEADER_PULLBACK": cheap_leader_pullback_signal,
    "OPEN_FLIP_VALUE": open_flip_value_signal,
}


def consensus_signal(w: Window) -> tuple[str, int] | None:
    raw = [
        opening_range_signal(w),
        volume_weighted_momentum_signal(w),
        renko_signal(w),
    ]
    votes = [item for item in raw if item is not None]
    up = [idx for side, idx in votes if side == "UP"]
    down = [idx for side, idx in votes if side == "DOWN"]
    if len(up) >= 2:
        return "UP", max(up)
    if len(down) >= 2:
        return "DOWN", max(down)
    return None


def run_strategy(
    windows: list[Window],
    name: str,
    signal_fn,
    slippage: float,
) -> tuple[np.ndarray, list[Entry], list[float]]:
    pnl = np.zeros(len(windows), dtype=float)
    entries: list[Entry] = []
    trade_pnls: list[float] = []
    for i, w in enumerate(windows):
        signal = signal_fn(w)
        if signal is None:
            continue
        side, idx = signal
        entry = guarded_entry(w, name, side, idx, slippage)
        if entry is None:
            continue
        value = settlement_pnl(w, entry)
        pnl[i] = value
        entries.append(entry)
        trade_pnls.append(value)
    return pnl, entries, trade_pnls


def run_priority_portfolio(
    windows: list[Window],
    slippage: float,
) -> tuple[np.ndarray, list[Entry], list[float]]:
    # Mirrors the live config: only one strategy per market. Priority favors the
    # strongest live paper performer first, then selective backups.
    priority = [
        ("LIVE_ORB", opening_range_signal),
        ("LIVE_VWM", volume_weighted_momentum_signal),
        ("LIVE_RENKO", renko_signal),
    ]
    pnl = np.zeros(len(windows), dtype=float)
    entries: list[Entry] = []
    trade_pnls: list[float] = []
    for i, w in enumerate(windows):
        for name, fn in priority:
            signal = fn(w)
            if signal is None:
                continue
            side, idx = signal
            entry = guarded_entry(w, name, side, idx, slippage)
            if entry is None:
                continue
            value = settlement_pnl(w, entry)
            pnl[i] = value
            entries.append(entry)
            trade_pnls.append(value)
            break
    return pnl, entries, trade_pnls


def metrics(sample: str, name: str, pnl: np.ndarray, trade_pnls: list[float], entries: list[Entry]) -> dict:
    equity = STARTING_CAPITAL + np.cumsum(pnl)
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
    raw_sharpe = float(returns.mean() / returns.std(ddof=1)) if len(returns) > 1 and returns.std(ddof=1) > 0 else 0.0
    wins = [x for x in trade_pnls if x > 0]
    losses = [x for x in trade_pnls if x < 0]
    gross_profit = float(sum(wins))
    gross_loss = float(-sum(losses))
    avg_entry = float(np.mean([e.price for e in entries])) if entries else 0.0
    avg_wipe = float(np.mean([e.one_loss_wipes_out_wins for e in entries])) if entries else 0.0
    return {
        "sample": sample,
        "strategy": name,
        "total_pnl": float(equity[-1] - STARTING_CAPITAL),
        "trades": len(trade_pnls),
        "win_rate": len(wins) / len(trade_pnls) if trade_pnls else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
        "avg_profit_per_trade": float(np.mean(trade_pnls)) if trade_pnls else 0.0,
        "max_drawdown": max_drawdown(equity),
        "raw_5m_sharpe": raw_sharpe,
        "avg_entry_price": avg_entry,
        "avg_one_loss_wipes_out_wins": avg_wipe,
    }


def run_suite(windows: list[Window], sample: str, slippage: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = []
    curves = {"window_start": [w.start for w in windows]}
    suite = {
        "LIVE_ORB": opening_range_signal,
        "LIVE_VWM": volume_weighted_momentum_signal,
        "LIVE_RENKO": renko_signal,
        "CONSENSUS_2_OF_3": consensus_signal,
        "CHEAP_LEADER_PULLBACK": cheap_leader_pullback_signal,
        "OPEN_FLIP_VALUE": open_flip_value_signal,
    }
    for name, fn in suite.items():
        pnl, entries, trade_pnls = run_strategy(windows, name, fn, slippage)
        runs.append(metrics(sample, name, pnl, trade_pnls, entries))
        curves[name] = STARTING_CAPITAL + np.cumsum(pnl)

    pnl, entries, trade_pnls = run_priority_portfolio(windows, slippage)
    runs.append(metrics(sample, "LIVE_PRIORITY_PORTFOLIO", pnl, trade_pnls, entries))
    curves["LIVE_PRIORITY_PORTFOLIO"] = STARTING_CAPITAL + np.cumsum(pnl)

    return pd.DataFrame(runs).sort_values("total_pnl", ascending=False), pd.DataFrame(curves)


def print_table(title: str, df: pd.DataFrame) -> None:
    print("\n" + title)
    out = df.copy()
    for col in ["total_pnl", "avg_profit_per_trade"]:
        out[col] = out[col].map(lambda x: f"${x:,.2f}")
    for col in ["win_rate", "max_drawdown"]:
        out[col] = out[col].map(lambda x: f"{100*x:.2f}%")
    out["profit_factor"] = out["profit_factor"].map(lambda x: "inf" if np.isinf(x) else f"{x:.2f}")
    out["raw_5m_sharpe"] = out["raw_5m_sharpe"].map(lambda x: f"{x:.3f}")
    out["avg_entry_price"] = out["avg_entry_price"].map(lambda x: f"{x:.3f}")
    out["avg_one_loss_wipes_out_wins"] = out["avg_one_loss_wipes_out_wins"].map(lambda x: f"{x:.2f}")
    print(out.to_string(index=False))


def save_curves(curves: pd.DataFrame, suffix: str) -> None:
    csv_path = Path(f"live_guarded_equity_{suffix}.csv")
    png_path = Path(f"live_guarded_equity_{suffix}.png")
    curves.to_csv(csv_path, index=False)
    plt.figure(figsize=(14, 8))
    for col in curves.columns:
        if col == "window_start":
            continue
        plt.plot(pd.to_datetime(curves["window_start"]), curves[col], label=col, linewidth=1.2)
    plt.axhline(STARTING_CAPITAL, color="black", linestyle="--", linewidth=1)
    plt.title(f"Live-Guarded Strategy Equity - {suffix}")
    plt.xlabel("Window start")
    plt.ylabel("Equity ($)")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"saved {csv_path} and {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument("--source", default="binance")
    parser.add_argument("--dataset-source", default="aliplayer_spot")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--slippage-cents", type=float, default=0.0)
    parser.add_argument(
        "--slippage-cents-list",
        default=None,
        help="Comma-separated slippage stress values in cents, e.g. 0,1,2,5.",
    )
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()

    price_df = load_spot_prices(args.symbol, None if args.source.lower() in {"none", "all"} else args.source, dataset_source=args.dataset_source)
    windows = build_windows(price_df, args.max_windows)
    train, test = split_windows_chronologically(windows, args.train_frac)

    if args.slippage_cents_list:
        slippage_values = [float(item.strip()) for item in args.slippage_cents_list.split(",") if item.strip()]
    else:
        slippage_values = [args.slippage_cents]

    stress_rows = []
    for slippage_cents in slippage_values:
        slippage = slippage_cents / 100.0
        train_metrics, train_curves = run_suite(train, "TRAIN", slippage)
        test_metrics, test_curves = run_suite(test, "TEST", slippage)
        all_metrics = pd.concat([train_metrics, test_metrics], ignore_index=True)
        all_metrics.to_csv(f"live_guarded_metrics_slip_{slippage_cents:.2f}c.csv", index=False)
        save_curves(train_curves, f"train_slip_{slippage_cents:.2f}c")
        save_curves(test_curves, f"test_slip_{slippage_cents:.2f}c")

        print_table(f"TRAIN live-guarded metrics, slippage={slippage_cents:.2f}c", train_metrics)
        print_table(f"TEST live-guarded metrics, slippage={slippage_cents:.2f}c", test_metrics)

        merged = train_metrics.merge(test_metrics, on="strategy", suffixes=("_train", "_test"))
        merged["slippage_cents"] = slippage_cents
        merged["survived"] = (
            (merged["total_pnl_train"] > 0)
            & (merged["total_pnl_test"] > 0)
            & (merged["trades_test"] >= 20)
        )
        merged = merged.sort_values(["survived", "total_pnl_test"], ascending=[False, False])
        print("\nSURVIVAL SUMMARY")
        print(merged[[
            "strategy",
            "slippage_cents",
            "total_pnl_train",
            "total_pnl_test",
            "trades_test",
            "win_rate_test",
            "avg_entry_price_test",
            "avg_one_loss_wipes_out_wins_test",
            "survived",
        ]].to_string(index=False))
        stress_rows.append(merged)

    stress = pd.concat(stress_rows, ignore_index=True)
    stress.to_csv("live_guarded_slippage_stress_summary.csv", index=False)
    print("\nsaved live_guarded_slippage_stress_summary.csv")


if __name__ == "__main__":
    main()
