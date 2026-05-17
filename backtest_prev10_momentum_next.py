"""
Backtest previous-market last-10s momentum into the next 5-minute market.

This explores whether the final 10 seconds of one BTC up/down window can choose
the side for the next window, then sweeps entry caps and entry windows to find a
historical sweet spot.

The data path is intentionally the same synthetic spot-derived Polymarket model
used by the other research scripts in this folder. Good results here are paper
run candidates, not proof of live orderbook edge.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
)


DEFAULT_MOMENTUM_BPS = [0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0]
DEFAULT_ENTRY_CAPS = [0.50, 0.51, 0.52, 0.53, 0.55, 0.57, 0.60, 0.65]
DEFAULT_ENTRY_WINDOWS = [5, 10, 20, 30, 60]
DEFAULT_SLIPPAGE_CENTS = [0.0, 1.0, 2.0, 5.0]


@dataclass(frozen=True)
class Combo:
    variant: str
    momentum_bps: float
    entry_cap: float
    entry_window_seconds: int
    slippage_cents: float

    @property
    def strategy(self) -> str:
        return (
            f"{self.variant}_mom{self.momentum_bps:g}bps_"
            f"cap{self.entry_cap:.2f}_win{self.entry_window_seconds}s_"
            f"slip{self.slippage_cents:.2f}c"
        )


@dataclass(frozen=True)
class Trade:
    sample: str
    strategy: str
    variant: str
    momentum_bps: float
    entry_cap: float
    entry_window_seconds: int
    slippage_cents: float
    momentum_lookback_seconds: int
    prev_market_start: pd.Timestamp
    market_start: pd.Timestamp
    prev10_ret: float
    side: str
    entry_idx: int
    entry_price: float
    settlement_value: float
    pnl: float


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def side_ask(w: Window, side: str) -> np.ndarray:
    if side == "UP":
        return w.up_ask
    if side == "DOWN":
        return w.down_ask
    raise ValueError(side)


def settlement_value(w: Window, side: str) -> float:
    if side == "UP":
        return float(w.outcome_up)
    if side == "DOWN":
        return float(1 - w.outcome_up)
    raise ValueError(side)


def side_from_momentum(prev10_ret: float, variant: str) -> str:
    if variant == "follow":
        return "UP" if prev10_ret > 0 else "DOWN"
    if variant == "fade":
        return "DOWN" if prev10_ret > 0 else "UP"
    raise ValueError(variant)


def previous_momentum_returns(windows: list[Window], momentum_lookback_seconds: int) -> np.ndarray:
    returns = np.zeros(len(windows), dtype=float)
    lookback_idx = max(0, WINDOW_SECONDS - momentum_lookback_seconds - 1)
    for i in range(1, len(windows)):
        prev = windows[i - 1]
        returns[i] = float(prev.prices[-1] / prev.prices[lookback_idx] - 1.0)
    return returns


def find_entry(
    w: Window,
    side: str,
    entry_window_seconds: int,
    entry_cap: float,
    slippage: float,
) -> tuple[int, float] | None:
    last_idx = min(max(entry_window_seconds, 1), WINDOW_SECONDS)
    ask = side_ask(w, side)[:last_idx]
    stressed_ask = np.minimum(0.999, ask + slippage)
    hits = np.flatnonzero(stressed_ask <= entry_cap)
    if len(hits) == 0:
        return None
    idx = int(hits[0])
    return idx, float(stressed_ask[idx])


def combo_grid(
    momentum_bps_values: Iterable[float],
    entry_caps: Iterable[float],
    entry_windows: Iterable[int],
    slippage_cents_values: Iterable[float],
) -> list[Combo]:
    combos: list[Combo] = []
    for variant in ["follow", "fade"]:
        for momentum_bps in momentum_bps_values:
            for entry_cap in entry_caps:
                for entry_window_seconds in entry_windows:
                    for slippage_cents in slippage_cents_values:
                        combos.append(
                            Combo(
                                variant=variant,
                                momentum_bps=momentum_bps,
                                entry_cap=entry_cap,
                                entry_window_seconds=entry_window_seconds,
                                slippage_cents=slippage_cents,
                            )
                        )
    return combos


def backtest_combo(
    sample: str,
    windows: list[Window],
    prev_returns: np.ndarray,
    combo: Combo,
    stake_usd: float,
    fee_rate: float,
    momentum_lookback_seconds: int,
    collect_trades: bool = False,
) -> tuple[dict, pd.DataFrame]:
    pnl_by_window = np.zeros(len(windows), dtype=float)
    trades: list[Trade] = []
    trade_pnls: list[float] = []
    entry_prices: list[float] = []
    signals = 0
    threshold = combo.momentum_bps / 10_000.0
    slippage = combo.slippage_cents / 100.0
    signal_indices = np.flatnonzero((prev_returns != 0.0) & (np.abs(prev_returns) >= threshold))

    for i in signal_indices:
        prev = windows[i - 1]
        current = windows[i]
        prev10_ret = float(prev_returns[i])
        signals += 1
        side = side_from_momentum(prev10_ret, combo.variant)
        entry = find_entry(
            current,
            side,
            combo.entry_window_seconds,
            combo.entry_cap,
            slippage,
        )
        if entry is None:
            continue

        entry_idx, entry_price = entry
        shares = stake_usd / entry_price
        entry_fee = stake_usd * fee_rate
        final_value = settlement_value(current, side)
        pnl = final_value * shares - stake_usd - entry_fee
        pnl_by_window[i] = pnl
        trade_pnls.append(float(pnl))
        entry_prices.append(entry_price)
        if collect_trades:
            trades.append(
                Trade(
                    sample=sample,
                    strategy=combo.strategy,
                    variant=combo.variant,
                    momentum_bps=combo.momentum_bps,
                    entry_cap=combo.entry_cap,
                    entry_window_seconds=combo.entry_window_seconds,
                    slippage_cents=combo.slippage_cents,
                    momentum_lookback_seconds=momentum_lookback_seconds,
                    prev_market_start=prev.start,
                    market_start=current.start,
                    prev10_ret=prev10_ret,
                    side=side,
                    entry_idx=entry_idx,
                    entry_price=entry_price,
                    settlement_value=final_value,
                    pnl=float(pnl),
                )
            )

    equity = STARTING_CAPITAL + np.cumsum(pnl_by_window)
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
    raw_sharpe = float(returns.mean() / returns.std(ddof=1)) if len(returns) > 1 and returns.std(ddof=1) > 0 else 0.0
    wins = [value for value in trade_pnls if value > 0]
    losses = [value for value in trade_pnls if value < 0]
    gross_profit = float(sum(wins))
    gross_loss = float(-sum(losses))

    metrics = {
        "sample": sample,
        "strategy": combo.strategy,
        "variant": combo.variant,
        "momentum_bps": combo.momentum_bps,
        "entry_cap": combo.entry_cap,
        "entry_window_seconds": combo.entry_window_seconds,
        "slippage_cents": combo.slippage_cents,
        "momentum_lookback_seconds": momentum_lookback_seconds,
        "total_pnl": float(equity[-1] - STARTING_CAPITAL),
        "trades": len(trade_pnls),
        "signals": signals,
        "fill_rate": len(trade_pnls) / signals if signals else 0.0,
        "win_rate": len(wins) / len(trade_pnls) if trade_pnls else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
        "avg_entry_price": float(np.mean(entry_prices)) if entry_prices else 0.0,
        "avg_pnl": float(np.mean(trade_pnls)) if trade_pnls else 0.0,
        "max_drawdown": max_drawdown(equity),
        "raw_5m_sharpe": raw_sharpe,
    }
    return metrics, pd.DataFrame([trade.__dict__ for trade in trades])


def print_table(title: str, df: pd.DataFrame, limit: int = 12) -> None:
    print("\n" + title)
    if df.empty:
        print("(none)")
        return

    out = df.head(limit).copy()
    for col in ["total_pnl", "avg_pnl"]:
        out[col] = out[col].map(lambda value: f"${value:,.2f}")
    for col in ["fill_rate", "win_rate", "max_drawdown"]:
        out[col] = out[col].map(lambda value: f"{100 * value:.2f}%")
    out["profit_factor"] = out["profit_factor"].map(lambda value: "inf" if np.isinf(value) else f"{value:.2f}")
    out["avg_entry_price"] = out["avg_entry_price"].map(lambda value: f"{value:.3f}")
    out["raw_5m_sharpe"] = out["raw_5m_sharpe"].map(lambda value: f"{value:.3f}")
    cols = [
        "sample",
        "variant",
        "momentum_bps",
        "entry_cap",
        "entry_window_seconds",
        "slippage_cents",
        "total_pnl",
        "trades",
        "fill_rate",
        "win_rate",
        "profit_factor",
        "avg_entry_price",
        "raw_5m_sharpe",
    ]
    print(out[cols].to_string(index=False))


def survival_table(metrics: pd.DataFrame) -> pd.DataFrame:
    train = metrics[metrics["sample"] == "TRAIN"]
    test = metrics[metrics["sample"] == "TEST"]
    merged = train.merge(
        test,
        on=[
            "strategy",
            "variant",
            "momentum_bps",
            "entry_cap",
            "entry_window_seconds",
            "slippage_cents",
            "momentum_lookback_seconds",
        ],
        suffixes=("_train", "_test"),
    )
    merged["survived"] = (
        (merged["total_pnl_train"] > 0)
        & (merged["total_pnl_test"] > 0)
        & (merged["trades_test"] >= 20)
    )
    merged["min_pnl"] = merged[["total_pnl_train", "total_pnl_test"]].min(axis=1)
    return merged.sort_values(["survived", "total_pnl_test", "min_pnl"], ascending=[False, False, False])


def choose_trade_log_strategies(survival: pd.DataFrame, per_group: int) -> set[str]:
    selected: set[str] = set()
    survived = survival[survival["survived"]]
    selected.update(survived.head(per_group)["strategy"].tolist())
    selected.update(survival.sort_values("total_pnl_train", ascending=False).head(per_group)["strategy"].tolist())
    selected.update(survival.sort_values("total_pnl_test", ascending=False).head(per_group)["strategy"].tolist())
    return selected


def format_money(value: float) -> str:
    return f"${value:,.2f}"


def format_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def markdown_table(df: pd.DataFrame, cols: list[str], limit: int = 12) -> str:
    if df.empty:
        return "_No rows._"
    out = df.head(limit).copy()
    for col in out.columns:
        if col in {"total_pnl", "total_pnl_train", "total_pnl_test", "avg_pnl", "avg_pnl_test"}:
            out[col] = out[col].map(format_money)
        elif col in {"fill_rate", "fill_rate_test", "win_rate", "win_rate_test", "max_drawdown", "max_drawdown_test"}:
            out[col] = out[col].map(format_pct)
        elif col in {"avg_entry_price", "avg_entry_price_test", "raw_5m_sharpe", "raw_5m_sharpe_test"}:
            out[col] = out[col].map(lambda value: f"{value:.3f}")
        elif col in {"profit_factor", "profit_factor_test"}:
            out[col] = out[col].map(lambda value: "inf" if np.isinf(value) else f"{value:.2f}")

    rendered = out[cols].astype(str)
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in rendered.to_numpy()]
    return "\n".join([header, separator, *rows])


def write_summary(
    path: Path,
    metrics: pd.DataFrame,
    survival: pd.DataFrame,
    baseline_path: Path,
    momentum_lookback_seconds: int,
    output_prefix: str,
) -> None:
    train_top = metrics[metrics["sample"] == "TRAIN"].sort_values("total_pnl", ascending=False)
    test_top = metrics[metrics["sample"] == "TEST"].sort_values("total_pnl", ascending=False)
    survived = survival[survival["survived"]]
    died_under_slip = survival[
        (survival["slippage_cents"] >= 2.0)
        & ((survival["total_pnl_train"] <= 0) | (survival["total_pnl_test"] <= 0))
    ].sort_values(["slippage_cents", "total_pnl_test"], ascending=[False, True])

    best_test = test_top.iloc[0] if not test_top.empty else None
    best_survivor = survived.iloc[0] if not survived.empty else None
    best_slippage_survivor = survived[survived["slippage_cents"] > 0].sort_values("total_pnl_test", ascending=False)
    best_5c_survivor = survived[survived["slippage_cents"] == 5.0].sort_values("total_pnl_test", ascending=False)

    baseline_note = "Baseline file not found."
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        cheap = baseline[(baseline["strategy"] == "CHEAP_LEADER_PULLBACK") & (baseline["slippage_cents"] == 5.0)]
        if not cheap.empty:
            row = cheap.iloc[0]
            if "total_pnl_test" in row.index:
                baseline_note = (
                    f"`CHEAP_LEADER_PULLBACK` held-out 5c stress baseline: "
                    f"{format_money(float(row['total_pnl_test']))}, "
                    f"{int(row['trades_test'])} trades, "
                    f"{format_pct(float(row['win_rate_test']))} win rate."
                )
            elif "sample" in baseline.columns:
                test_row = cheap[cheap["sample"] == "TEST"]
                if not test_row.empty:
                    row = test_row.iloc[0]
                    baseline_note = (
                        f"`CHEAP_LEADER_PULLBACK` held-out 5c stress baseline: "
                        f"{format_money(float(row['total_pnl']))}, "
                        f"{int(row['trades'])} trades, "
                        f"{format_pct(float(row['win_rate']))} win rate."
                    )

    if best_survivor is not None:
        slippage_sentence = "No positive-slippage combo survived train/test."
        if not best_slippage_survivor.empty:
            row = best_slippage_survivor.iloc[0]
            slippage_sentence = (
                f"Best positive-slippage survivor is `{row['strategy']}` with held-out test PnL "
                f"{format_money(float(row['total_pnl_test']))}."
            )
        if not best_5c_survivor.empty:
            row = best_5c_survivor.iloc[0]
            slippage_sentence += (
                f" Best 5c survivor is `{row['strategy']}` with held-out test PnL "
                f"{format_money(float(row['total_pnl_test']))}."
            )
        else:
            slippage_sentence += " No 5c slippage combo survived."
        recommendation = (
            "Paper-run candidate only if it remains competitive after live orderbook checks. "
            f"Best survivor is `{best_survivor['strategy']}` with held-out test PnL "
            f"{format_money(float(best_survivor['total_pnl_test']))}. {slippage_sentence}"
        )
    elif best_test is not None and float(best_test["total_pnl"]) > 0:
        recommendation = (
            "Refine before paper-running: the best held-out row is positive, but no combo passed "
            "the train/test survival gate."
        )
    else:
        recommendation = "Reject v1: no held-out evidence strong enough to justify paper-running."

    lines = [
        "# Previous-10s Momentum Next-Market Backtest",
        "",
        f"This tests whether the last {momentum_lookback_seconds} seconds of one 5-minute BTC market can choose the side for the next market.",
        "It sweeps follow/fade direction, previous-window momentum threshold, next-market entry cap, entry-window length, and entry slippage stress.",
        "",
        "## Caveat",
        "",
        "This uses synthetic Polymarket prices derived from spot data, not historical real orderbooks. Treat promising rows as paper-run candidates, not live-proof edges.",
        "",
        "## Baseline Check",
        "",
        baseline_note,
        "",
        "## Top Train Rows",
        "",
        markdown_table(
            train_top,
            [
                "variant",
                "momentum_bps",
                "entry_cap",
                "entry_window_seconds",
                "slippage_cents",
                "total_pnl",
                "trades",
                "fill_rate",
                "win_rate",
                "avg_entry_price",
            ],
        ),
        "",
        "## Top Held-Out Test Rows",
        "",
        markdown_table(
            test_top,
            [
                "variant",
                "momentum_bps",
                "entry_cap",
                "entry_window_seconds",
                "slippage_cents",
                "total_pnl",
                "trades",
                "fill_rate",
                "win_rate",
                "avg_entry_price",
            ],
        ),
        "",
        "## Train/Test Survivors",
        "",
        markdown_table(
            survived,
            [
                "variant",
                "momentum_bps",
                "entry_cap",
                "entry_window_seconds",
                "slippage_cents",
                "total_pnl_train",
                "total_pnl_test",
                "trades_test",
                "fill_rate_test",
                "win_rate_test",
                "avg_entry_price_test",
            ],
        ),
        "",
        "## Weak Under Slippage",
        "",
        markdown_table(
            died_under_slip,
            [
                "variant",
                "momentum_bps",
                "entry_cap",
                "entry_window_seconds",
                "slippage_cents",
                "total_pnl_train",
                "total_pnl_test",
                "trades_test",
                "fill_rate_test",
                "win_rate_test",
            ],
        ),
        "",
        "## Recommendation",
        "",
        recommendation,
        "",
        "## Files",
        "",
        f"- Metrics: `{output_prefix}_metrics.csv`",
        f"- Survival summary: `{output_prefix}_survival.csv`",
        f"- Selected trade logs: `{output_prefix}_best_trades.csv`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument("--source", default="binance")
    parser.add_argument("--dataset-source", default="aliplayer_spot")
    parser.add_argument("--stake-usd", type=float, default=10.0)
    parser.add_argument("--fee-rate", type=float, default=TAKER_FEE_RATE)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--momentum-bps-list", default=",".join(str(x) for x in DEFAULT_MOMENTUM_BPS))
    parser.add_argument("--entry-caps", default=",".join(str(x) for x in DEFAULT_ENTRY_CAPS))
    parser.add_argument("--entry-windows", default=",".join(str(x) for x in DEFAULT_ENTRY_WINDOWS))
    parser.add_argument("--slippage-cents-list", default=",".join(str(x) for x in DEFAULT_SLIPPAGE_CENTS))
    parser.add_argument("--momentum-lookback-seconds", type=int, default=10)
    parser.add_argument("--top-trade-logs", type=int, default=5)
    parser.add_argument("--output-prefix", default="prev10_momentum_next")
    args = parser.parse_args()

    momentum_bps_values = parse_float_list(args.momentum_bps_list)
    entry_caps = parse_float_list(args.entry_caps)
    entry_windows = parse_int_list(args.entry_windows)
    slippage_cents_values = parse_float_list(args.slippage_cents_list)

    price_df = load_spot_prices(args.symbol, None if args.source.lower() in {"none", "all"} else args.source, dataset_source=args.dataset_source)
    windows = build_windows(price_df, args.max_windows)
    train, test = split_windows_chronologically(windows, args.train_frac)
    combos = combo_grid(momentum_bps_values, entry_caps, entry_windows, slippage_cents_values)
    combos_by_strategy = {combo.strategy: combo for combo in combos}
    sample_sets = [("TRAIN", train), ("TEST", test)]
    prev_returns_by_sample = {
        sample: previous_momentum_returns(sample_windows, args.momentum_lookback_seconds)
        for sample, sample_windows in sample_sets
    }
    print(f"\nRunning {len(combos):,} parameter combos for TRAIN and TEST")

    metric_rows: list[dict] = []
    for sample, sample_windows in sample_sets:
        prev_returns = prev_returns_by_sample[sample]
        for combo in combos:
            metrics, trades = backtest_combo(
                sample,
                sample_windows,
                prev_returns,
                combo,
                args.stake_usd,
                args.fee_rate,
                args.momentum_lookback_seconds,
            )
            metric_rows.append(metrics)

    metrics = pd.DataFrame(metric_rows)
    survival = survival_table(metrics)
    selected_strategies = choose_trade_log_strategies(survival, args.top_trade_logs)

    selected_trade_frames = []
    for sample, sample_windows in sample_sets:
        prev_returns = prev_returns_by_sample[sample]
        for strategy in selected_strategies:
            combo = combos_by_strategy[strategy]
            _, trades = backtest_combo(
                sample,
                sample_windows,
                prev_returns,
                combo,
                args.stake_usd,
                args.fee_rate,
                args.momentum_lookback_seconds,
                collect_trades=True,
            )
            if not trades.empty:
                selected_trade_frames.append(trades)
    selected_trades = pd.concat(selected_trade_frames, ignore_index=True) if selected_trade_frames else pd.DataFrame()

    prefix = args.output_prefix
    metrics_path = Path(f"{prefix}_metrics.csv")
    survival_path = Path(f"{prefix}_survival.csv")
    trades_path = Path(f"{prefix}_best_trades.csv")
    report_path = Path(f"{prefix}_SUMMARY.md")

    metrics.to_csv(metrics_path, index=False)
    survival.to_csv(survival_path, index=False)
    selected_trades.to_csv(trades_path, index=False)
    write_summary(
        report_path,
        metrics,
        survival,
        Path("live_guarded_slippage_stress_summary.csv"),
        args.momentum_lookback_seconds,
        prefix,
    )

    print_table("Top TRAIN rows", metrics[metrics["sample"] == "TRAIN"].sort_values("total_pnl", ascending=False))
    print_table("Top TEST rows", metrics[metrics["sample"] == "TEST"].sort_values("total_pnl", ascending=False))
    print("\nTop train/test survivors")
    cols = [
        "variant",
        "momentum_bps",
        "entry_cap",
        "entry_window_seconds",
        "slippage_cents",
        "total_pnl_train",
        "total_pnl_test",
        "trades_test",
        "fill_rate_test",
        "win_rate_test",
        "survived",
    ]
    print(survival.head(12)[cols].to_string(index=False))
    print(f"\nSaved {metrics_path}, {survival_path}, {trades_path}, and {report_path}")


if __name__ == "__main__":
    main()
