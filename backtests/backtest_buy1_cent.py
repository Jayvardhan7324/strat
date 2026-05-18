"""
Backtest the lottery-style idea: buy a 1 cent Polymarket 5-minute BTC
up/down contract and hold it to settlement.

This uses the same synthetic spot-derived contract prices as the other scripts
in this folder. It is useful for checking the break-even win-rate math, not for
proving real 1c orderbook fills are available live.
"""

from __future__ import annotations

import argparse
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
)


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Trade:
    sample: str
    strategy: str
    market_start: pd.Timestamp
    side: str
    entry_idx: int
    entry_price: float
    ask_price: float
    settlement_value: float
    pnl: float


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


def first_eligible_touch(
    w: Window,
    buy_price: float,
    min_seconds_left: int,
) -> tuple[str, int, float] | None:
    max_entry_idx = WINDOW_SECONDS - min_seconds_left - 1
    if max_entry_idx < 0:
        return None

    candidates: list[tuple[int, str, float]] = []
    epsilon = 1e-9
    for side in ["UP", "DOWN"]:
        ask = side_ask(w, side)[: max_entry_idx + 1]
        hits = np.flatnonzero(ask <= buy_price + epsilon)
        if len(hits) > 0:
            idx = int(hits[0])
            candidates.append((idx, side, float(ask[idx])))

    if not candidates:
        return None

    idx, side, ask_price = min(candidates, key=lambda item: (item[0], item[2]))
    return side, idx, ask_price


def trade_for_window(
    sample: str,
    strategy: str,
    w: Window,
    buy_price: float,
    stake_usd: float,
    min_seconds_left: int,
    fill_mode: str,
    fee_rate: float,
) -> Trade | None:
    touch = first_eligible_touch(w, buy_price, min_seconds_left)
    if touch is None:
        return None

    side, idx, ask_price = touch
    entry_price = buy_price if fill_mode == "fixed" else ask_price
    shares = stake_usd / entry_price
    entry_fee = stake_usd * fee_rate
    final_value = settlement_value(w, side)
    pnl = final_value * shares - stake_usd - entry_fee

    return Trade(
        sample=sample,
        strategy=strategy,
        market_start=w.start,
        side=side,
        entry_idx=idx,
        entry_price=float(entry_price),
        ask_price=float(ask_price),
        settlement_value=final_value,
        pnl=float(pnl),
    )


def forced_loser_trade(
    sample: str,
    w: Window,
    buy_price: float,
    stake_usd: float,
    seconds_left: int,
    fee_rate: float,
) -> Trade | None:
    idx = WINDOW_SECONDS - 1 - seconds_left
    if idx < 0 or idx >= WINDOW_SECONDS:
        return None
    if w.prices[idx] == w.open_price:
        return None

    side = "UP" if w.prices[idx] < w.open_price else "DOWN"
    shares = stake_usd / buy_price
    entry_fee = stake_usd * fee_rate
    final_value = settlement_value(w, side)
    pnl = final_value * shares - stake_usd - entry_fee
    ask_price = float(side_ask(w, side)[idx])
    return Trade(
        sample=sample,
        strategy=f"forced_current_loser_fixed_1c_{seconds_left}s_left",
        market_start=w.start,
        side=side,
        entry_idx=idx,
        entry_price=float(buy_price),
        ask_price=ask_price,
        settlement_value=final_value,
        pnl=float(pnl),
    )


def summarize(sample: str, strategy: str, windows: list[Window], trades: list[Trade]) -> dict:
    pnl_by_window = np.zeros(len(windows), dtype=float)
    start_to_idx = {w.start: idx for idx, w in enumerate(windows)}
    for trade in trades:
        pnl_by_window[start_to_idx[trade.market_start]] += trade.pnl

    equity = STARTING_CAPITAL + np.cumsum(pnl_by_window)
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
    raw_sharpe = float(returns.mean() / returns.std(ddof=1)) if len(returns) > 1 and returns.std(ddof=1) > 0 else 0.0
    pnls = [t.pnl for t in trades]
    winning_tickets = [t for t in trades if t.settlement_value == 1.0]
    positive_pnls = [x for x in pnls if x > 0]
    negative_pnls = [x for x in pnls if x < 0]
    gross_profit = float(sum(positive_pnls))
    gross_loss = float(-sum(negative_pnls))

    return {
        "sample": sample,
        "strategy": strategy,
        "total_pnl": float(equity[-1] - STARTING_CAPITAL),
        "trades": len(trades),
        "markets": len(windows),
        "touch_rate": len(trades) / len(windows) if windows else 0.0,
        "win_rate": len(winning_tickets) / len(trades) if trades else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
        "avg_pnl": float(np.mean(pnls)) if pnls else 0.0,
        "max_drawdown": max_drawdown(equity),
        "raw_5m_sharpe": raw_sharpe,
        "wins": len(winning_tickets),
        "losses": len(trades) - len(winning_tickets),
        "avg_entry_price": float(np.mean([t.entry_price for t in trades])) if trades else 0.0,
        "avg_ask_price": float(np.mean([t.ask_price for t in trades])) if trades else 0.0,
        "avg_entry_seconds_left": float(np.mean([WINDOW_SECONDS - 1 - t.entry_idx for t in trades])) if trades else 0.0,
    }


def print_metrics(df: pd.DataFrame) -> None:
    out = df.copy()
    for col in ["total_pnl", "avg_pnl"]:
        out[col] = out[col].map(lambda x: f"${x:,.2f}")
    for col in ["touch_rate", "win_rate", "max_drawdown"]:
        out[col] = out[col].map(lambda x: f"{100*x:.3f}%")
    out["profit_factor"] = out["profit_factor"].map(lambda x: "inf" if np.isinf(x) else f"{x:.3f}")
    out["raw_5m_sharpe"] = out["raw_5m_sharpe"].map(lambda x: f"{x:.3f}")
    out["avg_entry_price"] = out["avg_entry_price"].map(lambda x: f"{x:.4f}")
    out["avg_ask_price"] = out["avg_ask_price"].map(lambda x: f"{x:.4f}")
    out["avg_entry_seconds_left"] = out["avg_entry_seconds_left"].map(lambda x: f"{x:.1f}")
    print(out.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument("--source", default="binance")
    parser.add_argument("--dataset-source", default="aliplayer_spot")
    parser.add_argument("--stake-usd", type=float, default=10.0)
    parser.add_argument("--buy-price", type=float, default=0.01)
    parser.add_argument("--min-seconds-left", type=int, default=0)
    parser.add_argument("--forced-loser-seconds-left", default="5,10,15,30,60,120")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()

    price_df = load_spot_prices(args.symbol, None if args.source.lower() in {"none", "all"} else args.source, dataset_source=args.dataset_source)
    windows = build_windows(price_df, args.max_windows)
    train, test = split_windows_chronologically(windows, args.train_frac)

    metric_rows: list[dict] = []
    all_trades: list[pd.DataFrame] = []
    break_even_fixed = args.buy_price * (1.0 + TAKER_FEE_RATE)
    print(f"\nFixed 1c break-even win rate with {TAKER_FEE_RATE:.3%} taker fee: {100*break_even_fixed:.3f}%")

    for sample, sample_windows in [("TRAIN", train), ("TEST", test)]:
        for fill_mode in ["fixed", "ask"]:
            strategy = f"first_touch_1c_{fill_mode}_fill"
            trades = [
                trade
                for w in sample_windows
                if (
                    trade := trade_for_window(
                        sample,
                        strategy,
                        w,
                        args.buy_price,
                        args.stake_usd,
                        args.min_seconds_left,
                        fill_mode,
                        TAKER_FEE_RATE,
                    )
                )
                is not None
            ]
            metric_rows.append(summarize(sample, strategy, sample_windows, trades))
            if trades:
                all_trades.append(pd.DataFrame([t.__dict__ for t in trades]))

        for seconds_left in parse_int_list(args.forced_loser_seconds_left):
            strategy = f"forced_current_loser_fixed_1c_{seconds_left}s_left"
            trades = [
                trade
                for w in sample_windows
                if (
                    trade := forced_loser_trade(
                        sample,
                        w,
                        args.buy_price,
                        args.stake_usd,
                        seconds_left,
                        TAKER_FEE_RATE,
                    )
                )
                is not None
            ]
            metric_rows.append(summarize(sample, strategy, sample_windows, trades))
            if trades:
                all_trades.append(pd.DataFrame([t.__dict__ for t in trades]))

    metrics = pd.DataFrame(metric_rows)
    print_metrics(metrics)
    metrics.to_csv("buy1_cent_metrics.csv", index=False)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv("buy1_cent_trades.csv", index=False)
    else:
        pd.DataFrame(
            columns=[
                "sample",
                "strategy",
                "market_start",
                "side",
                "entry_idx",
                "entry_price",
                "ask_price",
                "settlement_value",
                "pnl",
            ]
        ).to_csv("buy1_cent_trades.csv", index=False)
    print("\nSaved buy1_cent_metrics.csv and buy1_cent_trades.csv")


if __name__ == "__main__":
    main()
