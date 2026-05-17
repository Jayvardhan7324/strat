"""
Backtest a simple "buy at 97, sell at 99" Polymarket scalp idea.

Interpretation tested:
- In each 5-minute BTC up/down window, watch both UP and DOWN synthetic books.
- When the current leading side's ask reaches at least 0.97 and is still below 0.99,
  buy that side at the ask.
- After entry, try to sell at a 0.99 bid.
- If 0.99 never appears before the window ends, hold to settlement.

This is intentionally conservative about the risk:
- Includes 0.2% taker fee on entry and exit.
- Reports how many trades actually hit the 99c exit versus settlement.
- Uses the same synthetic price model as polymarket_updown_backtest.py, not real
  historical Polymarket orderbooks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

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
    if side == "UP":
        return float(w.outcome_up)
    if side == "DOWN":
        return float(1 - w.outcome_up)
    raise ValueError(side)


def scalp_97_99(
    w: Window,
    stake_usd: float,
    buy_price: float,
    sell_price: float,
    min_seconds_left: int,
    fee_rate: float,
) -> Trade | None:
    max_entry_idx = 300 - min_seconds_left

    for idx in range(1, max_entry_idx + 1):
        up_bid, up_ask = side_prices(w, "UP")
        down_bid, down_ask = side_prices(w, "DOWN")
        leader = "UP" if w.prices[idx] >= w.open_price else "DOWN"
        bid, ask = (up_bid, up_ask) if leader == "UP" else (down_bid, down_ask)
        entry_price = float(ask[idx])

        if entry_price < buy_price or entry_price >= sell_price:
            continue

        shares = stake_usd / entry_price
        entry_fee = stake_usd * fee_rate

        future_hit = np.flatnonzero(bid[idx + 1 :] >= sell_price)
        if len(future_hit) > 0:
            exit_idx = int(idx + 1 + future_hit[0])
            exit_value = sell_price * shares
            exit_fee = exit_value * fee_rate
            pnl = exit_value - stake_usd - entry_fee - exit_fee
            return Trade(
                strategy="leader_97_to_99",
                entry_idx=idx,
                exit_idx=exit_idx,
                side=leader,
                entry_price=entry_price,
                exit_price=sell_price,
                exit_type="sold_99",
                pnl=float(pnl),
            )

        final_value = outcome_value(w, leader) * shares
        pnl = final_value - stake_usd - entry_fee
        return Trade(
            strategy="leader_97_to_99",
            entry_idx=idx,
            exit_idx=None,
            side=leader,
            entry_price=entry_price,
            exit_price=outcome_value(w, leader),
            exit_type="settlement",
            pnl=float(pnl),
        )

    return None


def end_window_always_buy_97(
    w: Window,
    stake_usd: float,
    buy_price: float,
    sell_price: float,
    entry_seconds_left: int,
    fee_rate: float,
) -> Trade | None:
    # Alternative interpretation: at a fixed late time, buy the current leader
    # only if it can be bought at 97c or better, then try to sell at 99c.
    idx = 300 - entry_seconds_left
    leader = "UP" if w.prices[idx] >= w.open_price else "DOWN"
    bid, ask = side_prices(w, leader)
    entry_price = float(ask[idx])
    if entry_price > buy_price:
        return None

    shares = stake_usd / entry_price
    entry_fee = stake_usd * fee_rate
    future_hit = np.flatnonzero(bid[idx + 1 :] >= sell_price)
    if len(future_hit) > 0:
        exit_idx = int(idx + 1 + future_hit[0])
        exit_value = sell_price * shares
        exit_fee = exit_value * fee_rate
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return Trade("late_limit_97_to_99", idx, exit_idx, leader, entry_price, sell_price, "sold_99", float(pnl))

    final_value = outcome_value(w, leader) * shares
    pnl = final_value - stake_usd - entry_fee
    return Trade(
        "late_limit_97_to_99",
        idx,
        None,
        leader,
        entry_price,
        outcome_value(w, leader),
        "settlement",
        float(pnl),
    )


def fixed_price_97_to_99(
    w: Window,
    stake_usd: float,
    buy_price: float,
    sell_price: float,
    entry_seconds_left: int,
    fee_rate: float,
) -> Trade:
    # Literal interpretation: at a fixed late time, buy the current leader
    # at exactly 97c and try to sell at exactly 99c. If 99c does not fill,
    # hold to settlement.
    idx = 300 - entry_seconds_left
    leader = "UP" if w.prices[idx] >= w.open_price else "DOWN"
    bid, _ = side_prices(w, leader)
    shares = stake_usd / buy_price
    entry_fee = stake_usd * fee_rate
    future_hit = np.flatnonzero(bid[idx + 1 :] >= sell_price)
    if len(future_hit) > 0:
        exit_idx = int(idx + 1 + future_hit[0])
        exit_value = sell_price * shares
        exit_fee = exit_value * fee_rate
        pnl = exit_value - stake_usd - entry_fee - exit_fee
        return Trade("fixed_97_to_99", idx, exit_idx, leader, buy_price, sell_price, "sold_99", float(pnl))

    final_value = outcome_value(w, leader) * shares
    pnl = final_value - stake_usd - entry_fee
    return Trade(
        "fixed_97_to_99",
        idx,
        None,
        leader,
        buy_price,
        outcome_value(w, leader),
        "settlement",
        float(pnl),
    )


def run_strategy(windows: list[Window], strategy_name: str, trade_fn) -> tuple[pd.DataFrame, pd.DataFrame]:
    pnl_by_window = np.zeros(len(windows), dtype=float)
    trades: list[Trade] = []
    for i, w in enumerate(windows):
        trade = trade_fn(w)
        if trade is None:
            continue
        pnl_by_window[i] = trade.pnl
        trades.append(trade)

    equity = STARTING_CAPITAL + np.cumsum(pnl_by_window)
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
    raw_sharpe = float(returns.mean() / returns.std(ddof=1)) if len(returns) > 1 and returns.std(ddof=1) > 0 else 0.0
    trade_pnls = [t.pnl for t in trades]
    wins = [x for x in trade_pnls if x > 0]
    losses = [x for x in trade_pnls if x < 0]
    gross_profit = float(sum(wins))
    gross_loss = float(-sum(losses))
    metrics = pd.DataFrame([
        {
            "strategy": strategy_name,
            "total_pnl": float(equity[-1] - STARTING_CAPITAL),
            "trades": len(trades),
            "win_rate": len(wins) / len(trades) if trades else 0.0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0),
            "avg_pnl": float(np.mean(trade_pnls)) if trade_pnls else 0.0,
            "max_drawdown": max_drawdown(equity),
            "raw_5m_sharpe": raw_sharpe,
            "sold_99": sum(1 for t in trades if t.exit_type == "sold_99"),
            "settled": sum(1 for t in trades if t.exit_type == "settlement"),
            "avg_entry_price": float(np.mean([t.entry_price for t in trades])) if trades else 0.0,
            "avg_entry_seconds_left": float(np.mean([300 - t.entry_idx for t in trades])) if trades else 0.0,
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
    out["avg_entry_seconds_left"] = out["avg_entry_seconds_left"].map(lambda x: f"{x:.1f}")
    print(out.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument("--source", default="binance")
    parser.add_argument("--dataset-source", default="aliplayer_spot")
    parser.add_argument("--stake-usd", type=float, default=10.0)
    parser.add_argument("--buy-price", type=float, default=0.97)
    parser.add_argument("--sell-price", type=float, default=0.99)
    parser.add_argument("--min-seconds-left", type=int, default=5)
    parser.add_argument("--late-entry-seconds-left", type=int, default=30)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()

    price_df = load_spot_prices(args.symbol, None if args.source.lower() in {"none", "all"} else args.source, dataset_source=args.dataset_source)
    windows = build_windows(price_df, args.max_windows)
    train, test = split_windows_chronologically(windows, args.train_frac)

    rows = []
    for sample, sample_windows in [("TRAIN", train), ("TEST", test)]:
        m1, t1 = run_strategy(
            sample_windows,
            "leader_97_to_99",
            lambda w: scalp_97_99(
                w,
                args.stake_usd,
                args.buy_price,
                args.sell_price,
                args.min_seconds_left,
                TAKER_FEE_RATE,
            ),
        )
        m2, t2 = run_strategy(
            sample_windows,
            "late_limit_97_to_99",
            lambda w: end_window_always_buy_97(
                w,
                args.stake_usd,
                args.buy_price,
                args.sell_price,
                args.late_entry_seconds_left,
                TAKER_FEE_RATE,
            ),
        )
        m3, t3 = run_strategy(
            sample_windows,
            "fixed_97_to_99",
            lambda w: fixed_price_97_to_99(
                w,
                args.stake_usd,
                args.buy_price,
                args.sell_price,
                args.late_entry_seconds_left,
                TAKER_FEE_RATE,
            ),
        )
        m1.insert(0, "sample", sample)
        m2.insert(0, "sample", sample)
        m3.insert(0, "sample", sample)
        t1.insert(0, "sample", sample)
        t2.insert(0, "sample", sample)
        t3.insert(0, "sample", sample)
        rows.extend([m1, m2, m3])
        print_table(f"{sample} metrics", pd.concat([m1, m2, m3], ignore_index=True))
        pd.concat([t1, t2, t3], ignore_index=True).to_csv(f"buy97_sell99_trades_{sample.lower()}.csv", index=False)

    all_metrics = pd.concat(rows, ignore_index=True)
    all_metrics.to_csv("buy97_sell99_metrics.csv", index=False)
    print("\nSaved buy97_sell99_metrics.csv, buy97_sell99_trades_train.csv, buy97_sell99_trades_test.csv")


if __name__ == "__main__":
    main()
