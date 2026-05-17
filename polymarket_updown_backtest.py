"""
Polymarket crypto up/down 5-minute market backtester.

Loads the Hugging Face dataset `aliplayer1/polymarket-crypto-updown`
config `spot_prices`, builds aligned 5-minute binary markets, analyzes
target-line crossings, and backtests a basket of directional, reversal,
market-making, and experimental strategies.

Run:
    python polymarket_updown_backtest.py

Optional:
    python polymarket_updown_backtest.py --symbol btc/usd --source chainlink_proxy
    python polymarket_updown_backtest.py --max-windows 2000
    python polymarket_updown_backtest.py --train-frac 0.70
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from datasets import load_dataset, load_dataset_builder
from scipy.stats import norm

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


DATASET_NAME = "aliplayer1/polymarket-crypto-updown"
DATASET_CONFIG = "spot_prices"
DATASET_SPLIT = "train"

WINDOW_SECONDS = 300
SIGMA = 0.60
HALF_SPREAD = 0.005
STARTING_CAPITAL = 10_000.0
DEFAULT_TRADE_QTY = 1_000.0
MM_ORDER_QTY = 10.0
TAKER_FEE_RATE = 0.002
MAKER_REBATE_RATE = 0.001
WINDOWS_PER_YEAR = 365 * 24 * 12
TARGET_THRESHOLDS = [0.001, 0.002, 0.003, 0.005, 0.010, 0.020]


@dataclass(slots=True)
class Window:
    start: pd.Timestamp
    prices: np.ndarray
    open_price: float
    close_price: float
    outcome_up: int
    fair_up: np.ndarray
    up_bid: np.ndarray
    up_ask: np.ndarray
    down_bid: np.ndarray
    down_ask: np.ndarray


@dataclass(slots=True)
class StrategyRun:
    name: str
    pnl_by_window: np.ndarray
    trade_pnls: list[float]
    fills: int = 0
    signals: int = 0
    wins: int = 0
    extra: dict | None = None


def money(x: float) -> str:
    return f"${x:,.2f}"


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def clip_contract_price(x: np.ndarray | float) -> np.ndarray | float:
    return np.clip(x, 0.001, 0.999)


def fair_prices_for_window(prices: np.ndarray, open_price: float) -> np.ndarray:
    elapsed = np.arange(len(prices), dtype=float)
    tau = np.maximum((WINDOW_SECONDS - elapsed) / WINDOW_SECONDS, 1e-9)
    log_moneyness = np.log(prices / open_price)
    z = log_moneyness / (SIGMA * np.sqrt(tau))
    return clip_contract_price(norm.cdf(z))


def make_market_arrays(fair_up: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    up_bid = clip_contract_price(fair_up - HALF_SPREAD)
    up_ask = clip_contract_price(fair_up + HALF_SPREAD)
    fair_down = 1.0 - fair_up
    down_bid = clip_contract_price(fair_down - HALF_SPREAD)
    down_ask = clip_contract_price(fair_down + HALF_SPREAD)
    return up_bid, up_ask, down_bid, down_ask


def side_arrays(w: Window, side: str) -> tuple[np.ndarray, np.ndarray]:
    if side == "up":
        return w.up_bid, w.up_ask
    if side == "down":
        return w.down_bid, w.down_ask
    raise ValueError(f"Unknown side: {side}")


def settlement_value(w: Window, side: str) -> float:
    if side == "up":
        return float(w.outcome_up)
    if side == "down":
        return float(1 - w.outcome_up)
    raise ValueError(f"Unknown side: {side}")


def market_buy_pnl(w: Window, idx: int, side: str, qty: float = DEFAULT_TRADE_QTY) -> float:
    _, ask = side_arrays(w, side)
    entry = float(ask[idx])
    fee = entry * qty * TAKER_FEE_RATE
    return (settlement_value(w, side) - entry) * qty - fee


def limit_buy_fill_idx(w: Window, start_idx: int, side: str, limit_price: float) -> int | None:
    _, ask = side_arrays(w, side)
    hit = np.flatnonzero(ask[start_idx:] <= limit_price)
    if len(hit) == 0:
        return None
    return int(start_idx + hit[0])


def limit_buy_settlement_pnl(
    w: Window,
    start_idx: int,
    side: str,
    limit_price: float,
    qty: float = DEFAULT_TRADE_QTY,
) -> tuple[float, int | None]:
    fill_idx = limit_buy_fill_idx(w, start_idx, side, limit_price)
    if fill_idx is None:
        return 0.0, None
    rebate = limit_price * qty * MAKER_REBATE_RATE
    pnl = (settlement_value(w, side) - limit_price) * qty + rebate
    return pnl, fill_idx


def rsi(values: np.ndarray, period: int = 14) -> float:
    if len(values) <= period:
        return 50.0
    diffs = np.diff(values[-(period + 1) :])
    gains = np.clip(diffs, 0, None)
    losses = -np.clip(diffs, None, 0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def ema(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def max_drawdown(equity: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    return float(drawdowns.min()) if len(drawdowns) else 0.0


def strategy_metrics(run: StrategyRun, equity: np.ndarray) -> dict:
    total_pnl = float(equity[-1] - STARTING_CAPITAL)
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
    sharpe = 0.0
    if len(returns) > 1 and returns.std(ddof=1) > 0:
        sharpe = float((returns.mean() / returns.std(ddof=1)) * math.sqrt(WINDOWS_PER_YEAR))
    wins = [x for x in run.trade_pnls if x > 0]
    losses = [x for x in run.trade_pnls if x < 0]
    gross_profit = float(sum(wins))
    gross_loss = float(-sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)
    n_trades = len(run.trade_pnls)
    win_rate = len(wins) / n_trades if n_trades else 0.0
    avg_profit = total_pnl / n_trades if n_trades else 0.0
    return {
        "strategy": run.name,
        "total_pnl": total_pnl,
        "final_equity": float(equity[-1]),
        "sharpe_ann": sharpe,
        "max_drawdown": max_drawdown(equity),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_profit_per_trade": avg_profit,
        "trades": n_trades,
        "signals": run.signals,
        "fills": run.fills,
    }


def load_spot_prices(symbol: str, source: str | None, dataset_source: str = "aliplayer_spot") -> pd.DataFrame:
    """Load spot prices from the specified Hugging Face dataset source.

    Args:
        symbol: Asset symbol (e.g., "btcusdt", "btc", "ethusdt").
        source: Filter by data source within the dataset (e.g., "binance", "chainlink_proxy").
        dataset_source: Which Hugging Face dataset to use:
            - "aliplayer_spot": aliplayer1/polymarket-crypto-updown (spot_prices config)
            - "bmoney_crypto": bmoney1321/polymarket-crypto-5m-15m (crypto_prices config)
            - "aliplayer_prices": aliplayer1/polymarket-crypto-updown (prices config)
    """
    print("=" * 100)
    print("DATA LOADING")

    if dataset_source == "bmoney_crypto":
        return _load_bmoney_crypto_prices(symbol)
    elif dataset_source == "aliplayer_prices":
        return _load_aliplayer_prices(symbol, source)
    else:
        return _load_aliplayer_spot_prices(symbol, source)


def _load_aliplayer_spot_prices(symbol: str, source: str | None) -> pd.DataFrame:
    """Load from aliplayer1/polymarket-crypto-updown spot_prices config (default)."""
    builder = load_dataset_builder(DATASET_NAME, DATASET_CONFIG)
    full_rows = builder.info.splits[DATASET_SPLIT].num_examples
    print(f"Dataset: {DATASET_NAME} / config={DATASET_CONFIG} / split={DATASET_SPLIT}")
    print(f"Full split rows: {full_rows:,}")
    print(f"Features: {builder.info.features}")

    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    print(f"Loaded dataset shape: ({ds.num_rows:,}, {len(ds.column_names)})")
    print("Head:")
    print(pd.DataFrame(ds[:10]).to_string(index=False))

    timestamps: list[np.ndarray] = []
    prices: list[np.ndarray] = []
    sources_seen: dict[str, int] = {}
    symbols_seen: dict[str, int] = {}
    selected_rows = 0

    print(f"\nFiltering rows for symbol={symbol!r}" + (f", source={source!r}" if source else ""))
    for batch in ds.iter(batch_size=250_000):
        symbols = np.asarray(batch["symbol"], dtype=object)
        mask = np.char.lower(symbols.astype(str)) == symbol.lower()
        for s in symbols[:1000]:
            symbols_seen[str(s)] = symbols_seen.get(str(s), 0) + 1
        if source is not None:
            src = np.asarray(batch["source"], dtype=object)
            mask &= np.char.lower(src.astype(str)) == source.lower()
            for s in src[:1000]:
                sources_seen[str(s)] = sources_seen.get(str(s), 0) + 1
        if not mask.any():
            continue
        ts = np.asarray(batch["ts_ms"], dtype=np.int64)[mask]
        px = np.asarray(batch["price"], dtype=np.float64)[mask]
        timestamps.append(ts)
        prices.append(px)
        selected_rows += len(ts)

    if selected_rows == 0:
        raise RuntimeError(
            f"No rows found for symbol={symbol!r}, source={source!r}. "
            "Try --source none, --symbol btcusdt, or inspect the printed dataset head."
        )

    raw = pd.DataFrame(
        {
            "ts_ms": np.concatenate(timestamps),
            "price": np.concatenate(prices),
        }
    )
    raw["timestamp"] = pd.to_datetime(raw["ts_ms"], unit="ms", utc=True)
    raw = raw.sort_values("timestamp")

    print(f"Filtered rows: {len(raw):,}")
    print(f"Raw time range: {raw['timestamp'].min()} -> {raw['timestamp'].max()}")
    print("Filtered head:")
    print(raw.head(10).to_string(index=False))

    duplicate_ts = int(raw.duplicated("timestamp").sum())
    print(f"Duplicate timestamp rows before aggregation: {duplicate_ts:,}")

    raw = raw.groupby("timestamp", as_index=True)["price"].last().to_frame()
    diffs = raw.index.to_series().diff().dropna().dt.total_seconds()
    gap_count = int((diffs > 1).sum())
    max_gap = float(diffs.max()) if len(diffs) else 0.0
    print(f"Observed >1s gaps before resampling: {gap_count:,}")
    print(f"Max observed gap before resampling: {max_gap:,.0f}s")

    resampled = raw.resample("1s").last()
    missing_seconds = int(resampled["price"].isna().sum())
    resampled["price"] = resampled["price"].ffill().bfill()
    print(f"Uniform 1-second shape after resampling: {resampled.shape}")
    print(f"Forward/back-filled seconds: {missing_seconds:,}")
    print(f"Uniform time range: {resampled.index.min()} -> {resampled.index.max()}")
    print("=" * 100)
    return resampled


def _load_bmoney_crypto_prices(symbol: str) -> pd.DataFrame:
    """Load from bmoney1321/polymarket-crypto-5m-15m crypto_prices config.

    This provides 1-minute OHLCV candles from Binance for BTC, ETH, SOL, XRP.
    """
    asset_map = {
        "btcusdt": "BTC",
        "ethusdt": "ETH",
        "solusdt": "SOL",
        "xrpusdt": "XRP",
        "btc": "BTC",
        "eth": "ETH",
        "sol": "SOL",
        "xrp": "XRP",
    }
    asset = asset_map.get(symbol.lower(), symbol.upper())
    print(f"Loading bmoney crypto_prices for asset={asset}")

    ds = load_dataset("bmoney1321/polymarket-crypto-5m-15m", "crypto_prices", split="train")
    print(f"Loaded dataset shape: ({ds.num_rows:,}, {len(ds.column_names)})")

    df = ds.to_pandas()
    mask = df["asset"].str.upper() == asset
    df = df[mask].copy()
    print(f"Filtered to {asset}: {len(df):,} rows")

    if df.empty:
        raise RuntimeError(f"No rows found for asset={asset}. Available assets: {df['asset'].unique().tolist()}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    print(f"Time range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Columns: {df.columns.tolist()}")

    resampled = df.set_index("timestamp")["close"].resample("1s").last()
    missing_seconds = int(resampled.isna().sum())
    resampled = resampled.ffill().bfill().to_frame(name="price")
    print(f"Uniform 1-second shape after resampling: {resampled.shape}")
    print(f"Forward/back-filled seconds: {missing_seconds:,}")
    print("=" * 100)
    return resampled


def _load_aliplayer_prices(symbol: str, source: str | None) -> pd.DataFrame:
    """Load from aliplayer1/polymarket-crypto-updown prices config (OHLC from CLOB)."""
    asset_map = {
        "btcusdt": "BTC",
        "ethusdt": "ETH",
        "solusdt": "SOL",
        "bnbusdt": "BNB",
        "xrpusdt": "XRP",
        "dogeusdt": "DOGE",
        "hypeusdt": "HYPE",
        "btc": "BTC",
        "eth": "ETH",
        "sol": "SOL",
        "bnb": "BNB",
        "xrp": "XRP",
        "doge": "DOGE",
        "hype": "HYPE",
    }
    crypto = asset_map.get(symbol.lower(), symbol.upper())
    print(f"Loading aliplayer prices for crypto={crypto}")

    ds = load_dataset("aliplayer1/polymarket-crypto-updown", "prices", split="train")
    print(f"Loaded dataset shape: ({ds.num_rows:,}, {len(ds.column_names)})")

    df = ds.to_pandas()
    mask = df["crypto"].str.upper() == crypto
    df = df[mask].copy()
    print(f"Filtered to {crypto}: {len(df):,} rows")

    if df.empty:
        raise RuntimeError(f"No rows found for crypto={crypto}. Available: {df['crypto'].unique().tolist()}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values("timestamp")

    resampled = df.set_index("timestamp")["up_price"].resample("1s").last()
    missing_seconds = int(resampled.isna().sum())
    resampled = resampled.ffill().bfill().to_frame(name="price")
    print(f"Uniform 1-second shape after resampling: {resampled.shape}")
    print(f"Forward/back-filled seconds: {missing_seconds:,}")
    print("=" * 100)
    return resampled


def build_windows(price_df: pd.DataFrame, max_windows: int | None = None) -> list[Window]:
    print("\nBUILDING 5-MINUTE WINDOWS")
    windows: list[Window] = []
    tied = 0
    partial = 0
    for start, group in price_df.groupby(pd.Grouper(freq="5min", origin="epoch")):
        if len(group) != WINDOW_SECONDS:
            partial += 1
            continue
        prices = group["price"].to_numpy(dtype=float)
        open_price = float(prices[0])
        close_price = float(prices[-1])
        if close_price == open_price:
            tied += 1
            continue
        outcome_up = int(close_price > open_price)
        fair_up = fair_prices_for_window(prices, open_price)
        up_bid, up_ask, down_bid, down_ask = make_market_arrays(fair_up)
        windows.append(
            Window(
                start=start,
                prices=prices,
                open_price=open_price,
                close_price=close_price,
                outcome_up=outcome_up,
                fair_up=fair_up,
                up_bid=up_bid,
                up_ask=up_ask,
                down_bid=down_bid,
                down_ask=down_ask,
            )
        )
        if max_windows is not None and len(windows) >= max_windows:
            break

    if not windows:
        raise RuntimeError("No complete non-tie 5-minute windows were produced.")

    up_rate = np.mean([w.outcome_up for w in windows])
    print(f"Complete non-tie windows used: {len(windows):,}")
    print(f"Excluded partial windows: {partial:,}")
    print(f"Excluded exact ties: {tied:,}")
    print(f"Outcome Up rate: {pct(float(up_rate))}")
    print("Window sample:")
    sample = pd.DataFrame(
        {
            "start": [w.start for w in windows[:5]],
            "open_price": [w.open_price for w in windows[:5]],
            "close_price": [w.close_price for w in windows[:5]],
            "outcome_up": [w.outcome_up for w in windows[:5]],
        }
    )
    print(sample.to_string(index=False))
    print("=" * 100)
    return windows


def count_crossings(prices: np.ndarray, line: float, direction: str) -> int:
    if direction == "above":
        state = prices > line
    elif direction == "below":
        state = prices < line
    else:
        raise ValueError(direction)
    return int(np.sum(state & ~np.r_[False, state[:-1]]))


def analyze_target_crossings(windows: list[Window]) -> pd.DataFrame:
    print("\nTARGET-LINE CROSSING ANALYSIS")
    rows: list[dict] = []
    distributions: list[dict] = []

    for threshold in TARGET_THRESHOLDS:
        up_counts = []
        down_counts = []
        either_counts = []
        for w in windows:
            up_line = w.open_price * (1.0 + threshold)
            down_line = w.open_price * (1.0 - threshold)
            up_c = count_crossings(w.prices, up_line, "above")
            down_c = count_crossings(w.prices, down_line, "below")
            up_counts.append(up_c)
            down_counts.append(down_c)
            either_counts.append(up_c + down_c)

        for label, counts in [("up", up_counts), ("down", down_counts), ("either", either_counts)]:
            arr = np.asarray(counts)
            row = {
                "threshold": f"{threshold:.2%}",
                "direction": label,
                "windows_>=1": int(np.sum(arr >= 1)),
                "windows_>=2": int(np.sum(arr >= 2)),
                "windows_>=3": int(np.sum(arr >= 3)),
                "windows_>=4": int(np.sum(arr >= 4)),
                "windows_>=5": int(np.sum(arr >= 5)),
                "mean_crossings": float(arr.mean()),
                "max_crossings": int(arr.max()),
            }
            rows.append(row)
            values, freqs = np.unique(arr, return_counts=True)
            dist = {"threshold": f"{threshold:.2%}", "direction": label}
            dist.update({f"count={int(v)}": int(f) for v, f in zip(values, freqs)})
            distributions.append(dist)

    summary = pd.DataFrame(rows)
    dist_df = pd.DataFrame(distributions).fillna(0)
    print("\nCrossing threshold summary:")
    print(summary.to_string(index=False))
    print("\nCrossing count distributions:")
    print(dist_df.to_string(index=False))
    print("=" * 100)
    return summary


def strat_1_end_momentum_sniper(w: Window) -> tuple[float, float | None, bool]:
    idx = WINDOW_SECONDS - 10
    fair_down = 1.0 - w.fair_up[idx]
    if w.fair_up[idx] > 0.95:
        pnl, fill_idx = limit_buy_settlement_pnl(w, idx, "up", 0.90)
        return pnl, pnl if fill_idx is not None else None, True
    if fair_down > 0.95:
        pnl, fill_idx = limit_buy_settlement_pnl(w, idx, "down", 0.90)
        return pnl, pnl if fill_idx is not None else None, True
    return 0.0, None, False


def strat_2_composite_score(w: Window) -> tuple[float, float | None, bool]:
    idx = 120
    p = w.prices
    window_delta = (p[idx] - p[0]) / p[0]
    micro_momentum = (p[idx] - p[idx - 60]) / p[idx - 60]
    rsi_val = rsi(p[: idx + 1], 14)
    rsi_score = 1.0 if rsi_val < 30 else (-1.0 if rsi_val > 70 else 0.0)
    active_last_10 = np.count_nonzero(np.diff(p[idx - 10 : idx + 1]))
    active_prev = np.count_nonzero(np.diff(p[: idx - 10]))
    avg_active_10 = active_prev / max((idx - 10) / 10.0, 1.0)
    spike_ratio = active_last_10 / max(avg_active_10, 1e-9)
    vol_score = np.sign(p[idx] - p[idx - 10]) if spike_ratio > 1.5 else 0.0
    score = 5.0 * window_delta + 2.0 * micro_momentum + rsi_score + vol_score
    if score > 0:
        pnl = market_buy_pnl(w, idx, "up")
        return pnl, pnl, True
    if score < 0:
        pnl = market_buy_pnl(w, idx, "down")
        return pnl, pnl, True
    return 0.0, None, False


def strat_3_oracle_lag_arbitrage(w: Window) -> tuple[float, float | None, bool]:
    p = w.prices
    for idx in range(2, WINDOW_SECONDS):
        slow = p[idx - 2]
        fast = p[idx]
        if abs(fast / slow - 1.0) > 0.002:
            side = "up" if fast > slow else "down"
            pnl = market_buy_pnl(w, idx, side)
            return pnl, pnl, True
    return 0.0, None, False


def strat_4_vol_spike_reversal(w: Window) -> tuple[float, float | None, bool]:
    p = w.prices
    pending: tuple[str, float] | None = None
    entry: tuple[str, float, int] | None = None

    for idx in range(1, WINDOW_SECONDS):
        if entry is None and pending is None:
            ret = p[idx] / p[idx - 1] - 1.0
            if abs(ret) > 0.0015:
                side = "down" if ret > 0 else "up"
                limit_price = float((1.0 - w.fair_up[idx]) if side == "down" else w.fair_up[idx])
                pending = (side, limit_price)

        if entry is None and pending is not None:
            side, limit_price = pending
            _, ask = side_arrays(w, side)
            if ask[idx] <= limit_price:
                entry = (side, limit_price, idx)
                pending = None

        if entry is not None:
            side, entry_price, _ = entry
            bid, _ = side_arrays(w, side)
            exit_price = float(bid[idx])
            ret_on_contract = exit_price / entry_price - 1.0
            if ret_on_contract >= 0.03 or ret_on_contract <= -0.02:
                buy_rebate = entry_price * DEFAULT_TRADE_QTY * MAKER_REBATE_RATE
                sell_fee = exit_price * DEFAULT_TRADE_QTY * TAKER_FEE_RATE
                pnl = (exit_price - entry_price) * DEFAULT_TRADE_QTY + buy_rebate - sell_fee
                return pnl, pnl, True

    if entry is not None:
        side, entry_price, _ = entry
        rebate = entry_price * DEFAULT_TRADE_QTY * MAKER_REBATE_RATE
        pnl = (settlement_value(w, side) - entry_price) * DEFAULT_TRADE_QTY + rebate
        return pnl, pnl, True
    return 0.0, None, pending is not None


def strat_5_market_maker(w: Window) -> tuple[float, list[float], bool, dict]:
    cash = 0.0
    inv_up = 0.0
    inv_down = 0.0
    rebates = 0.0
    spread_cash = 0.0
    fills = 0

    for q in range(0, WINDOW_SECONDS, 5):
        quote = {
            "up_bid": float(clip_contract_price(w.fair_up[q] - 0.01)),
            "up_ask": float(clip_contract_price(w.fair_up[q] + 0.01)),
            "down_bid": float(clip_contract_price((1.0 - w.fair_up[q]) - 0.01)),
            "down_ask": float(clip_contract_price((1.0 - w.fair_up[q]) + 0.01)),
        }
        filled = {k: False for k in quote}
        for idx in range(q + 1, min(q + 5, WINDOW_SECONDS)):
            if not filled["up_bid"] and w.up_ask[idx] <= quote["up_bid"]:
                price = quote["up_bid"]
                cash -= price * MM_ORDER_QTY
                inv_up += MM_ORDER_QTY
                rebates += price * MM_ORDER_QTY * MAKER_REBATE_RATE
                fills += 1
                filled["up_bid"] = True
            if not filled["up_ask"] and w.up_bid[idx] >= quote["up_ask"]:
                price = quote["up_ask"]
                cash += price * MM_ORDER_QTY
                inv_up -= MM_ORDER_QTY
                rebates += price * MM_ORDER_QTY * MAKER_REBATE_RATE
                spread_cash += price * MM_ORDER_QTY
                fills += 1
                filled["up_ask"] = True
            if not filled["down_bid"] and w.down_ask[idx] <= quote["down_bid"]:
                price = quote["down_bid"]
                cash -= price * MM_ORDER_QTY
                inv_down += MM_ORDER_QTY
                rebates += price * MM_ORDER_QTY * MAKER_REBATE_RATE
                fills += 1
                filled["down_bid"] = True
            if not filled["down_ask"] and w.down_bid[idx] >= quote["down_ask"]:
                price = quote["down_ask"]
                cash += price * MM_ORDER_QTY
                inv_down -= MM_ORDER_QTY
                rebates += price * MM_ORDER_QTY * MAKER_REBATE_RATE
                spread_cash += price * MM_ORDER_QTY
                fills += 1
                filled["down_ask"] = True

    settlement = inv_up * w.outcome_up + inv_down * (1 - w.outcome_up)
    pnl = cash + settlement + rebates
    # Count the whole window as one trade event if at least one quote filled.
    trade_pnls = [pnl] if fills else []
    return pnl, trade_pnls, fills > 0, {"fills": fills, "rebates": rebates, "spread_cash": spread_cash}


def strat_6_double_ma_crossover(w: Window) -> tuple[float, float | None, bool]:
    p = w.prices[:271]
    fast = ema(p, 15)
    slow = ema(p, 45)
    diff = fast - slow
    last_signal: tuple[int, str] | None = None
    last_signal_idx = -10_000
    for idx in range(1, len(diff)):
        if idx - last_signal_idx < 8:
            continue
        if diff[idx - 1] <= 0 < diff[idx]:
            last_signal = (idx, "up")
            last_signal_idx = idx
        elif diff[idx - 1] >= 0 > diff[idx]:
            last_signal = (idx, "down")
            last_signal_idx = idx
    if last_signal is None:
        return 0.0, None, False
    side = last_signal[1]
    pnl = market_buy_pnl(w, 270, side)
    return pnl, pnl, True


def strat_7_rsi_bollinger_reversion(w: Window) -> tuple[float, float | None, bool]:
    idx = 180
    p = w.prices[: idx + 1]
    rsi_val = rsi(p, 14)
    last20 = p[-20:]
    mid = float(last20.mean())
    std = float(last20.std(ddof=1))
    lower = mid - 2.0 * std
    upper = mid + 2.0 * std
    price = p[-1]
    if rsi_val < 30 and price <= lower * 1.001:
        limit_price = float(clip_contract_price(w.fair_up[idx] - 0.005))
        pnl, fill_idx = limit_buy_settlement_pnl(w, idx, "up", limit_price)
        return pnl, pnl if fill_idx is not None else None, True
    if rsi_val > 70 and price >= upper * 0.999:
        limit_price = float(clip_contract_price((1.0 - w.fair_up[idx]) - 0.005))
        pnl, fill_idx = limit_buy_settlement_pnl(w, idx, "down", limit_price)
        return pnl, pnl if fill_idx is not None else None, True
    return 0.0, None, False


def strat_8_opening_range_breakout(w: Window) -> tuple[float, float | None, bool]:
    high = float(w.prices[:30].max())
    low = float(w.prices[:30].min())
    for idx in range(120, WINDOW_SECONDS - 1):
        if w.prices[idx] > high and w.prices[idx + 1] > high:
            pnl = market_buy_pnl(w, idx + 1, "up")
            return pnl, pnl, True
        if w.prices[idx] < low and w.prices[idx + 1] < low:
            pnl = market_buy_pnl(w, idx + 1, "down")
            return pnl, pnl, True
    return 0.0, None, False


def strat_9_renko_sequence(w: Window) -> tuple[float, float | None, bool]:
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
        pnl = market_buy_pnl(w, idx, "up")
        return pnl, pnl, True
    if down_bricks - up_bricks >= 2:
        pnl = market_buy_pnl(w, idx, "down")
        return pnl, pnl, True
    return 0.0, None, False


def strat_10_volume_weighted_momentum(w: Window) -> tuple[float, float | None, bool]:
    idx = 120
    diffs = np.diff(w.prices[: idx + 1])
    up_vol = float(np.clip(diffs, 0, None).sum())
    down_vol = float((-np.clip(diffs, None, 0)).sum())
    total = up_vol + down_vol
    if total <= 0:
        return 0.0, None, False
    ratio = (up_vol - down_vol) / total
    sma30 = float(w.prices[idx - 29 : idx + 1].mean())
    if ratio > 0.2 and w.prices[idx] > sma30:
        pnl = market_buy_pnl(w, idx, "up")
        return pnl, pnl, True
    if ratio < -0.2 and w.prices[idx] < sma30:
        pnl = market_buy_pnl(w, idx, "down")
        return pnl, pnl, True
    return 0.0, None, False


def creative_11_late_shock_continuation(w: Window) -> tuple[float, float | None, bool]:
    idx = 255
    recent = w.prices[idx] / w.prices[idx - 15] - 1.0
    fair = w.fair_up[idx]
    if recent > 0.0008 and 0.55 <= fair <= 0.90:
        pnl = market_buy_pnl(w, idx, "up")
        return pnl, pnl, True
    if recent < -0.0008 and 0.10 <= fair <= 0.45:
        pnl = market_buy_pnl(w, idx, "down")
        return pnl, pnl, True
    return 0.0, None, False


def creative_12_wick_reclaim_fade(w: Window) -> tuple[float, float | None, bool]:
    idx = 200
    p = w.prices[: idx + 1]
    upper = w.open_price * 1.002
    lower = w.open_price * 0.998
    neutral_upper = w.open_price * 1.0005
    neutral_lower = w.open_price * 0.9995
    breached_up = np.any(p[:150] > upper)
    breached_down = np.any(p[:150] < lower)
    now = p[-1]
    if breached_up and neutral_lower <= now <= neutral_upper:
        pnl = market_buy_pnl(w, idx, "down")
        return pnl, pnl, True
    if breached_down and neutral_lower <= now <= neutral_upper:
        pnl = market_buy_pnl(w, idx, "up")
        return pnl, pnl, True
    return 0.0, None, False


def creative_13_chop_box_contrarian(w: Window) -> tuple[float, float | None, bool]:
    idx = 240
    p = w.prices
    crossed_up = np.any(p[:180] > w.open_price * 1.001)
    crossed_down = np.any(p[:180] < w.open_price * 0.999)
    if not (crossed_up and crossed_down):
        return 0.0, None, False
    last30 = p[idx] / p[idx - 30] - 1.0
    if last30 > 0:
        pnl = market_buy_pnl(w, idx, "down")
        return pnl, pnl, True
    if last30 < 0:
        pnl = market_buy_pnl(w, idx, "up")
        return pnl, pnl, True
    return 0.0, None, False


def creative_14_settlement_gravity(w: Window) -> tuple[float, float | None, bool]:
    idx = 285
    delta = w.prices[idx] / w.open_price - 1.0
    last10 = w.prices[idx] / w.prices[idx - 10] - 1.0
    # Tiny lead close to settlement often behaves like a coin-flip with expensive contracts.
    # Fade only when price is barely above/below the open and recent momentum is weakening.
    if 0.00005 < delta < 0.0004 and last10 < 0:
        pnl = market_buy_pnl(w, idx, "down", qty=500.0)
        return pnl, pnl, True
    if -0.0004 < delta < -0.00005 and last10 > 0:
        pnl = market_buy_pnl(w, idx, "up", qty=500.0)
        return pnl, pnl, True
    return 0.0, None, False


StrategyFn = Callable[[Window], tuple]


def run_simple_strategy(name: str, windows: list[Window], fn: StrategyFn) -> StrategyRun:
    pnl_by_window = np.zeros(len(windows), dtype=float)
    trade_pnls: list[float] = []
    signals = 0
    fills = 0
    for i, w in enumerate(windows):
        result = fn(w)
        pnl = float(result[0])
        trade_pnl = result[1]
        signaled = bool(result[2])
        pnl_by_window[i] = pnl
        if signaled:
            signals += 1
        if trade_pnl is not None:
            fills += 1
            trade_pnls.append(float(trade_pnl))
    wins = sum(1 for x in trade_pnls if x > 0)
    return StrategyRun(name=name, pnl_by_window=pnl_by_window, trade_pnls=trade_pnls, fills=fills, signals=signals, wins=wins)


def run_market_maker(windows: list[Window]) -> StrategyRun:
    pnl_by_window = np.zeros(len(windows), dtype=float)
    trade_pnls: list[float] = []
    signals = 0
    fills = 0
    total_rebates = 0.0
    total_spread_cash = 0.0
    for i, w in enumerate(windows):
        pnl, trade_list, signaled, extra = strat_5_market_maker(w)
        pnl_by_window[i] = pnl
        if signaled:
            signals += 1
        if trade_list:
            trade_pnls.extend(float(x) for x in trade_list)
        fills += int(extra["fills"])
        total_rebates += float(extra["rebates"])
        total_spread_cash += float(extra["spread_cash"])
    return StrategyRun(
        name="05 Market Making & Spread Capture",
        pnl_by_window=pnl_by_window,
        trade_pnls=trade_pnls,
        fills=fills,
        signals=signals,
        wins=sum(1 for x in trade_pnls if x > 0),
        extra={"rebates": total_rebates, "spread_cash": total_spread_cash},
    )


def run_backtests(windows: list[Window], label: str = "FULL") -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\nRUNNING STRATEGY BACKTESTS: {label}")
    strategies: list[tuple[str, StrategyFn]] = [
        ("01 End-of-Window Momentum Sniper", strat_1_end_momentum_sniper),
        ("02 Composite Technical Score", strat_2_composite_score),
        ("03 Oracle Lag Arbitrage", strat_3_oracle_lag_arbitrage),
        ("04 Volatility Spike Reversal", strat_4_vol_spike_reversal),
        ("06 Double MA Crossover", strat_6_double_ma_crossover),
        ("07 RSI + Bollinger Reversion", strat_7_rsi_bollinger_reversion),
        ("08 Opening Range Breakout", strat_8_opening_range_breakout),
        ("09 Renko Change Sequencing", strat_9_renko_sequence),
        ("10 Volume Weighted Momentum", strat_10_volume_weighted_momentum),
        ("11 Creative Late Shock Continuation", creative_11_late_shock_continuation),
        ("12 Creative Wick Reclaim Fade", creative_12_wick_reclaim_fade),
        ("13 Creative Chop Box Contrarian", creative_13_chop_box_contrarian),
        ("14 Creative Settlement Gravity", creative_14_settlement_gravity),
    ]

    runs: list[StrategyRun] = []
    for name, fn in strategies:
        print(f"  {name}")
        runs.append(run_simple_strategy(name, windows, fn))
    print("  05 Market Making & Spread Capture")
    runs.insert(4, run_market_maker(windows))

    equity_data: dict[str, np.ndarray] = {}
    metrics_rows: list[dict] = []
    for run in runs:
        equity = STARTING_CAPITAL + np.cumsum(run.pnl_by_window)
        equity_data[run.name] = equity
        metrics_rows.append(strategy_metrics(run, equity))

    metrics = pd.DataFrame(metrics_rows).sort_values("total_pnl", ascending=False).reset_index(drop=True)
    metrics.insert(0, "sample", label)
    metrics["rating"] = rate_strategies(metrics)
    equity_df = pd.DataFrame(equity_data)
    equity_df.insert(0, "window_start", [w.start for w in windows])

    print(f"\nComparative strategy table ({label}):")
    printable = metrics.copy()
    money_cols = ["total_pnl", "final_equity", "avg_profit_per_trade"]
    for col in money_cols:
        printable[col] = printable[col].map(money)
    printable["max_drawdown"] = printable["max_drawdown"].map(pct)
    printable["win_rate"] = printable["win_rate"].map(pct)
    printable["sharpe_ann"] = printable["sharpe_ann"].map(lambda x: f"{x:.2f}")
    printable["profit_factor"] = printable["profit_factor"].map(lambda x: "inf" if np.isinf(x) else f"{x:.2f}")
    print(printable.to_string(index=False))
    print("=" * 100)
    return metrics, equity_df


def rate_strategies(metrics: pd.DataFrame) -> list[str]:
    ratings: list[str] = []
    for _, row in metrics.iterrows():
        pnl = row["total_pnl"]
        sharpe = row["sharpe_ann"]
        dd = row["max_drawdown"]
        trades = row["trades"]
        if trades < 10:
            ratings.append("D - too few trades")
        elif pnl > 0 and sharpe > 1.0 and dd > -0.25:
            ratings.append("A - promising")
        elif pnl > 0 and sharpe > 0.25:
            ratings.append("B - workable")
        elif pnl > 0:
            ratings.append("C - positive but weak")
        elif sharpe > -0.25:
            ratings.append("D - flat/noisy")
        else:
            ratings.append("F - avoid")
    return ratings


def split_windows_chronologically(windows: list[Window], train_frac: float) -> tuple[list[Window], list[Window]]:
    if not 0.05 <= train_frac <= 0.95:
        raise ValueError("--train-frac must be between 0.05 and 0.95")
    split_idx = int(len(windows) * train_frac)
    split_idx = min(max(split_idx, 1), len(windows) - 1)
    train_windows = windows[:split_idx]
    test_windows = windows[split_idx:]

    print("\nNO-MEMORY CHRONOLOGICAL SPLIT")
    print(f"Train fraction: {train_frac:.2%}")
    print(f"Train windows: {len(train_windows):,} | {train_windows[0].start} -> {train_windows[-1].start}")
    print(f"Test windows:  {len(test_windows):,} | {test_windows[0].start} -> {test_windows[-1].start}")
    print("No-memory rule: train and test are backtested in separate calls with fresh $10,000 capital.")
    print("No positions, equity, trade bookkeeping, fitted params, or indicator buffers cross the split boundary.")
    print("=" * 100)
    return train_windows, test_windows


def compare_train_test(train_metrics: pd.DataFrame, test_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "strategy",
        "total_pnl",
        "sharpe_ann",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "avg_profit_per_trade",
        "trades",
        "signals",
        "fills",
        "rating",
    ]
    merged = train_metrics[metric_cols].merge(
        test_metrics[metric_cols],
        on="strategy",
        suffixes=("_train", "_test"),
    )
    merged["pnl_decay_test_vs_train"] = merged["total_pnl_test"] / merged["total_pnl_train"].replace(0, np.nan)
    merged["survived_oos"] = (
        (merged["total_pnl_train"] > 0)
        & (merged["total_pnl_test"] > 0)
        & (merged["trades_test"] >= 10)
    )
    merged["verdict"] = np.where(
        merged["survived_oos"],
        "PASS - profitable in train and test",
        "FAIL - did not survive cleanly",
    )
    return merged.sort_values(["survived_oos", "total_pnl_test"], ascending=[False, False]).reset_index(drop=True)


def print_split_comparison(split_comparison: pd.DataFrame) -> None:
    print("\nTRAIN/TEST NO-MEMORY COMPARISON")
    printable = split_comparison.copy()
    for col in ["total_pnl_train", "total_pnl_test", "avg_profit_per_trade_train", "avg_profit_per_trade_test"]:
        printable[col] = printable[col].map(money)
    for col in ["max_drawdown_train", "max_drawdown_test", "win_rate_train", "win_rate_test"]:
        printable[col] = printable[col].map(pct)
    for col in ["sharpe_ann_train", "sharpe_ann_test", "pnl_decay_test_vs_train"]:
        printable[col] = printable[col].map(lambda x: "nan" if pd.isna(x) else f"{x:.2f}")
    for col in ["profit_factor_train", "profit_factor_test"]:
        printable[col] = printable[col].map(lambda x: "inf" if np.isinf(x) else f"{x:.2f}")
    print(printable.to_string(index=False))
    print("=" * 100)


def print_oos_findings(split_comparison: pd.DataFrame) -> None:
    print("\nOUT-OF-SAMPLE SURVIVORS")
    survivors = split_comparison[split_comparison["survived_oos"]].copy()
    if survivors.empty:
        print("No strategy was profitable in both train and test with at least 10 test trades.")
    else:
        for i, row in survivors.iterrows():
            print(
                f"{i + 1:02d}. {row['strategy']}: "
                f"train={money(row['total_pnl_train'])}, test={money(row['total_pnl_test'])}, "
                f"test win={pct(row['win_rate_test'])}, test trades={int(row['trades_test'])}"
            )

    failed_train_winners = split_comparison[
        (split_comparison["total_pnl_train"] > 0) & (~split_comparison["survived_oos"])
    ]
    if not failed_train_winners.empty:
        print("\nTrain winners that did not cleanly survive test:")
        for _, row in failed_train_winners.iterrows():
            print(
                f"- {row['strategy']}: train={money(row['total_pnl_train'])}, "
                f"test={money(row['total_pnl_test'])}, test trades={int(row['trades_test'])}"
            )
    print("=" * 100)


def plot_equity_curves(equity_df: pd.DataFrame, output_path: Path, title_suffix: str = "") -> None:
    plt.figure(figsize=(16, 9))
    x = equity_df["window_start"]
    for col in equity_df.columns:
        if col == "window_start":
            continue
        plt.plot(x, equity_df[col], linewidth=1.2, alpha=0.9, label=col)
    plt.axhline(STARTING_CAPITAL, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
    title = "Simulated Polymarket 5-Minute Up/Down Strategy Equity Curves"
    if title_suffix:
        title += f" - {title_suffix}"
    plt.title(title)
    plt.xlabel("Window start")
    plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"Saved equity curve chart: {output_path.resolve()}")


def save_outputs(metrics: pd.DataFrame, equity_df: pd.DataFrame, suffix: str = "full") -> None:
    metrics_path = f"strategy_comparison_{suffix}.csv"
    equity_path = f"equity_curves_{suffix}.csv"
    chart_path = f"equity_curves_{suffix}.png"
    metrics.to_csv(metrics_path, index=False)
    equity_df.to_csv(equity_path, index=False)
    plot_equity_curves(equity_df, Path(chart_path), suffix.upper())
    print(f"Saved metrics CSV: {metrics_path}")
    print(f"Saved equity CSV: {equity_path}")


def print_key_findings(metrics: pd.DataFrame) -> None:
    print("\nKEY FINDINGS / STRATEGY RATINGS")
    ranked = metrics.sort_values("total_pnl", ascending=False).reset_index(drop=True)
    for i, row in ranked.iterrows():
        print(
            f"{i + 1:02d}. {row['strategy']}: {row['rating']} | "
            f"PnL={money(row['total_pnl'])}, Sharpe={row['sharpe_ann']:.2f}, "
            f"Win={pct(row['win_rate'])}, Trades={int(row['trades'])}"
        )

    best = ranked.iloc[0]
    worst = ranked.iloc[-1]
    print("\nInterpretation:")
    print(
        f"- Best by total PnL: {best['strategy']} ({money(best['total_pnl'])}). "
        "Treat this as a simulated edge only; the fill model and fair-value model are simplified."
    )
    print(
        f"- Weakest by total PnL: {worst['strategy']} ({money(worst['total_pnl'])}). "
        "Negative results usually mean the signal is paying too much spread/taker fee or buying late."
    )
    print(
        "- Creative strategies tested here: Late Shock Continuation, Wick Reclaim Fade, "
        "Chop Box Contrarian, and Settlement Gravity. Keep any positive one as a research candidate, "
        "not as live-trading proof."
    )
    print("=" * 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest Polymarket crypto up/down strategies.")
    parser.add_argument("--symbol", default="btcusdt", help="Spot symbol to backtest, e.g. btcusdt or btc/usd.")
    parser.add_argument(
        "--source",
        default="binance",
        help="Source filter. Use 'none' to combine all sources after duplicate aggregation.",
    )
    parser.add_argument(
        "--dataset-source",
        default="aliplayer_spot",
        choices=["aliplayer_spot", "bmoney_crypto", "aliplayer_prices"],
        help="Which Hugging Face dataset to use for price data.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Optional cap for quicker experiments. Default uses all complete windows.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.70,
        help="Chronological train fraction for no-memory out-of-sample validation.",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Run one full-sample backtest instead of the default train/test no-memory split.",
    )
    parser.add_argument(
        "--also-full",
        action="store_true",
        help="After train/test validation, also run and save a full-sample backtest.",
    )
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    source = None if str(args.source).lower() in {"none", "all", ""} else args.source

    price_df = load_spot_prices(args.symbol, source, dataset_source=args.dataset_source)
    windows = build_windows(price_df, args.max_windows)
    analyze_target_crossings(windows)

    if args.no_split:
        metrics, equity_df = run_backtests(windows, "FULL")
        save_outputs(metrics, equity_df, "full")
        print_key_findings(metrics)
        return

    train_windows, test_windows = split_windows_chronologically(windows, args.train_frac)
    train_metrics, train_equity_df = run_backtests(train_windows, "TRAIN")
    test_metrics, test_equity_df = run_backtests(test_windows, "TEST")

    save_outputs(train_metrics, train_equity_df, "train")
    save_outputs(test_metrics, test_equity_df, "test")

    split_comparison = compare_train_test(train_metrics, test_metrics)
    split_comparison.to_csv("strategy_train_test_comparison.csv", index=False)
    print("Saved split comparison CSV: strategy_train_test_comparison.csv")
    print_split_comparison(split_comparison)
    print_oos_findings(split_comparison)

    if args.also_full:
        full_metrics, full_equity_df = run_backtests(windows, "FULL")
        save_outputs(full_metrics, full_equity_df, "full")
        print_key_findings(full_metrics)


if __name__ == "__main__":
    main()
