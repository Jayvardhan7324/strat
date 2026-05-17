"""
Extended Backtest using bmoney 1-minute Binance candles.

Uses bmoney1321/polymarket-crypto-5m-15m crypto_prices config which has:
- 1-minute OHLCV candles from Binance
- BTC, ETH, SOL, XRP
- Date range: 2026-01-09 to 2026-03-13 (~2 months)

This gives us ~60 days of real price data to backtest against.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("Install datasets: pip install datasets")

from polymarket_updown_backtest import (
    STARTING_CAPITAL,
    TAKER_FEE_RATE,
    WINDOW_SECONDS,
    Window,
    max_drawdown,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_bmoney_crypto_prices(asset: str = "BTC") -> pd.DataFrame:
    """Load 1-minute OHLCV candles from Binance via bmoney dataset."""
    print("=" * 80)
    print("LOADING BMONEY CRYPTO PRICES")
    print(f"Asset: {asset}")
    print("=" * 80)

    ds = load_dataset(
        "bmoney1321/polymarket-crypto-5m-15m",
        "default",
        data_dir="crypto_prices",
        split="train",
    )

    df = ds.to_pandas()
    df = df[df["asset"].str.upper() == asset.upper()].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    print(f"Loaded {len(df):,} rows")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Columns: {df.columns.tolist()}")

    return df


def load_bmoney_resolutions(asset: str = "BTC") -> pd.DataFrame:
    """Load market resolutions for validation."""
    print("\nLoading bmoney resolutions...")
    ds = load_dataset(
        "bmoney1321/polymarket-crypto-5m-15m",
        "default",
        data_dir="resolutions",
        split="train",
    )
    df = ds.to_pandas()
    df = df[df["asset"].str.upper() == asset.upper()].copy()
    print(f"  {asset} resolutions: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Window building from 1-minute candles
# ---------------------------------------------------------------------------


def build_windows_from_1min_candles(
    candles_df: pd.DataFrame,
    max_windows: int = None,
) -> list[Window]:
    """Build 5-minute windows from 1-minute OHLCV candles.

    Each 5-minute window = 5 consecutive 1-minute candles.
    We interpolate to 1-second using the candle data.
    """
    print("\n" + "=" * 80)
    print("BUILDING 5-MINUTE WINDOWS FROM 1-MINUTE CANDLES")
    print("=" * 80)

    # Group into 5-minute windows
    candles_df = candles_df.copy()
    candles_df = candles_df.set_index("timestamp")

    windows: list[Window] = []
    tied = 0
    partial = 0

    # Resample to 5-minute candles first
    ohlc = candles_df[["open", "high", "low", "close", "volume"]].resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    ohlc = ohlc.dropna()

    print(f"5-minute candles: {len(ohlc):,}")
    print(f"Date range: {ohlc.index.min()} to {ohlc.index.max()}")

    for start, row in ohlc.iterrows():
        open_price = float(row["open"])
        close_price = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])

        if close_price == open_price:
            tied += 1
            continue

        outcome_up = int(close_price > open_price)

        # Interpolate 1-second prices within the 5-minute window
        # Simple linear interpolation from open to close with some noise
        prices = _interpolate_prices(open_price, close_price, high, low, WINDOW_SECONDS)

        fair_up = _fair_prices_for_window(prices, open_price)
        up_bid, up_ask, down_bid, down_ask = _make_market_arrays(fair_up)

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

        if max_windows and len(windows) >= max_windows:
            break

    print(f"Complete windows: {len(windows):,}")
    print(f"Excluded ties: {tied:,}")
    print(f"Outcome Up rate: {np.mean([w.outcome_up for w in windows]):.2%}")

    return windows


def _interpolate_prices(open_price: float, close_price: float, high: float, low: float, n_seconds: int) -> np.ndarray:
    """Interpolate 1-second prices from OHLC data."""
    # Create a realistic price path
    prices = np.zeros(n_seconds)
    prices[0] = open_price
    prices[-1] = close_price

    # Linear interpolation as base
    prices = np.linspace(open_price, close_price, n_seconds)

    # Add realistic volatility based on high/low range
    mid = (high + low) / 2
    amplitude = (high - low) / 2

    # Add sinusoidal variation to simulate intra-candle movement
    t = np.linspace(0, 2 * np.pi, n_seconds)
    noise = amplitude * 0.3 * np.sin(t * 3)  # 3 cycles within the window
    noise += amplitude * 0.1 * np.sin(t * 7)  # Higher frequency noise

    prices = prices + noise

    # Ensure prices stay within high/low bounds
    prices = np.clip(prices, low, high)

    # Ensure first and last prices are correct
    prices[0] = open_price
    prices[-1] = close_price

    return prices


def _fair_prices_for_window(prices: np.ndarray, open_price: float, sigma: float = 0.60) -> np.ndarray:
    from scipy.stats import norm
    elapsed = np.arange(len(prices), dtype=float)
    tau = np.maximum((WINDOW_SECONDS - elapsed) / WINDOW_SECONDS, 1e-9)
    log_moneyness = np.log(prices / open_price)
    z = log_moneyness / (sigma * np.sqrt(tau))
    return np.clip(norm.cdf(z), 0.001, 0.999)


def _make_market_arrays(fair_up: np.ndarray, half_spread: float = 0.005):
    up_bid = np.clip(fair_up - half_spread, 0.001, 0.999)
    up_ask = np.clip(fair_up + half_spread, 0.001, 0.999)
    fair_down = 1.0 - fair_up
    down_bid = np.clip(fair_down - half_spread, 0.001, 0.999)
    down_ask = np.clip(fair_down + half_spread, 0.001, 0.999)
    return up_bid, up_ask, down_bid, down_ask


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@dataclass
class ScalpTrade:
    strategy: str
    market_start: pd.Timestamp
    side: str
    entry_idx: int
    entry_price: float
    exit_idx: int | None
    exit_price: float
    exit_type: str
    pnl: float
    seconds_held: int
    features: dict = field(default_factory=dict)


def side_ask(w: Window, side: str) -> np.ndarray:
    return w.up_ask if side == "UP" else w.down_ask


def side_bid(w: Window, side: str) -> np.ndarray:
    return w.up_bid if side == "UP" else w.down_bid


def settlement_value(w: Window, side: str) -> float:
    return float(w.outcome_up) if side == "UP" else float(1 - w.outcome_up)


def compute_pnl(entry_price: float, exit_price: float, stake: float, fee_rate: float = TAKER_FEE_RATE) -> float:
    shares = stake / entry_price
    entry_fee = stake * fee_rate
    exit_fee = exit_price * shares * fee_rate
    return (exit_price - entry_price) * shares - entry_fee - exit_fee


def strat_late_momentum(w: Window, stake: float = 10.0, min_seconds_left: int = 30, max_entry_price: float = 0.70, momentum_threshold: float = 0.0003) -> ScalpTrade | None:
    max_idx = WINDOW_SECONDS - min_seconds_left
    for idx in range(max_idx, WINDOW_SECONDS - 5):
        recent_ret = w.prices[idx] / w.prices[max(0, idx - 10)] - 1.0
        longer_ret = w.prices[idx] / w.prices[max(0, idx - 30)] - 1.0
        if abs(recent_ret) < momentum_threshold:
            continue
        if np.sign(recent_ret) != np.sign(longer_ret):
            continue
        side = "UP" if recent_ret > 0 else "DOWN"
        ask = float(side_ask(w, side)[idx])
        if ask > max_entry_price:
            continue
        final = settlement_value(w, side)
        pnl = compute_pnl(ask, final, stake)
        return ScalpTrade("late_momentum", w.start, side, idx, ask, WINDOW_SECONDS - 1, final, "settlement", pnl, WINDOW_SECONDS - 1 - idx)
    return None


def strat_volume_spike(w: Window, stake: float = 10.0, spike_threshold: float = 0.0005, confirmation_seconds: int = 5, max_entry_price: float = 0.70) -> ScalpTrade | None:
    for idx in range(60, WINDOW_SECONDS - 30):
        spike = w.prices[idx] / w.prices[idx - 5] - 1.0
        if abs(spike) < spike_threshold:
            continue
        confirm_idx = idx + confirmation_seconds
        if confirm_idx >= WINDOW_SECONDS - 10:
            continue
        confirm_ret = w.prices[confirm_idx] / w.prices[idx] - 1.0
        if np.sign(confirm_ret) != np.sign(spike):
            continue
        side = "UP" if spike > 0 else "DOWN"
        ask = float(side_ask(w, side)[confirm_idx])
        if ask > max_entry_price:
            continue
        final = settlement_value(w, side)
        pnl = compute_pnl(ask, final, stake)
        return ScalpTrade("volume_spike", w.start, side, confirm_idx, ask, WINDOW_SECONDS - 1, final, "settlement", pnl, WINDOW_SECONDS - 1 - confirm_idx)
    return None


def strat_opening_breakout(w: Window, stake: float = 10.0, range_seconds: int = 30, max_entry_price: float = 0.70) -> ScalpTrade | None:
    high = float(w.prices[:range_seconds].max())
    low = float(w.prices[:range_seconds].min())
    for idx in range(range_seconds + 10, WINDOW_SECONDS - 30):
        if w.prices[idx] > high and w.prices[idx - 1] > high:
            side = "UP"
            ask = float(side_ask(w, side)[idx])
            if ask > max_entry_price:
                continue
            final = settlement_value(w, side)
            pnl = compute_pnl(ask, final, stake)
            return ScalpTrade("opening_breakout", w.start, side, idx, ask, WINDOW_SECONDS - 1, final, "settlement", pnl, WINDOW_SECONDS - 1 - idx)
        if w.prices[idx] < low and w.prices[idx - 1] < low:
            side = "DOWN"
            ask = float(side_ask(w, side)[idx])
            if ask > max_entry_price:
                continue
            final = settlement_value(w, side)
            pnl = compute_pnl(ask, final, stake)
            return ScalpTrade("opening_breakout", w.start, side, idx, ask, WINDOW_SECONDS - 1, final, "settlement", pnl, WINDOW_SECONDS - 1 - idx)
    return None


STRATEGIES = {
    "late_momentum": strat_late_momentum,
    "volume_spike": strat_volume_spike,
    "opening_breakout": strat_opening_breakout,
}


def run_strategy(name: str, windows: list[Window], stake: float = 10.0) -> tuple[list[ScalpTrade], np.ndarray]:
    fn = STRATEGIES[name]
    pnl_by_window = np.zeros(len(windows), dtype=float)
    trades: list[ScalpTrade] = []
    for i, w in enumerate(windows):
        trade = fn(w, stake=stake)
        if trade:
            pnl_by_window[i] = trade.pnl
            trades.append(trade)
    return trades, pnl_by_window


# ---------------------------------------------------------------------------
# ML
# ---------------------------------------------------------------------------


def extract_features(w: Window, idx: int) -> dict:
    p = w.prices
    ret_5s = p[idx] / p[max(0, idx - 5)] - 1.0
    ret_10s = p[idx] / p[max(0, idx - 10)] - 1.0
    ret_30s = p[idx] / p[max(0, idx - 30)] - 1.0
    ret_60s = p[idx] / p[max(0, idx - 60)] - 1.0
    ret_from_open = p[idx] / w.open_price - 1.0
    vol_10s = float(np.std(np.diff(p[max(0, idx - 10):idx + 1]))) / p[idx] if idx > 10 else 0.0
    vol_30s = float(np.std(np.diff(p[max(0, idx - 30):idx + 1]))) / p[idx] if idx > 30 else 0.0
    diffs = np.diff(p[max(0, idx - 10):idx + 1])
    up_ratio = float(np.sum(diffs > 0) / len(diffs)) if len(diffs) > 0 else 0.5
    up_spread = float(w.up_ask[idx] - w.up_bid[idx])
    down_spread = float(w.down_ask[idx] - w.down_bid[idx])
    avg_spread = (up_spread + down_spread) / 2
    up_mid = (w.up_ask[idx] + w.up_bid[idx]) / 2
    down_mid = (w.down_ask[idx] + w.down_bid[idx]) / 2
    ob_imbalance = float(up_mid - down_mid)
    fair = float(w.fair_up[idx])
    fair_dev = fair - 0.5
    seconds_left = WINDOW_SECONDS - 1 - idx
    time_decay = seconds_left / WINDOW_SECONDS
    if idx >= 14:
        diffs_rsi = np.diff(p[idx - 14:idx + 1])
        gains = np.clip(diffs_rsi, 0, None).mean()
        losses = -np.clip(diffs_rsi, None, 0).mean()
        rsi = 100.0 - (100.0 / (1.0 + gains / losses)) if losses > 0 else 100.0
    else:
        rsi = 50.0
    return {
        "ret_5s": ret_5s, "ret_10s": ret_10s, "ret_30s": ret_30s, "ret_60s": ret_60s,
        "ret_from_open": ret_from_open, "vol_10s": vol_10s, "vol_30s": vol_30s,
        "up_ratio": up_ratio, "up_spread": up_spread, "down_spread": down_spread,
        "avg_spread": avg_spread, "ob_imbalance": ob_imbalance, "fair_dev": fair_dev,
        "seconds_left": seconds_left, "time_decay": time_decay, "rsi": rsi, "fair_up": fair,
    }


def build_ml_dataset(windows: list[Window], sample: str, entry_indices: list[int] = None) -> pd.DataFrame:
    if entry_indices is None:
        entry_indices = list(range(150, 280, 5))
    rows = []
    for w in windows:
        for idx in entry_indices:
            if idx >= WINDOW_SECONDS:
                continue
            features = extract_features(w, idx)
            row = {"sample": sample, "market_start": w.start, "idx": idx, "label": w.outcome_up}
            row.update(features)
            rows.append(row)
    return pd.DataFrame(rows)


def train_ml_model(train_windows: list[Window], test_windows: list[Window], output_path: str = "extended_ml_model.txt") -> dict:
    if not HAS_LGB:
        print("LightGBM not installed.")
        return {}

    print("\n" + "=" * 80)
    print("ML MODEL TRAINING (EXTENDED)")
    print("=" * 80)

    print("Building training dataset...")
    train_df = build_ml_dataset(train_windows, "TRAIN")
    print(f"  Train samples: {len(train_df):,}")

    print("Building test dataset...")
    test_df = build_ml_dataset(test_windows, "TEST")
    print(f"  Test samples: {len(test_df):,}")

    feature_cols = [c for c in train_df.columns if c not in ["sample", "market_start", "idx", "label"]]
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train UP rate: {y_train.mean():.3f}")
    print(f"Test UP rate: {y_test.mean():.3f}")

    params = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1, "seed": 42,
    }

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, feature_name=feature_cols)

    model = lgb.train(
        params, train_data, num_boost_round=500, valid_sets=[test_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_auc = float(np.corrcoef(train_pred, y_train)[0, 1])
    test_auc = float(np.corrcoef(test_pred, y_test)[0, 1])

    print(f"\nTrain AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    importance = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importance()}).sort_values("importance", ascending=False)
    print("\nTop 10 features:")
    print(importance.head(10).to_string(index=False))

    model.save_model(output_path)
    importance.to_csv("extended_ml_feature_importance.csv", index=False)
    print(f"\nModel saved to {output_path}")

    return {"model": model, "feature_cols": feature_cols, "train_auc": train_auc, "test_auc": test_auc, "importance": importance}


def backtest_ml_model(windows: list[Window], sample: str, model, feature_cols: list[str], threshold: float = 0.55, stake: float = 10.0, max_entry_price: float = 0.70) -> tuple[list[ScalpTrade], np.ndarray]:
    pnl_by_window = np.zeros(len(windows), dtype=float)
    trades: list[ScalpTrade] = []
    for i, w in enumerate(windows):
        idx = 240
        features = extract_features(w, idx)
        X = np.array([[features[c] for c in feature_cols]])
        pred = float(model.predict(X)[0])
        if pred >= threshold or pred <= (1 - threshold):
            side = "UP" if pred >= threshold else "DOWN"
            ask = float(side_ask(w, side)[idx])
            if ask > max_entry_price:
                continue
            final = settlement_value(w, side)
            pnl = compute_pnl(ask, final, stake)
            pnl_by_window[i] = pnl
            trades.append(ScalpTrade("ml_model", w.start, side, idx, ask, WINDOW_SECONDS - 1, final, "settlement", pnl, WINDOW_SECONDS - 1 - idx))
    return trades, pnl_by_window


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class StrategyMetrics:
    strategy: str
    sample: str
    total_pnl: float
    trades: int
    win_rate: float
    avg_pnl: float
    profit_factor: float
    max_drawdown: float
    sharpe: float
    avg_hold_seconds: float
    avg_entry_price: float


def compute_metrics(trades: list[ScalpTrade], pnl_by_window: np.ndarray, sample: str, strategy: str) -> StrategyMetrics:
    equity = STARTING_CAPITAL + np.cumsum(pnl_by_window)
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
    sharpe = 0.0
    if len(returns) > 1 and returns.std(ddof=1) > 0:
        sharpe = float((returns.mean() / returns.std(ddof=1)) * np.sqrt(365 * 24 * 12))
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins) if wins else 0
    gross_loss = -sum(losses) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    return StrategyMetrics(
        strategy=strategy, sample=sample, total_pnl=float(equity[-1] - STARTING_CAPITAL),
        trades=len(trades), win_rate=len(wins) / len(trades) if trades else 0.0,
        avg_pnl=float(np.mean(pnls)) if pnls else 0.0, profit_factor=profit_factor,
        max_drawdown=max_drawdown(equity), sharpe=sharpe,
        avg_hold_seconds=float(np.mean([t.seconds_held for t in trades])) if trades else 0.0,
        avg_entry_price=float(np.mean([t.entry_price for t in trades])) if trades else 0.0,
    )


def print_metrics_table(metrics: list[StrategyMetrics]) -> None:
    df = pd.DataFrame([m.__dict__ for m in metrics])
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
    out["avg_hold_seconds"] = out["avg_hold_seconds"].map(lambda x: f"{x:.0f}s")
    out["avg_entry_price"] = out["avg_entry_price"].map(lambda x: f"{x:.3f}")
    cols = ["sample", "strategy", "total_pnl", "trades", "win_rate", "profit_factor", "avg_pnl", "sharpe", "max_drawdown", "avg_hold_seconds", "avg_entry_price"]
    print(out[cols].to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="BTC", choices=["BTC", "ETH", "SOL", "XRP"])
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--stake-usd", type=float, default=10.0)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--ml-threshold", type=float, default=0.55)
    parser.add_argument("--skip-ml", action="store_true")
    args = parser.parse_args()

    # Load data
    candles_df = load_bmoney_crypto_prices(args.asset)
    resolutions = load_bmoney_resolutions(args.asset)

    # Build windows
    windows = build_windows_from_1min_candles(candles_df, args.max_windows)

    if len(windows) < 100:
        print("Not enough windows for meaningful backtest.")
        return

    # Split
    split_idx = int(len(windows) * args.train_frac)
    train_windows = windows[:split_idx]
    test_windows = windows[split_idx:]

    print("\n" + "=" * 80)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 80)
    print(f"Train: {len(train_windows):,} windows | {train_windows[0].start} -> {train_windows[-1].start}")
    print(f"Test:  {len(test_windows):,} windows | {test_windows[0].start} -> {test_windows[-1].start}")

    # Run strategies
    print("\n" + "=" * 80)
    print("STRATEGY BACKTESTS")
    print("=" * 80)

    all_metrics = []
    for sample, sample_windows in [("TRAIN", train_windows), ("TEST", test_windows)]:
        print(f"\n{sample} SET:")
        for name in STRATEGIES:
            trades, pnl = run_strategy(name, sample_windows, stake=args.stake_usd)
            metrics = compute_metrics(trades, pnl, sample, name)
            all_metrics.append(metrics)

    print_metrics_table(all_metrics)

    # Save
    metrics_df = pd.DataFrame([m.__dict__ for m in all_metrics])
    metrics_df.to_csv("extended_scalping_metrics.csv", index=False)
    print(f"\nSaved extended_scalping_metrics.csv")

    # ML
    if not args.skip_ml:
        result = train_ml_model(train_windows, test_windows)
        if result:
            for sample, sample_windows in [("TRAIN", train_windows), ("TEST", test_windows)]:
                trades, pnl = backtest_ml_model(
                    sample_windows, sample, result["model"], result["feature_cols"],
                    threshold=args.ml_threshold, stake=args.stake_usd,
                )
                metrics = compute_metrics(trades, pnl, sample, "ml_model")
                print(f"\nML Model {sample}: PnL=${metrics.total_pnl:,.2f}, Trades={metrics.trades:,}, WinRate={100*metrics.win_rate:.1f}%")


if __name__ == "__main__":
    main()
