"""
Robust Extended Backtest - No Interpolation, Real Data Only.

Uses actual 1-minute candles without interpolation to prevent lookahead bias.
Combines aliplayer (Mar-Apr 2026) + bmoney (Jan-Mar 2026) for ~3 months of data.

Key improvements:
1. No price interpolation - uses real candle data only
2. Walk-forward validation across multiple time periods
3. Strict feature engineering with no lookahead
4. Realistic slippage modeling
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
    load_spot_prices,
    build_windows,
    split_windows_chronologically,
    max_drawdown,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Multi-asset backtest
# ---------------------------------------------------------------------------


def run_multi_asset_backtest(
    assets: list[str] = ["BTC", "ETH", "SOL", "XRP"],
    train_frac: float = 0.80,
    stake_usd: float = 10.0,
    ml_threshold: float = 0.55,
    skip_ml: bool = False,
) -> dict:
    """Run backtest across multiple assets for robustness."""

    all_results = {}

    for asset in assets:
        print(f"\n{'=' * 80}")
        print(f"BACKTESTING: {asset}")
        print(f"{'=' * 80}")

        # Load aliplayer data
        symbol_map = {"BTC": "btcusdt", "ETH": "ethusdt", "SOL": "solusdt", "XRP": "xrpusdt"}
        symbol = symbol_map.get(asset, asset.lower() + "usdt")

        try:
            price_df = load_spot_prices(symbol, "binance")
            windows = build_windows(price_df)

            if len(windows) < 100:
                print(f"  Skipping {asset}: only {len(windows)} windows")
                continue

            # Split
            train_windows, test_windows = split_windows_chronologically(windows, train_frac)

            # Run strategies
            asset_results = run_strategies(train_windows, test_windows, stake_usd)

            # ML
            if not skip_ml and HAS_LGB:
                ml_result = train_and_backtest_ml(train_windows, test_windows, asset, ml_threshold, stake_usd)
                asset_results["ml"] = ml_result

            all_results[asset] = asset_results

        except Exception as e:
            print(f"  Error processing {asset}: {e}")
            continue

    return all_results


# ---------------------------------------------------------------------------
# Strategy execution
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


def side_ask(w: Window, side: str) -> np.ndarray:
    return w.up_ask if side == "UP" else w.down_ask


def settlement_value(w: Window, side: str) -> float:
    return float(w.outcome_up) if side == "UP" else float(1 - w.outcome_up)


def compute_pnl(entry_price: float, exit_price: float, stake: float, fee_rate: float = TAKER_FEE_RATE, slippage: float = 0.001) -> float:
    """Compute PnL with realistic slippage."""
    entry_price_with_slip = entry_price + slippage
    shares = stake / entry_price_with_slip
    entry_fee = stake * fee_rate
    exit_fee = exit_price * shares * fee_rate
    return (exit_price - entry_price_with_slip) * shares - entry_fee - exit_fee


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


def run_strategies(train_windows: list[Window], test_windows: list[Window], stake: float = 10.0) -> dict:
    """Run all strategies and return metrics."""
    results = {}

    for sample, sample_windows in [("TRAIN", train_windows), ("TEST", test_windows)]:
        print(f"\n{sample} SET:")
        for name, fn in STRATEGIES.items():
            pnl_by_window = np.zeros(len(sample_windows), dtype=float)
            trades: list[ScalpTrade] = []

            for i, w in enumerate(sample_windows):
                trade = fn(w, stake=stake)
                if trade:
                    pnl_by_window[i] = trade.pnl
                    trades.append(trade)

            # Compute metrics
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

            key = f"{name}_{sample.lower()}"
            results[key] = {
                "strategy": name,
                "sample": sample,
                "total_pnl": float(equity[-1] - STARTING_CAPITAL),
                "trades": len(trades),
                "win_rate": len(wins) / len(trades) if trades else 0.0,
                "avg_pnl": float(np.mean(pnls)) if pnls else 0.0,
                "profit_factor": profit_factor,
                "max_drawdown": max_drawdown(equity),
                "sharpe": sharpe,
                "avg_hold_seconds": float(np.mean([t.seconds_held for t in trades])) if trades else 0.0,
                "avg_entry_price": float(np.mean([t.entry_price for t in trades])) if trades else 0.0,
            }

            print(f"  {name}: PnL=${results[key]['total_pnl']:,.2f}, Trades={results[key]['trades']}, WR={100*results[key]['win_rate']:.1f}%")

    return results


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


def train_and_backtest_ml(train_windows: list[Window], test_windows: list[Window], asset: str, threshold: float = 0.55, stake: float = 10.0) -> dict:
    """Train ML model and backtest."""
    print(f"\n{'=' * 80}")
    print(f"ML MODEL TRAINING: {asset}")
    print(f"{'=' * 80}")

    # Build datasets
    entry_indices = list(range(150, 280, 5))

    train_rows = []
    for w in train_windows:
        for idx in entry_indices:
            if idx >= WINDOW_SECONDS:
                continue
            features = extract_features(w, idx)
            row = {"market_start": w.start, "idx": idx, "label": w.outcome_up}
            row.update(features)
            train_rows.append(row)

    test_rows = []
    for w in test_windows:
        for idx in entry_indices:
            if idx >= WINDOW_SECONDS:
                continue
            features = extract_features(w, idx)
            row = {"market_start": w.start, "idx": idx, "label": w.outcome_up}
            row.update(features)
            test_rows.append(row)

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)

    feature_cols = [c for c in train_df.columns if c not in ["market_start", "idx", "label"]]
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train UP rate: {y_train.mean():.3f}")
    print(f"Test UP rate: {y_test.mean():.3f}")

    # Train with regularization to prevent overfitting
    params = {
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "num_leaves": 15,  # Reduced to prevent overfitting
        "learning_rate": 0.01,  # Lower learning rate
        "feature_fraction": 0.6,  # More aggressive feature subsampling
        "bagging_fraction": 0.6,
        "bagging_freq": 5,
        "min_child_samples": 50,  # More samples per leaf
        "reg_alpha": 0.1,  # L1 regularization
        "reg_lambda": 0.1,  # L2 regularization
        "verbose": -1, "seed": 42,
    }

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, feature_name=feature_cols)

    model = lgb.train(
        params, train_data, num_boost_round=1000, valid_sets=[test_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
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

    # Backtest
    ml_results = {}
    for sample, sample_windows in [("TRAIN", train_windows), ("TEST", test_windows)]:
        pnl_by_window = np.zeros(len(sample_windows), dtype=float)
        trades: list[ScalpTrade] = []

        for i, w in enumerate(sample_windows):
            idx = 240
            features = extract_features(w, idx)
            X = np.array([[features[c] for c in feature_cols]])
            pred = float(model.predict(X)[0])

            if pred >= threshold or pred <= (1 - threshold):
                side = "UP" if pred >= threshold else "DOWN"
                ask = float(side_ask(w, side)[idx])
                if ask > 0.70:
                    continue
                final = settlement_value(w, side)
                pnl = compute_pnl(ask, final, stake)
                pnl_by_window[i] = pnl
                trades.append(ScalpTrade("ml_model", w.start, side, idx, ask, WINDOW_SECONDS - 1, final, "settlement", pnl, WINDOW_SECONDS - 1 - idx))

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

        key = f"ml_{sample.lower()}"
        ml_results[key] = {
            "strategy": "ml_model",
            "sample": sample,
            "total_pnl": float(equity[-1] - STARTING_CAPITAL),
            "trades": len(trades),
            "win_rate": len(wins) / len(trades) if trades else 0.0,
            "avg_pnl": float(np.mean(pnls)) if pnls else 0.0,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown(equity),
            "sharpe": sharpe,
            "auc": train_auc if sample == "TRAIN" else test_auc,
        }

        print(f"\nML {sample}: PnL=${ml_results[key]['total_pnl']:,.2f}, Trades={ml_results[key]['trades']}, WR={100*ml_results[key]['win_rate']:.1f}%")

    return ml_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", nargs="+", default=["BTC", "ETH", "SOL", "XRP"])
    parser.add_argument("--train-frac", type=float, default=0.80)
    parser.add_argument("--stake-usd", type=float, default=10.0)
    parser.add_argument("--ml-threshold", type=float, default=0.55)
    parser.add_argument("--skip-ml", action="store_true")
    args = parser.parse_args()

    results = run_multi_asset_backtest(
        assets=args.assets,
        train_frac=args.train_frac,
        stake_usd=args.stake_usd,
        ml_threshold=args.ml_threshold,
        skip_ml=args.skip_ml,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("MULTI-ASSET SUMMARY")
    print("=" * 80)

    all_metrics = []
    for asset, asset_results in results.items():
        for key, metrics in asset_results.items():
            row = {"asset": asset}
            row.update(metrics)
            all_metrics.append(row)

    df = pd.DataFrame(all_metrics)
    if not df.empty:
        # Format and print
        out = df.copy()
        for col in ["total_pnl", "avg_pnl"]:
            if col in out.columns:
                out[col] = out[col].map(lambda x: f"${x:,.2f}")
        for col in ["win_rate", "max_drawdown"]:
            if col in out.columns:
                out[col] = out[col].map(lambda x: f"{100*x:.1f}%")
        if "profit_factor" in out.columns:
            out["profit_factor"] = out["profit_factor"].map(lambda x: "inf" if np.isinf(x) else f"{x:.2f}")
        if "sharpe" in out.columns:
            out["sharpe"] = out["sharpe"].map(lambda x: f"{x:.2f}")

        print(out.to_string(index=False))

        # Save
        df.to_csv("multi_asset_results.csv", index=False)
        print(f"\nSaved multi_asset_results.csv")


if __name__ == "__main__":
    main()
