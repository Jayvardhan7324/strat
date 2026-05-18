"""
ML-Enhanced Chop Direction Predictor

Uses the proven chop pattern edge (~70% win rate) and enhances it with:
1. ML model to predict direction with higher accuracy
2. Confidence-based filtering (only trade when model is sure)
3. Dynamic position sizing based on confidence
4. Smart entry timing

Features engineered from BTC price action in the first 2 minutes:
- Pattern classification (chop_up_first, chop_down_first, trend_up, trend_down)
- Momentum at multiple timeframes
- Volatility measures
- RSI, MACD, Bollinger Bands
- Volume proxy (price activity)
- Time decay features

Usage:
    python ml_chop_scalper.py
    python ml_chop_scalper.py --train
    python ml_chop_scalper.py --backtest
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtests.polymarket_updown_backtest import (
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

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def extract_features(w: Window, entry_idx: int = 120) -> dict:
    """Extract features from BTC price action up to entry_idx."""
    p = w.prices[:entry_idx + 1]
    open_price = w.open_price

    # Basic stats
    high = float(p.max())
    low = float(p.min())
    close = float(p[-1])
    range_pct = (high - low) / open_price
    net_move = (close - open_price) / open_price

    # Pattern classification
    high_idx = int(np.argmax(p))
    low_idx = int(np.argmin(p))

    if high_idx < low_idx:
        pattern_up_first = 1.0
        pattern_down_first = 0.0
    elif low_idx < high_idx:
        pattern_up_first = 0.0
        pattern_down_first = 1.0
    else:
        pattern_up_first = 0.0
        pattern_down_first = 0.0

    is_chop = 1.0 if (pattern_up_first or pattern_down_first) and range_pct > 0.0005 else 0.0
    is_trend = 1.0 if abs(net_move) > 0.0003 and range_pct < 0.001 else 0.0
    is_flat = 1.0 if range_pct < 0.0003 else 0.0

    # Momentum at multiple timeframes
    ret_10s = p[-1] / p[max(0, -10)] - 1.0 if len(p) >= 10 else 0.0
    ret_30s = p[-1] / p[max(0, -30)] - 1.0 if len(p) >= 30 else 0.0
    ret_60s = p[-1] / p[max(0, -60)] - 1.0 if len(p) >= 60 else 0.0
    ret_120s = p[-1] / p[0] - 1.0 if len(p) >= 120 else net_move

    # Recent momentum direction (last 20s vs previous 20s)
    if len(p) >= 40:
        recent_20 = p[-20:]
        prev_20 = p[-40:-20]
        recent_ret = recent_20[-1] / recent_20[0] - 1.0
        prev_ret = prev_20[-1] / prev_20[0] - 1.0
        momentum_accel = recent_ret - prev_ret
    else:
        recent_ret = ret_30s
        prev_ret = ret_60s
        momentum_accel = 0.0

    # Volatility
    if len(p) > 10:
        returns = np.diff(np.log(p))
        vol_10s = float(np.std(returns[-10:])) if len(returns) >= 10 else 0.0
        vol_30s = float(np.std(returns[-30:])) if len(returns) >= 30 else 0.0
        vol_60s = float(np.std(returns[-60:])) if len(returns) >= 60 else 0.0
        vol_ratio = vol_10s / vol_60s if vol_60s > 0 else 1.0
    else:
        vol_10s = vol_30s = vol_60s = vol_ratio = 0.0

    # RSI
    if len(p) >= 15:
        diffs = np.diff(p[-15:])
        gains = np.clip(diffs, 0, None).mean()
        losses = -np.clip(diffs, None, 0).mean()
        rsi = 100.0 - (100.0 / (1.0 + gains / losses)) if losses > 0 else 100.0
    else:
        rsi = 50.0

    # EMA crossover
    if len(p) >= 20:
        ema_fast = pd.Series(p).ewm(span=8).mean().iloc[-1]
        ema_slow = pd.Series(p).ewm(span=20).mean().iloc[-1]
        ema_cross = float((ema_fast - ema_slow) / open_price)
    else:
        ema_cross = 0.0

    # Bollinger Bands
    if len(p) >= 20:
        bb_mid = float(p[-20:].mean())
        bb_std = float(p[-20:].std())
        bb_pct = float((close - bb_mid) / (2 * bb_std)) if bb_std > 0 else 0.0
        bb_width = float(bb_std / open_price)
    else:
        bb_pct = bb_width = 0.0

    # Price position within range
    price_in_range = float((close - low) / (high - low)) if high > low else 0.5

    # How far into the window we are
    time_frac = float(entry_idx / WINDOW_SECONDS)

    # Number of direction changes (choppiness)
    if len(p) > 2:
        diffs = np.diff(p)
        signs = np.sign(diffs)
        changes = float(np.sum(np.abs(np.diff(signs)) > 0))
        chop_ratio = changes / max(len(diffs) - 1, 1)
    else:
        chop_ratio = 0.0

    # Max up move and max down move
    cummax = np.maximum.accumulate(p)
    cummin = np.minimum.accumulate(p)
    max_up = float((cummax[-1] - p[0]) / p[0])
    max_down = float((p[0] - cummin[-1]) / p[0])

    # Second half momentum (after the high/low was hit)
    if high_idx < entry_idx and low_idx < entry_idx:
        # Both extremes already happened - measure what happened after
        last_extreme_idx = max(high_idx, low_idx)
        if last_extreme_idx < entry_idx:
            post_move = float((p[-1] - p[last_extreme_idx]) / p[last_extreme_idx])
        else:
            post_move = 0.0
    else:
        post_move = 0.0

    return {
        "range_pct": range_pct,
        "net_move": net_move,
        "pattern_up_first": pattern_up_first,
        "pattern_down_first": pattern_down_first,
        "is_chop": is_chop,
        "is_trend": is_trend,
        "is_flat": is_flat,
        "ret_10s": ret_10s,
        "ret_30s": ret_30s,
        "ret_60s": ret_60s,
        "ret_120s": ret_120s,
        "recent_ret": recent_ret,
        "prev_ret": prev_ret,
        "momentum_accel": momentum_accel,
        "vol_10s": vol_10s,
        "vol_30s": vol_30s,
        "vol_60s": vol_60s,
        "vol_ratio": vol_ratio,
        "rsi": rsi,
        "ema_cross": ema_cross,
        "bb_pct": bb_pct,
        "bb_width": bb_width,
        "price_in_range": price_in_range,
        "time_frac": time_frac,
        "chop_ratio": chop_ratio,
        "max_up": max_up,
        "max_down": max_down,
        "post_extreme_move": post_move,
    }


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def build_ml_dataset(
    windows: list[Window],
    entry_idx: int = 120,
    sample: str = "",
) -> pd.DataFrame:
    """Build ML dataset from windows."""
    rows = []
    for w in windows:
        features = extract_features(w, entry_idx)
        # Label: 1 if UP wins, 0 if DOWN wins
        # We want to predict if the 2nd movement direction wins
        high_idx = int(np.argmax(w.prices[:entry_idx + 1]))
        low_idx = int(np.argmin(w.prices[:entry_idx + 1]))

        if high_idx < low_idx:
            # Up then down -> label is whether DOWN wins (close < open)
            label = 1 if w.outcome_up == 0 else 0
            pattern = "chop_up_first"
        elif low_idx < high_idx:
            # Down then up -> label is whether UP wins (close > open)
            label = 1 if w.outcome_up == 1 else 0
            pattern = "chop_down_first"
        elif w.prices[entry_idx] > w.open_price:
            label = w.outcome_up
            pattern = "trend_up"
        else:
            label = 1 - w.outcome_up
            pattern = "trend_down"

        row = {"sample": sample, "label": label, "pattern": pattern, "outcome_up": w.outcome_up}
        row.update(features)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ML Training
# ---------------------------------------------------------------------------

def train_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Train LightGBM model."""
    if not HAS_LGB:
        raise ImportError("Install lightgbm: pip install lightgbm")

    feature_cols = [c for c in train_df.columns if c not in ["sample", "label", "pattern", "outcome_up"]]

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train positive rate: {y_train.mean():.3f}")
    print(f"Test positive rate: {y_test.mean():.3f}")

    # Use scale_pos_weight for class balance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_weight = n_neg / max(n_pos, 1)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": scale_weight,
        "verbose": -1,
        "seed": 42,
    }

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, feature_name=feature_cols)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Accuracy at different thresholds
    for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
        train_acc = np.mean((train_pred >= threshold) == y_train)
        test_acc = np.mean((test_pred >= threshold) == y_test)
        train_up_mask = train_pred >= threshold
        if train_up_mask.sum() > 0:
            train_up_acc = np.mean(y_train[train_up_mask])
        else:
            train_up_acc = 0
        test_up_mask = test_pred >= threshold
        if test_up_mask.sum() > 0:
            test_up_acc = np.mean(y_test[test_up_mask])
        else:
            test_up_acc = 0
        print(f"  Threshold {threshold:.2f}: Train acc={train_acc:.3f}, Test acc={test_acc:.3f}, "
              f"Train UP acc={train_up_acc:.3f} ({train_up_mask.sum()} trades), "
              f"Test UP acc={test_up_acc:.3f} ({test_up_mask.sum()} trades)")

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(),
    }).sort_values("importance", ascending=False)

    print("\nTop 10 features:")
    print(importance.head(10).to_string(index=False))

    return {
        "model": model,
        "feature_cols": feature_cols,
        "importance": importance,
    }


# ---------------------------------------------------------------------------
# ML Backtest
# ---------------------------------------------------------------------------

@dataclass
class MLTrade:
    window_start: pd.Timestamp
    pattern: str
    side: str
    confidence: float
    entry_idx: int
    entry_price: float
    exit_price: float
    pnl: float
    outcome: int
    stake: float


def backtest_ml(
    windows: list[Window],
    model,
    feature_cols: list[str],
    sample: str,
    entry_idx: int = 120,
    confidence_threshold: float = 0.60,
    base_stake: float = 1000.0,
    max_stake_multiplier: float = 3.0,
) -> tuple[list[MLTrade], np.ndarray]:
    """Backtest ML model with confidence-based filtering and position sizing."""
    trades = []
    pnl_by_window = np.zeros(len(windows), dtype=float)

    for i, w in enumerate(windows):
        features = extract_features(w, entry_idx)
        X = np.array([[features[c] for c in feature_cols]])
        pred = float(model.predict(X)[0])

        # Only trade when confident
        if pred < confidence_threshold and pred > (1 - confidence_threshold):
            continue

        # Determine side and confidence
        if pred >= confidence_threshold:
            side = "UP"
            confidence = pred
        else:
            side = "DOWN"
            confidence = 1 - pred

        # Dynamic position sizing: more confidence = bigger position
        stake = base_stake * (1 + max_stake_multiplier * (confidence - confidence_threshold) / (1 - confidence_threshold))
        stake = min(stake, base_stake * max_stake_multiplier)

        # Enter trade
        idx = min(entry_idx, WINDOW_SECONDS - 10)
        _, ask = side_arrays(w, side.lower())
        entry_price = float(ask[idx])

        # Hold to settlement
        final = settlement_value(w, side.lower())

        shares = stake / entry_price
        entry_fee = entry_price * shares * TAKER_FEE_RATE
        exit_fee = final * shares * TAKER_FEE_RATE
        pnl = (final - entry_price) * shares - entry_fee - exit_fee

        pnl_by_window[i] = pnl
        trades.append(MLTrade(
            window_start=w.start,
            pattern=features["pattern_up_first"] > 0.5 and "chop_up_first" or (features["pattern_down_first"] > 0.5 and "chop_down_first" or "other"),
            side=side,
            confidence=confidence,
            entry_idx=idx,
            entry_price=entry_price,
            exit_price=final,
            pnl=pnl,
            outcome=1 if pnl > 0 else 0,
            stake=stake,
        ))

    return trades, pnl_by_window


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class MLMetrics:
    sample: str
    confidence_threshold: float
    total_pnl: float
    total_trades: int
    windows_traded: int
    windows_total: int
    win_rate: float
    avg_pnl: float
    avg_stake: float
    profit_factor: float
    max_drawdown: float
    sharpe: float
    avg_confidence: float


def compute_ml_metrics(
    trades: list[MLTrade],
    windows: list[Window],
    pnl_by_window: np.ndarray,
    sample: str,
    confidence_threshold: float,
) -> MLMetrics:
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

    windows_traded = len(set(t.window_start for t in trades))

    return MLMetrics(
        sample=sample,
        confidence_threshold=confidence_threshold,
        total_pnl=float(equity[-1] - STARTING_CAPITAL),
        total_trades=len(trades),
        windows_traded=windows_traded,
        windows_total=len(windows),
        win_rate=len(wins) / len(trades) if trades else 0.0,
        avg_pnl=float(np.mean(pnls)) if pnls else 0.0,
        avg_stake=float(np.mean([t.stake for t in trades])) if trades else 0.0,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown(equity),
        sharpe=sharpe,
        avg_confidence=float(np.mean([t.confidence for t in trades])) if trades else 0.0,
    )


def print_ml_metrics(metrics_list: list[MLMetrics]) -> None:
    df = pd.DataFrame([m.__dict__ for m in metrics_list])
    if df.empty:
        print("No trades generated.")
        return

    out = df.copy()
    for col in ["total_pnl", "avg_pnl", "avg_stake"]:
        out[col] = out[col].map(lambda x: f"${x:,.2f}")
    for col in ["win_rate", "max_drawdown"]:
        out[col] = out[col].map(lambda x: f"{100*x:.1f}%")
    out["profit_factor"] = out["profit_factor"].map(lambda x: "inf" if np.isinf(x) else f"{x:.2f}")
    out["sharpe"] = out["sharpe"].map(lambda x: f"{x:.2f}")
    out["avg_confidence"] = out["avg_confidence"].map(lambda x: f"{100*x:.1f}%")

    cols = [
        "sample", "confidence_threshold", "total_pnl", "total_trades",
        "windows_traded", "win_rate", "profit_factor", "avg_pnl",
        "avg_stake", "sharpe", "max_drawdown", "avg_confidence",
    ]
    print(out[cols].to_string(index=False))


# ---------------------------------------------------------------------------
# Confidence sweep
# ---------------------------------------------------------------------------

def confidence_sweep(
    windows: list[Window],
    model,
    feature_cols: list[str],
    sample: str,
    base_stake: float,
) -> pd.DataFrame:
    """Sweep confidence thresholds."""
    results = []

    for threshold in np.arange(0.50, 0.85, 0.02):
        threshold = round(threshold, 2)
        trades, pnl_by_window = backtest_ml(
            windows, model, feature_cols, sample,
            confidence_threshold=threshold, base_stake=base_stake,
        )
        if trades:
            metrics = compute_ml_metrics(trades, windows, pnl_by_window, sample, threshold)
            results.append({
                "threshold": threshold,
                "total_pnl": metrics.total_pnl,
                "trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor if not np.isinf(metrics.profit_factor) else 999,
                "sharpe": metrics.sharpe,
                "max_drawdown": metrics.max_drawdown,
                "avg_stake": metrics.avg_stake,
                "avg_confidence": metrics.avg_confidence,
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not HAS_LGB:
        print("LightGBM not installed. Install: pip install lightgbm")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument("--source", default="binance")
    parser.add_argument("--dataset-source", default="aliplayer_spot")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--stake-usd", type=float, default=1000.0)
    parser.add_argument("--entry-idx", type=int, default=120)
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--backtest", action="store_true", help="Backtest model")
    parser.add_argument("--sweep", action="store_true", help="Sweep confidence thresholds")
    parser.add_argument("--threshold", type=float, default=0.60, help="Confidence threshold")
    args = parser.parse_args()

    price_df = load_spot_prices(args.symbol, None if args.source.lower() in {"none", "all"} else args.source, dataset_source=args.dataset_source)
    windows = build_windows(price_df, args.max_windows)
    train_windows, test_windows = split_windows_chronologically(windows, args.train_frac)

    if args.train:
        print("\n" + "=" * 80)
        print("ML MODEL TRAINING")
        print("=" * 80)

        train_df = build_ml_dataset(train_windows, args.entry_idx, "TRAIN")
        test_df = build_ml_dataset(test_windows, args.entry_idx, "TEST")

        result = train_model(train_df, test_df)

        # Save model
        result["model"].save_model("ml_chop_model.txt")
        result["importance"].to_csv("ml_chop_feature_importance.csv", index=False)
        print("\nSaved ml_chop_model.txt and ml_chop_feature_importance.csv")

    if args.backtest or args.sweep:
        # Load model
        model = lgb.Booster(model_file="ml_chop_model.txt")
        importance = pd.read_csv("ml_chop_feature_importance.csv")
        feature_cols = importance["feature"].tolist()

        if args.sweep:
            print("\n" + "=" * 80)
            print("CONFIDENCE THRESHOLD SWEEP")
            print("=" * 80)

            print("\nTRAIN sweep...")
            train_results = confidence_sweep(train_windows, model, feature_cols, "TRAIN", args.stake_usd)
            if not train_results.empty:
                train_results = train_results.sort_values("total_pnl", ascending=False)
                print(f"\nTop configs by TRAIN PnL:")
                print(train_results.head(10).to_string(index=False))
                train_results.to_csv("ml_chop_sweep_train.csv", index=False)

            print("\nTEST sweep (top 5 train thresholds)...")
            if not train_results.empty:
                top_thresholds = train_results.head(5)["threshold"].tolist()
                test_results = []
                for threshold in top_thresholds:
                    trades, pnl_by_window = backtest_ml(
                        test_windows, model, feature_cols, "TEST",
                        confidence_threshold=threshold, base_stake=args.stake_usd,
                    )
                    if trades:
                        metrics = compute_ml_metrics(trades, test_windows, pnl_by_window, "TEST", threshold)
                        test_results.append({
                            "threshold": threshold,
                            "total_pnl": metrics.total_pnl,
                            "trades": metrics.total_trades,
                            "win_rate": metrics.win_rate,
                            "profit_factor": metrics.profit_factor if not np.isinf(metrics.profit_factor) else 999,
                            "sharpe": metrics.sharpe,
                            "max_drawdown": metrics.max_drawdown,
                            "avg_stake": metrics.avg_stake,
                        })

                if test_results:
                    test_df = pd.DataFrame(test_results).sort_values("total_pnl", ascending=False)
                    print(f"\nTEST results:")
                    print(test_df.to_string(index=False))
                    test_df.to_csv("ml_chop_sweep_test.csv", index=False)

        if args.backtest:
            print("\n" + "=" * 80)
            print(f"ML BACKTEST (threshold={args.threshold})")
            print("=" * 80)

            all_metrics = []
            for sample, sample_windows in [("TRAIN", train_windows), ("TEST", test_windows)]:
                trades, pnl_by_window = backtest_ml(
                    sample_windows, model, feature_cols, sample,
                    confidence_threshold=args.threshold, base_stake=args.stake_usd,
                )
                metrics = compute_ml_metrics(trades, sample_windows, pnl_by_window, sample, args.threshold)
                all_metrics.append(metrics)

                print(f"\n{sample}: {metrics.total_trades} trades, Win rate={100*metrics.win_rate:.1f}%, "
                      f"PnL=${metrics.total_pnl:,.2f}, Sharpe={metrics.sharpe:.2f}")

            print_ml_metrics(all_metrics)

            # Save trades
            for sample, sample_windows in [("TRAIN", train_windows), ("TEST", test_windows)]:
                trades, _ = backtest_ml(
                    sample_windows, model, feature_cols, sample,
                    confidence_threshold=args.threshold, base_stake=args.stake_usd,
                )
                if trades:
                    trades_df = pd.DataFrame([t.__dict__ for t in trades])
                    trades_df.to_csv(f"ml_chop_trades_{sample.lower()}.csv", index=False)


if __name__ == "__main__":
    main()
