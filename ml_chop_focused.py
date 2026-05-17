"""
ML-Enhanced Chop Scalper - Focused on proven chop patterns only.

Strategy:
1. Only trade when BTC shows a clear chop pattern in first 2 minutes
2. ML model predicts how strong the 2nd-direction edge is
3. High confidence = bigger position, skip low confidence
4. Always hold to settlement (binary outcome, no spread risk)

Proven edge: chop patterns resolve in 2nd direction ~70% of the time.
ML enhances this by filtering to the strongest setups.
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

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------

def detect_chop(w: Window, measure_seconds: int = 120) -> tuple[str, float] | None:
    """Detect chop pattern and return (direction, strength).

    direction: "UP" if chop_down_first (bet on UP), "DOWN" if chop_up_first
    strength: how clear the chop is (higher = better)
    """
    p = w.prices[:measure_seconds]
    open_price = w.open_price
    high = float(p.max())
    low = float(p.min())
    range_pct = (high - low) / open_price

    # Need meaningful range
    if range_pct < 0.0004:
        return None

    high_idx = int(np.argmax(p))
    low_idx = int(np.argmin(p))

    if high_idx < low_idx:
        # Up then down -> bet DOWN
        direction = "DOWN"
        # Strength: how far did it go up, and how far did it come back?
        up_move = (high - open_price) / open_price
        down_move = (high - p[-1]) / high
        strength = range_pct * (1 + up_move + down_move)
    elif low_idx < high_idx:
        # Down then up -> bet UP
        direction = "UP"
        down_move = (open_price - low) / open_price
        up_move = (p[-1] - low) / low
        strength = range_pct * (1 + down_move + up_move)
    else:
        return None

    return direction, strength


# ---------------------------------------------------------------------------
# Feature engineering (chop-specific)
# ---------------------------------------------------------------------------

def extract_chop_features(w: Window, measure_seconds: int = 120) -> dict | None:
    """Extract features only for chop windows."""
    chop_result = detect_chop(w, measure_seconds)
    if chop_result is None:
        return None

    direction, strength = chop_result
    p = w.prices[:measure_seconds + 1]
    open_price = w.open_price

    high = float(p[:measure_seconds].max())
    low = float(p[:measure_seconds].min())
    close = float(p[-1])
    range_pct = (high - low) / open_price
    high_idx = int(np.argmax(p[:measure_seconds]))
    low_idx = int(np.argmin(p[:measure_seconds]))

    # Chop characteristics
    up_move = (high - open_price) / open_price
    down_move = (open_price - low) / open_price
    asymmetry = (up_move - down_move) / max(range_pct, 1e-9)

    # Where in the range are we at measure_seconds?
    price_in_range = (close - low) / max(high - low, 1e-9)

    # Momentum in last 30s of measure period
    if measure_seconds >= 30:
        ret_30s = p[-1] / p[-30] - 1.0
        ret_15s = p[-1] / p[-15] - 1.0
    else:
        ret_30s = ret_15s = 0.0

    # Volatility
    if len(p) > 10:
        returns = np.diff(np.log(p))
        vol_30s = float(np.std(returns[-30:])) if len(returns) >= 30 else 0.0
        vol_60s = float(np.std(returns[-60:])) if len(returns) >= 60 else 0.0
        vol_120s = float(np.std(returns[-120:])) if len(returns) >= 120 else 0.0
        vol_accel = vol_30s / max(vol_120s, 1e-9)
    else:
        vol_30s = vol_60s = vol_120s = vol_accel = 0.0

    # RSI at measure point
    if len(p) >= 15:
        diffs = np.diff(p[-15:])
        gains = np.clip(diffs, 0, None).mean()
        losses = -np.clip(diffs, None, 0).mean()
        rsi = 100.0 - (100.0 / (1.0 + gains / losses)) if losses > 0 else 100.0
    else:
        rsi = 50.0

    # EMA
    if len(p) >= 20:
        ema_fast = pd.Series(p).ewm(span=8).mean().iloc[-1]
        ema_slow = pd.Series(p).ewm(span=20).mean().iloc[-1]
        ema_cross = float((ema_fast - ema_slow) / open_price)
    else:
        ema_cross = 0.0

    # Bollinger
    if len(p) >= 20:
        bb_mid = float(p[-20:].mean())
        bb_std = float(p[-20:].std())
        bb_pct = float((close - bb_mid) / (2 * bb_std)) if bb_std > 0 else 0.0
    else:
        bb_pct = 0.0

    # Choppiness (number of direction changes)
    if len(p) > 2:
        diffs = np.diff(p)
        signs = np.sign(diffs)
        changes = float(np.sum(np.abs(np.diff(signs)) > 0))
        chop_ratio = changes / max(len(diffs) - 1, 1)
    else:
        chop_ratio = 0.0

    # Time to reach high/low
    time_to_high = high_idx / measure_seconds
    time_to_low = low_idx / measure_seconds

    # Recovery strength (how much of the initial move was recovered)
    if direction == "UP":
        initial_move = down_move
        recovery = (close - low) / max(high - low, 1e-9)
    else:
        initial_move = up_move
        recovery = (high - close) / max(high - low, 1e-9)

    return {
        "direction": direction,
        "strength": strength,
        "range_pct": range_pct,
        "up_move": up_move,
        "down_move": down_move,
        "asymmetry": asymmetry,
        "price_in_range": price_in_range,
        "ret_30s": ret_30s,
        "ret_15s": ret_15s,
        "vol_30s": vol_30s,
        "vol_60s": vol_60s,
        "vol_120s": vol_120s,
        "vol_accel": vol_accel,
        "rsi": rsi,
        "ema_cross": ema_cross,
        "bb_pct": bb_pct,
        "chop_ratio": chop_ratio,
        "time_to_high": time_to_high,
        "time_to_low": time_to_low,
        "recovery": recovery,
        "initial_move": initial_move,
    }


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def build_chop_dataset(
    windows: list[Window],
    measure_seconds: int = 120,
    sample: str = "",
) -> pd.DataFrame:
    """Build ML dataset from chop windows only."""
    rows = []
    for w in windows:
        features = extract_chop_features(w, measure_seconds)
        if features is None:
            continue

        direction = features.pop("direction")
        # Label: 1 if our directional bet wins
        if direction == "UP":
            label = w.outcome_up
        else:
            label = 1 - w.outcome_up

        row = {"sample": sample, "label": label, "direction": direction}
        row.update(features)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ML Training
# ---------------------------------------------------------------------------

def train_chop_model(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Train LightGBM on chop windows."""
    if not HAS_LGB:
        raise ImportError("pip install lightgbm")

    feature_cols = [c for c in train_df.columns if c not in ["sample", "label", "direction"]]

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    print(f"\nChop windows - Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Train win rate: {y_train.mean():.3f}")
    print(f"Test win rate: {y_test.mean():.3f}")

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_weight = n_neg / max(n_pos, 1)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 15,
        "learning_rate": 0.05,
        "feature_fraction": 0.7,
        "min_child_samples": 20,
        "scale_pos_weight": scale_weight,
        "verbose": -1,
        "seed": 42,
    }

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, feature_name=feature_cols)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    # Evaluate at thresholds
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print(f"\n{'Threshold':>10} {'Train Acc':>10} {'Train Trades':>12} {'Test Acc':>10} {'Test Trades':>12}")
    print("-" * 58)
    for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        train_mask = train_pred >= threshold
        test_mask = test_pred >= threshold
        train_acc = y_train[train_mask].mean() if train_mask.sum() > 0 else 0
        test_acc = y_test[test_mask].mean() if test_mask.sum() > 0 else 0
        print(f"{threshold:>10.2f} {train_acc:>10.3f} {train_mask.sum():>12} {test_acc:>10.3f} {test_mask.sum():>12}")

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(),
    }).sort_values("importance", ascending=False)

    print("\nTop 10 features:")
    print(importance.head(10).to_string(index=False))

    return {"model": model, "feature_cols": feature_cols, "importance": importance}


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

@dataclass
class ChopTrade:
    window_start: pd.Timestamp
    direction: str
    ml_confidence: float
    entry_price: float
    exit_price: float
    pnl: float
    outcome: int
    stake: float


def backtest_chop_ml(
    windows: list[Window],
    model,
    feature_cols: list[str],
    sample: str,
    measure_seconds: int = 120,
    confidence_threshold: float = 0.55,
    base_stake: float = 1000.0,
    max_stake_multiplier: float = 2.0,
) -> tuple[list[ChopTrade], np.ndarray]:
    """Backtest ML-enhanced chop strategy."""
    trades = []
    pnl_by_window = np.zeros(len(windows), dtype=float)

    for i, w in enumerate(windows):
        features = extract_chop_features(w, measure_seconds)
        if features is None:
            continue

        direction = features.pop("direction")
        X = np.array([[features[c] for c in feature_cols]])
        pred = float(model.predict(X)[0])

        # Put direction back
        features["direction"] = direction

        # Only trade when model agrees with the pattern
        # For UP bets: we want high pred (model thinks UP wins)
        # For DOWN bets: we want low pred (model thinks DOWN wins = label=1 means DOWN wins)
        if direction == "UP":
            confidence = pred
        else:
            confidence = 1 - pred

        if confidence < confidence_threshold:
            continue

        # Dynamic sizing
        stake = base_stake * (1 + max_stake_multiplier * (confidence - confidence_threshold) / (1 - confidence_threshold))
        stake = min(stake, base_stake * max_stake_multiplier)

        # Enter at measure_seconds
        idx = min(measure_seconds, WINDOW_SECONDS - 10)
        _, ask = side_arrays(w, direction.lower())
        entry_price = float(ask[idx])

        final = settlement_value(w, direction.lower())

        shares = stake / entry_price
        entry_fee = entry_price * shares * TAKER_FEE_RATE
        exit_fee = final * shares * TAKER_FEE_RATE
        pnl = (final - entry_price) * shares - entry_fee - exit_fee

        pnl_by_window[i] = pnl
        outcome = 1 if pnl > 0 else 0

        trades.append(ChopTrade(
            window_start=w.start,
            direction=direction,
            ml_confidence=confidence,
            entry_price=entry_price,
            exit_price=final,
            pnl=pnl,
            outcome=outcome,
            stake=stake,
        ))

    return trades, pnl_by_window


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class ChopMetrics:
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


def compute_chop_metrics(
    trades: list[ChopTrade],
    windows: list[Window],
    pnl_by_window: np.ndarray,
    sample: str,
    threshold: float,
) -> ChopMetrics:
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

    return ChopMetrics(
        sample=sample,
        confidence_threshold=threshold,
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
        avg_confidence=float(np.mean([t.ml_confidence for t in trades])) if trades else 0.0,
    )


def print_chop_metrics(metrics_list: list[ChopMetrics]) -> None:
    df = pd.DataFrame([m.__dict__ for m in metrics_list])
    if df.empty:
        print("No trades.")
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
        "sharpe", "max_drawdown", "avg_confidence",
    ]
    print(out[cols].to_string(index=False))


# ---------------------------------------------------------------------------
# Baseline comparison (no ML, just pattern rule)
# ---------------------------------------------------------------------------

def backtest_baseline(
    windows: list[Window],
    measure_seconds: int = 120,
    stake: float = 1000.0,
) -> tuple[list[ChopTrade], np.ndarray]:
    """Baseline: trade all chop patterns without ML."""
    trades = []
    pnl_by_window = np.zeros(len(windows), dtype=float)

    for i, w in enumerate(windows):
        chop_result = detect_chop(w, measure_seconds)
        if chop_result is None:
            continue

        direction, _ = chop_result
        idx = min(measure_seconds, WINDOW_SECONDS - 10)
        _, ask = side_arrays(w, direction.lower())
        entry_price = float(ask[idx])

        final = settlement_value(w, direction.lower())

        shares = stake / entry_price
        entry_fee = entry_price * shares * TAKER_FEE_RATE
        exit_fee = final * shares * TAKER_FEE_RATE
        pnl = (final - entry_price) * shares - entry_fee - exit_fee

        pnl_by_window[i] = pnl
        outcome = 1 if pnl > 0 else 0

        trades.append(ChopTrade(
            window_start=w.start,
            direction=direction,
            ml_confidence=0.5,  # baseline
            entry_price=entry_price,
            exit_price=final,
            pnl=pnl,
            outcome=outcome,
            stake=stake,
        ))

    return trades, pnl_by_window


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not HAS_LGB:
        print("pip install lightgbm")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument("--source", default="binance")
    parser.add_argument("--dataset-source", default="aliplayer_spot")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--stake-usd", type=float, default=1000.0)
    parser.add_argument("--measure-seconds", type=int, default=120)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--baseline", action="store_true", help="Run baseline (no ML)")
    args = parser.parse_args()

    price_df = load_spot_prices(args.symbol, None if args.source.lower() in {"none", "all"} else args.source, dataset_source=args.dataset_source)
    windows = build_windows(price_df, args.max_windows)
    train_windows, test_windows = split_windows_chronologically(windows, args.train_frac)

    if args.baseline:
        print("\n" + "=" * 80)
        print("BASELINE: Chop patterns without ML")
        print("=" * 80)

        all_metrics = []
        for sample, sample_windows in [("TRAIN", train_windows), ("TEST", test_windows)]:
            trades, pnl_by_window = backtest_baseline(sample_windows, args.measure_seconds, args.stake_usd)
            metrics = compute_chop_metrics(trades, sample_windows, pnl_by_window, sample, 0.5)
            all_metrics.append(metrics)
            print(f"{sample}: {metrics.total_trades} trades, Win={100*metrics.win_rate:.1f}%, PnL=${metrics.total_pnl:,.2f}")

        print_chop_metrics(all_metrics)
        return

    if args.train:
        print("\n" + "=" * 80)
        print("TRAINING CHOP-Focused ML MODEL")
        print("=" * 80)

        train_df = build_chop_dataset(train_windows, args.measure_seconds, "TRAIN")
        test_df = build_chop_dataset(test_windows, args.measure_seconds, "TEST")

        result = train_chop_model(train_df, test_df)
        result["model"].save_model("ml_chop_focused_model.txt")
        result["importance"].to_csv("ml_chop_focused_importance.csv", index=False)
        print("\nSaved model files.")

    if args.backtest or args.sweep:
        model = lgb.Booster(model_file="ml_chop_focused_model.txt")
        importance = pd.read_csv("ml_chop_focused_importance.csv")
        feature_cols = importance["feature"].tolist()

        if args.sweep:
            print("\n" + "=" * 80)
            print("CONFIDENCE THRESHOLD SWEEP")
            print("=" * 80)

            results = []
            for threshold in np.arange(0.50, 0.80, 0.02):
                threshold = round(threshold, 2)
                for sample, sample_windows in [("TRAIN", train_windows), ("TEST", test_windows)]:
                    trades, pnl_by_window = backtest_chop_ml(
                        sample_windows, model, feature_cols, sample,
                        confidence_threshold=threshold, base_stake=args.stake_usd,
                    )
                    if trades:
                        metrics = compute_chop_metrics(trades, sample_windows, pnl_by_window, sample, threshold)
                        results.append({
                            "sample": sample,
                            "threshold": threshold,
                            "total_pnl": metrics.total_pnl,
                            "trades": metrics.total_trades,
                            "win_rate": metrics.win_rate,
                            "profit_factor": metrics.profit_factor if not np.isinf(metrics.profit_factor) else 999,
                            "sharpe": metrics.sharpe,
                            "max_drawdown": metrics.max_drawdown,
                        })

            results_df = pd.DataFrame(results)
            train_results = results_df[results_df["sample"] == "TRAIN"].sort_values("total_pnl", ascending=False)
            print(f"\nTRAIN results:")
            print(train_results.to_string(index=False))

            test_results = results_df[results_df["sample"] == "TEST"].sort_values("total_pnl", ascending=False)
            print(f"\nTEST results:")
            print(test_results.to_string(index=False))

            train_results.to_csv("ml_chop_focused_sweep_train.csv", index=False)
            test_results.to_csv("ml_chop_focused_sweep_test.csv", index=False)

        if args.backtest:
            print(f"\nML BACKTEST (threshold={args.threshold})")
            all_metrics = []
            for sample, sample_windows in [("TRAIN", train_windows), ("TEST", test_windows)]:
                trades, pnl_by_window = backtest_chop_ml(
                    sample_windows, model, feature_cols, sample,
                    confidence_threshold=args.threshold, base_stake=args.stake_usd,
                )
                metrics = compute_chop_metrics(trades, sample_windows, pnl_by_window, sample, args.threshold)
                all_metrics.append(metrics)
                print(f"{sample}: {metrics.total_trades} trades, Win={100*metrics.win_rate:.1f}%, PnL=${metrics.total_pnl:,.2f}")

            print_chop_metrics(all_metrics)


if __name__ == "__main__":
    main()
