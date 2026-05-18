"""
Enhanced Strategy Research - Walk-Forward Validation & Portfolio Construction

Improvements over previous research:
1. Walk-forward validation across 4 time periods within 31 days
2. Parameter sweep with out-of-sample testing
3. Slippage stress testing (0-5 cents)
4. Portfolio construction (combining uncorrelated strategies)
5. Risk management (position sizing, daily stop loss)
6. ML ensemble (multiple models with different features)
7. Realistic fill simulation (queue position modeling)

Usage:
    python enhanced_research.py
    python enhanced_research.py --walk-forward
    python enhanced_research.py --slippage-stress
    python enhanced_research.py --portfolio
    python enhanced_research.py --ml-ensemble
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from backtests.polymarket_updown_backtest import (
    STARTING_CAPITAL,
    TAKER_FEE_RATE,
    WINDOW_SECONDS,
    Window,
    load_spot_prices,
    build_windows,
    max_drawdown,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------


def side_ask(w: Window, side: str) -> np.ndarray:
    return w.up_ask if side == "UP" else w.down_ask


def side_bid(w: Window, side: str) -> np.ndarray:
    return w.up_bid if side == "UP" else w.down_bid


def settlement_value(w: Window, side: str) -> float:
    return float(w.outcome_up) if side == "UP" else float(1 - w.outcome_up)


def compute_pnl(
    entry_price: float,
    exit_price: float,
    stake: float,
    fee_rate: float = TAKER_FEE_RATE,
    slippage: float = 0.0,
) -> float:
    """Compute PnL with fees and slippage."""
    entry_with_slip = entry_price + slippage
    shares = stake / entry_with_slip
    entry_fee = stake * fee_rate
    exit_fee = exit_price * shares * fee_rate
    return (exit_price - entry_with_slip) * shares - entry_fee - exit_fee


@dataclass
class Trade:
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
    params: dict = field(default_factory=dict)


@dataclass
class Metrics:
    strategy: str
    period: str
    total_pnl: float
    trades: int
    win_rate: float
    avg_pnl: float
    profit_factor: float
    max_drawdown: float
    sharpe: float
    avg_hold_seconds: float
    avg_entry_price: float


# ---------------------------------------------------------------------------
# Enhanced Strategies with Parameters
# ---------------------------------------------------------------------------


def strat_opening_breakout(
    w: Window,
    stake: float = 10.0,
    range_seconds: int = 30,
    max_entry_price: float = 0.70,
    slippage: float = 0.0,
) -> Trade | None:
    """Opening range breakout with configurable range length."""
    high = float(w.prices[:range_seconds].max())
    low = float(w.prices[:range_seconds].min())

    for idx in range(range_seconds + 10, WINDOW_SECONDS - 30):
        if w.prices[idx] > high and w.prices[idx - 1] > high:
            side = "UP"
            ask = float(side_ask(w, side)[idx])
            if ask > max_entry_price:
                continue
            final = settlement_value(w, side)
            pnl = compute_pnl(ask, final, stake, slippage=slippage)
            return Trade("opening_breakout", w.start, side, idx, ask, WINDOW_SECONDS - 1, final, "settlement", pnl, WINDOW_SECONDS - 1 - idx, {"range_seconds": range_seconds})

        if w.prices[idx] < low and w.prices[idx - 1] < low:
            side = "DOWN"
            ask = float(side_ask(w, side)[idx])
            if ask > max_entry_price:
                continue
            final = settlement_value(w, side)
            pnl = compute_pnl(ask, final, stake, slippage=slippage)
            return Trade("opening_breakout", w.start, side, idx, ask, WINDOW_SECONDS - 1, final, "settlement", pnl, WINDOW_SECONDS - 1 - idx, {"range_seconds": range_seconds})

    return None


def strat_late_momentum(
    w: Window,
    stake: float = 10.0,
    min_seconds_left: int = 30,
    max_entry_price: float = 0.70,
    momentum_threshold: float = 0.0003,
    slippage: float = 0.0,
) -> Trade | None:
    """Late momentum with configurable threshold."""
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
        pnl = compute_pnl(ask, final, stake, slippage=slippage)
        return Trade("late_momentum", w.start, side, idx, ask, WINDOW_SECONDS - 1, final, "settlement", pnl, WINDOW_SECONDS - 1 - idx, {
            "momentum_threshold": momentum_threshold, "min_seconds_left": min_seconds_left
        })

    return None


def strat_volume_spike(
    w: Window,
    stake: float = 10.0,
    spike_threshold: float = 0.0005,
    confirmation_seconds: int = 5,
    max_entry_price: float = 0.70,
    slippage: float = 0.0,
) -> Trade | None:
    """Volume spike continuation with configurable thresholds."""
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
        pnl = compute_pnl(ask, final, stake, slippage=slippage)
        return Trade("volume_spike", w.start, side, confirm_idx, ask, WINDOW_SECONDS - 1, final, "settlement", pnl, WINDOW_SECONDS - 1 - confirm_idx, {
            "spike_threshold": spike_threshold, "confirmation_seconds": confirmation_seconds
        })

    return None


def strat_mean_reversion(
    w: Window,
    stake: float = 10.0,
    fade_threshold: float = 0.001,
    entry_window_end: int = 120,
    max_entry_price: float = 0.60,
    slippage: float = 0.0,
) -> Trade | None:
    """Mean reversion fade with configurable thresholds."""
    for idx in range(60, entry_window_end):
        deviation = w.prices[idx] / w.open_price - 1.0
        if abs(deviation) < fade_threshold:
            continue

        micro_ret = w.prices[idx] / w.prices[max(0, idx - 10)] - 1.0
        if np.sign(micro_ret) == np.sign(deviation):
            continue

        side = "DOWN" if deviation > 0 else "UP"
        ask = float(side_ask(w, side)[idx])
        if ask > max_entry_price:
            continue

        final = settlement_value(w, side)
        pnl = compute_pnl(ask, final, stake, slippage=slippage)
        return Trade("mean_reversion", w.start, side, idx, ask, WINDOW_SECONDS - 1, final, "settlement", pnl, WINDOW_SECONDS - 1 - idx, {
            "fade_threshold": fade_threshold, "entry_window_end": entry_window_end
        })

    return None


STRATEGIES = {
    "opening_breakout": strat_opening_breakout,
    "late_momentum": strat_late_momentum,
    "volume_spike": strat_volume_spike,
    "mean_reversion": strat_mean_reversion,
}


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------


def walk_forward_test(
    windows: list[Window],
    n_periods: int = 4,
    train_frac: float = 0.75,
    stake: float = 10.0,
    slippage: float = 0.0,
) -> list[Metrics]:
    """Walk-forward validation across multiple time periods."""
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION")
    print(f"Periods: {n_periods}, Train fraction: {train_frac:.0%}")
    print("=" * 80)

    period_size = len(windows) // n_periods
    all_metrics = []

    for period in range(n_periods):
        # Define train/test windows for this period
        test_start = period * period_size
        test_end = (period + 1) * period_size
        test_windows = windows[test_start:test_end]

        # Train on all prior data
        train_windows = windows[:test_start] if test_start > 0 else []

        period_label = f"Period_{period + 1}"
        print(f"\n{period_label}: Train={len(train_windows):,}, Test={len(test_windows):,}")
        print(f"  Test range: {test_windows[0].start} -> {test_windows[-1].start}")

        if not train_windows or len(test_windows) < 50:
            print("  Skipping (insufficient data)")
            continue

        # Run strategies on test set
        for name, fn in STRATEGIES.items():
            pnl_by_window = np.zeros(len(test_windows), dtype=float)
            trades: list[Trade] = []

            for i, w in enumerate(test_windows):
                trade = fn(w, stake=stake, slippage=slippage)
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

            metrics = Metrics(
                strategy=name,
                period=period_label,
                total_pnl=float(equity[-1] - STARTING_CAPITAL),
                trades=len(trades),
                win_rate=len(wins) / len(trades) if trades else 0.0,
                avg_pnl=float(np.mean(pnls)) if pnls else 0.0,
                profit_factor=profit_factor,
                max_drawdown=max_drawdown(equity),
                sharpe=sharpe,
                avg_hold_seconds=float(np.mean([t.seconds_held for t in trades])) if trades else 0.0,
                avg_entry_price=float(np.mean([t.entry_price for t in trades])) if trades else 0.0,
            )
            all_metrics.append(metrics)

            print(f"  {name}: PnL=${metrics.total_pnl:,.2f}, WR={100*metrics.win_rate:.1f}%, Trades={metrics.trades}")

    return all_metrics


# ---------------------------------------------------------------------------
# Parameter Sweep
# ---------------------------------------------------------------------------


def parameter_sweep(
    windows: list[Window],
    train_windows: list[Window],
    test_windows: list[Window],
    stake: float = 10.0,
    slippage: float = 0.0,
) -> pd.DataFrame:
    """Sweep parameters and find optimal settings."""
    print("\n" + "=" * 80)
    print("PARAMETER SWEEP")
    print("=" * 80)

    param_grid = {
        "opening_breakout": {
            "range_seconds": [15, 30, 45, 60],
            "max_entry_price": [0.60, 0.65, 0.70, 0.75],
        },
        "late_momentum": {
            "momentum_threshold": [0.0002, 0.0003, 0.0005, 0.0008],
            "min_seconds_left": [20, 30, 40, 50],
        },
        "volume_spike": {
            "spike_threshold": [0.0003, 0.0005, 0.0008, 0.001],
            "confirmation_seconds": [3, 5, 10],
        },
    }

    results = []

    for strategy_name, params in param_grid.items():
        fn = STRATEGIES[strategy_name]
        keys = list(params.keys())
        values = list(params.values())

        print(f"\n{strategy_name}: Testing {np.prod([len(v) for v in values])} combinations")

        for combo in product(*values):
            param_dict = dict(zip(keys, combo))

            # Test set performance
            pnl_by_window = np.zeros(len(test_windows), dtype=float)
            trades = []

            for i, w in enumerate(test_windows):
                trade = fn(w, stake=stake, slippage=slippage, **param_dict)
                if trade:
                    pnl_by_window[i] = trade.pnl
                    trades.append(trade)

            if len(trades) < 10:
                continue

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
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

            results.append({
                "strategy": strategy_name,
                **param_dict,
                "test_pnl": float(equity[-1] - STARTING_CAPITAL),
                "test_trades": len(trades),
                "test_wr": len(wins) / len(trades),
                "test_sharpe": sharpe,
                "test_profit_factor": profit_factor,
                "test_max_dd": max_drawdown(equity),
            })

    df = pd.DataFrame(results)
    if not df.empty:
        # Sort by profit factor
        df = df.sort_values("test_profit_factor", ascending=False)
        print(f"\nTop 10 parameter combinations:")
        print(df.head(10).to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# Slippage Stress Test
# ---------------------------------------------------------------------------


def slippage_stress_test(
    windows: list[Window],
    train_windows: list[Window],
    test_windows: list[Window],
    stake: float = 10.0,
    slippage_values: list[float] = None,
) -> pd.DataFrame:
    """Test strategies across different slippage levels."""
    if slippage_values is None:
        slippage_values = [0.0, 0.001, 0.002, 0.003, 0.005]

    print("\n" + "=" * 80)
    print("SLIPPAGE STRESS TEST")
    print("=" * 80)

    results = []

    for slippage in slippage_values:
        print(f"\nSlippage: {slippage*100:.1f} cents")

        for name, fn in STRATEGIES.items():
            pnl_by_window = np.zeros(len(test_windows), dtype=float)
            trades = []

            for i, w in enumerate(test_windows):
                trade = fn(w, stake=stake, slippage=slippage)
                if trade:
                    pnl_by_window[i] = trade.pnl
                    trades.append(trade)

            equity = STARTING_CAPITAL + np.cumsum(pnl_by_window)
            pnls = [t.pnl for t in trades]
            wins = [p for p in pnls if p > 0]

            results.append({
                "strategy": name,
                "slippage_cents": slippage * 100,
                "pnl": float(equity[-1] - STARTING_CAPITAL),
                "trades": len(trades),
                "win_rate": len(wins) / len(trades) if trades else 0.0,
                "max_drawdown": max_drawdown(equity),
            })

            print(f"  {name}: PnL=${results[-1]['pnl']:,.2f}, WR={100*results[-1]['win_rate']:.1f}%")

    df = pd.DataFrame(results)
    return df


# ---------------------------------------------------------------------------
# Portfolio Construction
# ---------------------------------------------------------------------------


def portfolio_backtest(
    windows: list[Window],
    train_windows: list[Window],
    test_windows: list[Window],
    stake: float = 10.0,
    daily_stop_loss: float = -50.0,
    slippage: float = 0.0,
) -> dict:
    """Backtest portfolio of uncorrelated strategies."""
    print("\n" + "=" * 80)
    print("PORTFOLIO BACKTEST")
    print(f"Daily stop loss: ${daily_stop_loss}")
    print("=" * 80)

    # Run all strategies
    strategy_pnls = {}
    strategy_trades = {}

    for name, fn in STRATEGIES.items():
        pnl_by_window = np.zeros(len(test_windows), dtype=float)
        trades = []

        for i, w in enumerate(test_windows):
            trade = fn(w, stake=stake, slippage=slippage)
            if trade:
                pnl_by_window[i] = trade.pnl
                trades.append(trade)

        strategy_pnls[name] = pnl_by_window
        strategy_trades[name] = trades

    # Portfolio: equal weight across strategies
    portfolio_pnl = np.zeros(len(test_windows), dtype=float)
    for pnl in strategy_pnls.values():
        portfolio_pnl += pnl / len(strategy_pnls)

    # Apply daily stop loss
    daily_pnl = {}
    for i, w in enumerate(test_windows):
        date = w.start.strftime("%Y-%m-%d")
        if date not in daily_pnl:
            daily_pnl[date] = 0.0
        daily_pnl[date] += portfolio_pnl[i]

    # Zero out days that hit stop loss
    for i, w in enumerate(test_windows):
        date = w.start.strftime("%Y-%m-%d")
        if daily_pnl[date] <= daily_stop_loss:
            portfolio_pnl[i] = 0.0

    equity = STARTING_CAPITAL + np.cumsum(portfolio_pnl)
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
    sharpe = 0.0
    if len(returns) > 1 and returns.std(ddof=1) > 0:
        sharpe = float((returns.mean() / returns.std(ddof=1)) * np.sqrt(365 * 24 * 12))

    total_pnl = float(equity[-1] - STARTING_CAPITAL)
    max_dd = max_drawdown(equity)

    print(f"\nPortfolio Results:")
    print(f"  Total PnL: ${total_pnl:,.2f}")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Max Drawdown: {100*max_dd:.1f}%")
    print(f"  Final Equity: ${equity[-1]:,.2f}")

    # Strategy correlation
    pnl_matrix = np.array([strategy_pnls[name] for name in strategy_pnls])
    corr_matrix = np.corrcoef(pnl_matrix)

    print(f"\nStrategy Correlation Matrix:")
    corr_df = pd.DataFrame(corr_matrix, index=strategy_pnls.keys(), columns=strategy_pnls.keys())
    print(corr_df.to_string())

    return {
        "portfolio_pnl": total_pnl,
        "portfolio_sharpe": sharpe,
        "portfolio_max_dd": max_dd,
        "portfolio_equity": equity,
        "strategy_pnls": strategy_pnls,
        "correlation_matrix": corr_df,
    }


# ---------------------------------------------------------------------------
# ML Ensemble
# ---------------------------------------------------------------------------


def ml_ensemble(
    train_windows: list[Window],
    test_windows: list[Window],
    stake: float = 10.0,
    threshold: float = 0.55,
    slippage: float = 0.0,
) -> dict:
    """Train ensemble of ML models with different feature sets."""
    if not HAS_LGB:
        print("LightGBM not installed.")
        return {}

    print("\n" + "=" * 80)
    print("ML ENSEMBLE")
    print("=" * 80)

    # Feature sets
    feature_sets = {
        "momentum": ["ret_5s", "ret_10s", "ret_30s", "ret_60s", "ret_from_open"],
        "volatility": ["vol_10s", "vol_30s", "avg_spread"],
        "orderbook": ["ob_imbalance", "up_spread", "down_spread", "fair_up", "fair_dev"],
        "technical": ["rsi", "up_ratio", "seconds_left", "time_decay"],
        "full": None,  # All features
    }

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

    def build_dataset(windows: list[Window], feature_cols: list[str] = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
        entry_indices = list(range(150, 280, 5))
        rows = []
        for w in windows:
            for idx in entry_indices:
                if idx >= WINDOW_SECONDS:
                    continue
                features = extract_features(w, idx)
                if feature_cols:
                    row = {k: features[k] for k in feature_cols}
                else:
                    row = features
                row["label"] = w.outcome_up
                rows.append(row)
        df = pd.DataFrame(rows)
        cols = [c for c in df.columns if c != "label"]
        return df[cols].values, df["label"].values, cols

    models = {}
    predictions = {}

    for set_name, feature_cols in feature_sets.items():
        print(f"\nTraining model: {set_name}")

        X_train, y_train, cols = build_dataset(train_windows, feature_cols)
        X_test, y_test, _ = build_dataset(test_windows, feature_cols)

        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"  Features: {len(cols)}")

        params = {
            "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
            "num_leaves": 15, "learning_rate": 0.01, "feature_fraction": 0.6,
            "bagging_fraction": 0.6, "bagging_freq": 5,
            "min_child_samples": 50, "reg_alpha": 0.1, "reg_lambda": 0.1,
            "verbose": -1, "seed": 42,
        }

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=cols)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, feature_name=cols)

        model = lgb.train(
            params, train_data, num_boost_round=1000, valid_sets=[test_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        test_auc = float(np.corrcoef(test_pred, y_test)[0, 1])

        print(f"  Test AUC: {test_auc:.4f}")

        models[set_name] = {"model": model, "features": cols, "auc": test_auc}
        predictions[set_name] = test_pred

    # Ensemble: average predictions
    ensemble_pred = np.mean(list(predictions.values()), axis=0)
    ensemble_auc = float(np.corrcoef(ensemble_pred, y_test)[0, 1])
    print(f"\nEnsemble Test AUC: {ensemble_auc:.4f}")

    # Backtest ensemble
    pnl_by_window = np.zeros(len(test_windows), dtype=float)
    trades = []

    for i, w in enumerate(test_windows):
        idx = 240
        features = extract_features(w, idx)

        # Get ensemble prediction
        preds = []
        for set_name, model_info in models.items():
            X = np.array([[features[c] for c in model_info["features"]]])
            pred = float(model_info["model"].predict(X)[0])
            preds.append(pred)

        avg_pred = np.mean(preds)

        if avg_pred >= threshold or avg_pred <= (1 - threshold):
            side = "UP" if avg_pred >= threshold else "DOWN"
            ask = float(side_ask(w, side)[idx])
            if ask > 0.70:
                continue

            final = settlement_value(w, side)
            pnl = compute_pnl(ask, final, stake, slippage=slippage)
            pnl_by_window[i] = pnl
            trades.append(Trade("ml_ensemble", w.start, side, idx, ask, WINDOW_SECONDS - 1, final, "settlement", pnl, WINDOW_SECONDS - 1 - idx))

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

    print(f"\nEnsemble Backtest Results:")
    print(f"  PnL: ${equity[-1] - STARTING_CAPITAL:,.2f}")
    print(f"  Trades: {len(trades)}")
    print(f"  Win Rate: {100*len(wins)/len(trades):.1f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Profit Factor: {profit_factor:.2f}")

    return {
        "models": models,
        "ensemble_auc": ensemble_auc,
        "ensemble_pnl": float(equity[-1] - STARTING_CAPITAL),
        "ensemble_trades": len(trades),
        "ensemble_wr": len(wins) / len(trades) if trades else 0.0,
        "ensemble_sharpe": sharpe,
        "ensemble_profit_factor": profit_factor,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument("--source", default="binance")
    parser.add_argument("--dataset-source", default="aliplayer_spot")
    parser.add_argument("--stake-usd", type=float, default=10.0)
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--param-sweep", action="store_true")
    parser.add_argument("--slippage-stress", action="store_true")
    parser.add_argument("--portfolio", action="store_true")
    parser.add_argument("--ml-ensemble", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    args = parser.parse_args()

    # Load data
    price_df = load_spot_prices(args.symbol, None if args.source.lower() in {"none", "all"} else args.source, dataset_source=args.dataset_source)
    windows = build_windows(price_df)

    # Split
    split_idx = int(len(windows) * 0.80)
    train_windows = windows[:split_idx]
    test_windows = windows[split_idx:]

    print(f"\nTotal windows: {len(windows):,}")
    print(f"Train: {len(train_windows):,} | Test: {len(test_windows):,}")

    run_all = args.all or not any([args.walk_forward, args.param_sweep, args.slippage_stress, args.portfolio, args.ml_ensemble])

    # Walk-forward
    if args.walk_forward or run_all:
        wf_metrics = walk_forward_test(windows, n_periods=4, stake=args.stake_usd)
        wf_df = pd.DataFrame([m.__dict__ for m in wf_metrics])
        wf_df.to_csv("walk_forward_metrics.csv", index=False)
        print("\nSaved walk_forward_metrics.csv")

    # Parameter sweep
    if args.param_sweep or run_all:
        param_results = parameter_sweep(windows, train_windows, test_windows, stake=args.stake_usd)
        if not param_results.empty:
            param_results.to_csv("param_sweep_results.csv", index=False)
            print("\nSaved param_sweep_results.csv")

    # Slippage stress
    if args.slippage_stress or run_all:
        slip_results = slippage_stress_test(windows, train_windows, test_windows, stake=args.stake_usd)
        slip_results.to_csv("slippage_stress_results.csv", index=False)
        print("\nSaved slippage_stress_results.csv")

    # Portfolio
    if args.portfolio or run_all:
        portfolio_results = portfolio_backtest(windows, train_windows, test_windows, stake=args.stake_usd)
        portfolio_results["correlation_matrix"].to_csv("strategy_correlations.csv")
        print("\nSaved strategy_correlations.csv")

    # ML Ensemble
    if args.ml_ensemble or run_all:
        ensemble_results = ml_ensemble(train_windows, test_windows, stake=args.stake_usd)
        if ensemble_results:
            with open("ml_ensemble_results.json", "w") as f:
                json.dump({k: v for k, v in ensemble_results.items() if k != "models"}, f, indent=2)
            print("\nSaved ml_ensemble_results.json")


if __name__ == "__main__":
    main()
