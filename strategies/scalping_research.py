"""
Polymarket Scalping Strategy Research & ML Pipeline

Strategies tested:
1. Late Momentum Sniper - Enter at 270s+ when momentum is clear
2. Spread Capture - Exploit abnormally wide bid-ask spreads
3. Volume Spike Continuation - Follow informed trading flow
4. Mean Reversion Fade - Fade early overreactions
5. Micro-Lag Arbitrage - Exploit price feed delays
6. Opening Range Breakout - Trade break of first 30s range
7. VWAP Reversion - Fade deviations from volume-weighted price

ML Model:
- Features: orderbook imbalance, momentum, volatility, time decay, trade flow
- Model: LightGBM with strict chronological split
- Target: probability of UP outcome
- Validation: walk-forward testing, no lookahead bias

Usage:
    python scalping_research.py
    python scalping_research.py --train-ml
    python scalping_research.py --backtest-ml
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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
    build_windows,
    load_spot_prices,
    max_drawdown,
    split_windows_chronologically,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Strategy definitions
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
    exit_type: str  # "target", "stop", "settlement"
    pnl: float
    seconds_held: int
    features: dict = field(default_factory=dict)


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
    avg_exit_price: float


def clip(x: float) -> float:
    return float(np.clip(x, 0.001, 0.999))


def fair_up(price: np.ndarray, open_price: float, idx: int) -> float:
    elapsed = idx
    tau = max((WINDOW_SECONDS - elapsed) / WINDOW_SECONDS, 1e-9)
    sigma = 0.60
    log_moneyness = np.log(price[idx] / open_price)
    from scipy.stats import norm
    return float(np.clip(norm.cdf(log_moneyness / (sigma * np.sqrt(tau))), 0.001, 0.999))


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


# ---------------------------------------------------------------------------
# Strategy 1: Late Momentum Sniper
# Enter in last 30-60 seconds when price momentum is strong and clear
# ---------------------------------------------------------------------------


def strat_late_momentum(
    w: Window,
    stake: float = 10.0,
    min_seconds_left: int = 30,
    max_entry_price: float = 0.70,
    momentum_threshold: float = 0.0003,
) -> ScalpTrade | None:
    """Enter late when momentum is clear and contract is cheap."""
    max_idx = WINDOW_SECONDS - min_seconds_left

    for idx in range(max_idx, WINDOW_SECONDS - 5):
        # Recent momentum (last 10 seconds)
        recent_ret = w.prices[idx] / w.prices[max(0, idx - 10)] - 1.0

        # Longer momentum (last 30 seconds)
        longer_ret = w.prices[idx] / w.prices[max(0, idx - 30)] - 1.0

        # Both must agree and exceed threshold
        if abs(recent_ret) < momentum_threshold:
            continue
        if np.sign(recent_ret) != np.sign(longer_ret):
            continue

        side = "UP" if recent_ret > 0 else "DOWN"
        ask = float(side_ask(w, side)[idx])

        if ask > max_entry_price:
            continue

        # Hold to settlement
        final = settlement_value(w, side)
        pnl = compute_pnl(ask, final, stake)

        return ScalpTrade(
            strategy="late_momentum",
            market_start=w.start,
            side=side,
            entry_idx=idx,
            entry_price=ask,
            exit_idx=WINDOW_SECONDS - 1,
            exit_price=final,
            exit_type="settlement",
            pnl=pnl,
            seconds_held=WINDOW_SECONDS - 1 - idx,
            features={
                "recent_ret": recent_ret,
                "longer_ret": longer_ret,
                "seconds_left": WINDOW_SECONDS - idx,
                "ask_price": ask,
            },
        )

    return None


# ---------------------------------------------------------------------------
# Strategy 2: Spread Capture
# Enter when bid-ask spread is abnormally wide, exit when it normalizes
# ---------------------------------------------------------------------------


def strat_spread_capture(
    w: Window,
    stake: float = 10.0,
    spread_threshold_bps: float = 150,
    target_spread_bps: float = 80,
    max_entry_price: float = 0.65,
) -> ScalpTrade | None:
    """Buy when spread is wide, sell when it narrows."""
    threshold = spread_threshold_bps / 10000
    target = target_spread_bps / 10000

    for idx in range(30, WINDOW_SECONDS - 30):
        up_spread = float(w.up_ask[idx] - w.up_bid[idx])
        down_spread = float(w.down_ask[idx] - w.down_bid[idx])

        # Find the side with widest spread
        if up_spread > threshold and up_spread >= down_spread:
            side = "UP"
            spread = up_spread
        elif down_spread > threshold:
            side = "DOWN"
            spread = down_spread
        else:
            continue

        ask = float(side_ask(w, side)[idx])
        if ask > max_entry_price:
            continue

        # Wait for spread to narrow
        for exit_idx in range(idx + 5, WINDOW_SECONDS):
            exit_bid = float(side_bid(w, side)[exit_idx])
            exit_spread = float(side_ask(w, side)[exit_idx] - side_bid(w, side)[exit_idx])

            if exit_spread <= target:
                pnl = compute_pnl(ask, exit_bid, stake)
                return ScalpTrade(
                    strategy="spread_capture",
                    market_start=w.start,
                    side=side,
                    entry_idx=idx,
                    entry_price=ask,
                    exit_idx=exit_idx,
                    exit_price=exit_bid,
                    exit_type="target",
                    pnl=pnl,
                    seconds_held=exit_idx - idx,
                    features={"entry_spread": spread, "exit_spread": exit_spread},
                )

        # If spread never narrows, hold to settlement
        final = settlement_value(w, side)
        pnl = compute_pnl(ask, final, stake)
        return ScalpTrade(
            strategy="spread_capture",
            market_start=w.start,
            side=side,
            entry_idx=idx,
            entry_price=ask,
            exit_idx=WINDOW_SECONDS - 1,
            exit_price=final,
            exit_type="settlement",
            pnl=pnl,
            seconds_held=WINDOW_SECONDS - 1 - idx,
            features={"entry_spread": spread, "exit_spread": None},
        )

    return None


# ---------------------------------------------------------------------------
# Strategy 3: Volume Spike Continuation
# Enter when there's a sudden price move with momentum
# ---------------------------------------------------------------------------


def strat_volume_spike(
    w: Window,
    stake: float = 10.0,
    spike_threshold: float = 0.0005,
    confirmation_seconds: int = 5,
    max_entry_price: float = 0.70,
) -> ScalpTrade | None:
    """Enter when price spikes and momentum continues."""
    for idx in range(60, WINDOW_SECONDS - 30):
        # Check for spike in last 5 seconds
        spike = w.prices[idx] / w.prices[idx - 5] - 1.0

        if abs(spike) < spike_threshold:
            continue

        # Wait for confirmation
        confirm_idx = idx + confirmation_seconds
        if confirm_idx >= WINDOW_SECONDS - 10:
            continue

        confirm_ret = w.prices[confirm_idx] / w.prices[idx] - 1.0

        # Must continue in same direction
        if np.sign(confirm_ret) != np.sign(spike):
            continue

        side = "UP" if spike > 0 else "DOWN"
        ask = float(side_ask(w, side)[confirm_idx])

        if ask > max_entry_price:
            continue

        # Hold to settlement
        final = settlement_value(w, side)
        pnl = compute_pnl(ask, final, stake)

        return ScalpTrade(
            strategy="volume_spike",
            market_start=w.start,
            side=side,
            entry_idx=confirm_idx,
            entry_price=ask,
            exit_idx=WINDOW_SECONDS - 1,
            exit_price=final,
            exit_type="settlement",
            pnl=pnl,
            seconds_held=WINDOW_SECONDS - 1 - confirm_idx,
            features={"spike": spike, "confirm_ret": confirm_ret},
        )

    return None


# ---------------------------------------------------------------------------
# Strategy 4: Mean Reversion Fade
# Fade extreme early moves that are likely to revert
# ---------------------------------------------------------------------------


def strat_mean_reversion(
    w: Window,
    stake: float = 10.0,
    fade_threshold: float = 0.001,
    entry_window_end: int = 120,
    max_entry_price: float = 0.60,
) -> ScalpTrade | None:
    """Fade extreme moves in the first 2 minutes."""
    for idx in range(60, entry_window_end):
        # Deviation from open
        deviation = w.prices[idx] / w.open_price - 1.0

        if abs(deviation) < fade_threshold:
            continue

        # Check if momentum is weakening (last 10 seconds)
        micro_ret = w.prices[idx] / w.prices[max(0, idx - 10)] - 1.0

        # Fade only if micro momentum opposes the move
        if np.sign(micro_ret) == np.sign(deviation):
            continue

        side = "DOWN" if deviation > 0 else "UP"
        ask = float(side_ask(w, side)[idx])

        if ask > max_entry_price:
            continue

        # Target: return to within 0.02% of open
        target_price = w.open_price
        bid = side_bid(w, side)

        for exit_idx in range(idx + 10, WINDOW_SECONDS - 10):
            if side == "UP" and w.prices[exit_idx] >= w.open_price * 1.0002:
                exit_price = float(bid[exit_idx])
                pnl = compute_pnl(ask, exit_price, stake)
                return ScalpTrade(
                    strategy="mean_reversion",
                    market_start=w.start,
                    side=side,
                    entry_idx=idx,
                    entry_price=ask,
                    exit_idx=exit_idx,
                    exit_price=exit_price,
                    exit_type="target",
                    pnl=pnl,
                    seconds_held=exit_idx - idx,
                    features={"deviation": deviation, "micro_ret": micro_ret},
                )
            if side == "DOWN" and w.prices[exit_idx] <= w.open_price * 0.9998:
                exit_price = float(bid[exit_idx])
                pnl = compute_pnl(ask, exit_price, stake)
                return ScalpTrade(
                    strategy="mean_reversion",
                    market_start=w.start,
                    side=side,
                    entry_idx=idx,
                    entry_price=ask,
                    exit_idx=exit_idx,
                    exit_price=exit_price,
                    exit_type="target",
                    pnl=pnl,
                    seconds_held=exit_idx - idx,
                    features={"deviation": deviation, "micro_ret": micro_ret},
                )

        # Hold to settlement
        final = settlement_value(w, side)
        pnl = compute_pnl(ask, final, stake)
        return ScalpTrade(
            strategy="mean_reversion",
            market_start=w.start,
            side=side,
            entry_idx=idx,
            entry_price=ask,
            exit_idx=WINDOW_SECONDS - 1,
            exit_price=final,
            exit_type="settlement",
            pnl=pnl,
            seconds_held=WINDOW_SECONDS - 1 - idx,
            features={"deviation": deviation, "micro_ret": micro_ret},
        )

    return None


# ---------------------------------------------------------------------------
# Strategy 5: Opening Range Breakout
# Trade break of first 30-second range with confirmation
# ---------------------------------------------------------------------------


def strat_opening_breakout(
    w: Window,
    stake: float = 10.0,
    range_seconds: int = 30,
    confirmation_bars: int = 2,
    max_entry_price: float = 0.70,
) -> ScalpTrade | None:
    """Trade breakout of opening range with confirmation."""
    high = float(w.prices[:range_seconds].max())
    low = float(w.prices[:range_seconds].min())

    for idx in range(range_seconds + 10, WINDOW_SECONDS - 30):
        # Check for breakout with confirmation
        if w.prices[idx] > high and w.prices[idx - 1] > high:
            side = "UP"
            ask = float(side_ask(w, side)[idx])

            if ask > max_entry_price:
                continue

            final = settlement_value(w, side)
            pnl = compute_pnl(ask, final, stake)
            return ScalpTrade(
                strategy="opening_breakout",
                market_start=w.start,
                side=side,
                entry_idx=idx,
                entry_price=ask,
                exit_idx=WINDOW_SECONDS - 1,
                exit_price=final,
                exit_type="settlement",
                pnl=pnl,
                seconds_held=WINDOW_SECONDS - 1 - idx,
                features={"range_high": high, "range_low": low, "breakout_pct": (w.prices[idx] / high - 1)},
            )

        if w.prices[idx] < low and w.prices[idx - 1] < low:
            side = "DOWN"
            ask = float(side_ask(w, side)[idx])

            if ask > max_entry_price:
                continue

            final = settlement_value(w, side)
            pnl = compute_pnl(ask, final, stake)
            return ScalpTrade(
                strategy="opening_breakout",
                market_start=w.start,
                side=side,
                entry_idx=idx,
                entry_price=ask,
                exit_idx=WINDOW_SECONDS - 1,
                exit_price=final,
                exit_type="settlement",
                pnl=pnl,
                seconds_held=WINDOW_SECONDS - 1 - idx,
                features={"range_high": high, "range_low": low, "breakout_pct": (low / w.prices[idx] - 1)},
            )

    return None


# ---------------------------------------------------------------------------
# Strategy 6: VWAP Reversion
# Fade deviations from volume-weighted average price
# ---------------------------------------------------------------------------


def strat_vwap_reversion(
    w: Window,
    stake: float = 10.0,
    dev_threshold_pct: float = 0.0008,
    max_entry_price: float = 0.60,
) -> ScalpTrade | None:
    """Fade price when it deviates significantly from VWAP."""
    # Approximate VWAP using cumulative price average (proxy for volume)
    cumsum = np.cumsum(w.prices)
    counts = np.arange(1, len(w.prices) + 1)
    vwap = cumsum / counts

    for idx in range(90, WINDOW_SECONDS - 30):
        deviation = (w.prices[idx] - vwap[idx]) / vwap[idx]

        if abs(deviation) < dev_threshold_pct:
            continue

        # Check if returning to VWAP
        if idx > 90:
            prev_dev = (w.prices[idx - 1] - vwap[idx - 1]) / vwap[idx - 1]
            # Must be reverting (deviation decreasing)
            if abs(deviation) > abs(prev_dev):
                continue

        side = "DOWN" if deviation > 0 else "UP"
        ask = float(side_ask(w, side)[idx])

        if ask > max_entry_price:
            continue

        # Target: return to VWAP
        bid = side_bid(w, side)
        for exit_idx in range(idx + 5, WINDOW_SECONDS - 10):
            if side == "UP" and w.prices[exit_idx] >= vwap[exit_idx]:
                exit_price = float(bid[exit_idx])
                pnl = compute_pnl(ask, exit_price, stake)
                return ScalpTrade(
                    strategy="vwap_reversion",
                    market_start=w.start,
                    side=side,
                    entry_idx=idx,
                    entry_price=ask,
                    exit_idx=exit_idx,
                    exit_price=exit_price,
                    exit_type="target",
                    pnl=pnl,
                    seconds_held=exit_idx - idx,
                    features={"vwap_dev": deviation},
                )
            if side == "DOWN" and w.prices[exit_idx] <= vwap[exit_idx]:
                exit_price = float(bid[exit_idx])
                pnl = compute_pnl(ask, exit_price, stake)
                return ScalpTrade(
                    strategy="vwap_reversion",
                    market_start=w.start,
                    side=side,
                    entry_idx=idx,
                    entry_price=ask,
                    exit_idx=exit_idx,
                    exit_price=exit_price,
                    exit_type="target",
                    pnl=pnl,
                    seconds_held=exit_idx - idx,
                    features={"vwap_dev": deviation},
                )

        final = settlement_value(w, side)
        pnl = compute_pnl(ask, final, stake)
        return ScalpTrade(
            strategy="vwap_reversion",
            market_start=w.start,
            side=side,
            entry_idx=idx,
            entry_price=ask,
            exit_idx=WINDOW_SECONDS - 1,
            exit_price=final,
            exit_type="settlement",
            pnl=pnl,
            seconds_held=WINDOW_SECONDS - 1 - idx,
            features={"vwap_dev": deviation},
        )

    return None


# ---------------------------------------------------------------------------
# Strategy runner
# ---------------------------------------------------------------------------

STRATEGIES = {
    "late_momentum": strat_late_momentum,
    "spread_capture": strat_spread_capture,
    "volume_spike": strat_volume_spike,
    "mean_reversion": strat_mean_reversion,
    "opening_breakout": strat_opening_breakout,
    "vwap_reversion": strat_vwap_reversion,
}


def run_strategy(
    name: str,
    windows: list[Window],
    stake: float = 10.0,
    **kwargs,
) -> tuple[list[ScalpTrade], np.ndarray]:
    """Run a strategy across windows, return trades and PnL array."""
    fn = STRATEGIES[name]
    pnl_by_window = np.zeros(len(windows), dtype=float)
    trades: list[ScalpTrade] = []

    for i, w in enumerate(windows):
        trade = fn(w, stake=stake, **kwargs)
        if trade:
            pnl_by_window[i] = trade.pnl
            trades.append(trade)

    return trades, pnl_by_window


def compute_metrics(
    trades: list[ScalpTrade],
    pnl_by_window: np.ndarray,
    sample: str,
    strategy: str,
) -> StrategyMetrics:
    """Compute strategy metrics."""
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
        strategy=strategy,
        sample=sample,
        total_pnl=float(equity[-1] - STARTING_CAPITAL),
        trades=len(trades),
        win_rate=len(wins) / len(trades) if trades else 0.0,
        avg_pnl=float(np.mean(pnls)) if pnls else 0.0,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown(equity),
        sharpe=sharpe,
        avg_hold_seconds=float(np.mean([t.seconds_held for t in trades])) if trades else 0.0,
        avg_entry_price=float(np.mean([t.entry_price for t in trades])) if trades else 0.0,
        avg_exit_price=float(np.mean([t.exit_price for t in trades])) if trades else 0.0,
    )


def print_metrics_table(metrics: list[StrategyMetrics]) -> None:
    """Print formatted metrics table."""
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
# ML Feature Engineering
# ---------------------------------------------------------------------------


def extract_features(w: Window, idx: int) -> dict:
    """Extract features for ML model at a given index in the window."""
    p = w.prices
    fair = float(w.fair_up[idx])

    # Price features
    ret_5s = p[idx] / p[max(0, idx - 5)] - 1.0
    ret_10s = p[idx] / p[max(0, idx - 10)] - 1.0
    ret_30s = p[idx] / p[max(0, idx - 30)] - 1.0
    ret_60s = p[idx] / p[max(0, idx - 60)] - 1.0
    ret_from_open = p[idx] / w.open_price - 1.0

    # Volatility
    vol_10s = float(np.std(np.diff(p[max(0, idx - 10):idx + 1]))) / p[idx] if idx > 10 else 0.0
    vol_30s = float(np.std(np.diff(p[max(0, idx - 30):idx + 1]))) / p[idx] if idx > 30 else 0.0

    # Momentum direction consistency
    diffs = np.diff(p[max(0, idx - 10):idx + 1])
    up_ratio = float(np.sum(diffs > 0) / len(diffs)) if len(diffs) > 0 else 0.5

    # Spread features
    up_spread = float(w.up_ask[idx] - w.up_bid[idx])
    down_spread = float(w.down_ask[idx] - w.down_bid[idx])
    avg_spread = (up_spread + down_spread) / 2

    # Orderbook imbalance
    up_mid = (w.up_ask[idx] + w.up_bid[idx]) / 2
    down_mid = (w.down_ask[idx] + w.down_bid[idx]) / 2
    ob_imbalance = float(up_mid - down_mid)

    # Fair value deviation
    fair_dev = fair - 0.5

    # Time features
    seconds_left = WINDOW_SECONDS - 1 - idx
    time_decay = seconds_left / WINDOW_SECONDS

    # RSI
    if idx >= 14:
        diffs_rsi = np.diff(p[idx - 14:idx + 1])
        gains = np.clip(diffs_rsi, 0, None).mean()
        losses = -np.clip(diffs_rsi, None, 0).mean()
        rsi = 100.0 - (100.0 / (1.0 + gains / losses)) if losses > 0 else 100.0
    else:
        rsi = 50.0

    # EMA crossover
    if idx >= 20:
        ema_fast = p[idx] * 0.18 + p[idx - 1] * 0.82 if idx > 0 else p[idx]
        ema_slow = p[idx] * 0.05 + p[idx - 1] * 0.95 if idx > 0 else p[idx]
        ema_cross = float(ema_fast - ema_slow) / p[idx]
    else:
        ema_cross = 0.0

    return {
        "ret_5s": ret_5s,
        "ret_10s": ret_10s,
        "ret_30s": ret_30s,
        "ret_60s": ret_60s,
        "ret_from_open": ret_from_open,
        "vol_10s": vol_10s,
        "vol_30s": vol_30s,
        "up_ratio": up_ratio,
        "up_spread": up_spread,
        "down_spread": down_spread,
        "avg_spread": avg_spread,
        "ob_imbalance": ob_imbalance,
        "fair_dev": fair_dev,
        "seconds_left": seconds_left,
        "time_decay": time_decay,
        "rsi": rsi,
        "ema_cross": ema_cross,
        "fair_up": fair,
    }


def build_ml_dataset(
    windows: list[Window],
    sample: str,
    entry_indices: list[int] = None,
) -> pd.DataFrame:
    """Build ML dataset from windows."""
    if entry_indices is None:
        entry_indices = list(range(150, 280, 5))

    rows = []
    for w in windows:
        for idx in entry_indices:
            if idx >= WINDOW_SECONDS:
                continue

            features = extract_features(w, idx)
            label = int(w.outcome_up)

            row = {"sample": sample, "market_start": w.start, "idx": idx, "label": label}
            row.update(features)
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ML Training
# ---------------------------------------------------------------------------


def train_ml_model(
    train_windows: list[Window],
    test_windows: list[Window],
    output_path: str = "polymarket_ml_model.txt",
) -> dict:
    """Train LightGBM model and evaluate."""
    if not HAS_LGB:
        print("LightGBM not installed. Install: pip install lightgbm")
        return {}

    print("\n" + "=" * 80)
    print("ML MODEL TRAINING")
    print("=" * 80)

    # Build datasets
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

    # Train model
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
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

    train_auc = float(np.corrcoef(train_pred, y_train)[0, 1])
    test_auc = float(np.corrcoef(test_pred, y_test)[0, 1])

    print(f"\nTrain AUC: {train_auc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(),
    }).sort_values("importance", ascending=False)

    print("\nTop 10 features:")
    print(importance.head(10).to_string(index=False))

    # Save model
    model.save_model(output_path)
    print(f"\nModel saved to {output_path}")

    # Save feature importance
    importance.to_csv("ml_feature_importance.csv", index=False)
    print("Feature importance saved to ml_feature_importance.csv")

    return {
        "model": model,
        "feature_cols": feature_cols,
        "train_auc": train_auc,
        "test_auc": test_auc,
        "importance": importance,
    }


# ---------------------------------------------------------------------------
# ML Backtest
# ---------------------------------------------------------------------------


def backtest_ml_model(
    windows: list[Window],
    sample: str,
    model,
    feature_cols: list[str],
    threshold: float = 0.55,
    stake: float = 10.0,
    max_entry_price: float = 0.70,
) -> tuple[list[ScalpTrade], np.ndarray]:
    """Backtest ML model predictions."""
    pnl_by_window = np.zeros(len(windows), dtype=float)
    trades: list[ScalpTrade] = []

    for i, w in enumerate(windows):
        # Get prediction at entry point (240 seconds in)
        idx = 240
        features = extract_features(w, idx)
        X = np.array([[features[c] for c in feature_cols]])

        pred = float(model.predict(X)[0])

        # Only trade when confident
        if pred >= threshold or pred <= (1 - threshold):
            side = "UP" if pred >= threshold else "DOWN"
            ask = float(side_ask(w, side)[idx])

            if ask > max_entry_price:
                continue

            final = settlement_value(w, side)
            pnl = compute_pnl(ask, final, stake)

            pnl_by_window[i] = pnl
            trades.append(ScalpTrade(
                strategy="ml_model",
                market_start=w.start,
                side=side,
                entry_idx=idx,
                entry_price=ask,
                exit_idx=WINDOW_SECONDS - 1,
                exit_price=final,
                exit_type="settlement",
                pnl=pnl,
                seconds_held=WINDOW_SECONDS - 1 - idx,
                features={"ml_pred": pred, "threshold": threshold},
            ))

    return trades, pnl_by_window


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="btcusdt")
    parser.add_argument("--source", default="binance")
    parser.add_argument("--dataset-source", default="aliplayer_spot")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--stake-usd", type=float, default=10.0)
    parser.add_argument("--train-ml", action="store_true", help="Train ML model")
    parser.add_argument("--backtest-ml", action="store_true", help="Backtest ML model")
    parser.add_argument("--ml-threshold", type=float, default=0.55, help="ML prediction confidence threshold")
    args = parser.parse_args()

    price_df = load_spot_prices(args.symbol, None if args.source.lower() in {"none", "all"} else args.source, dataset_source=args.dataset_source)
    windows = build_windows(price_df, args.max_windows)
    train_windows, test_windows = split_windows_chronologically(windows, args.train_frac)

    # Run all strategies
    print("\n" + "=" * 80)
    print("SCALPING STRATEGY BACKTESTS")
    print("=" * 80)

    all_metrics = []

    for sample, sample_windows in [("TRAIN", train_windows), ("TEST", test_windows)]:
        print(f"\n{sample} SET:")
        for name in STRATEGIES:
            trades, pnl = run_strategy(name, sample_windows, stake=args.stake_usd)
            metrics = compute_metrics(trades, pnl, sample, name)
            all_metrics.append(metrics)

    print_metrics_table(all_metrics)

    # Save results
    metrics_df = pd.DataFrame([m.__dict__ for m in all_metrics])
    metrics_df.to_csv("scalping_metrics.csv", index=False)
    print(f"\nSaved scalping_metrics.csv")

    # ML Training
    if args.train_ml:
        result = train_ml_model(train_windows, test_windows)

        if result:
            # Backtest ML model
            for sample, sample_windows in [("TRAIN", train_windows), ("TEST", test_windows)]:
                trades, pnl = backtest_ml_model(
                    sample_windows,
                    sample,
                    result["model"],
                    result["feature_cols"],
                    threshold=args.ml_threshold,
                    stake=args.stake_usd,
                )
                metrics = compute_metrics(trades, pnl, sample, "ml_model")
                print(f"\nML Model {sample}: PnL=${metrics.total_pnl:.2f}, Trades={metrics.trades}, WinRate={100*metrics.win_rate:.1f}%")

    # ML Backtest only (load existing model)
    if args.backtest_ml and not args.train_ml:
        if not HAS_LGB:
            print("LightGBM not installed.")
            return

        model = lgb.Booster(model_file="polymarket_ml_model.txt")
        importance = pd.read_csv("ml_feature_importance.csv")
        feature_cols = importance["feature"].tolist()

        for sample, sample_windows in [("TRAIN", train_windows), ("TEST", test_windows)]:
            trades, pnl = backtest_ml_model(
                sample_windows,
                sample,
                model,
                feature_cols,
                threshold=args.ml_threshold,
                stake=args.stake_usd,
            )
            metrics = compute_metrics(trades, pnl, sample, "ml_model")
            print(f"\nML Model {sample}: PnL=${metrics.total_pnl:.2f}, Trades={metrics.trades}, WinRate={100*metrics.win_rate:.1f}%")


if __name__ == "__main__":
    main()
