"""
Enhanced Buy97 Strategy Suite v2

Addresses the core problem: 87% WR with 0.50:1 payoff ratio is fragile.

New strategies:
1. MomentumReversalDetector - Neural net to predict reversals
2. ConfidenceScorer - ML model to score entry quality
3. OppositeSideHedge - Buy opposite side when reversal detected
4. MultiFeatureFilter - Combine momentum, volatility, time-decay
5. AdaptivePositionSizing - Size based on predicted confidence

Usage:
    python enhanced_buy97_strategies_v2.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

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
    confidence: float = 0.0


def side_prices(w, side):
    if side == "UP":
        return w.up_bid, w.up_ask
    return w.down_bid, w.down_ask


def outcome_value(w, side):
    return float(w.outcome_up) if side == "UP" else float(1 - w.outcome_up)


def extract_features(w: Window, entry_idx: int = 270):
    """Extract comprehensive features for ML."""
    p = w.prices[:entry_idx + 1]
    if len(p) < 30:
        return None
    
    open_p = w.open_price
    
    # Price action
    high = float(p.max())
    low = float(p.min())
    close = float(p[-1])
    range_pct = (high - low) / open_p
    net_move = (close - open_p) / open_p
    
    # Momentum (first derivative)
    ret_10s = p[-1] / p[max(0, len(p)-10)] - 1.0 if len(p) >= 10 else 0.0
    ret_30s = p[-1] / p[max(0, len(p)-30)] - 1.0 if len(p) >= 30 else 0.0
    ret_60s = p[-1] / p[0] - 1.0 if len(p) >= 60 else ret_30s
    
    # Acceleration (second derivative)
    if len(p) >= 20:
        ret_first_half = p[len(p)//2] / p[0] - 1.0
        ret_second_half = p[-1] / p[len(p)//2] - 1.0
        acceleration = ret_second_half - ret_first_half
    else:
        acceleration = 0.0
    
    # Volatility
    if len(p) > 10:
        returns = np.diff(np.log(p))
        vol_10s = float(np.std(returns[-10:])) if len(returns) >= 10 else 0.0
        vol_30s = float(np.std(returns[-30:])) if len(returns) >= 30 else 0.0
        vol_ratio = vol_10s / max(vol_30s, 1e-9)
    else:
        vol_10s = vol_30s = vol_ratio = 0.0
    
    # Trend strength (how linear is the move?)
    if len(p) >= 10:
        x = np.arange(len(p))
        slope = np.polyfit(x, p, 1)[0]
        trend_strength = slope / open_p
    else:
        trend_strength = 0.0
    
    # Distance to settlement
    seconds_left = 300 - entry_idx
    time_frac = entry_idx / 300.0
    
    # Where in range are we?
    price_in_range = (close - low) / max(high - low, 1e-9)
    
    # RSI-like
    if len(p) >= 15:
        diffs = np.diff(p[-15:])
        gains = np.clip(diffs, 0, None).mean()
        losses = -np.clip(diffs, None, 0).mean()
        rsi = 100.0 - (100.0 / (1.0 + gains / max(losses, 1e-9))) if losses > 0 else 100.0
    else:
        rsi = 50.0
    
    features = {
        "range_pct": range_pct,
        "net_move": net_move,
        "ret_10s": ret_10s,
        "ret_30s": ret_30s,
        "ret_60s": ret_60s,
        "acceleration": acceleration,
        "vol_10s": vol_10s,
        "vol_ratio": vol_ratio,
        "trend_strength": trend_strength,
        "seconds_left": seconds_left,
        "time_frac": time_frac,
        "price_in_range": price_in_range,
        "rsi": rsi,
    }
    return features


def train_reversal_detector(train_windows):
    """Train MLP to detect momentum reversals."""
    X, y = [], []
    
    for w in train_windows:
        for entry_sl in [10, 20, 30]:
            idx = 300 - entry_sl
            if idx > len(w.prices) - 10 or idx < 10:
                continue
            
            features = extract_features(w, idx)
            if features is None:
                continue
            
            # Label: 1 if reversal happens (price continues opposite direction)
            future_price = w.prices[-1]
            current_price = w.prices[idx]
            initial_direction = np.sign(current_price - w.prices[max(0, idx-10)])
            future_direction = np.sign(future_price - current_price)
            
            reversal = 1 if initial_direction != 0 and future_direction != initial_direction else 0
            
            X.append(list(features.values()))
            y.append(reversal)
    
    if len(X) < 50:
        return None, None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler


def train_confidence_scorer(train_windows):
    """Train RF to score entry confidence (probability of win)."""
    X, y = [], []
    
    for w in train_windows:
        for entry_sl in [5, 10, 15, 20, 25, 30]:
            idx = 300 - entry_sl
            if idx >= len(w.prices) or idx < 0:
                continue
            
            features = extract_features(w, idx)
            if features is None:
                continue
            
            # Label: 1 if we picked the winning side
            leader = "UP" if w.prices[idx] >= w.open_price else "DOWN"
            correct = (leader == "UP" and w.outcome_up == 1) or (leader == "DOWN" and w.outcome_up == 0)
            
            X.append(list(features.values()))
            y.append(1 if correct else 0)
    
    if len(X) < 50:
        return None, None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler


# ========================================================================
# STRATEGY: Base buy97 (late entry, hold to settlement)
# ========================================================================

def base_strategy(w, stake_usd=10.0, entry_sl=30):
    idx = 300 - entry_sl
    leader = "UP" if w.prices[idx] >= w.open_price else "DOWN"
    bid, ask = side_prices(w, leader)
    entry_price = float(ask[idx])
    
    if entry_price > 0.97:
        return None
    
    shares = stake_usd / entry_price
    entry_fee = stake_usd * TAKER_FEE_RATE
    final = outcome_value(w, leader) * shares
    pnl = final - stake_usd - entry_fee
    
    return Trade("base", idx, None, leader, entry_price, float(w.outcome_up if leader == "UP" else 1-w.outcome_up), "settlement", float(pnl))


# ========================================================================
# STRATEGY 1: Momentum Reversal Detector
# If NN predicts reversal, skip trade
# ========================================================================

def reversal_filter_strategy(w, model, scaler, threshold=0.5, stake_usd=10.0, entry_sl=30):
    if model is None:
        return base_strategy(w, stake_usd, entry_sl)
    
    idx = 300 - entry_sl
    features = extract_features(w, idx)
    if features is None:
        return None
    
    X = scaler.transform([list(features.values())])
    reversal_prob = model.predict_proba(X)[0][1]  # probability of reversal
    
    if reversal_prob > threshold:
        # Skip this trade - predicted reversal
        return None
    
    return base_strategy(w, stake_usd, entry_sl)


# ========================================================================
# STRATEGY 2: Confidence Scorer
# Only trade when confidence > threshold
# ========================================================================

def confidence_scorer_strategy(w, model, scaler, threshold=0.90, stake_usd=10.0, entry_sl=30):
    if model is None:
        return base_strategy(w, stake_usd, entry_sl)
    
    idx = 300 - entry_sl
    features = extract_features(w, idx)
    if features is None:
        return None
    
    X = scaler.transform([list(features.values())])
    confidence = model.predict_proba(X)[0][1]  # probability of correct side
    
    if confidence < threshold:
        return None
    
    trade = base_strategy(w, stake_usd, entry_sl)
    if trade is not None:
        trade.confidence = confidence
    return trade


# ========================================================================
# STRATEGY 3: Opposite Side Hedge
# When reversal predicted, buy the OPPOSITE side instead
# ========================================================================

def opposite_hedge_strategy(w, model, scaler, threshold=0.5, stake_usd=10.0, entry_sl=30):
    if model is None:
        return base_strategy(w, stake_usd, entry_sl)
    
    idx = 300 - entry_sl
    features = extract_features(w, idx)
    if features is None:
        return None
    
    X = scaler.transform([list(features.values())])
    reversal_prob = model.predict_proba(X)[0][1]
    
    leader = "UP" if w.prices[idx] >= w.open_price else "DOWN"
    
    if reversal_prob > threshold:
        # Buy the OPPOSITE side
        opposite = "DOWN" if leader == "UP" else "UP"
        bid, ask = side_prices(w, opposite)
        entry_price = float(ask[idx])
        
        if entry_price > 0.97:
            return None
        
        shares = stake_usd / entry_price
        entry_fee = stake_usd * TAKER_FEE_RATE
        final = outcome_value(w, opposite) * shares
        pnl = final - stake_usd - entry_fee
        
        return Trade("opposite_hedge", idx, None, opposite, entry_price, 
                     float(w.outcome_up if opposite == "UP" else 1-w.outcome_up), 
                     "settlement", float(pnl), reversal_prob)
    
    return base_strategy(w, stake_usd, entry_sl)


# ========================================================================
# STRATEGY 4: Adaptive Position Sizing
# Size proportional to predicted confidence
# ========================================================================

def adaptive_size_strategy(w, model, scaler, stake_usd=10.0, entry_sl=30):
    if model is None:
        return base_strategy(w, stake_usd, entry_sl)
    
    idx = 300 - entry_sl
    features = extract_features(w, idx)
    if features is None:
        return None
    
    X = scaler.transform([list(features.values())])
    confidence = model.predict_proba(X)[0][1]
    
    # Size inversely to failure probability
    size_multiplier = max(0.2, confidence / 0.87)  # scale around base WR of 87%
    adjusted_stake = stake_usd * size_multiplier
    
    trade = base_strategy(w, adjusted_stake, entry_sl)
    if trade is not None:
        trade.confidence = confidence
    return trade


# ========================================================================
# STRATEGY 5: Multi-Feature Filter (no ML, just logic)
# Combine multiple signals for stricter entry criteria
# ========================================================================

def multi_feature_filter(w, stake_usd=10.0, entry_sl=30):
    idx = 300 - entry_sl
    features = extract_features(w, idx)
    if features is None:
        return None
    
    # STRICT entry criteria:
    # 1. Must have clear momentum (large net move)
    # 2. RSI not in extreme territory (not overbought/oversold)
    # 3. Low volatility (not choppy)
    # 4. Time fraction > 0.80 (late in window)
    # 5. Acceleration in same direction as net move
    
    net_move_ok = abs(features["net_move"]) > 0.0005  # meaningful price move
    rsi_ok = 30 < features["rsi"] < 70  # not extreme
    vol_ok = features["vol_ratio"] < 2.0  # volatility not spiking
    time_ok = features["time_frac"] > 0.85  # late entry
    accel_ok = features["acceleration"] * features["net_move"] > 0  # acceleration confirms direction
    
    score = sum([net_move_ok, rsi_ok, vol_ok, time_ok, accel_ok])
    
    if score < 4:  # Require at least 4/5 criteria
        return None
    
    return base_strategy(w, stake_usd, entry_sl)


# ========================================================================
# Runner
# ========================================================================

def run_strategy(windows, name, strategy_fn):
    trades = []
    pnls = []
    for w in windows:
        try:
            trade = strategy_fn(w)
            if trade is not None:
                trades.append(trade)
                pnls.append(trade.pnl)
        except Exception as e:
            continue
    
    if not trades:
        return None, None
    
    equity = STARTING_CAPITAL + np.cumsum(pnls)
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
    raw_sharpe = float(returns.mean() / returns.std(ddof=1) if len(returns) > 1 and returns.std(ddof=1) > 0 else 0.0)
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl < 0]
    
    metrics = pd.DataFrame([{
        "strategy": name,
        "trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades),
        "total_pnl": float(equity[-1] - STARTING_CAPITAL),
        "avg_pnl": float(np.mean(pnls)),
        "max_drawdown": float(max_drawdown(equity)),
        "raw_sharpe": raw_sharpe,
        "profit_factor": abs(sum(wins) / sum(losses)) if losses else float("inf"),
        "avg_win": float(np.mean(wins)) if wins else 0,
        "avg_loss": float(np.mean(losses)) if losses else 0,
        "payoff_ratio": abs(np.mean(wins)) / abs(np.mean(losses)) if losses else float("inf"),
    }])
    
    return metrics, trades


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-windows", type=int, default=500)
    parser.add_argument("--train-frac", type=float, default=0.70)
    args = parser.parse_args()
    
    print("Loading data...")
    price_df = load_spot_prices("btcusdt", "binance", dataset_source="aliplayer_spot")
    windows = build_windows(price_df, args.max_windows)
    train, test = split_windows_chronologically(windows, args.train_frac)
    
    print(f"Training on {len(train)} windows, testing on {len(test)}\n")
    
    # Train ML models on train set
    print("Training reversal detector (MLP)...")
    reversal_model, reversal_scaler = train_reversal_detector(train)
    
    print("Training confidence scorer (Random Forest)...")
    confidence_model, confidence_scaler = train_confidence_scorer(train)
    
    # Run all strategies
    strategies = {
        "base": (base_strategy, {}),
        "reversal_filter": (reversal_filter_strategy, {"model": reversal_model, "scaler": reversal_scaler, "threshold": 0.3}),
        "confidence_90": (confidence_scorer_strategy, {"model": confidence_model, "scaler": confidence_scaler, "threshold": 0.90}),
        "confidence_95": (confidence_scorer_strategy, {"model": confidence_model, "scaler": confidence_scaler, "threshold": 0.95}),
        "opposite_hedge": (opposite_hedge_strategy, {"model": reversal_model, "scaler": reversal_scaler, "threshold": 0.4}),
        "adaptive_size": (adaptive_size_strategy, {"model": confidence_model, "scaler": confidence_scaler}),
        "multi_filter": (multi_feature_filter, {}),
    }
    
    all_results = []
    
    for sample_name, sample_windows in [("TRAIN", train), ("TEST", test)]:
        print(f"\n{'='*80}")
        print(f"{sample_name} Results")
        print(f"{'='*80}\n")
        
        results = []
        for name, (fn, kwargs) in strategies.items():
            print(f"Running {name}...")
            
            def strategy_fn(w, fn=fn, kwargs=kwargs):
                return fn(w, **kwargs)
            
            metrics, trades = run_strategy(sample_windows, name, strategy_fn)
            if metrics is not None:
                metrics.insert(0, "sample", sample_name)
                results.append(metrics)
        
        if results:
            df = pd.concat(results, ignore_index=True)
            all_results.append(df)
            
            print(f"\n{'Strategy':<18} {'Trades':>7} {'Wins':>6} {'Losses':>7} {'WR':>7} {'PnL':>12} {'Avg':>8} {'Sharpe':>7} {'PF':>7} {'Payoff':>7}")
            print("-" * 95)
            for _, row in df.iterrows():
                print(f"{row['strategy']:<18} {row['trades']:>7} {row['wins']:>6} {row['losses']:>7} {row['win_rate']*100:>6.1f}% ${row['total_pnl']:>10,.2f} ${row['avg_pnl']:>6.2f} {row['raw_sharpe']:>6.2f} {row['profit_factor']:>6.2f} {row['payoff_ratio']:>6.2f}")
    
    if all_results:
        final = pd.concat(all_results, ignore_index=True)
        final.to_csv("enhanced_buy97_v2_metrics.csv", index=False)
        print(f"\nSaved to enhanced_buy97_v2_metrics.csv")


if __name__ == "__main__":
    main()
