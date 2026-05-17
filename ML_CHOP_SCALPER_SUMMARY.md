# ML Chop Scalper Strategy

## Summary

A machine learning-enhanced strategy that exploits BTC price chop patterns in the first 2 minutes of each 5-minute Polymarket binary window.

## Core Edge

When BTC price "chops" (moves in one direction then reverses) in the first 2 minutes, the **second movement direction wins ~70% of the time** at settlement.

- `chop_down_first` (down then up) → UP wins 71.8%
- `chop_up_first` (up then down) → DOWN wins 70.6%

## How It Works

1. **Detect chop pattern** in first 120 seconds of BTC price
2. **ML model scores** how strong the edge is (20 features)
3. **Filter by confidence** - only trade when ML agrees (threshold 0.60+)
4. **Enter directional contract** at 120s mark
5. **Hold to settlement** - binary outcome, no spread risk

## Performance

| Config | Win Rate | Trades | Profit Factor | Sharpe | Max DD |
|--------|----------|--------|--------------|--------|--------|
| Baseline (no ML) | 67-69% | 3,155 | 2.0-2.1 | 58 | -18% |
| ML threshold 0.60 | 75% | 1,037 | 3.0-3.2 | 36-56 | -18% |
| ML threshold 0.70 | 77% | 831 | 3.2-3.6 | 39-51 | -22% |
| ML threshold 0.78 | 79% | 465 | 3.3-4.6 | 43-50 | -16% |

## Key ML Features

1. **recovery** - How much of the initial move was recovered
2. **initial_move** - Size of the first directional move
3. **strength** - Overall chop strength metric
4. **time_to_low** - When the low was hit (timing matters)
5. **down_move** - Size of the downward move
6. **bb_pct** - Bollinger Band position
7. **ret_30s** - Recent 30s momentum

## Files

- `ml_chop_focused.py` - Main strategy script
- `ml_chop_focused_model.txt` - Trained LightGBM model
- `ml_chop_focused_importance.csv` - Feature importance

## Usage

```bash
# Train model
python ml_chop_focused.py --train

# Backtest with default threshold
python ml_chop_focused.py --backtest --threshold 0.60

# Sweep confidence thresholds
python ml_chop_focused.py --sweep

# Run baseline (no ML)
python ml_chop_focused.py --baseline
```

## Why This Works

The edge comes from **mean reversion in short-term BTC price action**. When BTC makes a sharp move then reverses within 2 minutes, it often indicates:
- Liquidity grabs / stop hunts
- Overreaction correction
- Market maker positioning

The 5-minute binary contract resolution tends to follow the **second direction** because the initial move was often a false breakout.

## Risk Notes

- This is a simulated backtest with simplified fill model
- Real execution would have slippage, latency, and order book impact
- The 1c bid-ask spread is baked into the model
- Past performance does not guarantee future results
- Model was trained on ~1 month of BTC data (Mar-Apr 2026)
