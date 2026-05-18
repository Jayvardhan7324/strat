# Polymarket Scalping Strategy Research - Results

## Dataset
- **Source**: aliplayer1/polymarket-crypto-updown (spot_prices)
- **Symbol**: BTC/USDT (Binance)
- **Period**: 2026-03-26 to 2026-04-26 (31 days)
- **Windows**: 8,964 complete 5-minute windows
- **Split**: 80/20 chronological (7,171 train / 1,793 test)

## Strategy Results

### Winners

| Strategy | Train PnL | Test PnL | Train WR | Test WR | Trades (T/Te) |
|----------|-----------|----------|----------|---------|---------------|
| opening_breakout | $18,102 | $5,314 | 63.7% | 66.0% | 7,075 / 1,764 |
| late_momentum | $5,305 | $1,795 | 66.1% | 69.1% | 1,784 / 505 |
| volume_spike | $4,709 | $1,366 | 66.3% | 68.8% | 1,543 / 385 |

### Losers (Rejected)

| Strategy | Train PnL | Test PnL | Issue |
|----------|-----------|----------|-------|
| mean_reversion | -$7,314 | -$1,933 | Fading trends doesn't work |
| vwap_reversion | -$16,629 | -$3,294 | VWAP mean reversion fails |
| spread_capture | $0 | $0 | No trades (synthetic spreads too tight) |

## ML Model (LightGBM)

### Performance
| Metric | Train | Test |
|--------|-------|------|
| PnL | $54,526 | $11,054 |
| Trades | 7,147 | 1,786 |
| Win Rate | 89.5% | 82.2% |
| AUC | 0.872 | 0.646 |

### Top Features
1. **ret_60s** - 60-second return (momentum)
2. **vol_30s** - 30-second volatility
3. **ob_imbalance** - Orderbook bid/ask imbalance
4. **rsi** - Relative Strength Index
5. **ret_from_open** - Return from window open

### Model Config
- Entry: 240 seconds into 300s window
- Threshold: 0.55 confidence
- Max entry price: 0.70
- Stake: $10/trade

## Key Findings

1. **Opening Breakout** is the most robust strategy - survives train/test split with consistent win rates
2. **Late Momentum** works well with very short hold times (20s average)
3. **Volume Spike** continuation is profitable but fewer signals
4. **Mean Reversion strategies fail** - trends persist in 5-minute windows
5. **ML Model** shows strong predictive power (82% test win rate)

## Caveats

- Synthetic prices, not real orderbook data
- No slippage modeling beyond taker fees
- Results may not translate to live trading
- Need real orderbook validation before paper trading

## Next Steps

1. Test with real orderbook data (bmoney dataset)
2. Add slippage stress testing
3. Walk-forward validation across multiple time periods
4. Paper trade with small stakes

## Files Generated

- `scalping_metrics.csv` - All strategy metrics
- `polymarket_ml_model.txt` - Trained LightGBM model
- `ml_feature_importance.csv` - Feature importance rankings
