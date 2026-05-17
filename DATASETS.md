# Polymarket Datasets - Streaming API

All datasets stream directly from Hugging Face. **Zero download required.**

## Quick Start

```python
from huggingface_datasets import load_polymarket_dataset, stream_dataset

# Small datasets - loads directly
resolutions = load_polymarket_dataset("bmoney_resolutions", asset="BTC")
print(resolutions.head())

# Large datasets - streams in batches
for batch in stream_dataset("bmoney_trades", asset="BTC", batch_size=10_000):
    print(f"Got {len(batch)} rows")
```

## Available Datasets

| Key | Source | Rows | Mode | Description |
|-----|--------|------|------|-------------|
| `bmoney_resolutions` | bmoney1321/polymarket-crypto-5m-15m | 18K | direct | Final market outcomes (BTC/ETH/SOL/XRP) |
| `bmoney_markets` | bmoney1321/polymarket-crypto-5m-15m | 17K | direct | Market metadata |
| `bmoney_crypto_prices` | bmoney1321/polymarket-crypto-5m-15m | 37K | direct | 1-min Binance OHLCV candles |
| `bmoney_price_history` | bmoney1321/polymarket-crypto-5m-15m | 304K | direct | 1-min CLOB mid-prices |
| `bmoney_trades` | bmoney1321/polymarket-crypto-5m-15m | 23M | **stream** | Individual trade executions |
| `bmoney_orderbooks` | bmoney1321/polymarket-crypto-5m-15m | 3.4M | **stream** | 10-level orderbook snapshots |
| `aliplayer_markets` | aliplayer1/polymarket-crypto-updown | - | direct | Market metadata with resolutions |
| `aliplayer_spot_prices` | aliplayer1/polymarket-crypto-updown | - | **stream** | Spot prices (Binance + Chainlink) |
| `aliplayer_prices` | aliplayer1/polymarket-crypto-updown | - | **stream** | CLOB OHLC price history |
| `aliplayer_ticks` | aliplayer1/polymarket-crypto-updown | - | **stream** | Trade-level fills |
| `aliplayer_orderbook` | aliplayer1/polymarket-crypto-updown | - | **stream** | Best bid/ask snapshots |
| `trade_capture_5mar` | PolyData/polymarket_trade_capture_5Mar2026 | 1-2B | **stream** | All CLOB fills through 2026-03-05 |
| `daxiongya_quant` | daxiongya/Polymarket_data | 170M | **stream** | Clean YES-perspective data |
| `daxiongya_trades` | daxiongya/Polymarket_data | 293M | **stream** | Processed trades |
| `daxiongya_markets` | daxiongya/Polymarket_data | 268K | direct | Market metadata |
| `daxiongya_orderfilled` | daxiongya/Polymarket_data | 293M | **stream** | Raw blockchain events |
| `daxiongya_users` | daxiongya/Polymarket_data | 340M | **stream** | User behavior data |
| `gamma_markets` | cognocracy-agent/polymarket-gamma-dataset | - | direct | Gamma API metadata |

## CLI Usage

```bash
# List all datasets
python huggingface_datasets.py --list

# Load small dataset
python huggingface_datasets.py --dataset bmoney_resolutions --asset BTC --head 10

# Stream large dataset to CSV
python huggingface_datasets.py --dataset bmoney_trades --asset BTC --stream --output btc_trades.csv

# Stream with custom batch size
python huggingface_datasets.py --dataset trade_capture_5mar --stream --batch-size 100000 --output all_trades.csv
```

## Backtester Integration

All backtesters now support `--dataset-source`:

```bash
# Default: aliplayer spot prices
python polymarket_updown_backtest.py --symbol btcusdt

# Use bmoney 1-min Binance candles
python polymarket_updown_backtest.py --symbol btcusdt --dataset-source bmoney_crypto

# Use aliplayer CLOB prices
python polymarket_updown_backtest.py --symbol btcusdt --dataset-source aliplayer_prices
```

Works with all backtesters:
- `polymarket_updown_backtest.py`
- `backtest_buy1_cent.py`
- `backtest_buy97_sell99.py`
- `backtest_prev10_momentum_next.py`
- `live_guarded_backtest.py`
