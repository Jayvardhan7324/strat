"""
Example: Using additional Hugging Face datasets with your Polymarket backtesters.

This script demonstrates how to switch between different data sources
for backtesting your Polymarket strategies.

Usage:
    python example_dataset_usage.py
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Using the main backtester with different datasets
# ---------------------------------------------------------------------------
#
# The main backtester (polymarket_updown_backtest.py) now supports a
# --dataset-source flag. Available options:
#
#   --dataset-source aliplayer_spot   (default)
#       Uses aliplayer1/polymarket-crypto-updown spot_prices config
#       Continuous spot price feed from Binance + Chainlink
#
#   --dataset-source bmoney_crypto
#       Uses bmoney1321/polymarket-crypto-5m-15m crypto_prices config
#       1-minute OHLCV candles from Binance for BTC, ETH, SOL, XRP
#
#   --dataset-source aliplayer_prices
#       Uses aliplayer1/polymarket-crypto-updown prices config
#       OHLC price history from CLOB API
#
# Examples:
#
#   # Default: aliplayer spot prices
#   python polymarket_updown_backtest.py --symbol btcusdt
#
#   # Use bmoney 1-minute Binance candles
#   python polymarket_updown_backtest.py --symbol btcusdt --dataset-source bmoney_crypto
#
#   # Use aliplayer CLOB price history
#   python polymarket_updown_backtest.py --symbol btcusdt --dataset-source aliplayer_prices
#
#   # Test with ETH instead of BTC
#   python polymarket_updown_backtest.py --symbol ethusdt --dataset-source bmoney_crypto
#
#   # Run with limited windows for quick testing
#   python polymarket_updown_backtest.py --symbol btcusdt --dataset-source bmoney_crypto --max-windows 500

# ---------------------------------------------------------------------------
# 2. Using the huggingface_datasets module directly
# ---------------------------------------------------------------------------
#
# You can also load datasets programmatically for custom analysis:

from huggingface_datasets import (
    DATASETS,
    list_available_datasets,
    load_bmoney_orderbooks,
    load_bmoney_trades,
    load_bmarket_resolutions,
    load_polymarket_dataset,
    load_quant_data,
)

# List all available datasets
print("Available datasets:")
list_available_datasets()

# Load real market resolutions
print("\n" + "=" * 80)
print("Loading BTC market resolutions from bmoney dataset...")
resolutions = load_bmarket_resolutions(asset="BTC")
print(f"\nResolution distribution:")
print(resolutions["outcome"].value_counts())

# Load real trade data
print("\n" + "=" * 80)
print("Loading BTC trades from bmoney dataset...")
trades = load_bmoney_trades(asset="BTC")
print(f"\nTrade columns: {trades.columns.tolist()}")
print(f"Trade count: {len(trades):,}")

# Load orderbook data
print("\n" + "=" * 80)
print("Loading BTC orderbooks from bmoney dataset...")
orderbooks = load_bmoney_orderbooks(asset="BTC")
print(f"\nOrderbook columns: {orderbooks.columns.tolist()}")
print(f"Orderbook snapshot count: {len(orderbooks):,}")

# ---------------------------------------------------------------------------
# 3. Combining datasets for enhanced analysis
# ---------------------------------------------------------------------------
#
# Example: Use real resolutions to validate synthetic backtest results
#
# resolutions = load_bmarket_resolutions(asset="BTC")
#
# # Compare your backtest predictions against actual outcomes
# # This lets you validate if your synthetic model matches real market resolutions
#
# Example: Use real trade data to calibrate your fill model
#
# trades = load_bmoney_trades(asset="BTC")
#
# # Analyze real trade prices and sizes to improve your backtest assumptions
# avg_trade_size = trades["size"].mean()
# price_distribution = trades["price"].describe()

# ---------------------------------------------------------------------------
# 4. Using the large quant dataset for comprehensive analysis
# ---------------------------------------------------------------------------
#
# The daxiongya dataset has 170M+ cleaned records with unified YES perspective
#
# quant_df = load_quant_data()
#
# This dataset includes:
# - All trades normalized to YES token perspective
# - Maker/taker roles preserved
# - Clean data with contract trades filtered out
# - Best for: Market analysis, price studies, time-series forecasting

print("\n" + "=" * 80)
print("Example complete!")
print("\nTo run your backtesters with different datasets:")
print("  python polymarket_updown_backtest.py --dataset-source bmoney_crypto")
print("  python backtest_buy1_cent.py --dataset-source bmoney_crypto")
print("  python backtest_buy97_sell99.py --dataset-source bmoney_crypto")
print("  python backtest_prev10_momentum_next.py --dataset-source bmoney_crypto")
print("  python live_guarded_backtest.py --dataset-source bmoney_crypto")
