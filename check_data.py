"""Check data availability across all Hugging Face Polymarket datasets."""

from huggingface_datasets import stream_dataset, load_polymarket_dataset
import pandas as pd

print("=" * 80)
print("DATA AVAILABILITY CHECK")
print("=" * 80)

# 1. Check bmoney resolutions
print("\n1. bmoney_resolutions (BTC):")
try:
    df = load_polymarket_dataset("bmoney_resolutions", asset="BTC")
    print(f"  Rows: {len(df):,}")
    print(f"  Date range: {df['start_time'].min()} to {df['end_time'].max()}")
    print(f"  Timeframes: {df['question'].head(3).tolist()}")
except Exception as e:
    print(f"  Error: {e}")

# 2. Check bmoney crypto prices
print("\n2. bmoney_crypto_prices (BTC):")
try:
    df = load_polymarket_dataset("bmoney_crypto_prices", asset="BTC")
    print(f"  Rows: {len(df):,}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
except Exception as e:
    print(f"  Error: {e}")

# 3. Check aliplayer markets
print("\n3. aliplayer_markets (BTC, 5-minute):")
try:
    df = load_polymarket_dataset("aliplayer_markets")
    btc_5m = df[(df['crypto'] == 'BTC') & (df['timeframe'] == '5-minute')]
    print(f"  BTC 5m rows: {len(btc_5m):,}")
    if len(btc_5m) > 0:
        print(f"  Date range: {pd.to_datetime(btc_5m['start_ts'], unit='s').min()} to {pd.to_datetime(btc_5m['end_ts'], unit='s').max()}")
except Exception as e:
    print(f"  Error: {e}")

# 4. Check PolyData trade capture
print("\n4. trade_capture_5mar (sample):")
try:
    count = 0
    for batch in stream_dataset("trade_capture_5mar", batch_size=1000):
        print(f"  First batch timestamp: {pd.to_datetime(batch['timestamp'].iloc[0], unit='s')}")
        print(f"  First batch rows: {len(batch)}")
        count += len(batch)
        break
    print(f"  (Streaming available - 1-2B total rows)")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 80)
print("CONCLUSION: We need to combine multiple datasets for full year coverage")
print("=" * 80)
