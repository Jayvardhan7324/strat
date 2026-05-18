"""Quick test to verify dataset integration works."""

from utils.huggingface_datasets import list_available_datasets, DATASETS

print("Testing huggingface_datasets module...")
print(f"\nNumber of available datasets: {len(DATASETS)}")
print("\nDataset keys:")
for key in DATASETS:
    print(f"  - {key}")

print("\nModule loaded successfully!")
print("\nTo use these datasets in your backtesters:")
print("  python polymarket_updown_backtest.py --dataset-source bmoney_crypto")
print("  python backtest_buy1_cent.py --dataset-source bmoney_crypto")
print("  python backtest_buy97_sell99.py --dataset-source bmoney_crypto")
print("  python backtest_prev10_momentum_next.py --dataset-source bmoney_crypto")
print("  python live_guarded_backtest.py --dataset-source bmoney_crypto")
