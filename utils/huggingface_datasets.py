"""
Additional Hugging Face dataset loaders for Polymarket research.

Streams data directly from Hugging Face via API - no download needed.
Data is fetched on-demand and filtered server-side where possible.

Usage:
    from huggingface_datasets import load_polymarket_dataset, stream_dataset

    # Load into DataFrame (streams and filters, returns only matching rows)
    df = load_polymarket_dataset("bmoney_resolutions", asset="BTC")

    # Stream large datasets row-by-row without loading into memory
    for batch in stream_dataset("trade_capture_5mar", batch_size=10000):
        process(batch)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterator

import pandas as pd

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("Install datasets: pip install datasets")


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "aliplayer_spot_prices": {
        "name": "aliplayer1/polymarket-crypto-updown",
        "config": "spot_prices",
        "description": "Continuous spot price feed (Binance + Chainlink)",
        "streaming_default": True,
    },
    "aliplayer_markets": {
        "name": "aliplayer1/polymarket-crypto-updown",
        "config": "markets",
        "description": "Market metadata with resolution outcomes",
        "streaming_default": False,
    },
    "aliplayer_prices": {
        "name": "aliplayer1/polymarket-crypto-updown",
        "config": "prices",
        "description": "OHLC price history from CLOB API",
        "streaming_default": True,
    },
    "aliplayer_ticks": {
        "name": "aliplayer1/polymarket-crypto-updown",
        "config": "ticks",
        "description": "Trade-level fills from on-chain + WebSocket",
        "streaming_default": True,
    },
    "aliplayer_orderbook": {
        "name": "aliplayer1/polymarket-crypto-updown",
        "config": "orderbook",
        "description": "Best bid/ask snapshots from CLOB WebSocket",
        "streaming_default": True,
    },
    "bmoney_trades": {
        "name": "bmoney1321/polymarket-crypto-5m-15m",
        "config": "default",
        "data_dir": "trades",
        "description": "Individual trade executions from Data API (23M rows)",
        "streaming_default": True,
    },
    "bmoney_orderbooks": {
        "name": "bmoney1321/polymarket-crypto-5m-15m",
        "config": "default",
        "data_dir": "orderbooks",
        "description": "10-level order book snapshots every 10s (3.4M rows)",
        "streaming_default": True,
    },
    "bmoney_price_history": {
        "name": "bmoney1321/polymarket-crypto-5m-15m",
        "config": "default",
        "data_dir": "price_history",
        "description": "1-minute mid-price points from CLOB API (304K rows)",
        "streaming_default": False,
    },
    "bmoney_crypto_prices": {
        "name": "bmoney1321/polymarket-crypto-5m-15m",
        "config": "default",
        "data_dir": "crypto_prices",
        "description": "1-minute OHLCV candles from Binance (37K rows)",
        "streaming_default": False,
    },
    "bmoney_resolutions": {
        "name": "bmoney1321/polymarket-crypto-5m-15m",
        "config": "default",
        "data_dir": "resolutions",
        "description": "Final outcomes for resolved markets (18K rows)",
        "streaming_default": False,
    },
    "bmoney_markets": {
        "name": "bmoney1321/polymarket-crypto-5m-15m",
        "config": "default",
        "data_dir": "markets",
        "description": "Market metadata and configuration (17K rows)",
        "streaming_default": False,
    },
    "trade_capture_5mar": {
        "name": "PolyData/polymarket_trade_capture_5Mar2026",
        "config": None,
        "description": "All CLOB fills from launch to 2026-03-05 (~1-2B rows)",
        "streaming_default": True,
    },
    "daxiongya_quant": {
        "name": "daxiongya/Polymarket_data",
        "config": "quant",
        "description": "Clean market data with unified YES perspective (170M records)",
        "streaming_default": True,
    },
    "daxiongya_trades": {
        "name": "daxiongya/Polymarket_data",
        "config": "trades",
        "description": "Processed trades with market metadata linkage (293M records)",
        "streaming_default": True,
    },
    "daxiongya_markets": {
        "name": "daxiongya/Polymarket_data",
        "config": "markets",
        "description": "Market information and metadata (268K markets)",
        "streaming_default": False,
    },
    "daxiongya_orderfilled": {
        "name": "daxiongya/Polymarket_data",
        "config": "orderfilled",
        "description": "Raw blockchain OrderFilled events (293M records)",
        "streaming_default": True,
    },
    "daxiongya_users": {
        "name": "daxiongya/Polymarket_data",
        "config": "users",
        "description": "User behavior data split by maker/taker roles (340M records)",
        "streaming_default": True,
    },
    "gamma_markets": {
        "name": "cognocracy-agent/polymarket-gamma-dataset",
        "config": None,
        "description": "Gamma API market/event metadata snapshot",
        "streaming_default": False,
    },
}


# ---------------------------------------------------------------------------
# Streaming iterator - fetches data in batches without loading everything
# ---------------------------------------------------------------------------


def stream_dataset(
    dataset_key: str,
    asset: str | None = None,
    timeframe: str | None = None,
    date: str | None = None,
    batch_size: int = 10_000,
    split: str = "train",
    **kwargs,
) -> Iterator[pd.DataFrame]:
    """Stream a dataset from Hugging Face in batches.

    Yields DataFrames of `batch_size` rows. Filters are applied to each batch.
    Memory usage stays constant regardless of dataset size.

    Args:
        dataset_key: Key from the DATASETS registry.
        asset: Filter by asset symbol.
        timeframe: Filter by timeframe.
        date: Filter by date (YYYY-MM-DD).
        batch_size: Number of rows per batch.
        split: Dataset split.
        **kwargs: Additional arguments to load_dataset.

    Yields:
        pandas DataFrame batches.
    """
    if dataset_key not in DATASETS:
        available = "\n".join(f"  - {k}: {v['description']}" for k, v in DATASETS.items())
        raise ValueError(f"Unknown dataset key: {dataset_key}\n\nAvailable datasets:\n{available}")

    info = DATASETS[dataset_key]
    ds_name = info["name"]
    config = info["config"]
    data_dir = info.get("data_dir")

    print(f"Streaming dataset: {ds_name}")
    if config:
        print(f"  Config: {config}")
    if data_dir:
        print(f"  Data dir: {data_dir}")
    print(f"  Batch size: {batch_size:,}")

    load_kwargs = {"split": split, "streaming": True, **kwargs}
    if data_dir:
        load_kwargs["data_dir"] = data_dir

    if config:
        ds = load_dataset(ds_name, config, **load_kwargs)
    else:
        ds = load_dataset(ds_name, **load_kwargs)

    if hasattr(ds, "keys"):
        ds = ds[split] if split in ds else list(ds.values())[0]

    batch: list[dict] = []
    total_yielded = 0

    for row in ds:
        # Apply filters
        if asset:
            row_asset = str(row.get("asset", row.get("crypto", row.get("symbol", "")))).lower()
            if row_asset != asset.lower():
                continue

        if timeframe and "timeframe" in row:
            if row["timeframe"] != timeframe:
                continue

        if date and "date" in row:
            if row["date"] != date:
                continue

        batch.append(row)

        if len(batch) >= batch_size:
            df = pd.DataFrame(batch)
            total_yielded += len(df)
            print(f"  Yielded batch #{total_yielded // batch_size}: {len(df):,} rows (total: {total_yielded:,})")
            yield df
            batch = []

    if batch:
        df = pd.DataFrame(batch)
        total_yielded += len(df)
        print(f"  Yielded final batch: {len(df):,} rows (total: {total_yielded:,})")
        yield df


# ---------------------------------------------------------------------------
# Load into DataFrame (streams, filters, returns all matching rows)
# ---------------------------------------------------------------------------


def load_polymarket_dataset(
    dataset_key: str,
    asset: str | None = None,
    timeframe: str | None = None,
    date: str | None = None,
    split: str = "train",
    streaming: bool | None = None,
    cache_dir: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Load a Polymarket dataset from Hugging Face and return as DataFrame.

    For large datasets, streams data in batches and filters on-the-fly.
    For small datasets, loads directly into memory.

    Args:
        dataset_key: Key from the DATASETS registry.
        asset: Filter by asset symbol (BTC, ETH, SOL, XRP, etc.).
        timeframe: Filter by timeframe (5-minute, 15-minute, 1-hour, 4-hour).
        date: Filter by date (YYYY-MM-DD) for date-partitioned datasets.
        split: Dataset split to load (default: "train").
        streaming: Force streaming mode. Auto-detected if None.
        cache_dir: Optional cache directory.
        **kwargs: Additional arguments passed to load_dataset.

    Returns:
        pandas DataFrame with the loaded and filtered data.
    """
    if dataset_key not in DATASETS:
        available = "\n".join(f"  - {k}: {v['description']}" for k, v in DATASETS.items())
        raise ValueError(f"Unknown dataset key: {dataset_key}\n\nAvailable datasets:\n{available}")

    info = DATASETS[dataset_key]
    ds_name = info["name"]
    config = info["config"]
    data_dir = info.get("data_dir")
    use_streaming = streaming if streaming is not None else info.get("streaming_default", False)

    print(f"Loading dataset: {ds_name}")
    if config:
        print(f"  Config: {config}")
    if data_dir:
        print(f"  Data dir: {data_dir}")
    print(f"  Mode: {'streaming' if use_streaming else 'direct'}")
    print(f"  Description: {info['description']}")

    if use_streaming:
        # Stream in batches and concatenate
        frames = []
        for batch_df in stream_dataset(
            dataset_key,
            asset=asset,
            timeframe=timeframe,
            date=date,
            batch_size=50_000,
            split=split,
            cache_dir=cache_dir,
            **kwargs,
        ):
            frames.append(batch_df)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        print(f"  Final shape: {df.shape}")
        return df
    else:
        # Load directly
        load_kwargs = {"split": split, "streaming": False, **kwargs}
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        if data_dir:
            load_kwargs["data_dir"] = data_dir

        if config:
            ds = load_dataset(ds_name, config, **load_kwargs)
        else:
            ds = load_dataset(ds_name, **load_kwargs)

        if hasattr(ds, "keys"):
            ds = ds[split] if split in ds else list(ds.values())[0]

        df = ds.to_pandas()

        # Apply filters
        if asset and "asset" in df.columns:
            mask = df["asset"].str.lower() == asset.lower()
            df = df[mask].copy()
            print(f"  Filtered to asset={asset}: {len(df):,} rows")
        elif asset and "crypto" in df.columns:
            mask = df["crypto"].str.lower() == asset.lower()
            df = df[mask].copy()
            print(f"  Filtered to crypto={asset}: {len(df):,} rows")
        elif asset and "symbol" in df.columns:
            mask = df["symbol"].str.lower() == asset.lower()
            df = df[mask].copy()
            print(f"  Filtered to symbol={asset}: {len(df):,} rows")

        if timeframe and "timeframe" in df.columns:
            mask = df["timeframe"] == timeframe
            df = df[mask].copy()
            print(f"  Filtered to timeframe={timeframe}: {len(df):,} rows")

        if date and "date" in df.columns:
            mask = df["date"] == date
            df = df[mask].copy()
            print(f"  Filtered to date={date}: {len(df):,} rows")

        print(f"  Final shape: {df.shape}")
        return df


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def load_real_orderbook_data(
    asset: str = "BTC",
    timeframe: str = "5-minute",
    hours: int | None = None,
) -> pd.DataFrame:
    """Load real CLOB orderbook snapshots from the aliplayer dataset."""
    df = load_polymarket_dataset(
        "aliplayer_orderbook",
        asset=asset,
        timeframe=timeframe,
    )
    if hours and "timestamp" in df.columns:
        cutoff = df["timestamp"].max() - hours * 3600
        df = df[df["timestamp"] >= cutoff].copy()
        print(f"  Filtered to last {hours} hours: {len(df):,} rows")
    return df


def load_real_trade_data(
    asset: str = "BTC",
    timeframe: str = "5-minute",
) -> pd.DataFrame:
    """Load real trade-level fills from the aliplayer ticks dataset."""
    return load_polymarket_dataset(
        "aliplayer_ticks",
        asset=asset,
        timeframe=timeframe,
    )


def load_bmoney_trades(asset: str = "BTC") -> pd.DataFrame:
    """Load individual trade executions from the bmoney dataset."""
    return load_polymarket_dataset("bmoney_trades", asset=asset)


def load_bmoney_orderbooks(asset: str = "BTC") -> pd.DataFrame:
    """Load 10-level orderbook snapshots from the bmoney dataset."""
    return load_polymarket_dataset("bmoney_orderbooks", asset=asset)


def load_bmarket_resolutions(asset: str = "BTC") -> pd.DataFrame:
    """Load resolved market outcomes from the bmoney dataset."""
    return load_polymarket_dataset("bmoney_resolutions", asset=asset)


def load_quant_data(asset: str | None = None) -> pd.DataFrame:
    """Load clean quant data with unified YES perspective from daxiongya."""
    return load_polymarket_dataset("daxiongya_quant", asset=asset)


def load_all_market_metadata() -> pd.DataFrame:
    """Load all market metadata from daxiongya (268K markets)."""
    return load_polymarket_dataset("daxiongya_markets")


def list_available_datasets() -> None:
    """Print all available dataset keys and descriptions."""
    print("\nAvailable Polymarket datasets on Hugging Face:\n")
    for key, info in DATASETS.items():
        mode = "stream" if info.get("streaming_default") else "direct"
        print(f"  {key} [{mode}]")
        print(f"    Source: {info['name']}")
        if info["config"]:
            print(f"    Config: {info['config']}")
        print(f"    Description: {info['description']}")
        print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Load Polymarket datasets from Hugging Face (streaming API)")
    parser.add_argument("--list", action="store_true", help="List all available datasets")
    parser.add_argument("--dataset", type=str, help="Dataset key to load")
    parser.add_argument("--asset", type=str, default=None, help="Filter by asset (BTC, ETH, SOL, XRP)")
    parser.add_argument("--timeframe", type=str, default=None, help="Filter by timeframe")
    parser.add_argument("--date", type=str, default=None, help="Filter by date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default=None, help="Save to CSV file")
    parser.add_argument("--stream", action="store_true", help="Stream in batches and save each batch")
    parser.add_argument("--batch-size", type=int, default=50_000, help="Batch size for streaming")
    parser.add_argument("--head", type=int, default=None, help="Print first N rows")
    args = parser.parse_args()

    if args.list:
        list_available_datasets()
        return

    if not args.dataset:
        parser.error("--dataset is required (use --list to see options)")

    if args.stream:
        # Streaming mode - process batches
        batch_num = 0
        for batch_df in stream_dataset(
            args.dataset,
            asset=args.asset,
            timeframe=args.timeframe,
            date=args.date,
            batch_size=args.batch_size,
        ):
            batch_num += 1
            if args.output:
                path = Path(args.output)
                batch_path = path.parent / f"{path.stem}_batch_{batch_num:04d}{path.suffix}"
                batch_df.to_csv(batch_path, index=False)
                print(f"  Saved {batch_path}")
            if args.head:
                print(f"\nBatch #{batch_num} - First {args.head} rows:")
                print(batch_df.head(args.head).to_string())
        print(f"\nTotal batches: {batch_num}")
    else:
        # Load into single DataFrame
        df = load_polymarket_dataset(
            args.dataset,
            asset=args.asset,
            timeframe=args.timeframe,
            date=args.date,
        )

        if args.head:
            print(f"\nFirst {args.head} rows:")
            print(df.head(args.head).to_string())
        else:
            print(f"\nDataset info:")
            print(df.info())
            print(f"\nFirst 5 rows:")
            print(df.head().to_string())

        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
