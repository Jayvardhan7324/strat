"""
Analyze contract price micro-movements to understand what's possible for scalping.

Looks at:
1. How much fair_up moves second-to-second
2. Bid/ask spread behavior
3. How often price moves enough to scalp after fees
4. What entry/exit timing works best
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtests.polymarket_updown_backtest import (
    WINDOW_SECONDS,
    Window,
    build_windows,
    load_spot_prices,
    side_arrays,
    fair_prices_for_window,
    make_market_arrays,
    clip_contract_price,
)

price_df = load_spot_prices("btcusdt", "binance", dataset_source="aliplayer_spot")
windows = build_windows(price_df, max_windows=5000)

print("\n" + "=" * 80)
print("CONTRACT PRICE MICRO-MOVEMENT ANALYSIS")
print("=" * 80)

# Sample a few windows to look at contract price behavior
sample_windows = windows[:10]

for wi, w in enumerate(sample_windows):
    print(f"\n--- Window {wi}: {w.start} (BTC open={w.open_price:.2f}, close={w.close_price:.2f}, outcome={'UP' if w.outcome_up else 'DOWN'}) ---")

    # Fair value movement
    fair = w.fair_up
    print(f"Fair UP: {fair[0]:.4f} -> {fair[-1]:.4f} (range: {fair.min():.4f} - {fair.max():.4f})")

    # Second-to-second changes in fair value
    fair_diffs = np.diff(fair)
    print(f"Fair changes: mean={np.mean(np.abs(fair_diffs)):.5f}, max={np.max(np.abs(fair_diffs)):.5f}, median={np.median(np.abs(fair_diffs)):.5f}")

    # How often does fair move by at least X cents?
    for threshold in [0.001, 0.002, 0.003, 0.005, 0.01]:
        big_moves = np.sum(np.abs(fair_diffs) >= threshold)
        print(f"  Moves >= {threshold:.3f}: {big_moves} times ({100*big_moves/len(fair_diffs):.1f}%)")

    # Bid/ask spread
    spread = w.up_ask - w.up_bid
    print(f"Bid/ask spread: mean={np.mean(spread):.4f}, min={np.min(spread):.4f}, max={np.max(spread):.4f}")

    # If we bought at ask[120] and sold at bid[X], what's the best exit?
    entry_idx = 120
    entry_ask = float(w.up_ask[entry_idx])
    best_bid_after = float(w.up_bid[entry_idx+1:].max())
    best_pnl = best_bid_after - entry_ask
    print(f"If bought UP at ask[{entry_idx}]={entry_ask:.4f}, best bid after={best_bid_after:.4f}, best PnL/share={best_pnl:.4f}")

    # What about DOWN contract?
    entry_ask_down = float(w.down_ask[entry_idx])
    best_bid_down = float(w.down_bid[entry_idx+1:].max())
    best_pnl_down = best_bid_down - entry_ask_down
    print(f"If bought DOWN at ask[{entry_idx}]={entry_ask_down:.4f}, best bid after={best_bid_down:.4f}, best PnL/share={best_pnl_down:.4f}")

# Aggregate analysis across all windows
print("\n" + "=" * 80)
print("AGGREGATE ANALYSIS (all windows)")
print("=" * 80)

all_fair_diffs = []
all_spreads = []
best_scalp_pnls = []

for w in windows:
    fair = w.fair_up
    fair_diffs = np.diff(fair)
    all_fair_diffs.append(fair_diffs)

    spread = w.up_ask - w.up_bid
    all_spreads.append(spread)

    # Best possible scalp from each second
    for entry_idx in range(60, 270):
        entry_ask = float(w.up_ask[entry_idx])
        remaining_bids = w.up_bid[entry_idx+1:]
        if len(remaining_bids) > 0:
            best_exit = float(remaining_bids.max())
            best_scalp_pnls.append(best_exit - entry_ask)

all_fair_diffs = np.concatenate(all_fair_diffs)
all_spreads = np.concatenate(all_spreads)

print(f"\nFair value second-to-second changes:")
print(f"  Mean abs change: {np.mean(np.abs(all_fair_diffs)):.5f}")
print(f"  Median abs change: {np.median(np.abs(all_fair_diffs)):.5f}")
print(f"  90th pct: {np.percentile(np.abs(all_fair_diffs), 90):.5f}")
print(f"  95th pct: {np.percentile(np.abs(all_fair_diffs), 95):.5f}")
print(f"  99th pct: {np.percentile(np.abs(all_fair_diffs), 99):.5f}")

print(f"\nBid/ask spread:")
print(f"  Mean: {np.mean(all_spreads):.4f}")
print(f"  Median: {np.median(all_spreads):.4f}")

print(f"\nBest possible scalp PnL per share (buy at ask, sell at best bid after):")
print(f"  Mean: {np.mean(best_scalp_pnls):.5f}")
print(f"  Median: {np.median(best_scalp_pnls):.5f}")
print(f"  Positive: {100*np.mean(np.array(best_scalp_pnls) > 0):.1f}%")
print(f"  >= 0.005: {100*np.mean(np.array(best_scalp_pnls) >= 0.005):.1f}%")
print(f"  >= 0.010: {100*np.mean(np.array(best_scalp_pnls) >= 0.010):.1f}%")
print(f"  >= 0.015: {100*np.mean(np.array(best_scalp_pnls) >= 0.015):.1f}%")

# After-fee analysis
TAKER_FEE = 0.002
print(f"\nAfter taker fees (0.2% both sides):")
for target in [0.005, 0.010, 0.015, 0.020, 0.030]:
    entry_prices = np.array([0.50] * len(best_scalp_pnls))  # approximate
    fee_per_share = entry_prices * TAKER_FEE * 2  # entry + exit
    net_pnls = np.array(best_scalp_pnls) - fee_per_share
    profitable = np.mean(net_pnls >= target)
    print(f"  Net PnL >= {target:.3f}: {100*profitable:.1f}%")

# Look at windows where BTC chopped a lot
print("\n" + "=" * 80)
print("HIGH CHOP WINDOWS ANALYSIS")
print("=" * 80)

chop_windows = []
for w in windows:
    p = w.prices[:120]
    range_pct = (p.max() - p.min()) / w.open_price
    if range_pct > 0.001:
        chop_windows.append(w)

print(f"Windows with >0.1% BTC range in first 120s: {len(chop_windows)}")

if chop_windows:
    chop_scalp_pnls = []
    for w in chop_windows:
        for entry_idx in range(120, 270):
            entry_ask = float(w.up_ask[entry_idx])
            remaining_bids = w.up_bid[entry_idx+1:]
            if len(remaining_bids) > 0:
                best_exit = float(remaining_bids.max())
                chop_scalp_pnls.append(best_exit - entry_ask)

    chop_scalp_pnls = np.array(chop_scalp_pnls)
    print(f"Best scalp PnL in chop windows:")
    print(f"  Mean: {np.mean(chop_scalp_pnls):.5f}")
    print(f"  Median: {np.median(chop_scalp_pnls):.5f}")
    print(f"  >= 0.005: {100*np.mean(chop_scalp_pnls >= 0.005):.1f}%")
    print(f"  >= 0.010: {100*np.mean(chop_scalp_pnls >= 0.010):.1f}%")
    print(f"  >= 0.015: {100*np.mean(chop_scalp_pnls >= 0.015):.1f}%")

    # DOWN contract too
    chop_scalp_pnls_down = []
    for w in chop_windows:
        for entry_idx in range(120, 270):
            entry_ask = float(w.down_ask[entry_idx])
            remaining_bids = w.down_bid[entry_idx+1:]
            if len(remaining_bids) > 0:
                best_exit = float(remaining_bids.max())
                chop_scalp_pnls_down.append(best_exit - entry_ask)

    chop_scalp_pnls_down = np.array(chop_scalp_pnls_down)
    print(f"\nDOWN contract best scalp PnL in chop windows:")
    print(f"  Mean: {np.mean(chop_scalp_pnls_down):.5f}")
    print(f"  >= 0.005: {100*np.mean(chop_scalp_pnls_down >= 0.005):.1f}%")
    print(f"  >= 0.010: {100*np.mean(chop_scalp_pnls_down >= 0.010):.1f}%")

    # Combined: can we ALWAYS scalp either UP or DOWN?
    combined_best = []
    for w in chop_windows:
        for entry_idx in range(120, 270):
            up_ask = float(w.up_ask[entry_idx])
            up_best = float(w.up_bid[entry_idx+1:].max()) if len(w.up_bid[entry_idx+1:]) > 0 else 0
            down_ask = float(w.down_ask[entry_idx])
            down_best = float(w.down_bid[entry_idx+1:].max()) if len(w.down_bid[entry_idx+1:]) > 0 else 0
            combined_best.append(max(up_best - up_ask, down_best - down_ask))

    combined_best = np.array(combined_best)
    print(f"\nBest of UP or DOWN contract scalp:")
    print(f"  Mean: {np.mean(combined_best):.5f}")
    print(f"  Median: {np.median(combined_best):.5f}")
    print(f"  >= 0.005: {100*np.mean(combined_best >= 0.005):.1f}%")
    print(f"  >= 0.010: {100*np.mean(combined_best >= 0.010):.1f}%")
    print(f"  >= 0.015: {100*np.mean(combined_best >= 0.015):.1f}%")
    print(f"  >= 0.020: {100*np.mean(combined_best >= 0.020):.1f}%")
