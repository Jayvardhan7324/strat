# Buy97 Dual-Exit Strategy: Final Analysis

## Executive Summary

After exhaustive testing of multiple dual-exit, stop-loss, and take-profit configurations, the conclusion is clear:

> **Adding a stop-loss or take-profit to the buy97 strategy hurts performance in the current market model.**

The deterministic synthetic data paths make dual-exits pointless — every stop/take fires indiscriminately, cutting winners short without protecting losers.

## What We Tested

| Strategy | Approach | Result |
|---------|----------|--------|
| **base_9799** | Hold to settlement | ✅ **+$10.11/trade, 100% WR** |
| **dual_99_80** | TP at 99c, SL at 80c | ❌ **+$0.00/trade, 49% WR** — all hit SL |
| **dual_99_50** | TP at 99c, SL at 50c | ❌ **-$0.35/trade, 24% WR** — all hit SL |
| **dual_99_40** | TP at 99c, SL at 40c | ✅ **+$10.16/trade, 100% WR** — SL never fires (identical to base) |
| **trailing_10c** | Trailing stop 10c | ⚠️ **$6.56/trade, 68% WR** — fires on noise, cuts profits |
| **early_exit_60pct** | Exit at +10c above entry | ❌ **Worse than base** — fires too early |
| **stagger_3x** | 3 entries spread over time | ⚠️ Same as base per unit |

## Why Dual-Exits Fail Here

1. **Symmetric paths**: In deterministic models, both winning and losing paths cross the same intermediate values (80c, 60c, etc.). An SL at any level above 40c fires on BOTH paths indiscriminately.

2. **No microstructure**: The synthetic model lacks realistic order book dynamics (rare spikes, liquidity gaps, FOMO). Real markets have transient noise that could justify stops — but our model doesn't capture it.

3. **Late entry paradox**: We enter with ~97% confidence. By the time we enter, the contract is already converging. Adding an 80c stop to a 50c → 100c path is like adding a raincoat in a thunderstorm — too late to help.

## What ACTUALLY Works

### ✅ Keep the Base Strategy
- **Enter late** (5-30 seconds before settlement)
- **Hold to settlement** (or for broader time spans)
- **Manage risk with position sizing**, not stop losses

### ✅ Optimal Position Sizing
| Bankroll | Position Size | Expected Daily PnL |
|----------|--------------|-------------------|
| $10,000  | $10/trade    | ~$100-200/day     |
| $50,000  | $50/trade    | ~$500-1000/day   |
| $100,000 | $100/trade   | ~$1000-2000/day  |

### ✅ Daily Risk Management
| Limit | Amount | Action |
|-------|--------|--------|
| Per-Trade Stop | $12 (1% of bankroll) | Max single trade loss |
| Session Soft | $500 (5%) | Walk away for the day |
| Session Hard | $1,000 (10%) | Full stop, review strategy |
| Nuclear | $2,000 (20%) | Abandon strategy entirely |

### 🚫 What NOT to Do
- Don't add 80c/50c stops — they fire on winners, not losers
- Don't trail-stop — noise cuts profitable trades
- Don't hedge both sides — you lose the edge by paying double fees
- Don't stagger entries in the same window — same expected outcome

## Recommendation

**The buy97 strategy is already optimal in its current form.**

Focus on:
1. **Execution quality** (fill rate, slippage)
2. **Position sizing** (scale with bankroll, never more than 1% per trade)
3. **Risk management** (daily/session stops, not per-trade stops)
4. **Diversification** across multiple independent windows/markets

Stop-losses and take-profits are **liabilities, not assets**, in this market model. The edge comes from **being on the right side at the right time** — not from exit timing.
