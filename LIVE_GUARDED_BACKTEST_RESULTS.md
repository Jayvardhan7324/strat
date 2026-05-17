# Live-Guarded Strategy Backtest Results

This is the stricter follow-up to the first high-Sharpe research backtest.

The goal was to test strategies closer to live Polymarket behavior:

- smaller fixed stakes
- no entries near `0.90+`
- hard max entry prices
- minimum profit-if-win
- max `oneLossWipesOutWins`
- one no-memory train/test split
- stress test with worse entry prices

## What Changed

The first backtest was too forgiving. It assumed synthetic fills and made the strategies look much cleaner than live trading. The live bot showed the real problem: if the bot buys a contract at `0.85` to `0.95`, a win makes very little and a single loss can erase many wins.

The guarded backtest rejects those trades.

## Strategies Tested

| Strategy | Idea | Live status |
|---|---|---|
| `LIVE_ORB` | Opening Range Breakout with max entry and payout guards. | Already live-config candidate |
| `LIVE_VWM` | Volume Weighted Momentum with max entry and payout guards. | Already live-config candidate |
| `LIVE_RENKO` | Renko sequencing with max entry and payout guards. | Already live-config candidate |
| `LIVE_PRIORITY_PORTFOLIO` | One trade per market, priority order: ORB, then VWM, then Renko. | Current safest portfolio structure |
| `CONSENSUS_2_OF_3` | Trade only when at least two of ORB, VWM, Renko agree. | New candidate |
| `CHEAP_LEADER_PULLBACK` | Buy the current winning side only if it is still cheap enough. | New strongest candidate |
| `OPEN_FLIP_VALUE` | If price crossed both sides of the open, buy the re-break side only if still cheap. | Weak research candidate |

## Stress-Test Result

The table below shows the held-out test period only. `5c` means the script added `+0.05` to every entry price before applying the gates.

| Strategy | Slippage | Test PnL | Test trades | Test win rate | Avg entry | Raw 5m Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| `CHEAP_LEADER_PULLBACK` | `5c` | `$10,226.74` | `2,153` | `81.98%` | `0.556` | `0.566` |
| `LIVE_PRIORITY_PORTFOLIO` | `5c` | `$8,102.13` | `2,615` | `72.77%` | `0.556` | `0.380` |
| `LIVE_ORB` | `5c` | `$7,886.18` | `2,579` | `72.55%` | `0.556` | `0.371` |
| `CONSENSUS_2_OF_3` | `5c` | `$6,181.63` | `1,326` | `81.52%` | `0.556` | `0.419` |
| `LIVE_VWM` | `5c` | `$4,369.87` | `1,082` | `78.00%` | `0.556` | `0.317` |
| `LIVE_RENKO` | `5c` | `$4,166.60` | `638` | `92.01%` | `0.557` | `0.420` |
| `OPEN_FLIP_VALUE` | `5c` | `$968.25` | `328` | `71.95%` | `0.556` | `0.120` |

## Interpretation

The best result was `CHEAP_LEADER_PULLBACK`. It is not implemented in the live bot yet. It only buys the side that is already winning when the contract is still cheap enough to pay a sane reward/risk.

The safest currently implemented structure is `LIVE_PRIORITY_PORTFOLIO`: ORB first, then VWM, then Renko, with one trade per market.

The most selective candidate is `CONSENSUS_2_OF_3`. It trades less often but has a stronger win rate and survived the harsh `5c` stress test.

## Caveat

This still does not prove an edge over the real Polymarket market. The test uses synthetic contract prices derived from spot, not historical real orderbook fills. A real edge requires paper logs with actual Polymarket best ask, depth, fillability, and latency.

Use this result to decide what to paper-run next, not to justify live sizing.

## Files

- Backtest script: `live_guarded_backtest.py`
- Stress summary CSV: `live_guarded_slippage_stress_summary.csv`
- Per-slippage metric CSVs: `live_guarded_metrics_slip_*.csv`
- Equity charts: `live_guarded_equity_*_slip_*.png`

