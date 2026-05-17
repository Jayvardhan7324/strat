# Previous-60s Momentum Next-Market Diagnostic Backtest

This tests whether the last 60 seconds of one 5-minute BTC market can choose the side for the next market.
It sweeps follow/fade direction, previous-window momentum threshold, next-market entry cap, entry-window length, and entry slippage stress.

## Caveat

This uses synthetic Polymarket prices derived from spot data, not historical real orderbooks. Treat promising rows as paper-run candidates, not live-proof edges.

## Baseline Check

`CHEAP_LEADER_PULLBACK` held-out 5c stress baseline: $10,226.74, 2153 trades, 81.98% win rate.

## Top Train Rows

| variant | momentum_bps | entry_cap | entry_window_seconds | slippage_cents | total_pnl | trades | fill_rate | win_rate | avg_entry_price |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fade | 0.0 | 0.57 | 60 | 0.0 | $1,104.65 | 6109 | 100.00% | 51.51% | 0.505 |
| fade | 0.0 | 0.57 | 30 | 0.0 | $1,104.65 | 6109 | 100.00% | 51.51% | 0.505 |
| fade | 0.0 | 0.6 | 10 | 0.0 | $1,104.65 | 6109 | 100.00% | 51.51% | 0.505 |
| fade | 0.0 | 0.6 | 5 | 0.0 | $1,104.65 | 6109 | 100.00% | 51.51% | 0.505 |
| fade | 0.0 | 0.55 | 20 | 0.0 | $1,104.65 | 6109 | 100.00% | 51.51% | 0.505 |
| fade | 0.0 | 0.55 | 5 | 0.0 | $1,104.65 | 6109 | 100.00% | 51.51% | 0.505 |
| fade | 0.0 | 0.57 | 10 | 0.0 | $1,104.65 | 6109 | 100.00% | 51.51% | 0.505 |
| fade | 0.0 | 0.57 | 20 | 0.0 | $1,104.65 | 6109 | 100.00% | 51.51% | 0.505 |
| fade | 0.0 | 0.55 | 60 | 0.0 | $1,104.65 | 6109 | 100.00% | 51.51% | 0.505 |
| fade | 0.0 | 0.57 | 5 | 0.0 | $1,104.65 | 6109 | 100.00% | 51.51% | 0.505 |
| fade | 0.0 | 0.65 | 5 | 0.0 | $1,104.65 | 6109 | 100.00% | 51.51% | 0.505 |
| fade | 0.0 | 0.6 | 60 | 0.0 | $1,104.65 | 6109 | 100.00% | 51.51% | 0.505 |

## Top Held-Out Test Rows

| variant | momentum_bps | entry_cap | entry_window_seconds | slippage_cents | total_pnl | trades | fill_rate | win_rate | avg_entry_price |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fade | 1.0 | 0.65 | 10 | 0.0 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |
| fade | 1.0 | 0.65 | 60 | 0.0 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |
| fade | 1.0 | 0.6 | 30 | 0.0 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |
| fade | 1.0 | 0.57 | 60 | 0.0 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |
| fade | 1.0 | 0.6 | 20 | 0.0 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |
| fade | 1.0 | 0.53 | 20 | 0.0 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |
| fade | 1.0 | 0.57 | 5 | 0.0 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |
| fade | 1.0 | 0.55 | 60 | 0.0 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |
| fade | 1.0 | 0.6 | 10 | 0.0 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |
| fade | 1.0 | 0.6 | 5 | 0.0 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |
| fade | 1.0 | 0.51 | 5 | 0.0 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |
| fade | 1.0 | 0.65 | 20 | 0.0 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |

## Train/Test Survivors

| variant | momentum_bps | entry_cap | entry_window_seconds | slippage_cents | total_pnl_train | total_pnl_test | trades_test | fill_rate_test | win_rate_test | avg_entry_price_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fade | 1.0 | 0.51 | 5 | 0.0 | $975.93 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |
| fade | 1.0 | 0.51 | 10 | 0.0 | $975.93 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |
| fade | 1.0 | 0.51 | 20 | 0.0 | $975.93 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |
| fade | 1.0 | 0.51 | 30 | 0.0 | $975.93 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |
| fade | 1.0 | 0.51 | 60 | 0.0 | $975.93 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |
| fade | 1.0 | 0.52 | 5 | 0.0 | $975.93 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |
| fade | 1.0 | 0.52 | 10 | 0.0 | $975.93 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |
| fade | 1.0 | 0.52 | 20 | 0.0 | $975.93 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |
| fade | 1.0 | 0.52 | 30 | 0.0 | $975.93 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |
| fade | 1.0 | 0.52 | 60 | 0.0 | $975.93 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |
| fade | 1.0 | 0.53 | 5 | 0.0 | $975.93 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |
| fade | 1.0 | 0.53 | 10 | 0.0 | $975.93 | $817.44 | 1950 | 100.00% | 52.72% | 0.505 |

## Weak Under Slippage

| variant | momentum_bps | entry_cap | entry_window_seconds | slippage_cents | total_pnl_train | total_pnl_test | trades_test | fill_rate_test | win_rate_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| follow | 0.0 | 0.57 | 5 | 5.0 | $-7,842.81 | $-3,609.46 | 2590 | 100.00% | 47.88% |
| follow | 0.0 | 0.57 | 10 | 5.0 | $-7,842.81 | $-3,609.46 | 2590 | 100.00% | 47.88% |
| follow | 0.0 | 0.57 | 20 | 5.0 | $-7,842.81 | $-3,609.46 | 2590 | 100.00% | 47.88% |
| follow | 0.0 | 0.57 | 30 | 5.0 | $-7,842.81 | $-3,609.46 | 2590 | 100.00% | 47.88% |
| follow | 0.0 | 0.57 | 60 | 5.0 | $-7,842.81 | $-3,609.46 | 2590 | 100.00% | 47.88% |
| follow | 0.0 | 0.6 | 5 | 5.0 | $-7,842.81 | $-3,609.46 | 2590 | 100.00% | 47.88% |
| follow | 0.0 | 0.6 | 10 | 5.0 | $-7,842.81 | $-3,609.46 | 2590 | 100.00% | 47.88% |
| follow | 0.0 | 0.6 | 20 | 5.0 | $-7,842.81 | $-3,609.46 | 2590 | 100.00% | 47.88% |
| follow | 0.0 | 0.6 | 30 | 5.0 | $-7,842.81 | $-3,609.46 | 2590 | 100.00% | 47.88% |
| follow | 0.0 | 0.6 | 60 | 5.0 | $-7,842.81 | $-3,609.46 | 2590 | 100.00% | 47.88% |
| follow | 0.0 | 0.65 | 5 | 5.0 | $-7,842.81 | $-3,609.46 | 2590 | 100.00% | 47.88% |
| follow | 0.0 | 0.65 | 10 | 5.0 | $-7,842.81 | $-3,609.46 | 2590 | 100.00% | 47.88% |

## Recommendation

Paper-run candidate only if it remains competitive after live orderbook checks. Best survivor is `fade_mom1bps_cap0.51_win5s_slip0.00c` with held-out test PnL $817.44. Best positive-slippage survivor is `fade_mom3bps_cap0.52_win5s_slip1.00c` with held-out test PnL $511.30. No 5c slippage combo survived.

## Files

- Metrics: `prev60_momentum_next_metrics.csv`
- Survival summary: `prev60_momentum_next_survival.csv`
- Selected trade logs: `prev60_momentum_next_best_trades.csv`
