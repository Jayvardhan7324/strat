# Previous-10s Momentum Next-Market Backtest

This tests whether the last 10 seconds of one 5-minute BTC market can choose the side for the next market.
It sweeps follow/fade direction, previous-window momentum threshold, next-market entry cap, entry-window length, and entry slippage stress.

## Caveat

This uses synthetic Polymarket prices derived from spot data, not historical real orderbooks. Treat promising rows as paper-run candidates, not live-proof edges.

## Baseline Check

`CHEAP_LEADER_PULLBACK` held-out 5c stress baseline: $10,226.74, 2153 trades, 81.98% win rate.

## Top Train Rows

| variant | momentum_bps | entry_cap | entry_window_seconds | slippage_cents | total_pnl | trades | fill_rate | win_rate | avg_entry_price |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| follow | 0.0 | 0.5 | 5 | 0.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |
| follow | 0.0 | 0.5 | 5 | 1.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |
| follow | 0.0 | 0.5 | 5 | 2.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |
| follow | 0.0 | 0.5 | 5 | 5.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |
| follow | 0.0 | 0.5 | 10 | 0.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |
| follow | 0.0 | 0.5 | 10 | 1.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |
| follow | 0.0 | 0.5 | 10 | 2.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |
| follow | 0.0 | 0.5 | 10 | 5.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |
| follow | 0.0 | 0.5 | 20 | 0.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |
| follow | 0.0 | 0.5 | 20 | 1.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |
| follow | 0.0 | 0.5 | 20 | 2.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |
| follow | 0.0 | 0.5 | 20 | 5.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |

## Top Held-Out Test Rows

| variant | momentum_bps | entry_cap | entry_window_seconds | slippage_cents | total_pnl | trades | fill_rate | win_rate | avg_entry_price |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| follow | 0.0 | 0.5 | 5 | 0.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |
| follow | 0.0 | 0.5 | 5 | 1.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |
| follow | 0.0 | 0.5 | 5 | 2.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |
| follow | 0.0 | 0.5 | 5 | 5.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |
| follow | 0.0 | 0.5 | 10 | 0.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |
| follow | 0.0 | 0.5 | 10 | 1.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |
| follow | 0.0 | 0.5 | 10 | 2.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |
| follow | 0.0 | 0.5 | 10 | 5.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |
| follow | 0.0 | 0.5 | 20 | 0.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |
| follow | 0.0 | 0.5 | 20 | 1.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |
| follow | 0.0 | 0.5 | 20 | 2.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |
| follow | 0.0 | 0.5 | 20 | 5.0 | $0.00 | 0 | 0.00% | 0.00% | 0.000 |

## Train/Test Survivors

_No rows._

## Weak Under Slippage

| variant | momentum_bps | entry_cap | entry_window_seconds | slippage_cents | total_pnl_train | total_pnl_test | trades_test | fill_rate_test | win_rate_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| follow | 0.0 | 0.5 | 5 | 5.0 | $0.00 | $0.00 | 0 | 0.00% | 0.00% |
| follow | 0.0 | 0.5 | 10 | 5.0 | $0.00 | $0.00 | 0 | 0.00% | 0.00% |
| follow | 0.0 | 0.5 | 20 | 5.0 | $0.00 | $0.00 | 0 | 0.00% | 0.00% |
| follow | 0.0 | 0.5 | 30 | 5.0 | $0.00 | $0.00 | 0 | 0.00% | 0.00% |
| follow | 0.0 | 0.5 | 60 | 5.0 | $0.00 | $0.00 | 0 | 0.00% | 0.00% |
| follow | 0.0 | 0.51 | 5 | 5.0 | $0.00 | $0.00 | 0 | 0.00% | 0.00% |
| follow | 0.0 | 0.51 | 10 | 5.0 | $0.00 | $0.00 | 0 | 0.00% | 0.00% |
| follow | 0.0 | 0.51 | 20 | 5.0 | $0.00 | $0.00 | 0 | 0.00% | 0.00% |
| follow | 0.0 | 0.51 | 30 | 5.0 | $0.00 | $0.00 | 0 | 0.00% | 0.00% |
| follow | 0.0 | 0.51 | 60 | 5.0 | $0.00 | $0.00 | 0 | 0.00% | 0.00% |
| follow | 0.0 | 0.52 | 5 | 5.0 | $0.00 | $0.00 | 0 | 0.00% | 0.00% |
| follow | 0.0 | 0.52 | 10 | 5.0 | $0.00 | $0.00 | 0 | 0.00% | 0.00% |

## Recommendation

Reject v1: no held-out evidence strong enough to justify paper-running.

## Files

- Metrics: `prev10_momentum_next_metrics.csv`
- Survival summary: `prev10_momentum_next_survival.csv`
- Selected trade logs: `prev10_momentum_next_best_trades.csv`
