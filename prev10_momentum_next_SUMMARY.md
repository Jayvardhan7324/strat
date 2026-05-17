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
| fade | 3.0 | 0.65 | 30 | 0.0 | $316.06 | 672 | 100.00% | 52.98% | 0.505 |
| fade | 3.0 | 0.55 | 30 | 0.0 | $316.06 | 672 | 100.00% | 52.98% | 0.505 |
| fade | 3.0 | 0.53 | 30 | 0.0 | $316.06 | 672 | 100.00% | 52.98% | 0.505 |
| fade | 3.0 | 0.51 | 20 | 0.0 | $316.06 | 672 | 100.00% | 52.98% | 0.505 |
| fade | 3.0 | 0.6 | 30 | 0.0 | $316.06 | 672 | 100.00% | 52.98% | 0.505 |
| fade | 3.0 | 0.65 | 10 | 0.0 | $316.06 | 672 | 100.00% | 52.98% | 0.505 |
| fade | 3.0 | 0.65 | 5 | 0.0 | $316.06 | 672 | 100.00% | 52.98% | 0.505 |
| fade | 3.0 | 0.53 | 10 | 0.0 | $316.06 | 672 | 100.00% | 52.98% | 0.505 |
| fade | 3.0 | 0.53 | 60 | 0.0 | $316.06 | 672 | 100.00% | 52.98% | 0.505 |
| fade | 3.0 | 0.51 | 5 | 0.0 | $316.06 | 672 | 100.00% | 52.98% | 0.505 |
| fade | 3.0 | 0.52 | 10 | 0.0 | $316.06 | 672 | 100.00% | 52.98% | 0.505 |
| fade | 3.0 | 0.52 | 5 | 0.0 | $316.06 | 672 | 100.00% | 52.98% | 0.505 |

## Top Held-Out Test Rows

| variant | momentum_bps | entry_cap | entry_window_seconds | slippage_cents | total_pnl | trades | fill_rate | win_rate | avg_entry_price |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fade | 2.0 | 0.65 | 60 | 0.0 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |
| fade | 2.0 | 0.65 | 10 | 0.0 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |
| fade | 2.0 | 0.65 | 30 | 0.0 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |
| fade | 2.0 | 0.6 | 60 | 0.0 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |
| fade | 2.0 | 0.6 | 30 | 0.0 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |
| fade | 2.0 | 0.57 | 10 | 0.0 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |
| fade | 2.0 | 0.57 | 30 | 0.0 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |
| fade | 2.0 | 0.65 | 5 | 0.0 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |
| fade | 2.0 | 0.65 | 20 | 0.0 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |
| fade | 2.0 | 0.6 | 5 | 0.0 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |
| fade | 2.0 | 0.6 | 20 | 0.0 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |
| fade | 2.0 | 0.6 | 10 | 0.0 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |

## Train/Test Survivors

| variant | momentum_bps | entry_cap | entry_window_seconds | slippage_cents | total_pnl_train | total_pnl_test | trades_test | fill_rate_test | win_rate_test | avg_entry_price_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fade | 2.0 | 0.51 | 5 | 0.0 | $87.62 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |
| fade | 2.0 | 0.51 | 10 | 0.0 | $87.62 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |
| fade | 2.0 | 0.51 | 20 | 0.0 | $87.62 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |
| fade | 2.0 | 0.51 | 30 | 0.0 | $87.62 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |
| fade | 2.0 | 0.51 | 60 | 0.0 | $87.62 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |
| fade | 2.0 | 0.52 | 5 | 0.0 | $87.62 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |
| fade | 2.0 | 0.52 | 10 | 0.0 | $87.62 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |
| fade | 2.0 | 0.52 | 20 | 0.0 | $87.62 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |
| fade | 2.0 | 0.52 | 30 | 0.0 | $87.62 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |
| fade | 2.0 | 0.52 | 60 | 0.0 | $87.62 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |
| fade | 2.0 | 0.53 | 5 | 0.0 | $87.62 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |
| fade | 2.0 | 0.53 | 10 | 0.0 | $87.62 | $550.79 | 530 | 100.00% | 55.85% | 0.505 |

## Weak Under Slippage

| variant | momentum_bps | entry_cap | entry_window_seconds | slippage_cents | total_pnl_train | total_pnl_test | trades_test | fill_rate_test | win_rate_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| follow | 0.0 | 0.57 | 5 | 5.0 | $-4,501.96 | $-2,181.54 | 2140 | 100.00% | 49.95% |
| follow | 0.0 | 0.57 | 10 | 5.0 | $-4,501.96 | $-2,181.54 | 2140 | 100.00% | 49.95% |
| follow | 0.0 | 0.57 | 20 | 5.0 | $-4,501.96 | $-2,181.54 | 2140 | 100.00% | 49.95% |
| follow | 0.0 | 0.57 | 30 | 5.0 | $-4,501.96 | $-2,181.54 | 2140 | 100.00% | 49.95% |
| follow | 0.0 | 0.57 | 60 | 5.0 | $-4,501.96 | $-2,181.54 | 2140 | 100.00% | 49.95% |
| follow | 0.0 | 0.6 | 5 | 5.0 | $-4,501.96 | $-2,181.54 | 2140 | 100.00% | 49.95% |
| follow | 0.0 | 0.6 | 10 | 5.0 | $-4,501.96 | $-2,181.54 | 2140 | 100.00% | 49.95% |
| follow | 0.0 | 0.6 | 20 | 5.0 | $-4,501.96 | $-2,181.54 | 2140 | 100.00% | 49.95% |
| follow | 0.0 | 0.6 | 30 | 5.0 | $-4,501.96 | $-2,181.54 | 2140 | 100.00% | 49.95% |
| follow | 0.0 | 0.6 | 60 | 5.0 | $-4,501.96 | $-2,181.54 | 2140 | 100.00% | 49.95% |
| follow | 0.0 | 0.65 | 5 | 5.0 | $-4,501.96 | $-2,181.54 | 2140 | 100.00% | 49.95% |
| follow | 0.0 | 0.65 | 10 | 5.0 | $-4,501.96 | $-2,181.54 | 2140 | 100.00% | 49.95% |

## Recommendation

Paper-run candidate only if it remains competitive after live orderbook checks. Best survivor is `fade_mom2bps_cap0.51_win5s_slip0.00c` with held-out test PnL $550.79. Best positive-slippage survivor is `fade_mom5bps_cap0.52_win5s_slip1.00c` with held-out test PnL $210.10. Best 5c survivor is `fade_mom7.5bps_cap0.57_win5s_slip5.00c` with held-out test PnL $77.69.

## Files

- Metrics: `prev10_momentum_next_metrics.csv`
- Survival summary: `prev10_momentum_next_survival.csv`
- Selected trade logs: `prev10_momentum_next_best_trades.csv`
