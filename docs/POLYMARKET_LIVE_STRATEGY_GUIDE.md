# Polymarket 5-Minute Crypto Up/Down Strategy Guide

This file explains the strategies from `polymarket_updown_backtest.py` in a format that can be handed to a live Polymarket trading project.

The backtest used simulated 5-minute binary markets built from BTC spot data. A live bot must replace simulated fills with real Polymarket CLOB orderbook data, real order IDs, private trade confirmations, cancel/replace logic, and explicit risk controls.

## Current Validation Snapshot

Dataset and split:

| Item | Value |
|---|---:|
| Symbol/source | `btcusdt` / `binance` |
| Complete non-tie 5-minute windows | `8,964` |
| Train windows | `6,274` |
| Test windows | `2,690` |
| Train period | `2026-03-26 03:05 UTC` to `2026-04-17 05:00 UTC` |
| Test period | `2026-04-17 05:05 UTC` to `2026-04-26 17:55 UTC` |
| Starting capital per split | `$10,000` |
| No-memory rule | Train and test were run separately with fresh capital and no state crossing the split |

Out-of-sample survivors:

| Rank | Strategy | Train PnL | Test PnL | Test win rate | Test trades | Verdict |
|---:|---|---:|---:|---:|---:|---|
| 1 | `08 Opening Range Breakout` | `$1,331,275.62` | `$564,839.31` | `72.08%` | `2,636` | PASS |
| 2 | `10 Volume Weighted Momentum` | `$768,339.58` | `$295,867.46` | `78.00%` | `1,082` | PASS |
| 3 | `09 Renko Change Sequencing` | `$699,109.14` | `$263,180.77` | `92.01%` | `638` | PASS |
| 4 | `06 Double MA Crossover` | `$606,939.31` | `$242,168.65` | `59.65%` | `2,689` | PASS |
| 5 | `02 Composite Technical Score` | `$374,640.27` | `$141,119.07` | `55.89%` | `2,675` | PASS |
| 6 | `03 Oracle Lag Arbitrage` | `$50,722.33` | `$8,331.82` | `100.00%` | `17` | PASS, low sample |
| 7 | `12 Creative Wick Reclaim Fade` | `$411.78` | `$1,916.53` | `64.29%` | `14` | PASS, low sample |

Important read:

The large PnL values are from a simplified simulated market model, not proof of real-world fillability. Treat the ranking as a research signal. The first live implementation should be paper-only, then tiny-size, then gradually scaled only if live slippage and fill data match expectations.

## Live Polymarket Integration Requirements

Use the official Polymarket CLOB documentation as the implementation reference:

| Area | Live requirement |
|---|---|
| Market metadata | Find the active 5-minute crypto up/down market and map the correct `UP` and `DOWN` token IDs. |
| Orderbook stream | Subscribe to the market WebSocket for orderbook, best bid/ask, price changes, trades, tick-size changes, new-market events, and market-resolved events. |
| Orders | Use the CLOB SDK or REST API for signed orders. Market-like orders should be sent as marketable orders with strict worst-price protection. |
| Private fills | Track private order confirmations, fills, cancels, rejects, partial fills, and replacements. Do not infer fills only from public book movement. |
| Tick size | React to `tick_size_change` events. If the bot quotes an old tick size, orders can be rejected. |
| Expiration | Any passive order must expire or be cancelled before the market becomes too close to settlement. |
| Spot feed | Use a low-latency BTC spot feed with heartbeat checks. If spot data is stale, stop trading. |
| Time sync | Use exchange/server time discipline. The 5-minute window open, T-minus triggers, and settlement logic depend on exact timing. |
| Paper mode | Required first. Log the signal, intended order, observed book, expected fill, actual live fill simulation, and settlement. |

Docs checked on `2026-05-12`:

| Source | Why it matters |
|---|---|
| [Polymarket CLOB trading overview](https://docs.polymarket.com/trading/overview) | CLOB architecture, SDKs, auth levels, and trading client setup. |
| [Polymarket orderbook docs](https://docs.polymarket.com/trading/orderbook) | Live WebSocket orderbook events, best bid/ask updates, trade events, and tick-size changes. |
| [Polymarket create order docs](https://polymarket-292d1b1b.mintlify.app/trading/orders/create) | Limit orders, marketable orders, FOK/FAK behavior, GTD orders, and slippage protection. |

## Shared Market Model

The backtest modeled each 5-minute market as:

```text
window_start = aligned 5-minute timestamp
S0 = first spot price in the window
St = current spot price
outcome_up = 1 if close_price > open_price else 0
```

Synthetic fair price used in the backtest:

```text
fair_up = NormalCDF(log(St / S0) / (sigma * sqrt(tau)))
sigma = 0.60 annualized
tau = seconds_remaining / 300
fair_down = 1 - fair_up
```

Live note:

Do not treat `fair_up` as the tradable Polymarket price. In live trading it is only a model-derived probability estimate. Actual entries and exits must use the real `UP` and `DOWN` token orderbooks.

## Recommended Rollout Order

| Tier | Strategies | Action |
|---|---|---|
| Tier 1 | `08`, `10`, `09`, `06`, `02` | Implement first in paper mode. These had the strongest train/test survival. |
| Tier 2 | `03`, `12` | Implement as research modules only. Good OOS result but low trade count. |
| Disabled | `01`, `04`, `05`, `07`, `11`, `13`, `14` | Do not trade live without redesign. Keep only for telemetry and research. |

## Core Live Risk Rules

| Risk control | Rule |
|---|---|
| One trade per strategy per market | Prevent duplicate entries from reconnects or repeated signals. |
| Max total exposure per market | Cap aggregate `UP` plus `DOWN` position across all strategies. |
| Max taker price | Reject marketable buys above the strategy's worst allowed price. |
| Min book depth | Reject entries if there is not enough size at or below the worst allowed price. |
| Stale spot cutoff | Disable signals if spot feed age exceeds `2s`. |
| Stale book cutoff | Disable orders if best bid/ask age exceeds `1s`. |
| Settlement cutoff | Do not enter new positions inside the final `5s` unless the strategy explicitly requires it. |
| Kill switch | Stop all new orders after daily loss, repeated rejects, websocket desync, or market metadata mismatch. |
| Paper-first gate | No live orders until paper mode logs at least several hundred eligible windows. |

## Strategy 01: End-of-Window Momentum Sniper

Status:

| Field | Value |
|---|---|
| Recommendation | Disabled |
| Train PnL | `$0.00` |
| Test PnL | `$0.00` |
| Test trades | `0` |
| Reason | No clean fills/signals under the tested limit rule. |

Thesis:

This strategy tries to buy the nearly certain side very late in the market, but only if it can get a cheap limit fill. It is trying to exploit stale liquidity near settlement.

Signal:

At `T-10s`, calculate model fair prices.

```text
if fair_up > 0.95:
    place limit buy UP at 0.90
elif fair_down > 0.95:
    place limit buy DOWN at 0.90
else:
    no trade
```

Order behavior:

The backtest used a passive limit buy at `0.90`. It only fills if the ask falls to or below `0.90`.

Exit:

Hold to settlement.

Live notes:

This did not trade in the validation run. The idea is only useful if the real book occasionally offers stale asks below true late-window probability. In a live bot, keep this disabled until telemetry proves stale asks actually appear.

Implementation warning:

Late-window markets can have changing tick sizes and thin liquidity. If used later, require strict order expiry, correct tick size, and private fill confirmation.

## Strategy 02: Composite Technical Score

Status:

| Field | Value |
|---|---|
| Recommendation | Tier 1 candidate |
| Train PnL | `$374,640.27` |
| Test PnL | `$141,119.07` |
| Test win rate | `55.89%` |
| Test trades | `2,675` |

Thesis:

Use a weighted score of short-window trend, recent momentum, RSI, and tick activity to choose the direction at the 2-minute mark. This is a broad directional classifier.

Inputs:

| Input | Definition |
|---|---|
| `window_delta` | `(current_price - open_price) / open_price` |
| `micro_momentum` | `(current_price - price_60s_ago) / price_60s_ago` |
| `rsi_14` | 14-period RSI on 1-second spot data |
| `volume_spike` | Last 10 seconds of active ticks compared with average prior 10-second activity |

Signal timing:

At `T+120s`.

Score logic:

```text
score = 5 * window_delta
      + 2 * micro_momentum
      + 1 * rsi_score
      + 1 * volume_score

rsi_score = +1 if RSI < 30
rsi_score = -1 if RSI > 70
rsi_score = 0 otherwise

volume_score = sign(price_now - price_10s_ago) if tick activity spike > 1.5
volume_score = 0 otherwise
```

Trade rule:

```text
if score > 0:
    buy UP
elif score < 0:
    buy DOWN
else:
    no trade
```

Order behavior:

The backtest used taker-style market buys at the current ask. Live should use FAK or FOK with slippage protection.

Exit:

Hold to settlement.

Live notes:

This traded often and survived OOS, but the edge is thinner than the best breakout strategies. Use it as a core signal, but cap size lower than Opening Range Breakout until live slippage is measured.

Main failure mode:

It can overtrade because any non-zero score triggers. Live implementation should consider a minimum absolute score threshold and a maximum ask price filter.

## Strategy 03: Oracle Lag Arbitrage

Status:

| Field | Value |
|---|---|
| Recommendation | Tier 2 research candidate |
| Train PnL | `$50,722.33` |
| Test PnL | `$8,331.82` |
| Test win rate | `100.00%` |
| Test trades | `17` |

Thesis:

If the market or oracle reacts slower than the spot feed, a sharp fast move can create a temporary mispricing. The strategy buys the side implied by the fast feed before the slow feed catches up.

Signal:

For each second from `T+2s` onward:

```text
slow_price = price_2s_ago
fast_price = current_price

if abs(fast_price / slow_price - 1) > 0.20%:
    if fast_price > slow_price:
        buy UP
    else:
        buy DOWN
```

Order behavior:

The backtest used taker-style market buys.

Exit:

Hold to settlement.

Live notes:

This had strong results but very low trade count. It should not be treated as proven. In a real Polymarket project, this needs live latency instrumentation:

| Needed measurement | Why |
|---|---|
| Spot feed timestamp | Confirms the move is truly fresh. |
| Polymarket best bid/ask timestamp | Confirms the market has not already repriced. |
| Order submit timestamp | Measures reaction time. |
| Fill timestamp | Measures whether the edge survived execution. |

Main failure mode:

The signal can disappear instantly. If the book has already moved, do not chase.

## Strategy 04: Volatility Spike Reversal

Status:

| Field | Value |
|---|---|
| Recommendation | Disabled |
| Train PnL | `-$2,508.62` |
| Test PnL | `-$257.68` |
| Test win rate | `0.00%` |
| Test trades | `24` |

Thesis:

Fade abrupt 1-second spot moves, assuming the spike mean-reverts.

Signal:

Monitor 1-second returns.

```text
if one_second_return > +0.15%:
    limit buy DOWN near post-spike mid/fair
elif one_second_return < -0.15%:
    limit buy UP near post-spike mid/fair
```

Order behavior:

The backtest used passive limit entries at the post-spike model mid.

Exit:

Exit if contract price reaches approximately `+3%` take profit or `-2%` stop loss. Otherwise hold to settlement.

Live notes:

This failed badly in both train and test. It appears the reversal premise is wrong under this market model, or the stop/take-profit structure is too tight.

Recommendation:

Do not trade. Keep only as a telemetry module to study whether real books show better fade opportunities.

## Strategy 05: Market Making and Spread Capture

Status:

| Field | Value |
|---|---|
| Recommendation | Disabled |
| Train PnL | `-$3,110.39` |
| Test PnL | `-$1,169.92` |
| Test win rate | `0.00%` |
| Test trades | `124` |

Thesis:

Continuously quote both `UP` and `DOWN`, collect spread and maker rebates, and let inventory settle.

Backtest quoting:

```text
every 5 seconds:
    UP bid = fair_up - 0.01
    UP ask = fair_up + 0.01
    DOWN bid = fair_down - 0.01
    DOWN ask = fair_down + 0.01
    size = 10 contracts
```

Exit:

Unclosed inventory is held to settlement.

Live notes:

The simplified backtest market-making model was negative. Real market making requires queue position, adverse selection controls, inventory limits, reward modeling, cancel/replace safety, and private fills. This script does not model those deeply enough.

Recommendation:

Do not deploy this as-is. If market making is needed, build it as a separate specialist system with:

| Required module | Purpose |
|---|---|
| Inventory skew | Avoid being one-sided into settlement. |
| Adverse selection detector | Stop quoting when spot is moving fast. |
| Queue/fill model | Estimate whether passive orders are likely to fill. |
| Reward model | Include actual maker/reward terms, not assumed rebates. |
| Cancel watchdog | Cancel stale quotes on disconnect or tick-size change. |

## Strategy 06: Double Moving Average Crossover

Status:

| Field | Value |
|---|---|
| Recommendation | Tier 1 candidate |
| Train PnL | `$606,939.31` |
| Test PnL | `$242,168.65` |
| Test win rate | `59.65%` |
| Test trades | `2,689` |

Thesis:

Use short-horizon trend structure near the end of the window. If the fast EMA has crossed above the slow EMA, buy `UP`. If it has crossed below, buy `DOWN`.

Inputs:

| Input | Value |
|---|---|
| Fast EMA | `15s` |
| Slow EMA | `45s` |
| Signal scan period | From start through `T-30s` |
| Cooldown | `8s` between signals |

Signal:

At `T-30s`, use the latest valid crossover observed before the trigger.

```text
if fast_ema crossed above slow_ema:
    buy UP
elif fast_ema crossed below slow_ema:
    buy DOWN
else:
    no trade
```

Order behavior:

The backtest used taker-style market buys at `T-30s`.

Exit:

Hold to settlement.

Live notes:

This was profitable with many trades, but it buys late and can pay expensive asks. Add a max ask filter before live trading.

Suggested live filters:

| Filter | Starting idea |
|---|---|
| Max ask | Do not buy above `0.92` without separate late-edge proof. |
| Min EMA separation | Require fast minus slow to exceed a small threshold. |
| Stale signal guard | Ignore crossovers older than `30s`. |

## Strategy 07: RSI Mean Reversion with Bollinger Bands

Status:

| Field | Value |
|---|---|
| Recommendation | Disabled |
| Train PnL | `-$450,716.35` |
| Test PnL | `-$134,672.08` |
| Test win rate | `0.00%` |
| Test trades | `273` |

Thesis:

Buy the opposite side when RSI and Bollinger Bands show an extreme move, expecting mean reversion.

Signal timing:

At `T+180s`.

Signal:

```text
if RSI_14 < 30 and price near/touches lower Bollinger Band:
    limit buy UP at fair_up - 0.005
elif RSI_14 > 70 and price near/touches upper Bollinger Band:
    limit buy DOWN at fair_down - 0.005
```

Order behavior:

The backtest used aggressive passive limits.

Exit:

Hold to settlement.

Live notes:

This was the worst strategy in the run. It often bought against a move that continued into settlement.

Recommendation:

Do not trade. If revisited, add trend-regime filters and require the price to actually reclaim the open before entering.

## Strategy 08: Opening Range Breakout

Status:

| Field | Value |
|---|---|
| Recommendation | Tier 1 candidate, highest priority |
| Train PnL | `$1,331,275.62` |
| Test PnL | `$564,839.31` |
| Test win rate | `72.08%` |
| Test trades | `2,636` |

Thesis:

The first 30 seconds define a local range. A confirmed break after the 2-minute mark often predicts the final 5-minute direction.

Inputs:

| Input | Definition |
|---|---|
| Opening high | Highest spot price in first `30s` |
| Opening low | Lowest spot price in first `30s` |
| Earliest breakout check | `T+120s` |
| Confirmation | Price remains beyond the range for `1s` |

Signal:

```text
if price[t] > opening_high and price[t + 1] > opening_high:
    buy UP at t + 1
elif price[t] < opening_low and price[t + 1] < opening_low:
    buy DOWN at t + 1
```

Order behavior:

The backtest used taker-style market buys.

Exit:

Hold to settlement.

Why it should be implemented first:

It had the best train and test PnL, strong test win rate, and thousands of test trades.

Live implementation notes:

This strategy needs exact window open detection. A wrong `S0` or wrong market mapping will destroy the signal. Store the first 30 seconds of spot ticks per market, then only allow one breakout entry per market.

Suggested live filters:

| Filter | Starting idea |
|---|---|
| Max ask | Reject entries above `0.95` unless testing proves late expensive buys still win. |
| Min breakout distance | Require price to exceed range by at least one spot tick or `0.005%`. |
| Book sanity | Do not enter if spread is too wide or book depth is too thin. |
| One-position rule | Only one entry per market per strategy. |

## Strategy 09: Renko-Style Price Change Sequencing

Status:

| Field | Value |
|---|---|
| Recommendation | Tier 1 candidate |
| Train PnL | `$699,109.14` |
| Test PnL | `$263,180.77` |
| Test win rate | `92.01%` |
| Test trades | `638` |

Thesis:

Count directional movement in fixed price-change bricks. If one side has clearly dominated by the 3-minute mark, follow that side.

Inputs:

| Input | Value |
|---|---|
| Brick size | `0.05%` of open price |
| Decision time | `T+180s` |
| Required edge | At least `2` more bricks in one direction |

Signal:

```text
brick = open_price * 0.0005
ref = open_price
up_bricks = 0
down_bricks = 0

for each price through T+180s:
    while price >= ref + brick:
        up_bricks += 1
        ref += brick
    while price <= ref - brick:
        down_bricks += 1
        ref -= brick

if up_bricks - down_bricks >= 2:
    buy UP
elif down_bricks - up_bricks >= 2:
    buy DOWN
```

Order behavior:

The backtest used taker-style market buys at `T+180s`.

Exit:

Hold to settlement.

Live notes:

This had a very high test win rate but fewer trades than the top high-frequency strategies. That makes it attractive as a selective signal.

Suggested live filters:

| Filter | Starting idea |
|---|---|
| Min trade count before scaling | Keep size small until live sample is large. |
| Ask cap | Avoid buying contracts already near certainty. |
| Volatility guard | Disable during extremely stale or gappy spot data. |

## Strategy 10: Volume Weighted Momentum

Status:

| Field | Value |
|---|---|
| Recommendation | Tier 1 candidate |
| Train PnL | `$768,339.58` |
| Test PnL | `$295,867.46` |
| Test win rate | `78.00%` |
| Test trades | `1,082` |

Thesis:

Use directional tick movement over the first 2 minutes. If most movement is from upticks and price is above its short moving average, buy `UP`. If most movement is from downticks and price is below its short moving average, buy `DOWN`.

Signal timing:

At `T+120s`.

Signal:

```text
up_volume = sum(all positive 1s price diffs from T+0s to T+120s)
down_volume = sum(abs(all negative 1s price diffs from T+0s to T+120s))
ratio = (up_volume - down_volume) / (up_volume + down_volume)
sma30 = mean(last 30 spot prices)

if ratio > 0.20 and current_price > sma30:
    buy UP
elif ratio < -0.20 and current_price < sma30:
    buy DOWN
```

Order behavior:

The backtest used taker-style market buys.

Exit:

Hold to settlement.

Live notes:

This is one of the strongest candidates because it is selective, survived test, and had a high test win rate. It should be implemented alongside Opening Range Breakout.

Main failure mode:

The strategy assumes 1-second price movement is meaningful. If the spot feed is forward-filled or stale, the signal quality collapses. Require live spot freshness.

## Strategy 11: Creative Late Shock Continuation

Status:

| Field | Value |
|---|---|
| Recommendation | Disabled |
| Train PnL | `$0.00` |
| Test PnL | `$0.00` |
| Test trades | `0` |

Thesis:

Look for a late 15-second spot shock and follow it if the modeled contract is not already too expensive.

Signal timing:

At `T+255s`.

Signal:

```text
recent = price_now / price_15s_ago - 1

if recent > 0.08% and 0.55 <= fair_up <= 0.90:
    buy UP
elif recent < -0.08% and 0.10 <= fair_up <= 0.45:
    buy DOWN
```

Order behavior:

The backtest used taker-style market buys.

Exit:

Hold to settlement.

Live notes:

This did not fire in the validation run. Keep disabled unless thresholds are redesigned.

## Strategy 12: Creative Wick Reclaim Fade

Status:

| Field | Value |
|---|---|
| Recommendation | Tier 2 research candidate |
| Train PnL | `$411.78` |
| Test PnL | `$1,916.53` |
| Test win rate | `64.29%` |
| Test trades | `14` |

Thesis:

If price makes an early extreme move away from the open, then reclaims back near the open by the mid-late window, fade the original extreme.

Signal timing:

At `T+200s`.

Signal:

```text
upper_extreme = open_price * 1.002
lower_extreme = open_price * 0.998
neutral_zone = open_price +/- 0.05%

if price breached upper_extreme before T+150s and now is back in neutral_zone:
    buy DOWN
elif price breached lower_extreme before T+150s and now is back in neutral_zone:
    buy UP
```

Order behavior:

The backtest used taker-style market buys.

Exit:

Hold to settlement.

Live notes:

This passed OOS, but only with `14` test trades. Treat it as a small-size research module. It may be useful as a confluence filter rather than a standalone strategy.

Potential improvement:

Require a second confirmation that the reclaim has held for `3s` to `5s`.

## Strategy 13: Creative Chop Box Contrarian

Status:

| Field | Value |
|---|---|
| Recommendation | Disabled |
| Train PnL | `-$3,554.87` |
| Test PnL | `-$4,662.78` |
| Test win rate | `36.36%` |
| Test trades | `33` |

Thesis:

If price crosses both above and below small open-relative bands, the market is choppy. Fade the latest 30-second direction.

Signal timing:

At `T+240s`.

Signal:

```text
crossed_up = any price before T+180s > open_price * 1.001
crossed_down = any price before T+180s < open_price * 0.999
last30 = price_now / price_30s_ago - 1

if crossed_up and crossed_down and last30 > 0:
    buy DOWN
elif crossed_up and crossed_down and last30 < 0:
    buy UP
```

Order behavior:

The backtest used taker-style market buys.

Exit:

Hold to settlement.

Live notes:

This failed train and test. The market may stay directional after fake chop, or the 30-second fade rule may be too naive.

Recommendation:

Do not trade.

## Strategy 14: Creative Settlement Gravity

Status:

| Field | Value |
|---|---|
| Recommendation | Disabled |
| Train PnL | `-$87,164.38` |
| Test PnL | `-$50,418.62` |
| Test win rate | `21.05%` |
| Test trades | `342` |

Thesis:

Near settlement, tiny leads around the open may be overbought. Fade the leading side if recent 10-second momentum weakens.

Signal timing:

At `T+285s`.

Signal:

```text
delta = price_now / open_price - 1
last10 = price_now / price_10s_ago - 1

if +0.005% < delta < +0.04% and last10 < 0:
    buy DOWN with half size
elif -0.04% < delta < -0.005% and last10 > 0:
    buy UP with half size
```

Order behavior:

The backtest used taker-style market buys with `500` contracts instead of `1000`.

Exit:

Hold to settlement.

Live notes:

This was strongly negative. The near-settlement side with even a tiny lead appears to retain too much advantage, or the market price is already too efficient.

Recommendation:

Do not trade.

## Live Implementation Checklist

Before live orders:

| Step | Requirement |
|---:|---|
| 1 | Build a market resolver that maps the current BTC 5-minute market to `UP` and `DOWN` token IDs. |
| 2 | Record the exact market open spot price and first 30 seconds of ticks. |
| 3 | Subscribe to the Polymarket market WebSocket for both token IDs. |
| 4 | Track best bid, best ask, spread, depth, tick size, last trade, and event timestamp. |
| 5 | Add private order tracking for submitted, filled, partially filled, cancelled, rejected, and replaced orders. |
| 6 | Implement the Tier 1 strategies in paper mode only. |
| 7 | For every signal, log the strategy, signal inputs, intended side, intended size, best ask, worst accepted price, simulated fill, and final settlement. |
| 8 | Compare paper fills to observed orderbook depth and slippage for at least several hundred windows. |
| 9 | Enable tiny-size live orders only after paper results match expectations. |
| 10 | Keep a global kill switch and daily loss cap. |

## Recommended Live Strategy Config

Starting config for a paper bot:

```json
{
  "mode": "paper",
  "symbol": "BTC",
  "windowSeconds": 300,
  "maxStrategiesPerMarket": 3,
  "maxTotalContractsPerMarket": 1000,
  "maxSingleStrategyContracts": 250,
  "minSpotFreshnessMs": 2000,
  "minOrderbookFreshnessMs": 1000,
  "maxAllowedSpread": 0.08,
  "disableInsideFinalSeconds": 5,
  "enabledStrategies": [
    "OPENING_RANGE_BREAKOUT",
    "VOLUME_WEIGHTED_MOMENTUM",
    "RENKO_CHANGE_SEQUENCING",
    "DOUBLE_MA_CROSSOVER",
    "COMPOSITE_TECHNICAL_SCORE"
  ],
  "researchStrategies": [
    "ORACLE_LAG_ARBITRAGE",
    "WICK_RECLAIM_FADE"
  ],
  "disabledStrategies": [
    "END_OF_WINDOW_MOMENTUM_SNIPER",
    "VOLATILITY_SPIKE_REVERSAL",
    "MARKET_MAKING_SPREAD_CAPTURE",
    "RSI_BOLLINGER_REVERSION",
    "LATE_SHOCK_CONTINUATION",
    "CHOP_BOX_CONTRARIAN",
    "SETTLEMENT_GRAVITY"
  ]
}
```

## Practical Build Order

| Priority | Build item | Why |
|---:|---|---|
| 1 | Market/window resolver | All strategies depend on exact market timing and token mapping. |
| 2 | Spot feed health layer | Bad spot data creates fake edge. |
| 3 | Orderbook WebSocket cache | Live price and depth must replace simulated asks. |
| 4 | Paper execution simulator | Needed before risking money. |
| 5 | Opening Range Breakout | Best OOS result and simple logic. |
| 6 | Volume Weighted Momentum | Strong OOS and selective. |
| 7 | Renko Sequencing | High win rate, fewer trades. |
| 8 | Double MA Crossover | Good broad directional coverage. |
| 9 | Composite Score | Useful as standalone or confluence. |
| 10 | Low-sample research modules | Oracle lag and wick reclaim only after the core system is stable. |

## Final Recommendation

Implement this as a portfolio of independent paper strategies, not as one merged black box.

Best first live candidates:

| Order | Strategy | Reason |
|---:|---|---|
| 1 | Opening Range Breakout | Best total OOS PnL with thousands of trades. |
| 2 | Volume Weighted Momentum | Strong OOS PnL and high test win rate. |
| 3 | Renko Change Sequencing | Selective and very high test win rate. |
| 4 | Double MA Crossover | High trade count and strong OOS survival. |
| 5 | Composite Technical Score | Lower edge but still robust. |

Use Oracle Lag Arbitrage and Wick Reclaim Fade as small-size research modules only. Keep all failed strategies disabled until they are redesigned and retested.

