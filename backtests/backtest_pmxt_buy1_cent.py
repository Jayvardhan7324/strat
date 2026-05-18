"""
Backtest buying real Polymarket CLOB 1 cent asks from the PMXT orderbook archive.

Data source:
- PMXT public hourly Parquet archive:
  https://archive.pmxt.dev/docs/v2-data-overview
- Polymarket Gamma event slugs for BTC 5m market metadata/resolution labels.

Rule:
- For each BTC 5m market in the requested UTC hour range, inspect both outcome
  tokens.
- If that token's real best_ask was <= --max-ask at least once during the hour,
  count one hypothetical buy-and-hold trade for that token.
- Label the trade as a win only if that token's outcome resolved to 1.
- Simulate the user's daily rule by sorting opportunities by first hit time and
  stopping each UTC day once realized PnL reaches --daily-profit-target.

This tests opportunity winrate, not guaranteed fill rate or queue priority.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import fsspec
import pandas as pd
import pyarrow.parquet as pq
import requests


PMXT_URL_TEMPLATE = "https://r2v2.pmxt.dev/polymarket_orderbook_{hour:%Y-%m-%dT%H}.parquet"
GAMMA_EVENT_URL_TEMPLATE = "https://gamma-api.polymarket.com/events/slug/{slug}"
PMXT_INDEX_URL = "https://archive.pmxt.dev/Polymarket/v2/"


@dataclass(frozen=True)
class MarketMeta:
    hour: dt.datetime
    start: dt.datetime
    slug: str
    condition: str
    tokens: list[str]
    outcomes: list[str]
    winner: str | None


def parse_utc_hour(raw: str) -> dt.datetime:
    value = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    value = value.astimezone(dt.timezone.utc)
    return value.replace(minute=0, second=0, microsecond=0)


def hour_range(start_hour: dt.datetime, hours: int) -> list[dt.datetime]:
    return [start_hour + dt.timedelta(hours=i) for i in range(hours)]


def fetch_available_pmxt_hours() -> list[dt.datetime]:
    hours: set[dt.datetime] = set()
    for page in range(1, 100):
        url = PMXT_INDEX_URL if page == 1 else f"{PMXT_INDEX_URL}?page={page}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        matches = re.findall(r"polymarket_orderbook_(\d{4}-\d{2}-\d{2}T\d{2})\.parquet", response.text)
        if not matches and page > 1:
            break
        for match in matches:
            hours.add(parse_utc_hour(match))
    return sorted(hours)


def btc_5m_slugs_for_hour(hour: dt.datetime) -> list[tuple[dt.datetime, str]]:
    return [
        (hour + dt.timedelta(minutes=5 * i), f"btc-updown-5m-{int((hour + dt.timedelta(minutes=5 * i)).timestamp())}")
        for i in range(12)
    ]


def fetch_market_meta(hour: dt.datetime, session: requests.Session) -> list[MarketMeta]:
    markets: list[MarketMeta] = []
    for start, slug in btc_5m_slugs_for_hour(hour):
        url = GAMMA_EVENT_URL_TEMPLATE.format(slug=slug)
        last_exc: Exception | None = None
        response: requests.Response | None = None
        for attempt in range(4):
            try:
                response = session.get(url, timeout=20)
                break
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(0.75 * (attempt + 1))
        if response is None:
            raise RuntimeError(f"Gamma fetch failed for {slug}: {last_exc}")
        if response.status_code != 200:
            continue
        event = response.json()
        event_markets = event.get("markets") or []
        if not event_markets:
            continue
        market = event_markets[0]
        tokens = json.loads(market["clobTokenIds"])
        outcomes = json.loads(market["outcomes"])
        prices = json.loads(market["outcomePrices"])
        winner = outcomes[prices.index("1")] if "1" in prices else None
        if winner is None:
            continue
        markets.append(
            MarketMeta(
                hour=hour,
                start=start,
                slug=slug,
                condition=market["conditionId"],
                tokens=tokens,
                outcomes=outcomes,
                winner=winner,
            )
        )
    return markets


def decimal_min(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(min(series))


def backtest_hour(hour: dt.datetime, markets: list[MarketMeta], max_ask: Decimal) -> pd.DataFrame:
    if not markets:
        return pd.DataFrame()

    url = PMXT_URL_TEMPLATE.format(hour=hour)
    filters = [("market", "in", [market.condition.encode("ascii") for market in markets])]
    with fsspec.open(url, "rb") as file_obj:
        table = pq.read_table(
            file_obj,
            columns=["timestamp_received", "market", "asset_id", "event_type", "best_ask"],
            filters=filters,
        )
    if table.num_rows == 0:
        return pd.DataFrame()

    df = table.to_pandas()
    rows: list[dict] = []
    for market in markets:
        for token, outcome in zip(market.tokens, market.outcomes):
            token_rows = df[df["asset_id"] == token]
            quote_rows = token_rows[token_rows["best_ask"].notna()].copy()
            hit_rows = quote_rows[quote_rows["best_ask"] <= max_ask]
            if hit_rows.empty:
                continue

            first_hit = hit_rows.sort_values("timestamp_received").iloc[0]
            entry_price = float(first_hit["best_ask"])
            rows.append(
                {
                    "hour": market.hour.isoformat(),
                    "market_start": market.start.isoformat(),
                    "slug": market.slug,
                    "condition": market.condition,
                    "token": token,
                    "side": outcome,
                    "winner": market.winner,
                    "won": outcome == market.winner,
                    "first_hit_ts": first_hit["timestamp_received"].isoformat(),
                    "entry_price": entry_price,
                    "min_best_ask": decimal_min(quote_rows["best_ask"]),
                    "hit_rows": len(hit_rows),
                    "quote_rows": len(quote_rows),
                }
            )
    return pd.DataFrame(rows)


def process_hour(hour: dt.datetime, max_ask_raw: str) -> tuple[dt.datetime, int, pd.DataFrame, str | None]:
    session = requests.Session()
    max_ask = Decimal(max_ask_raw)
    try:
        markets = fetch_market_meta(hour, session)
    except Exception as exc:
        return hour, 0, pd.DataFrame(), f"metadata failed: {type(exc).__name__}: {exc}"
    if not markets:
        return hour, 0, pd.DataFrame(), None
    try:
        trades = backtest_hour(hour, markets, max_ask)
    except FileNotFoundError:
        return hour, len(markets), pd.DataFrame(), "PMXT file missing"
    except Exception as exc:
        return hour, len(markets), pd.DataFrame(), f"{type(exc).__name__}: {exc}"
    return hour, len(markets), trades, None


def trade_pnl(row: pd.Series, stake_usd: float, fee_rate: float) -> float:
    entry_price = float(row["entry_price"])
    entry_fee = stake_usd * fee_rate
    if bool(row["won"]):
        shares = stake_usd / entry_price
        return shares - stake_usd - entry_fee
    return -stake_usd - entry_fee


def simulate_daily_profit_stop(
    trades: pd.DataFrame,
    stake_usd: float,
    fee_rate: float,
    daily_profit_target: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return trades.copy(), pd.DataFrame()

    sim = trades.copy()
    sim["first_hit_ts"] = pd.to_datetime(sim["first_hit_ts"], utc=True)
    sim["date_utc"] = sim["first_hit_ts"].dt.strftime("%Y-%m-%d")
    sim["pnl"] = sim.apply(lambda row: trade_pnl(row, stake_usd, fee_rate), axis=1)
    sim = sim.sort_values(["first_hit_ts", "slug", "side"]).reset_index(drop=True)

    taken_rows: list[pd.DataFrame] = []
    daily_rows: list[dict] = []
    for date_utc, group in sim.groupby("date_utc", sort=True):
        day_pnl = 0.0
        day_taken: list[pd.Series] = []
        stopped_at_target = False
        for _, row in group.iterrows():
            if day_pnl >= daily_profit_target:
                stopped_at_target = True
                break
            day_taken.append(row)
            day_pnl += float(row["pnl"])
        if day_taken:
            day_df = pd.DataFrame(day_taken)
            taken_rows.append(day_df)
            wins = int(day_df["won"].sum())
            daily_rows.append(
                {
                    "date_utc": date_utc,
                    "opportunities_seen": len(group),
                    "trades_taken": len(day_df),
                    "wins": wins,
                    "losses": len(day_df) - wins,
                    "win_rate": wins / len(day_df),
                    "pnl": day_pnl,
                    "stopped_at_target": stopped_at_target or day_pnl >= daily_profit_target,
                }
            )
        else:
            daily_rows.append(
                {
                    "date_utc": date_utc,
                    "opportunities_seen": len(group),
                    "trades_taken": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate": 0.0,
                    "pnl": 0.0,
                    "stopped_at_target": False,
                }
            )

    taken = pd.concat(taken_rows, ignore_index=True) if taken_rows else pd.DataFrame()
    daily = pd.DataFrame(daily_rows)
    return taken, daily


def summarize(
    trades: pd.DataFrame,
    taken: pd.DataFrame,
    daily: pd.DataFrame,
    markets_seen: int,
    max_ask: Decimal,
    stake_usd: float,
    daily_profit_target: float,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            [
                {
                    "max_ask": float(max_ask),
                    "stake_usd": stake_usd,
                    "daily_profit_target": daily_profit_target,
                    "markets_seen": markets_seen,
                    "opportunities": 0,
                    "trades_taken": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "profitable_days": 0,
                    "losing_days": 0,
                    "break_even_win_rate_no_fee": float(max_ask),
                    "profitable_vs_no_fee_break_even": False,
                }
            ]
        )
    wins = int(taken["won"].sum()) if not taken.empty else 0
    trades_taken = len(taken)
    raw_wins = int(trades["won"].sum())
    raw_opportunities = len(trades)
    total_pnl = float(daily["pnl"].sum()) if not daily.empty else 0.0
    return pd.DataFrame(
        [
            {
                "max_ask": float(max_ask),
                "stake_usd": stake_usd,
                "daily_profit_target": daily_profit_target,
                "markets_seen": markets_seen,
                "opportunities": raw_opportunities,
                "raw_wins": raw_wins,
                "raw_win_rate": raw_wins / raw_opportunities if raw_opportunities else 0.0,
                "trades_taken": trades_taken,
                "wins": wins,
                "losses": trades_taken - wins,
                "win_rate": wins / trades_taken if trades_taken else 0.0,
                "total_pnl": total_pnl,
                "profitable_days": int((daily["pnl"] > 0).sum()) if not daily.empty else 0,
                "losing_days": int((daily["pnl"] < 0).sum()) if not daily.empty else 0,
                "days_hit_target": int(daily["stopped_at_target"].sum()) if not daily.empty else 0,
                "break_even_win_rate_no_fee": float(max_ask),
                "profitable_vs_no_fee_break_even": (wins / trades_taken > float(max_ask)) if trades_taken else False,
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-hour", required=True, help="UTC hour, e.g. 2026-05-14T16:00:00Z")
    parser.add_argument("--hours", type=int, default=1)
    parser.add_argument("--available-pmxt", action="store_true", help="Use every hourly file listed in the PMXT v2 index.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-ask", default="0.01")
    parser.add_argument("--stake-usd", type=float, default=1.0)
    parser.add_argument("--fee-rate", type=float, default=0.002)
    parser.add_argument("--daily-profit-target", type=float, default=20.0)
    parser.add_argument("--out-prefix", default="pmxt_buy1_cent")
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()

    start_hour = parse_utc_hour(args.start_hour)
    max_ask = Decimal(args.max_ask)
    hours = fetch_available_pmxt_hours() if args.available_pmxt else hour_range(start_hour, args.hours)
    if args.available_pmxt:
        hours = [hour for hour in hours if hour >= start_hour]
    print(f"Hours to process: {len(hours)}")
    if hours:
        print(f"Hour range: {hours[0]:%Y-%m-%dT%H}:00Z -> {hours[-1]:%Y-%m-%dT%H}:00Z")

    all_trades: list[pd.DataFrame] = []
    markets_seen = 0
    failed_rows: list[dict] = []

    if args.workers <= 1:
        for hour in hours:
            print(f"Processing {hour:%Y-%m-%dT%H}:00Z")
            processed_hour, market_count, trades, error = process_hour(hour, args.max_ask)
            markets_seen += market_count
            if error:
                print(f"  skipped: {error}")
                failed_rows.append({"hour": processed_hour.isoformat(), "markets_seen": market_count, "error": error})
                continue
            print(f"  BTC 5m resolved markets: {market_count}")
            print(f"  1c opportunities: {len(trades)} | wins: {int(trades['won'].sum()) if not trades.empty else 0}")
            if not trades.empty:
                all_trades.append(trades)
            if args.checkpoint_dir:
                checkpoint_dir = Path(args.checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                safe_hour = processed_hour.strftime("%Y-%m-%dT%H")
                trades.to_csv(checkpoint_dir / f"{safe_hour}_trades.csv", index=False)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_hour = {
                executor.submit(process_hour, hour, args.max_ask): hour
                for hour in hours
            }
            completed = 0
            for future in as_completed(future_to_hour):
                hour, market_count, trades, error = future.result()
                completed += 1
                markets_seen += market_count
                if error:
                    print(f"[{completed}/{len(hours)}] {hour:%Y-%m-%dT%H}: skipped: {error}")
                    failed_rows.append({"hour": hour.isoformat(), "markets_seen": market_count, "error": error})
                    continue
                wins = int(trades["won"].sum()) if not trades.empty else 0
                print(f"[{completed}/{len(hours)}] {hour:%Y-%m-%dT%H}: markets={market_count} opportunities={len(trades)} wins={wins}")
                if not trades.empty:
                    all_trades.append(trades)
                if args.checkpoint_dir:
                    checkpoint_dir = Path(args.checkpoint_dir)
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    safe_hour = hour.strftime("%Y-%m-%dT%H")
                    trades.to_csv(checkpoint_dir / f"{safe_hour}_trades.csv", index=False)

    trade_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    taken_df, daily_df = simulate_daily_profit_stop(
        trade_df,
        args.stake_usd,
        args.fee_rate,
        args.daily_profit_target,
    )
    metrics = summarize(
        trade_df,
        taken_df,
        daily_df,
        markets_seen,
        max_ask,
        args.stake_usd,
        args.daily_profit_target,
    )
    trade_path = f"{args.out_prefix}_trades.csv"
    taken_path = f"{args.out_prefix}_taken_trades.csv"
    daily_path = f"{args.out_prefix}_daily.csv"
    failed_path = f"{args.out_prefix}_failed_hours.csv"
    metrics_path = f"{args.out_prefix}_metrics.csv"
    trade_df.to_csv(trade_path, index=False)
    taken_df.to_csv(taken_path, index=False)
    daily_df.to_csv(daily_path, index=False)
    pd.DataFrame(failed_rows).to_csv(failed_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    print(metrics.to_string(index=False))
    print(f"Saved {metrics_path}, {daily_path}, {taken_path}, {trade_path}, and {failed_path}")


if __name__ == "__main__":
    main()
