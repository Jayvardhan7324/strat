"""
Profit Calculator: What would you have if you gave each strategy $10?

For each strategy, we calculate:
1. $10 staked per trade (flat)
2. Kelly-sized positions (optimal bankroll growth)
3. Quarter-Kelly (conservative)

Then project to 1 year of trading (365 days * 288 windows/day for 5-min markets)
"""

import pandas as pd
import numpy as np

# Strategy data from backtests
trades_per_year = 365 * 24 * 12  # 5-minute windows, 365 days

strategies = {
    "buy97_sell99": {
        "avg_pnl": 7.45,
        "wr": 0.894,
        "avg_loss": -10.02,
        "avg_win": 5.00,
        "kelly": 0.6827,
        "max_streak": 4,
        "recommended_size": 0.10,  # 10% of bankroll (1/10th Kelly)
    },
    "prev10_momentum": {
        "avg_pnl": 0.46,
        "wr": 0.529,
        "avg_loss": -10.02,
        "avg_win": 9.78,
        "kelly": 0.0472,
        "max_streak": 9,
        "recommended_size": 0.0236,  # half Kelly
    },
    "buy1_cent": {
        "avg_pnl": 112.58,
        "wr": 0.123,
        "avg_loss": -10.02,
        "avg_win": 989.98,
        "kelly": 0.1137,
        "max_streak": 1202,
        "recommended_size": 0.01,  # tiny (streak risk)
    },
    "chop_direction_predictor": {
        "avg_pnl": 402.22,
        "wr": 0.712,
        "avg_loss": -1002.00,
        "avg_win": 971.25,
        "kelly": 0.4141,
        "max_streak": 6,
        "recommended_size": 0.10,  # 10% Kelly (high risk)
    },
    "chop_scalper_v1": {
        "avg_pnl": -22.99,
        "wr": 0.087,
        "avg_loss": -26.49,
        "avg_win": 13.73,
        "kelly": 0.0,
        "max_streak": 102,
        "recommended_size": 0.0,  # BROKEN
    },
    "chop_scalper_v2": {
        "avg_pnl": -19.37,
        "wr": 0.087,
        "avg_loss": -22.71,
        "avg_win": 15.61,
        "kelly": 0.0,
        "max_streak": 99,
        "recommended_size": 0.0,  # BROKEN
    },
}

print("=" * 80)
print("PROFIT CALCULATOR: What would you have with $10 per trade?")
print("=" * 80)
print(f"\nAssumptions based on TEST data:")
print(f"  - $10 staked per trade (flat)")
print(f"  - Fee: 0.2% taker")
print(f"  - {trades_per_year:,} trades per year (5-min windows, 24/7)")
print(f"  - No compounding for flat $10 sizing")

print(f"\n{'='*80}")
print("FLAT $10 PER TRADE (No compounding, same $10 every time)")
print(f"{'='*80}")
print(f"{'Strategy':<25} {'EV/Trade':>10} {'After 1 Trade':>14} {'After 100':>12} {'After 1000':>12} {'1 Year':>12} {'Status':>10}")
print("-" * 95)

results = []

for name, s in strategies.items():
    if s["avg_pnl"] <= 0:
        print(f"{name:<25} {'N/A':>10} {'BROKEN':>14} {'DO NOT':>12} {'TRADE':>12} {'NEGATIVE EV':>12} {'NO':>10}")
        continue
    
    ev = s["avg_pnl"]
    after_1 = ev
    after_100 = 100 * ev
    after_1000 = 1000 * ev
    after_year = trades_per_year * ev
    
    status = "PROFIT" if ev > 5 else "THIN"
    
    print(f"{name:<25} ${ev:>8.2f} ${after_1:>12.2f} ${after_100:>10.2f} ${after_1000:>10.0f} ${after_year:>10,.0f} {status:>10}")
    results.append({
        "strategy": name, "ev": ev, "1_trade": after_1, "100_trades": after_100,
        "1000_trades": after_1000, "1_year": after_year
    })

print(f"\n{'='*80}")
print("OPTIMAL KELLY SIZING (Compounding, optimal bankroll growth)")
print(f"{'='*80}")
print(f"{'Strategy':<25} {'Kelly %':>10} {'After 1 Trade':>14} {'After 100':>12} {'After 1000':>12} {'1 Year':>12}")
print("-" * 95)

for name, s in strategies.items():
    if s["avg_pnl"] <= 0:
        continue
    
    kelly = s["kelly"]
    bankroll = 10000.0  # Start with $10k
    
    # Compounded for 1 trade
    bankroll_1 = bankroll * (1 + s["avg_pnl"] / bankroll * kelly)
    
    # Compounded for 100 trades
    for _ in range(100):
        bankroll *= (1 + s["avg_pnl"] / bankroll * kelly)
    bankroll_100 = bankroll
    
    # Reset
    bankroll = 10000.0
    
    # Compounded for 1000 trades
    for _ in range(1000):
        bankroll *= (1 + s["avg_pnl"] / bankroll * kelly)
    bankroll_1000 = bankroll
    
    # Compounded for 1 year
    bankroll = 10000.0
    for _ in range(trades_per_year):
        bankroll *= (1 + s["avg_pnl"] / bankroll * kelly)
    bankroll_year = bankroll
    
    print(f"{name:<25} {kelly*100:>8.2f}% ${bankroll_1:>12.2f} ${bankroll_100:>10,.0f} ${bankroll_1000:>10,.0f} ${bankroll_year:>10,.0f}")

print(f"\n{'='*80}")
print("CONSERVATIVE KELLY (1/10th Kelly, safe but slow)")
print(f"{'='*80}")
print(f"{'Strategy':<25} {'10% Kelly':>10} {'After 1 Trade':>14} {'After 100':>12} {'After 1000':>12} {'1 Year':>12}")
print("-" * 95)

for name, s in strategies.items():
    if s["avg_pnl"] <= 0:
        continue
    
    kelly = s["kelly"] * 0.10  # 1/10th Kelly
    bankroll = 10000.0
    
    # 1 trade
    bankroll_1 = bankroll * (1 + s["avg_pnl"] / bankroll * kelly)
    
    # 100 trades
    for _ in range(100):
        bankroll *= (1 + s["avg_pnl"] / bankroll * kelly)
    bankroll_100 = bankroll
    
    # Reset
    bankroll = 10000.0
    
    # 1000 trades
    for _ in range(1000):
        bankroll *= (1 + s["avg_pnl"] / bankroll * kelly)
    bankroll_1000 = bankroll
    
    # 1 year
    bankroll = 10000.0
    for _ in range(trades_per_year):
        bankroll *= (1 + s["avg_pnl"] / bankroll * kelly)
    bankroll_year = bankroll
    
    print(f"{name:<25} {kelly*100:>8.2f}% ${bankroll_1:>12.2f} ${bankroll_100:>10,.0f} ${bankroll_1000:>10,.0f} ${bankroll_year:>10,.0f}")

print(f"\n{'='*80}")
print("REALISTIC SCENARIO: With session stops and variance")
print(f"{'='*80}")
print("Assuming you hit a session soft stop ($500) on bad days:")
print("  - 20% of days end at -$500 (hit stop)")
print("  - 80% of days end at average")
print("  - Flat $10 per trade, no compounding")
print()

for name, s in strategies.items():
    if s["avg_pnl"] <= 0:
        continue
    
    daily_avg = trades_per_year / 365 * s["avg_pnl"]  # avg daily trades
    daily_80 = daily_avg * 0.8
    daily_20 = -500.0
    
    daily_ev = daily_80 + daily_20
    yearly = daily_ev * 365
    
    print(f"  {name:<25}: ${daily_ev:>8.2f}/day = ${yearly:>12,.0f}/year")

print(f"\n{'='*80}")
print("BOTTOM LINE: With $10/stake, after 1 year:")
print(f"{'='*80}")
for r in results:
    name = r["strategy"]
    yearly = r["1_year"]
    status = "PROFIT" if yearly > 100_000 else "MODEST" if yearly > 50_000 else "THIN"
    print(f"  {name:<25}: ${yearly:>12,.0f}/year  {status}")
