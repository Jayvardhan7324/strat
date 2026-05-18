"""
Backtesting Framework for Sequential YES/NO Hedging Strategy

Runs the strategy against multiple realistic market scenarios and provides
detailed performance statistics.
"""

import random
import math
from strategies.sequential_hedge_strategy import SequentialHedgeStrategy


def generate_price_scenarios():
    """Generate realistic market scenarios for backtesting"""
    
    scenarios = {}
    
    # Scenario 1: Gradual YES increase (bullish)
    prices_1 = [0.30 + (i * 0.02) for i in range(35)]
    scenarios["Bullish Run"] = [min(p, 0.95) for p in prices_1]
    
    # Scenario 2: Gradual YES decrease (bearish)
    prices_2 = [0.70 - (i * 0.02) for i in range(35)]
    scenarios["Bearish Drop"] = [max(p, 0.05) for p in prices_2]
    
    # Scenario 3: Oscillating prices (volatile)
    scenarios["Oscillating"] = [
        0.30 if i % 4 == 0 else
        0.70 if i % 4 == 1 else
        0.35 if i % 4 == 2 else
        0.65
        for i in range(40)
    ]
    
    # Scenario 4: Sharp spike then crash
    scenarios["Spike & Crash"] = (
        [0.30] * 5 +
        [0.30 + (i * 0.10) for i in range(6)] +
        [0.90 - (i * 0.10) for i in range(8)]
    )
    
    # Scenario 5: Slow grind up
    scenarios["Slow Grind Up"] = [0.25 + (i * 0.015) for i in range(50)]
    
    # Scenario 6: Slow grind down
    scenarios["Slow Grind Down"] = [0.75 - (i * 0.015) for i in range(50)]
    
    # Scenario 7: Dead flat (no movement)
    scenarios["Flat Market"] = [0.50] * 30
    
    # Scenario 8: Random walk starting from 0.30
    random.seed(42)
    prices_8 = [0.30]
    for i in range(49):
        change = random.uniform(-0.05, 0.05)
        new_price = max(0.05, min(0.95, prices_8[-1] + change))
        prices_8.append(new_price)
    scenarios["Random Walk 1"] = prices_8
    
    # Scenario 9: Random walk starting from 0.50
    random.seed(123)
    prices_9 = [0.50]
    for i in range(49):
        change = random.uniform(-0.05, 0.05)
        new_price = max(0.05, min(0.95, prices_9[-1] + change))
        prices_9.append(new_price)
    scenarios["Random Walk 2"] = prices_9
    
    # Scenario 10: Quick pump and dump
    scenarios["Pump & Dump"] = (
        [0.30] * 3 +
        [0.35, 0.45, 0.55, 0.65, 0.75, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20]
    )
    
    # Scenario 11: V-shape recovery
    scenarios["V-Shape"] = (
        [0.50 - (i * 0.05) for i in range(8)] +
        [0.10 + (i * 0.05) for i in range(12)]
    )
    
    # Scenario 12: Inverted V
    scenarios["Inverted V"] = (
        [0.30 + (i * 0.05) for i in range(8)] +
        [0.70 - (i * 0.05) for i in range(12)]
    )
    
    return scenarios


def run_backtest():
    """Run comprehensive backtest across all scenarios"""
    
    scenarios = generate_price_scenarios()
    
    # Strategy parameters
    params = {
        "initial_capital": 1000.0,
        "yes_entry_threshold": 0.35,
        "no_entry_threshold": 0.35,
        "profit_target": 0.15,
        "max_position_size": 100.0,
    }
    
    print("="*80)
    print("BACKTESTING SEQUENTIAL HEDGE STRATEGY")
    print("="*80)
    print(f"\nParameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print(f"\nRunning {len(scenarios)} scenarios...\n")
    
    results = []
    
    for name, prices in scenarios.items():
        print(f"\n{'='*80}")
        print(f"SCENARIO: {name}")
        print(f"Price range: ${min(prices):.2f} - ${max(prices):.2f}")
        print(f"{'='*80}")
        
        strategy = SequentialHedgeStrategy(**params)
        result = strategy.execute_strategy(prices)
        result["scenario"] = name
        result["price_range"] = f"${min(prices):.2f} - ${max(prices):.2f}"
        results.append(result)
    
    # Print summary table
    print("\n" + "="*80)
    print("BACKTEST SUMMARY")
    print("="*80)
    
    print(f"\n{'Scenario':<20} {'Hedged':<8} {'Profit':<10} {'ROI':<10} {'Capital':<10}")
    print("-"*80)
    
    total_profit = 0
    hedged_count = 0
    
    for r in results:
        status = "YES" if r["is_hedged"] else "NO"
        profit = r["locked_profit"]
        roi = r["roi"]
        capital = r["remaining_capital"]
        
        print(f"{r['scenario']:<20} {status:<8} ${profit:<9.2f} {roi:<9.1f}% ${capital:<9.2f}")
        
        if r["is_hedged"]:
            total_profit += profit
            hedged_count += 1
    
    print("-"*80)
    print(f"\nTotal Scenarios: {len(results)}")
    print(f"Successfully Hedged: {hedged_count}/{len(results)} ({hedged_count/len(results)*100:.1f}%)")
    print(f"Total Profit from Hedged Positions: ${total_profit:.2f}")
    print(f"Average Profit per Hedge: ${total_profit/hedged_count:.2f}" if hedged_count > 0 else "")
    
    # Detailed analysis
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    profitable = [r for r in results if r["is_hedged"]]
    unprofitable = [r for r in results if not r["is_hedged"]]
    
    print(f"\nWinning Scenarios: {len(profitable)}")
    for r in profitable:
        print(f"  [WIN] {r['scenario']}: ${r['locked_profit']:.2f} profit ({r['roi']:.1f}% ROI)")
    
    print(f"\nUnhedged Scenarios: {len(unprofitable)}")
    for r in unprofitable:
        print(f"  [SKIP] {r['scenario']}: No hedge opportunity found")
    
    # Risk metrics
    if profitable:
        profits = [r["locked_profit"] for r in profitable]
        rois = [r["roi"] for r in profitable]
        
        print(f"\nRisk Metrics:")
        print(f"  Best Trade: ${max(profits):.2f} ({max(rois):.1f}% ROI)")
        print(f"  Worst Trade: ${min(profits):.2f} ({min(rois):.1f}% ROI)")
        print(f"  Average Trade: ${sum(profits)/len(profits):.2f} ({sum(rois)/len(rois):.1f}% ROI)")
        print(f"  Profit Std Dev: ${(sum((p - sum(profits)/len(profits))**2 for p in profits) / len(profits))**0.5:.2f}")


if __name__ == "__main__":
    run_backtest()
