"""
Test script to generate sample data for the dashboard and verify components.
"""

from core.polymarket_logging import TradeLogger, TradeEntry, SignalLog
from core.polymarket_execution import SessionRiskManager, MultiFilterStrategy
import random
import time

# Create logger
logger = TradeLogger(log_dir="./live_logs_demo", level="INFO")

# Simulate session
risk = SessionRiskManager({
    "starting_capital": 10000.0,
    "session_soft_stop": 500.0,
    "session_hard_stop": 1000.0,
    "nuclear_stop": 2000.0,
    "max_trades_per_session": 200,
    "max_drawdown_pct": 0.20,
})

capital = 10000.0
for i in range(50):
    # Simulate trade     
    pnl = random.uniform(-15, 25)
    capital += pnl
    risk.update(pnl)
    
    trade = TradeEntry(
        trade_id=f"T{i:04d}",
        position_id=f"P{i:04d}",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        action="BUY",
        side=random.choice(["UP", "DOWN"]),
        condition_id="btcusdt-001",
        token_id="tok-" + str(i % 10),
        price=random.uniform(0.85, 0.99),
        size=10.0,
        fee=0.02,
        pnl=pnl,
        pnl_pct=(pnl / 10.0) * 100,
        exit_type=random.choice(["target_hit", "stop_loss", "time_exit", "settlement"]),
    )
    logger.log_trade(trade)
    logger.log_equity_update(capital)
    
    # Log some signals    
    if i % 3 == 0:
        signal = SignalLog(
            signal_id=f"S{i:04d}",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            strategy="multi_filter",
            condition_id="btcusdt-001",
            side=random.choice(["UP", "DOWN"]),
            confidence=round(random.uniform(0.6, 0.95), 2),
            price=random.uniform(0.85, 0.99),
            filters_passed=random.randint(3, 5),
            filters_total=5,
            executed=random.choice([True, False]),
        )
        logger.log_signal(signal)

# Print summary
report = logger.generate_trade_report()
print("=" * 60)
print("  TRADING SYSTEM TEST REPORT")
print("=" * 60)
for key, value in report.items():
    print(f"  {key}: {value}")
print("=" * 60)

logger.close()
print("\nDemo data generated successfully. Start the dashboard with:")
print("  python polymarket_dashboard.py --log-dir ./live_logs_demo")
