from core.polymarket_dashboard import DashboardData

config = {
    "log_dir": "./live_logs_demo",
    "max_trades_display": 100,
    "max_signals_display": 50,
}

data = DashboardData(config)
result = data.refresh()

print("Session data keys:", list(result.get('session', {}).keys()))
print("Session PnL:", result.get('session', {}).get('session_pnl', 'N/A'))
print("Trades count:", len(result.get('trades', [])))
print("Signals count:", len(result.get('signals', [])))
print("Equity count:", len(result.get('equity', [])))

if result.get('trades'):
    print("First trade:", result['trades'][0])
else:
    print("No trades loaded")
