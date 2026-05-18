import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

from core.polymarket_dashboard import DashboardData, create_app

config = {
    "host": "127.0.0.1",
    "port": 5000,
    "debug": False,
    "log_dir": "./live_logs_demo",
    "max_trades_display": 100,
    "max_signals_display": 50,
    "refresh_interval": 5,
}

print("Testing DashboardData directly...")
data = DashboardData(config)
result = data.refresh()

print(f"Log dir: {data.log_dir}")
print(f"Trades loaded: {len(result.get('trades', []))}")
print(f"Signals loaded: {len(result.get('signals', []))}")
print(f"Session PnL: {result.get('session', {}).get('session_pnl', 0)}")

print("\nTesting create_app...")
app, socketio, app_data = create_app(config)

print(f"App data log_dir: {app_data.log_dir}")
print(f"App data trades: {len(app_data.trades)}")

# Test the Flask app directly
with app.test_client() as client:
    response = client.get('/api/data')
    api_data = response.get_json()
    print(f"\nAPI response trades: {len(api_data.get('trades', []))}")
    print(f"API response signals: {len(api_data.get('signals', []))}")
