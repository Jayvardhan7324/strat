import urllib.request
import json

# Test the dashboard API
try:
    print("Testing dashboard...")

    # Test API endpoint
    response = urllib.request.urlopen('http://127.0.0.1:5000/api/data')
    data = json.loads(response.read())
    print("Dashboard API - OK!")
    print("Keys returned:", list(data.keys()))

    if data.get('session'):
        print("Session PnL: $", data['session'].get('session_pnl', 0))
        print("Total Trades:", data['session'].get('total_trades', 0))
        print("Win Rate:", data['session'].get('win_rate', 0)*100, "%")

    print("Trades loaded:", len(data.get('trades', [])))
    print("Signals loaded:", len(data.get('signals', [])))

    print("Dashboard is running successfully!")
    print("Open your browser to: http://127.0.0.1:5000/")

except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()
