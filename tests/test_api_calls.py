import urllib.request
import json

# Call the API twice to see if data changes
for i in range(2):
    response = urllib.request.urlopen('http://127.0.0.1:5000/api/data')
    data = json.loads(response.read())
    trades_count = len(data.get('trades', []))
    signals_count = len(data.get('signals', []))
    print(f'Call {i+1} - Trades: {trades_count}, Signals: {signals_count}')
