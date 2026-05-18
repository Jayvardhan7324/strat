# Polymarket Live Trading Bot - Deployment Guide

## Quick Start

### 1. Configure

Edit `config/live_config.json`:
```json
{
  "mode": "paper",          // Start with paper mode
  "starting_capital": 10000,
  "stake_per_trade": 10,
  "session_soft_stop": 500,
  "session_hard_stop": 1000,
  "nuclear_stop": 2000
}
```

### 2. Run Paper Mode (Safe)

```bash
# Windows PowerShell
.\scripts\run_polymarket_bot.ps1

# Linux/macOS
python -m core.polymarket_live_trading_bot --mode paper --config config/live_config.json
```

### 3. View Logs

```bash
# Live tail
tail -f live_logs/live_trading_*.log

# Trade history
cat live_logs/trades.csv

# Session summary
cat live_logs/session_summary.json
```

### 4. Promote to Live (After Paper Validation)

```bash
# 1. Validate you've run paper mode for at least 500 trades
# 2. Update config: "mode": "live"
# 3. Add API credentials (see below)
# 4. Run with --mode live
```

## API Credentials Setup

### Polymarket Gamma API

1. Create account at https://polymarket.com
2. Navigate to Developer Settings
3. Generate API key
4. Add to environment:

```bash
export POLYMARKET_API_KEY="your_api_key_here"
export POLYMARKET_SECRET="your_secret_here"
```

### PMXT Archive (Optional)

No credentials needed for public archive URLs.

## Production Deployment

### Linux (systemd)

```bash
# Copy service file
sudo cp scripts/polymarket_bot.service /etc/systemd/system/

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable polymarket_bot
sudo systemctl start polymarket_bot

# Check status
sudo systemctl status polymarket_bot
sudo journalctl -u polymarket_bot -f
```

### Docker (Optional)

```bash
# Build
docker build -t polymarket-bot .

# Run
# docker run -d \
#   -v $(pwd)/live_logs:/app/live_logs \
#   -e POLYMARKET_MODE=paper \
#   polymarket-bot
```

## Monitoring

### Health Check

The bot logs a heartbeat every `heartbeat_interval` seconds (default: 60).

### Alerting

Set up alerts on:
- **STOP triggered** (soft/hard/nuclear)
- **Session PnL < -$1,000**
- **Max drawdown > 20%**
- **Bot crash / restart**

### Dashboard

```bash
# View real-time status
cat live_logs/session_summary.json

# Daily reports
python -c "
import json
with open('live_logs/session_summary.json') as f:
    data = json.load(f)
    print(f\"PnL: {data['session_pnl']:.2f}, WR: {data['win_rate']*100:.1f}%, Trades: {data['session_trades']}\")
"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Bot not starting | Check `config/live_config.json` exists and is valid JSON |
| No trades | Check filters are not too strict; lower thresholds in config |
| Stopped early | Check which stop triggered; review logs |
| API errors | Verify credentials; check rate limits |
| Memory high | Restart bot; check for memory leaks |

## File Structure

```
.
├── core/                              ← Core trading system
│   ├── polymarket_live_trading_bot.py    ← Main bot
│   ├── polymarket_api_client.py          ← API client
│   ├── polymarket_execution.py           ← Strategy engine
│   └── polymarket_dashboard.py           ← Web dashboard
├── config/
│   └── live_config.json                ← Config
├── scripts/
│   ├── run_polymarket_bot.ps1          ← Windows launcher
│   └── polymarket_bot.service          ← systemd service
├── tests/
│   └── test_live_bot.py               ← Test script
├── live_logs/                          ← Logs output
│   ├── live_trading_*.log
│   ├── trades.csv
│   └── session_summary.json
├── docs/
│   └── DEPLOYMENT_GUIDE.md            ← This file
└── README.md
```

## Support

For issues, check:
- `live_logs/*.log` for execution logs
- `live_logs/session_summary.json` for summary
- Polymarket API docs: https://docs.polymarket.com/
