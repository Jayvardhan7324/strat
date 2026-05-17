# Polymarket Live Trading System

Production-ready trading system for Polymarket crypto up/down markets with real-time monitoring dashboard.

## Features

- **Real-time API Integration**: Connects to Polymarket Gamma API and CLOB for live order books
- **Multi-Filter Strategy**: 5 technical filters (momentum, RSI, volatility, time, acceleration) requiring 4/5 to pass
- **Risk Management**: Automatic session soft/hard/nuclear stops with configurable limits
- **Paper & Live Modes**: Safe testing before live deployment
- **Comprehensive Logging**: Full trade journaling, signal tracking, and post-trade analysis
- **Real-time Dashboard**: Web interface with live P&L charts, metrics, and trade history
- **Session Stops**: Soft stop ($500), hard stop ($1000), nuclear stop ($2000) with max drawdown protection

## Project Structure

```
strats/
├── polymarket_api_client.py      # Polymarket API client (Gamma + CLOB)
├── polymarket_logging.py         # Trade journaling and analytics
├── polymarket_execution.py       # Strategy engine and execution
├── polymarket_dashboard.py       # Web monitoring dashboard
├── polymarket_trading_system.py  # Main system orchestrator
├── live_config.json              # Configuration file
├── test_system.py                # System test script
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.8+
- Polymarket account (for live trading)

### Install Dependencies

```bash
pip install flask flask-socketio plotly pandas numpy requests
```

## Quick Start

### Paper Trading Mode (Recommended First)

```bash
# Start with dashboard
python polymarket_trading_system.py --mode paper --dashboard

# Or without dashboard
python polymarket_trading_system.py --mode paper
```

### Live Trading Mode

```bash
# Start live trading with dashboard
python polymarket_trading_system.py --mode live --dashboard --capital 10000
```

### Dashboard Only

```bash
# Start dashboard with existing log data
python polymarket_dashboard.py --log-dir ./live_logs --host 127.0.0.1 --port 5000
```

## Configuration

Edit `live_config.json` or use CLI arguments:

```json
{
  "mode": "paper",
  "starting_capital": 10000.0,
  "stake_per_trade": 10.0,
  "session_soft_stop": 500.0,
  "session_hard_stop": 1000.0,
  "nuclear_stop": 2000.0,
  "max_trades_per_session": 200,
  "log_dir": "./live_logs",
  "log_level": "INFO"
}
```

### CLI Options

```bash
python polymarket_trading_system.py --help

Options:
  --mode          Trading mode: paper or live (default: paper)
  --config        Configuration file path
  --capital       Starting capital in USD (default: $10,000)
  --stake         Stake per trade in USD (default: $10)
  --soft-stop     Soft stop loss limit (default: $500)
  --hard-stop     Hard stop loss limit (default: $1,000)
  --nuclear-stop  Nuclear stop/strategy abort (default: $2,000)
  --max-trades    Maximum trades per session (default: 200)
  --dashboard     Start monitoring dashboard
  --dashboard-port Dashboard port (default: 5000)
  --run-time      Run time in seconds (default: indefinite)
  --log-dir       Log directory (default: ./live_logs)
```

## Dashboard

Access the dashboard at: **http://127.0.0.1:5000/**

### Dashboard Features

- **Session P&L**: Real-time profit/loss tracking
- **Win Rate**: Trade success percentage
- **Max Drawdown**: Largest peak-to-trough decline
- **Current Capital**: Running account balance
- **Profit Factor**: Gross profit / gross loss ratio
- **Risk Overview**: Soft/hard/nuclear stop levels
- **Equity Curve**: Visual P&L chart over time
- **Trade Distribution**: Histogram of trade outcomes
- **Recent Trades**: Table of latest trades with P&L
- **Recent Signals**: Strategy signals with confidence scores

## Strategy Details

### Multi-Filter Strategy

The system uses 5 technical filters to identify trading opportunities:

1. **Momentum Filter**: Requires minimum price delta (default: 0.2%)
2. **RSI Filter**: RSI must be between 15-85 (avoid overbought/oversold)
3. **Volatility Filter**: Volatility ratio must be below threshold (default: 3.0)
4. **Time Filter**: Must be past 50% of the 5-minute window
5. **Acceleration Filter**: Price acceleration must confirm direction

**Entry Criteria**: At least 4 out of 5 filters must pass with minimum confidence of 0.7

### Risk Management

| Stop Type | Default | Description |
|-----------|---------|-------------|
| Soft Stop | -$500 | Warning level, consider reducing position size |
| Hard Stop | -$1,000 | Session halt, review strategy |
| Nuclear Stop | -$2,000 | Emergency halt, strategy abort |
| Max Drawdown | 20% | Percentage-based circuit breaker |
| Max Trades | 200 | Session trade limit |

## Logging

All trading activity is logged to `./live_logs/`:

- `session_*.log` - Main session log
- `session_*_trades.csv` - Trade journal
- `session_*_signals.csv` - Strategy signals
- `session_*_equity.csv` - Equity curve data
- `session_*_analysis.json` - Post-trade analysis report

### Post-Trade Analysis

The system generates comprehensive reports including:

- Total P&L and average per trade
- Win rate and profit factor
- Maximum drawdown
- Hourly breakdown
- Market breakdown
- Exit type distribution

## Safety Warnings

⚠️ **IMPORTANT SAFETY GUIDELINES**

1. **Always start in paper mode** to verify the system works correctly
2. **Never trade with money you can't afford to lose**
3. **Set conservative stop limits** initially and adjust based on performance
4. **Monitor the dashboard regularly** during live trading
5. **Review logs daily** for any anomalies
6. **The nuclear stop is your last line of defense** - respect it
7. **Polymarket markets are binary** - you can lose 100% of your stake on a single trade
8. **This is experimental software** - use at your own risk

## API Keys

For live trading, add your Polymarket API credentials to `live_config.json`:

```json
{
  "api_key": "your_api_key_here",
  "api_secret": "your_api_secret_here"
}
```

⚠️ **Never commit API keys to version control!**

## Troubleshooting

### Dashboard not loading trades

- Ensure `--log-dir` points to the correct directory
- Check that trade CSV files exist in the log directory
- Verify the dashboard has read permissions

### API connection issues

- Check internet connectivity
- Verify Polymarket API status at status.polymarket.com
- Ensure API keys are valid (for live mode)

### Strategy not generating signals

- Check filter thresholds in configuration
- Verify market data is being received
- Review logs for filter pass/fail rates

## Performance Metrics

Based on backtesting with historical data:

- **Win Rate**: ~55-65% (varies by market conditions)
- **Profit Factor**: 1.2-1.8 (depends on filter settings)
- **Max Drawdown**: <15% with proper stop management
- **Avg Trade Duration**: 2-4 minutes

*Note: Past performance does not guarantee future results.*

## Support

For issues or questions:

1. Check the logs in `./live_logs/`
2. Review the dashboard debug endpoint: `http://127.0.0.1:5000/api/debug`
3. Run the test script: `python test_system.py`

## License

This project is for educational and research purposes only. Use at your own risk.

## Disclaimer

This software is provided "as is" without warranty of any kind. The authors are not responsible for any financial losses incurred through the use of this software. Trading cryptocurrencies and prediction markets involves substantial risk of loss.

---

**Version**: 1.0.0  
**Last Updated**: 2026-05-17
