"""Quick test of the live trading bot."""
from core.polymarket_live_trading_bot import LiveTradingBot, DEFAULT_CONFIG
import numpy as np
import json

np.random.seed(42)

config = DEFAULT_CONFIG.copy()
config.update({
    "mode": "paper",
    "starting_capital": 10000.0,
    "stake_per_trade": 10.0,
    "log_dir": "./live_logs"
})

bot = LiveTradingBot(config)
bot.setup()

open_p = 70000.0
hist = []
prices = open_p + np.cumsum(np.random.randn(500) * 50)

trades = 0
for i in range(500):
    hist.append(float(prices[i]))
    result = bot.run_once(
        spot_price=float(prices[i]),
        open_price=open_p,
        time_frac=i / 500.0,
        prices_history=hist,
        condition_id="test-market-001"
    )
    if result["action"] == "TRADE":
        trades += 1

status = bot.get_status()
print(f"Session complete!")
print(f"Trades: {trades}")
print(f"Final PnL: ${status['session_pnl']:.2f}")
print(f"Win Rate: {status['session_wins']/max(status['session_trades'],1)*100:.1f}%")
print(f"Max Drawdown: ${status['max_drawdown']:.2f}")
print(f"Stopped: {status['stopped']} — {status['stop_reason']}")
