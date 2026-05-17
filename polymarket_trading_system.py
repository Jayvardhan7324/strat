"""
Polymarket Live Trading System - Main Runner
=============================================
Unifies all components:
    - API Client (polymarket_api_client.py)
    - Logging (polymarket_logging.py)
    - Execution (polymarket_execution.py)
    - Dashboard (polymarket_dashboard.py)

Provides a single entry point with CLI and programmatic interfaces.

Author: OpenCode AI
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import logging
import subprocess
import threading
import signal as os_signal
from pathlib import Path
from typing import Optional, Dict, Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG_FILE = "polymarket_config.json"


def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        "mode": "paper",
        "symbol": "btcusdt",
        "source": "binance",
        "dataset_source": "bmoney_crypto",
        "max_windows": None,
        "stake_usd": 10.0,
        "entry_seconds_left": 30,
        "min_seconds_left": 5,
        "trail_bps": 50.0,
        "train_frac": 0.70,
        "starting_capital": 10000.0,
        "stake_per_trade": 10.0,
        "max_stake_per_trade": 20.0,
        "session_soft_stop": 500.0,
        "session_hard_stop": 1000.0,
        "nuclear_stop": 2000.0,
        "max_trades_per_session": 200,
        "require_min_delta": 0.002,
        "require_max_rsi": 85.0,
        "require_min_rsi": 15.0,
        "require_max_vol_ratio": 3.0,
        "require_time_frac": 0.50,
        "require_accel_confirms": True,
        "log_dir": "./live_logs",
        "log_level": "INFO",
        "heartbeat_interval": 60,
        "gamma_api_base": "https://gamma-api.polymarket.com/",
        "pmxt_base_url": "https://archive.pmxt.dev/Polymarket/v2/",
        "dashboard_host": "127.0.0.1",
        "dashboard_port": 5000,
        "api_key": "",
        "api_secret": "",
    }


# ---------------------------------------------------------------------------
# Dashboard Server Runner
# ---------------------------------------------------------------------------

class DashboardServer:
    """Manages the dashboard web server."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.running = False
    
    def start(self):
        """Start the dashboard server."""
        if self.running:
            print("Dashboard already running")
            return
        
        print("[START] Starting dashboard server...")
        
        try:
            # Use subprocess to run the dashboard in a separate process
            cmd = [
                sys.executable,
                "polymarket_dashboard.py",
                "--host", self.config.get("dashboard_host", "127.0.0.1"),
                "--port", str(self.config.get("dashboard_port", 5000)),
                "--log-dir", self.config.get("log_dir", "./live_logs"),
            ]
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            )
            self.running = True
            
            time.sleep(2)  # Give it time to start
            
            print(f"[OK] Dashboard running at http://{self.config['dashboard_host']}:{self.config['dashboard_port']}/")
            
        except Exception as e:
            print(f"[FAIL] Failed to start dashboard: {e}")
    
    def stop(self):
        """Stop the dashboard server."""
        if self.process:
            print("[STOP] Stopping dashboard...")
            try:
                if os.name == "nt":
                    self.process.terminate()
                else:
                    self.process.send_signal(os_signal.SIGTERM)
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            self.process = None
            self.running = False
            print("[OK] Dashboard stopped")


# ---------------------------------------------------------------------------
# Main System Runner
# ---------------------------------------------------------------------------

class PolymarketTradingSystem:
    """Orchestrates the entire trading system."""
    
    def __init__(self, config_file: str = None, **kwargs):
        self.config = create_default_config()
        
        # Load from file
        if config_file and Path(config_file).exists():
            with open(config_file) as f:
                self.config.update(json.load(f))
        
        # Override with kwargs
        self.config.update(kwargs)
        
        # Components
        self.api: Optional[Any] = None
        self.logger: Optional[Any] = None
        self.risk_manager: Optional[Any] = None
        self.strategy: Optional[Any] = None
        self.executor: Optional[Any] = None
        self.trader: Optional[Any] = None
        self.dashboard: Optional[DashboardServer] = None
        
        self._running = False
    
    def initialize(self):
        """Initialize all components."""
        print("[SETUP] Initializing trading system...")
        
        # Import here to avoid circular dependencies
        from polymarket_api_client import PolymarketAPI
        from polymarket_logging import TradeLogger
        from polymarket_execution import SessionRiskManager, MultiFilterStrategy, ExecutionEngine, PolymarketTrader
        
        # Store class references for later use
        self._PolymarketTrader = PolymarketTrader
        
        # Create API client
        self.api = PolymarketAPI(
            api_key=self.config.get("api_key"),
            api_secret=self.config.get("api_secret"),
        )
        
        # Create logger
        self.logger = TradeLogger(
            log_dir=self.config["log_dir"],
            level=self.config.get("log_level", "INFO"),
        )
        
        # Create risk manager
        self.risk_manager = SessionRiskManager(self.config)
        
        # Create strategy
        self.strategy = MultiFilterStrategy(self.config)
        
        print("[OK] All components initialized")
    
    def start_dashboard(self):
        """Start the monitoring dashboard."""
        self.dashboard = DashboardServer(self.config)
        self.dashboard.start()
    
    def run(self):
        """Run the trading bot."""
        if not self.trader:
            # Create trader
            self.trader = self._PolymarketTrader(config_file=None, **self.config)
        
        print(f"[START] Starting trading bot in {self.config['mode'].upper()} mode...")
        self.trader.start()
    
    def stop(self):
        """Stop all components."""
        print("[STOP] Stopping trading system...")
        
        if self.trader:
            self.trader.stop()
        
        if self.dashboard:
            self.dashboard.stop()
        
        if self.logger:
            self.logger.close()
        
        if self.api:
            self.api.close()
        
        print("[OK] System stopped")
    
    def get_status(self) -> Dict:
        """Get current system status."""
        status = {
            "running": self._running,
            "mode": self.config["mode"],
            "dashboard_running": self.dashboard.running if self.dashboard else False,
        }
        
        if self.risk_manager:
            status["risk"] = self.risk_manager.get_status()
        
        return status


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Live Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run in paper mode with dashboard
    python polymarket_trading_system.py --mode paper --dashboard
    
    # Run in paper mode only (no dashboard)
    python polymarket_trading_system.py --mode paper
    
    # Run in live mode (use with caution!)
    python polymarket_trading_system.py --mode live --capital 10000 --dashboard
    
    # Run for a limited time
    python polymarket_trading_system.py --mode paper --run-time 3600
    
    # Show status
    python polymarket_trading_system.py --status
        """
    )
    
    parser.add_argument("--mode", choices=["paper", "live"], default="paper",
                        help="Trading mode (default: paper)")
    parser.add_argument("--config", default=CONFIG_FILE,
                        help="Configuration file (default: polymarket_config.json)")
    parser.add_argument("--capital", type=float, default=10000.0,
                        help="Starting capital (default: $10,000)")
    parser.add_argument("--stake", type=float, default=10.0,
                        help="Stake per trade in USD (default: $10)")
    parser.add_argument("--soft-stop", type=float, default=500.0,
                        help="Soft stop loss limit (default: $500)")
    parser.add_argument("--hard-stop", type=float, default=1000.0,
                        help="Hard stop loss limit (default: $1,000)")
    parser.add_argument("--nuclear-stop", type=float, default=2000.0,
                        help="Nuclear stop/strategy abort (default: $2,000)")
    parser.add_argument("--max-trades", type=int, default=200,
                        help="Maximum trades per session (default: 200)")
    parser.add_argument("--dashboard", action="store_true",
                        help="Start monitoring dashboard")
    parser.add_argument("--dashboard-port", type=int, default=5000,
                        help="Dashboard port (default: 5000)")
    parser.add_argument("--run-time", type=int, default=None,
                        help="Run time in seconds (default: indefinite)")
    parser.add_argument("--status", action="store_true",
                        help="Show current status and exit")
    parser.add_argument("--backfill", action="store_true",
                        help="Run backfill before starting")
    parser.add_argument("--log-dir", default="./live_logs",
                        help="Log directory")
    
    args = parser.parse_args()
    
    # Show status and exit
    if args.status:
        print("[CHART] System Status")
        print("-" * 50)
        print(f"Mode: {args.mode.upper()}")
        if Path(args.config).exists():
            print(f"Config: {args.config} (exists)")
        else:
            print(f"Config: {args.config} (not found, using defaults)")
        print(f"Capital: ${args.capital:,.2f}")
        print(f"Stake: ${args.stake:,.2f}")
        print(f"Stops: Soft=${args.soft_stop}/Hard=${args.hard_stop}/Nuclear=${args.nuclear_stop}")
        print(f"Max Trades: {args.max_trades}")
        print(f"Dashboard: {'Yes (port {})'.format(args.dashboard_port) if args.dashboard else 'No'}")
        return
    
    # Create system
    config = {
        "mode": args.mode,
        "starting_capital": args.capital,
        "stake_per_trade": args.stake,
        "session_soft_stop": max(1, abs(args.soft_stop)),
        "session_hard_stop": max(1, abs(args.hard_stop)),
        "nuclear_stop": max(1, abs(args.nuclear_stop)),
        "max_trades_per_session": args.max_trades,
        "log_dir": args.log_dir,
        "dashboard_host": "127.0.0.1",
        "dashboard_port": args.dashboard_port,
    }
    
    system = PolymarketTradingSystem(config_file=args.config, **config)
    
    try:
        # Initialize components
        system.initialize()
        
        # Start dashboard if requested
        if args.dashboard:
            system.start_dashboard()
        
        print(f"\n{'=' * 70}")
        print(f"  POLYMARKET TRADING SYSTEM STARTING")
        print(f"{'=' * 70}")
        print(f"  Mode:            {args.mode.upper()}")
        print(f"  Capital:         ${args.capital:,.2f}")
        print(f"  Stake/Trade:     ${args.stake:,.2f}")
        print(f"  Soft Stop:       ${args.soft_stop:,.2f}")
        print(f"  Hard Stop:       ${args.hard_stop:,.2f}")
        print(f"  Nuclear Stop:    ${args.nuclear_stop:,.2f}")
        print(f"  Max Trades:      {args.max_trades}")
        print(f"  Dashboard:       {'Yes' if args.dashboard else 'No'}")
        print(f"\n  Press Ctrl+C to stop gracefully")
        print(f"{'=' * 70}\n")
        
        # Run the bot
        system.run()
        
        if args.run_time:
            print(f"\n[TIME] Running for {args.run_time} seconds...")
            try:
                time.sleep(args.run_time)
            except KeyboardInterrupt:
                print("\n[PAUSE] Interrupted by user")
        else:
            # Keep running until interrupted
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n[BYE] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        traceback.print_exc()
    finally:
        system.stop()


if __name__ == "__main__":
    main()
