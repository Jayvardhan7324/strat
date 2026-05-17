"""
Comprehensive Logging System for Post-Trade Analysis
====================================================
Provides structured logging, trade journaling, and analytics 
for post-trade analysis.

Author: OpenCode AI
"""

from __future__ import annotations

import json
import csv
import os
import time
import datetime
import logging
import uuid
import hashlib
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple
from enum import Enum, auto
from threading import Lock
from collections import deque

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Enums and Constants
# ---------------------------------------------------------------------------

class TradeAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    SETTLE = "SETTLE"
    CANCEL = "CANCEL"
    NOOP = "NOOP"


class ExitType(Enum):
    TARGET_HIT = "target_hit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    TIME_EXIT = "time_exit"
    MANUAL = "manual"
    SETTLEMENT = "settlement"
    MARGIN_CALL = "margin_call"
    STRATEGY_SIGNAL = "strategy_signal"


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Represents an open position."""
    entry_time: float
    entry_price: float
    side: str
    size: float
    condition_id: str
    token_id: str
    market_name: str = ""
    unrealized_pnl: float = 0.0
    peak_price: float = 0.0
    lowest_price: float = 0.0


@dataclass
class TradeEntry:
    """Represents a single trade entry/exit."""
    trade_id: str
    position_id: str
    timestamp: str
    action: str
    side: str
    condition_id: str
    token_id: str
    price: float
    size: float
    fee: float
    pnl: float
    pnl_pct: float
    exit_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionMetrics:
    """Session-level performance metrics."""
    session_id: str
    start_time: str
    end_time: str = ""
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_volume: float = 0.0
    total_fees: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    max_drawdown: float = 0.0
    starting_capital: float = 0.0
    final_capital: float = 0.0
    sharpe_ratio: float = 0.0
    avg_trade_duration_ms: float = 0.0
    stopped: bool = False
    stop_reason: str = ""
    strategies_used: List[str] = field(default_factory=list)
    filters_hit: Dict[str, int] = field(default_factory=dict)
    market_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class SignalLog:
    """Log of strategy signals (not necessarily executed)."""
    signal_id: str
    timestamp: str
    strategy: str
    condition_id: str
    side: str
    confidence: float
    price: float
    filters_passed: int
    filters_total: int
    filter_details: Dict[str, bool] = field(default_factory=dict)
    executed: bool = False
    trade_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Trade Logger
# ---------------------------------------------------------------------------

class TradeLogger:
    """Comprehensive logging for post-trade analysis."""
    
    def __init__(self, log_dir: str = "./live_logs", level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = f"session_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Setup logging
        self.logger = logging.getLogger("PolymarketTrader")
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        
        if not self.logger.handlers:
            # File handler
            log_file = self.log_dir / f"{self.session_id}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s"
            ))
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s"
            ))
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
        
        self.logger.info("=" * 80)
        self.logger.info(f"  SESSION INITIALIZED: {self.session_id}")
        self.logger.info("=" * 80)
        
        # Trade journal
        self.trades: List[Dict] = []
        self.trades_lock = Lock()
        self.trades_file = self.log_dir / f"{self.session_id}_trades.csv"
        
        # Signal log
        self.signals: List[Dict] = []
        self.signals_lock = Lock()
        self.signals_file = self.log_dir / f"{self.session_id}_signals.csv"
        
        # Session metrics
        self.session_start = time.time()
        self.session_metrics = SessionMetrics(
            session_id=self.session_id,
            start_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            starting_capital=0,
            strategies_used=[],
        )
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Dict] = []
        
        # P&L tracking
        self.equity_curve: List[Tuple[float, float]] = []  # timestamp, equity
        self.pnl_history: List[Dict] = []
        
        # API call logging
        self.api_calls: List[Dict] = []
        
        # Data backup
        self.raw_data: deque = deque(maxlen=10000)
        
        self._init_csv_files()
    
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        # Trades CSV
        trade_headers = [
            "trade_id", "timestamp", "action", "side", "condition_id", 
            "token_id", "price", "size", "fee", "pnl", "pnl_pct",
            "exit_type", "metadata"
        ]
        with open(self.trades_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(trade_headers)
        
        # Signals CSV
        signal_headers = [
            "signal_id", "timestamp", "strategy", "condition_id", "side",
            "confidence", "price", "filters_passed", "filters_total", "executed"
        ]
        with open(self.signals_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(signal_headers)
    
    def log_trade(self, trade: TradeEntry):
        """Log a trade."""
        trade_dict = asdict(trade)
        
        with self.trades_lock:
            self.trades.append(trade_dict)
            
            with open(self.trades_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    trade.trade_id, trade.timestamp, trade.action, trade.side,
                    trade.condition_id, trade.token_id, trade.price, trade.size,
                    trade.fee, trade.pnl, trade.pnl_pct, trade.exit_type,
                    json.dumps(trade.metadata)
                ])
        
        self.logger.info(
            f"TRADE | {trade.action} {trade.side} | ${trade.price:.4f} | "
            f"PnL: ${trade.pnl:+.2f} ({trade.pnl_pct:.2f}%) | Exit: {trade.exit_type}"
        )
    
    def log_signal(self, signal: SignalLog):
        """Log a strategy signal."""
        signal_dict = asdict(signal)
        
        with self.signals_lock:
            self.signals.append(signal_dict)
            
            with open(self.signals_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    signal.signal_id, signal.timestamp, signal.strategy,
                    signal.condition_id, signal.side, signal.confidence,
                    signal.price, signal.filters_passed, signal.filters_total,
                    signal.executed
                ])
        
        self.logger.debug(
            f"SIGNAL | {signal.strategy} | {signal.side} | conf={signal.confidence:.2f} | "
            f"filters={signal.filters_passed}/{signal.filters_total}"
        )
    
    def log_api_call(self, endpoint: str, request_data: dict, response_data: dict, 
                     status: str, latency_ms: float):
        """Log API calls for debugging and auditing."""
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "endpoint": endpoint,
            "status": status,
            "latency_ms": latency_ms,
            "request_hash": hashlib.sha256(json.dumps(request_data, sort_keys=True).encode()).hexdigest()[:16],
            "response_hash": hashlib.sha256(json.dumps(response_data, sort_keys=True).encode()).hexdigest()[:16],
        }
        self.api_calls.append(entry)
        self.logger.debug(f"API CALL | {endpoint} | {status} | {latency_ms:.2f}ms")
    
    def log_equity_update(self, equity: float):
        """Update equity curve."""
        self.equity_curve.append((time.time(), equity))
        
    def log_pnl(self, trade_pnl: float, equity: float):
        """Log a P&L update."""
        self.pnl_history.append({
            "timestamp": time.time(),
            "trade_pnl": trade_pnl,
            "equity": equity,
        })
        self.log_equity_update(equity)
    
    def log_error(self, error: Exception, context: str = ""):
        """Log an error with context."""
        self.logger.error(f"ERROR | {context} | {type(error).__name__}: {str(error)}", 
                         exc_info=True)
    
    def log_market_data(self, condition_id: str, data: dict):
        """Log market data for replay and analysis."""
        entry = {
            "timestamp": time.time(),
            "condition_id": condition_id,
            "data": data,
        }
        self.raw_data.append(entry)
    
    def log_info(self, message: str):
        """Log an info message."""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """Log a warning."""
        self.logger.warning(message)
    
    def log_debug(self, message: str):
        """Log a debug message."""
        self.logger.debug(message)
    
    # ---------------------------------------------------------------------------
    # Post-Trade Analysis
    # ---------------------------------------------------------------------------
    
    def generate_trade_report(self) -> Dict[str, Any]:
        """Generate comprehensive trade report."""
        if not self.trades:
            return {"error": "No trades recorded"}
        
        df = pd.DataFrame(self.trades)
        report = {
            "session_id": self.session_id,
            "duration_hours": (time.time() - self.session_start) / 3600,
            "total_trades": len(df),
            "total_pnl": float(df["pnl"].sum()),
            "avg_pnl_per_trade": float(df["pnl"].mean()),
            "win_rate": float(len(df[df["pnl"] > 0]) / len(df)),
            "profit_factor": self._calculate_profit_factor(df),
            "max_drawdown": self._calculate_drawdown(),
            "avg_trade_duration": self._calculate_avg_duration(df),
            "top_strategies": self._get_top_strategies(df),
            "hourly_breakdown": self._get_hourly_breakdown(df),
            "market_breakdown": self._get_market_breakdown(df),
        }
        
        self.logger.info(f"Trade report generated: {len(df)} trades, "
                        f"PnL: ${report['total_pnl']:.2f}")
        
        return report
    
    def _calculate_profit_factor(self, df: pd.DataFrame) -> float:
        profits = df[df["pnl"] > 0]["pnl"].sum()
        losses = abs(df[df["pnl"] < 0]["pnl"].sum())
        return profits / losses if losses > 0 else (float("inf") if profits > 0 else 0)
    
    def _calculate_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        equity = np.array([e for t, e in self.equity_curve])
        peak = np.maximum.accumulate(equity)
        drawdowns = (peak - equity) / peak
        return float(drawdowns.max()) if len(drawdowns) > 0 else 0.0
    
    def _calculate_avg_duration(self, df: pd.DataFrame) -> float:
        if "exit_time" in df.columns and "entry_time" in df.columns:
            return float(df["exit_time"].mean() - df["entry_time"].mean())
        return 0.0
    
    def _get_top_strategies(self, df: pd.DataFrame) -> List[Dict]:
        if "strategy" in df.columns:
            grouped = df.groupby("strategy").agg({
                "pnl": ["sum", "count", "mean"]
            }).reset_index()
            return grouped.to_dict("records")
        return []
    
    def _get_hourly_breakdown(self, df: pd.DataFrame) -> Dict:
        if "timestamp" in df.columns:
            df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
            grouped = df.groupby("hour").agg({"pnl": ["sum", "count"]}).to_dict()
            return {str(k): v for k, v in grouped.items()}
        return {}
    
    def _get_market_breakdown(self, df: pd.DataFrame) -> Dict:
        if "condition_id" in df.columns:
            grouped = df.groupby("condition_id").agg({"pnl": "sum"}).to_dict()
            return grouped
        return {}
    
    def save_analysis(self, filepath: str = None):
        """Save all analysis data to disk."""
        if filepath is None:
            filepath = self.log_dir / f"{self.session_id}_analysis.json"
        
        analysis = {
            "session_id": self.session_id,
            "started": self.session_start,
            "total_signals": len(self.signals),
            "total_trades": len(self.trades),
            "trade_report": self.generate_trade_report(),
            "exit_types": self._count_exit_types(),
        }
        
        with open(filepath, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        
        self.logger.info(f"Analysis saved to {filepath}")
        return filepath
    
    def _count_exit_types(self) -> Dict[str, int]:
        counts = {}
        for trade in self.trades:
            exit_type = trade.get("exit_type", "unknown")
            counts[exit_type] = counts.get(exit_type, 0) + 1
        return counts
    
    def get_session_summary(self) -> Dict:
        """Get real-time session summary."""
        return {
            "session_id": self.session_id,
            "runtime": time.time() - self.session_start,
            "total_signals": len(self.signals),
            "total_trades": len(self.trades),
            "total_pnl": sum(t["pnl"] for t in self.trades),
            "equity_current": self.equity_curve[-1][1] if self.equity_curve else 0,
            "max_drawdown": self._calculate_drawdown(),
        }
    
    def close(self):
        """Finalize logging and save data."""
        self.logger.info("=" * 80)
        self.logger.info(f"  SESSION CLOSED: {self.session_id}")
        self.logger.info("=" * 80)
        
        # Save final analysis
        self.save_analysis()
        
        # Save equity curve
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve, columns=["timestamp", "equity"])
            equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], unit="s")
            equity_df.to_csv(self.log_dir / f"{self.session_id}_equity.csv", index=False)


# ---------------------------------------------------------------------------
# CSV Export Utility
# ---------------------------------------------------------------------------

def export_trades_to_csv(trades: List[Dict], filepath: str):
    """Export trades to CSV."""
    df = pd.DataFrame(trades)
    df.to_csv(filepath, index=False)
    return filepath


def export_signals_to_csv(signals: List[Dict], filepath: str):
    """Export signals to CSV."""
    df = pd.DataFrame(signals)
    df.to_csv(filepath, index=False)
    return filepath


if __name__ == "__main__":
    # Test logger
    logger = TradeLogger(log_dir="./test_logs")
    
    # Simulate some trades
    import random
    for i in range(10):
        trade = TradeEntry(
            trade_id=f"T{i:04d}",
            position_id=f"P{i:04d}",
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            action="BUY",
            side="UP",
            condition_id="cond_001",
            token_id="token_001",
            price=0.97,
            size=10.0,
            fee=0.02,
            pnl=random.uniform(-5, 10),
            pnl_pct=random.uniform(-5, 10),
            exit_type=random.choice(["target_hit", "stop_loss", "time_exit"]),
        )
        logger.log_trade(trade)
    
    report = logger.generate_trade_report()
    print(json.dumps(report, indent=2, default=str))
    logger.close()
