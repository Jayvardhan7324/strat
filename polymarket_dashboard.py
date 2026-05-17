"""
Polymarket Live Trading Dashboard
===================================
Real-time web interface for monitoring trades, sessions, and bot performance.
Uses Flask for the web server, with real-time updates via Socket.IO.

Features:
- Real-time P&L chart
- Trade history table
- Session metrics cards
- Live bot status
- Risk overview (stop levels, drawdown)
- Signal log
- Automatic refresh

Author: OpenCode AI
"""

from __future__ import annotations

import json
import os
import sys
import glob
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from threading import Lock
import threading

# Try to import deps, install if missing
try:
    from flask import Flask, render_template, jsonify, request, render_template_string
    from flask_socketio import SocketIO, emit
    import plotly.graph_objects as go
    import plotly.utils
    import plotly.express as px
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "flask", "flask-socketio", "plotly>=5.0", "pandas"])
    from flask import Flask, render_template, jsonify, request, render_template_string
    from flask_socketio import SocketIO, emit
    import plotly.graph_objects as go
    import plotly.utils
    import plotly.express as px

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Dashboard Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "host": "127.0.0.1",
    "port": 5000,
    "debug": True,
    "log_dir": "./live_logs",
    "refresh_interval": 5,  # seconds
    "max_trades_display": 100,
    "max_signals_display": 50,
}


# ---------------------------------------------------------------------------
# Dashboard App
# ---------------------------------------------------------------------------

class DashboardData:
    """Manages dashboard data from log files."""
    
    def __init__(self, config: Dict = None):
        self.config = config or DEFAULT_CONFIG
        self.log_dir = Path(self.config["log_dir"])
        print(f"DashboardData initialized with log_dir: {self.log_dir.absolute()}")
        self.data_lock = Lock()
        
        self.session_data: Dict[str, Any] = {
            "session_id": "",
            "start_time": "",
            "running": False,
            "session_pnl": 0.0,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "current_capital": 0.0,
            "starting_capital": 0.0,
            "stopped": False,
            "stop_reason": "",
            "strategies_used": [],
            "stopped_date": "",  
        }
        self.trades: List[Dict] = []
        self.signals: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.api_stats: Dict[str, Any] = {"total_calls": 0, "avg_latency": 0.0}
        
        self.last_update = 0
    
    def refresh(self) -> Dict[str, Any]:
        """Refresh all data from log files."""
        print(f"refresh() called - log_dir: {self.log_dir}")
        print(f"Acquiring data_lock...")
        with self.data_lock:
            print(f"Lock acquired")
            # Find latest log files
            print("Calling _reload_session()...")
            self._reload_session()
            print("Calling _reload_trades()...")
            self._reload_trades()
            print(f"After _reload_trades - self.trades count: {len(self.trades)}")
            print("Calling _reload_signals()...")
            self._reload_signals()
            print("Calling _reload_equity()...")
            self._reload_equity()
            self.last_update = time.time()
            print(f"Returning data - trades: {len(self.trades)}")
            
            return self._get_dashboard_data()
    
    def _reload_session(self):
        """Reload session data from latest log."""
        log_files = sorted(self.log_dir.glob("session_*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not log_files:
            return
        
        # Try to find session_summary.json
        summary_files = sorted(self.log_dir.glob("*session_summary*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if summary_files:
            try:
                with open(summary_files[0]) as f:
                    data = json.load(f)
                self.session_data.update(data)
            except Exception:
                pass
        
        # Try to find analysis file
        analysis_files = sorted(self.log_dir.glob("*_analysis.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        if analysis_files:
            try:
                with open(analysis_files[0]) as f:
                    data = json.load(f)
                if "trade_report" in data:
                    tr = data["trade_report"]
                    self.session_data.update({
                        "session_pnl": tr.get("total_pnl", 0),
                        "total_trades": tr.get("total_trades", 0),
                    })
            except Exception:
                pass
        
        # Detect if bot is running by checking for recent log activity
        if log_files:
            latest_log = log_files[0]
            mtime = latest_log.stat().st_mtime
            age_seconds = time.time() - mtime
            # If log was modified in last 30 seconds, bot is likely running
            is_recently_active = age_seconds < 30
            self.session_data["running"] = is_recently_active and not self.session_data.get("stopped", False)
    
    def _reload_trades(self):
        """Reload trades from latest CSV."""
        try:
            trade_files = sorted(self.log_dir.glob("*_trades.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
            if not trade_files:
                print(f"No trade files found in {self.log_dir}")
                return False
            
            print(f"Loading trades from: {trade_files[0]}")
            df = pd.read_csv(trade_files[0])
            print(f"Loaded {len(df)} trades")
            if not df.empty:
                self.trades = df.to_dict("records")[-self.config["max_trades_display"]:]
                print(f"Stored {len(self.trades)} trades")
                return True
            return False
        except Exception as e:
            print(f"Error loading trades: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _reload_signals(self):
        """Reload signals from latest CSV."""
        signal_files = sorted(self.log_dir.glob("*_signals.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not signal_files:
            return
        
        try:
            df = pd.read_csv(signal_files[0])
            if not df.empty:
                self.signals = df.to_dict("records")[-self.config["max_signals_display"]:]
        except Exception:
            pass
    
    def _reload_equity(self):
        """Reload equity curve from latest CSV."""
        equity_files = sorted(self.log_dir.glob("*_equity.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not equity_files:
            return
        
        try:
            df = pd.read_csv(equity_files[0])
            if not df.empty:
                self.equity_curve = df.head(1000).to_dict("records")
        except Exception:
            pass
    
    def _get_dashboard_data(self) -> Dict[str, Any]:
        """Compile dashboard data."""
        return {
            "session": self.session_data,
            "trades": self.trades,
            "signals": self.signals,
            "equity": self.equity_curve,
            "stats": {
                "last_update": self.last_update,
                "total_trades": len(self.trades),
                "total_signals": len(self.signals),
            },
            "charts": {
                "equity": self._render_equity_chart(),
                "trade_distribution": self._render_trade_distribution(),
                "hourly_pnl": self._render_hourly_pnl(),
                "filters": self._render_filter_chart(),
            },
        }
    
    def _render_equity_chart(self) -> str:
        """Render equity curve as HTML."""
        if not self.equity_curve:
            return ""
        
        df = pd.DataFrame(self.equity_curve)
        if len(df) < 2:
            return ""
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["equity"],
            mode="lines",
            name="Equity",
            line=dict(color="#00ff88", width=2),
        ))
        
        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Time",
            yaxis_title="Equity ($)",
            template="plotly_dark",
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        
        return fig.to_html(full_html=False, include_plotlyjs="cdn")
    
    def _render_trade_distribution(self) -> str:
        """Render trade P&L distribution."""
        if not self.trades:
            return ""
        
        df = pd.DataFrame(self.trades)
        if "pnl" not in df.columns:
            return ""
        
        fig = go.Figure(data=[go.Histogram(
            x=df["pnl"],
            nbinsx=20,
            marker_color="#00ff88",
        )])
        
        fig.update_layout(
            title="Trade P&L Distribution",
            xaxis_title="P&L ($)",
            yaxis_title="Count",
            template="plotly_dark",
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        
        return fig.to_html(full_html=False, include_plotlyjs="cdn")
    
    def _render_hourly_pnl(self) -> str:
        """Render hourly P&L chart."""
        if not self.trades:
            return ""
        
        df = pd.DataFrame(self.trades)
        if "timestamp" not in df.columns or "pnl" not in df.columns:
            return ""
        
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["hour"] = df["timestamp"].dt.hour
            hourly = df.groupby("hour")["pnl"].sum().reset_index()
            
            fig = go.Figure(data=[go.Bar(
                x=hourly["hour"],
                y=hourly["pnl"],
                marker_color=["#00ff88" if v >= 0 else "#ff4444" for v in hourly["pnl"]],
            )])
            
            fig.update_layout(
                title="Hourly P&L",
                xaxis_title="Hour",
                yaxis_title="P&L ($)",
                template="plotly_dark",
                height=300,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            
            return fig.to_html(full_html=False, include_plotlyjs="cdn")
        except Exception:
            return ""
    
    def _render_filter_chart(self) -> str:
        """Render filter pass/fail chart."""
        if not self.signals:
            return ""
        
        df = pd.DataFrame(self.signals)
        if "filters_passed" not in df.columns:
            return ""
        
        pass_counts = df["filters_passed"].value_counts().sort_index()
        
        fig = go.Figure(data=[go.Bar(
            x=pass_counts.index,
            y=pass_counts.values,
            marker_color="#00ccff",
        )])
        
        fig.update_layout(
            title="Filter Results",
            xaxis_title="Filters Passed",
            yaxis_title="Count",
            template="plotly_dark",
            height=300,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        
        return fig.to_html(full_html=False, include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------

def create_app(config: Dict = None) -> Tuple[Flask, SocketIO, DashboardData]:
    """Create and configure the Flask app."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.urandom(24).hex()
    
    socketio = SocketIO(app, async_mode="threading")
    
    data = DashboardData(config or DEFAULT_CONFIG)
    
    @app.route("/")
    def index():
        """Main dashboard page."""
        result = data.refresh()
        is_running = result.get("session", {}).get("running", False)
        last_update_ts = result.get("stats", {}).get("last_update", 0)
        last_update_str = datetime.fromtimestamp(last_update_ts).strftime("%H:%M:%S") if last_update_ts else "Never"
        return render_template_string(DASHBOARD_HTML, config=config, is_running=is_running, last_update=last_update_str)
    
    @app.route("/api/debug")
    def debug_info():
        """Debug endpoint to check paths and data."""
        import os
        import traceback
        import io
        import sys
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        debug_result = {}
        
        try:
            print(f"Starting debug - log_dir: {data.log_dir}")
            print(f"log_dir exists: {data.log_dir.exists()}")
            print(f"Current trades count: {len(data.trades)}")
            
            # Try to manually load trades
            trade_files = sorted(data.log_dir.glob("*_trades.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
            print(f"Found {len(trade_files)} trade files")
            
            if trade_files:
                print(f"Latest file: {trade_files[0]}")
                try:
                    df = pd.read_csv(trade_files[0])
                    print(f"Loaded {len(df)} trades from CSV")
                    print(f"Columns: {list(df.columns)}")
                except Exception as e:
                    print(f"Error loading CSV: {e}")
            
            # Now try _reload_trades directly
            print("Calling _reload_trades() directly...")
            reload_result = data._reload_trades()
            print(f"_reload_trades returned: {reload_result}")
            print(f"data.trades count after reload: {len(data.trades)}")
            
            # Now try the actual refresh
            print("Calling data.refresh()...")
            result = data.refresh()
            print(f"After refresh - trades: {len(result.get('trades', []))}")
            print(f"data.trades count: {len(data.trades)}")
            
            debug_result = {
                "reload_result": reload_result,
                "trades_after_reload": len(data.trades),
                "refresh_trades": len(result.get('trades', [])),
            }
            
        except Exception as e:
            print(f"Exception: {e}")
            traceback.print_exc()
        
        # Restore stdout
        sys.stdout = old_stdout
        captured_text = captured_output.getvalue()
        
        return jsonify({
            "cwd": os.getcwd(),
            "log_dir": str(data.log_dir),
            "captured_output": captured_text.split('\n'),
            "trades_count": len(data.trades),
            **debug_result,
        })
    
    @app.route("/api/data")
    def get_data():
        """Get all dashboard data as JSON."""
        print(f"API /api/data called - log_dir: {data.log_dir}")
        print(f"Current trades count: {len(data.trades)}")
        result = data.refresh()
        print(f"After refresh - trades count: {len(result.get('trades', []))}")
        return jsonify(result)
    
    @app.route("/api/session")
    def get_session():
        """Get session data."""
        return jsonify(data.session_data)
    
    @app.route("/api/trades")
    def get_trades():
        """Get trade data."""
        return jsonify(data.trades)
    
    @app.route("/api/signals")
    def get_signals():
        """Get signal data."""
        return jsonify(data.signals)
    
    @app.route("/api/stats")
    def get_stats():
        """Get statistics."""
        return jsonify(data._get_dashboard_data()["stats"])
    
    # WebSocket events
    @socketio.on("connect")
    def handle_connect():
        """Handle client connection."""
        emit("update", data.refresh())
    
    @socketio.on("request_update")
    def handle_request_update():
        """Handle manual data refresh request."""
        emit("update", data.refresh())
    
    return app, socketio, data


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Polymarket Trading Dashboard</title>
    <script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            line-height: 1.6;
        }
        
        .header {
            background: #161b22;
            padding: 15px 30px;
            border-bottom: 1px solid #21262d;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .header h1 {
            color: #00ff88;
            font-size: 24px;
            font-weight: 600;
        }
        
        .status {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 16px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
        }
        
        .status-running {
            background: #1f2937;
            color: #00ff88;
            border: 1px solid #00ff88;
        }
        
        .status-stopped {
            background: #1f2937;
            color: #ff4444;
            border: 1px solid #ff4444;
        }
        
        .last-update {
            font-size: 12px;
            color: #8b949e;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px 30px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background: #161b22;
            border: 1px solid #21262d;
            border-radius: 10px;
            padding: 18px 22px;
            transition: transform 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: #30363d;
        }
        
        .metric-label {
            font-size: 12px;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 28px;
            font-weight: 700;
        }
        
        .metrics-grid .metric-card:nth-child(1) .metric-value { color: #00ff88; } /* P&L */
        .metrics-grid .metric-card:nth-child(2) .metric-value { color: #ffd700; } /* Trades */
        .metrics-grid .metric-card:nth-child(3) .metric-value { color: #00ccff; } /* Win Rate */
        .metrics-grid .metric-card:nth-child(4) .metric-value { color: #ff6b6b; } /* Max Drawdown */
        .metrics-grid .metric-card:nth-child(5) .metric-value { color: #c9d1d9; } /* Capital */
        .metrics-grid .metric-card:nth-child(6) .metric-value { color: #00ff88; } /* PF */
        
        .section {
            background: #161b22;
            border: 1px solid #21262d;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .section h2 {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
            color: #f0f6fc;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .chart-container {
            background: #0d1117;
            border: 1px solid #21262d;
            border-radius: 8px;
            padding: 15px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        
        th {
            text-align: left;
            padding: 10px 12px;
            border-bottom: 1px solid #30363d;
            color: #8b949e;
            font-weight: 600;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        td {
            padding: 10px 12px;
            border-bottom: 1px solid #21262d;
        }
        
        tr:hover {
            background: #1f2937;
        }
        
        .positive { color: #00ff88; }
        .negative { color: #ff4444; }
        .neutral { color: #8b949e; }
        
        .stop-levels {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stop-card {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 12px 15px;
            text-align: center;
        }
        
        .stop-label {
            font-size: 11px;
            color: #8b949e;
            text-transform: uppercase;
        }
        
        .stop-value {
            font-size: 22px;
            font-weight: 700;
            margin-top: 5px;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            font-size: 12px;
            color: #484f58;
        }
        
        @media (max-width: 768px) {
            .metrics-grid { grid-template-columns: repeat(2, 1fr); }
            .chart-grid { grid-template-columns: 1fr; }
            .stop-levels { grid-template-columns: 1fr; }
            .header { flex-direction: column; gap: 10px; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Polymarket Trading Dashboard</h1>
        <div class="status">
            <div class="status-indicator {{ 'status-running' if is_running else 'status-stopped' }}">
                <span>●</span> {{ 'Running' if is_running else 'Stopped' }}
            </div>
            <div class="last-update">
                Updated: {{ last_update }}
            </div>
        </div>
    </div>
    
    <div class="container">
        <!-- Metrics Cards -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Session P&L</div>
                <div class="metric-value" id="pnl">$0.00</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value" id="trades">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value" id="winrate">0%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value" id="drawdown">0%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Current Capital</div>
                <div class="metric-value" id="capital">$10,000</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value" id="pf">0.00</div>
            </div>
        </div>
        
        <!-- Risk Overview -->
        <div class="section">
            <h2>Risk Management</h2>
            <div class="stop-levels">
                <div class="stop-card">
                    <div class="stop-label">Soft Stop</div>
                    <div class="stop-value positive" id="soft_stop">$0</div>
                </div>
                <div class="stop-card">
                    <div class="stop-label">Hard Stop</div>
                    <div class="stop-value negative" id="hard_stop">$0</div>
                </div>
                <div class="stop-card">
                    <div class="stop-label">Nuclear Stop</div>
                    <div class="stop-value negative" id="nuclear_stop">$0</div>
                </div>
            </div>
        </div>
        
        <!-- Charts -->
        <div class="chart-grid">
            <div class="chart-container" id="equity-chart">
                <p class="neutral">Loading equity curve...</p>
            </div>
            <div class="chart-container" id="trade-dist">
                <p class="neutral">Loading trade distribution...</p>
            </div>
        </div>
        
        <!-- Recent Trades -->
        <div class="section">
            <h2>Recent Trades</h2>
            <div style="overflow-x: auto;">
                <table id="trades-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Time</th>
                            <th>Action</th>
                            <th>Side</th>
                            <th>Price</th>
                            <th>P&L</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td colspan="7" class="neutral">No trades yet</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Recent Signals -->
        <div class="section">
            <h2>Recent Signals</h2>
            <div style="overflow-x: auto;">
                <table id="signals-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Time</th>
                            <th>Strategy</th>
                            <th>Side</th>
                            <th>Confidence</th>
                            <th>Filters</th>
                            <th>Executed</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td colspan="7" class="neutral">No signals yet</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <div class="footer">
        Polymarket Live Trading Bot v1.0 | Dashboard auto-refreshes every 5s
    </div>
    
    <script>
        const socket = io();
        let currentData = {};
        
        function formatMoney(value) {
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD'
            }).format(value);
        }
        
        function formatPercent(value) {
            return new Intl.NumberFormat('en-US', {
                style: 'percent',
                minimumFractionDigits: 1,
                maximumFractionDigits: 1
            }).format(value);
        }
        
        function updateDashboard(data) {
            currentData = data;
            
            // Update metrics
            if (data.session) {
                const s = data.session;
                document.getElementById('pnl').textContent = formatMoney(s.session_pnl || 0);
                document.getElementById('trades').textContent = s.total_trades || 0;
                document.getElementById('winrate').textContent = formatPercent(s.win_rate || 0);
                document.getElementById('drawdown').textContent = formatPercent(s.max_drawdown || 0);
                document.getElementById('capital').textContent = formatMoney(s.current_capital || 10000);
                document.getElementById('pf').textContent = (s.profit_factor || 0).toFixed(2);
                
                // Stops
                document.getElementById('soft_stop').textContent = formatMoney(500);
                document.getElementById('hard_stop').textContent = formatMoney(1000);
                document.getElementById('nuclear_stop').textContent = formatMoney(2000);
            }
            
            // Update charts
            if (data.charts) {
                if (data.charts.equity) {
                    document.getElementById('equity-chart').innerHTML = data.charts.equity;
                }
                if (data.charts.trade_distribution) {
                    document.getElementById('trade-dist').innerHTML = data.charts.trade_distribution;
                }
            }
            
            // Update trades table
            if (data.trades && data.trades.length > 0) {
                const tbody = document.querySelector('#trades-table tbody');
                tbody.innerHTML = data.trades.slice(-20).reverse().map(t => `
                    <tr>
                        <td>${t.trade_id || t.tradeId || 'N/A'}</td>
                        <td>${t.timestamp || 'N/A'}</td>
                        <td>${t.action || 'BUY'}</td>
                        <td>${t.side || 'N/A'}</td>
                        <td>$${(t.price || 0).toFixed(4)}</td>
                        <td class="${(t.pnl || 0) >= 0 ? 'positive' : 'negative'}">${formatMoney(t.pnl || 0)}</td>
                        <td>${t.exit_type || 'OPEN'}</td>
                    </tr>
                `).join('');
            }
            
            // Update signals table
            if (data.signals && data.signals.length > 0) {
                const tbody = document.querySelector('#signals-table tbody');
                tbody.innerHTML = data.signals.slice(-20).reverse().map(s => `
                    <tr>
                        <td>${s.signal_id || s.signalId || 'N/A'}</td>
                        <td>${s.timestamp || 'N/A'}</td>
                        <td>${s.strategy || 'N/A'}</td>
                        <td>${s.side || 'N/A'}</td>
                        <td>${((s.confidence || 0) * 100).toFixed(0)}%</td>
                        <td>${s.filters_passed || 0}/${s.filters_total || 0}</td>
                        <td>${s.executed ? 'Yes' : 'No'}</td>
                    </tr>
                `).join('');
            }
        }
        
        socket.on('update', (data) => {
            updateDashboard(data);
        });
        
        // Initial data load
        fetch('/api/data')
            .then(r => r.json())
            .then(data => updateDashboard(data));
        
        // Auto-refresh via polling (fallback)
        setInterval(() => {
            fetch('/api/data')
                .then(r => r.json())
                .then(data => updateDashboard(data));
        }, 5000);
    </script>
</body>
</html>
'''


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Polymarket Trading Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--log-dir", default="./live_logs", help="Log directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    
    args = parser.parse_args()
    
    config = {
        "host": args.host,
        "port": args.port,
        "debug": args.debug,
        "log_dir": args.log_dir,
        "max_trades_display": 100,
        "max_signals_display": 50,
        "refresh_interval": 5,
    }
    
    app, socketio, data = create_app(config)
    
    print(f"Dashboard starting on http://{args.host}:{args.port}/")
    print(f"   Log directory: {args.log_dir}/")
    print(f"   Press Ctrl+C to stop")
    
    try:
        socketio.run(app, host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\n Shutting down dashboard...")


if __name__ == "__main__":
    main()
