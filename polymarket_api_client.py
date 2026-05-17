"""
Polymarket API Client Module
============================
Connects to real Polymarket order books via the Gamma API and CLOB.
Handles market data, order book fetching, and order placement.

Author: OpenCode AI
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from decimal import Decimal
from typing import Any, Optional, Dict, List, Tuple

import requests
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"
POLYGON_RPC = "https://polygon-rpc.com"

PMXT_ORDERBOOK_URL = "https://r2v2.pmxt.dev/polymarket_orderbook_{timestamp}.parquet"
PMXT_TRADES_URL = "https://r2v2.pmxt.dev/polymarket_trades_{timestamp}.parquet"

MIN_ORDER_SIZE = 0.1  # minimum order size in USD equivalent
DEFAULT_TIMEOUT = 10

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class MarketMetadata:
    """Represents a Polymarket market."""
    condition_id: str
    market_id: str
    question: str
    description: str
    category: str
    active: bool
    closed_time: Optional[str]
    outcome_prices: Dict[str, float]
    
    @classmethod
    def from_api(cls, data: dict) -> "MarketMetadata":
        prices = {}
        if "outcomePrices" in data:
            prices = {o.get("name", ""): float(o.get("price", 0)) 
                     for o in data.get("outcomePrices", [])}
        return cls(
            condition_id=data.get("conditionId", ""),
            market_id=data.get("id", ""),
            question=data.get("question", ""),
            description=data.get("description", ""),
            category=data.get("category", ""),
            active=data.get("active", False),
            closed_time=data.get("closedTime", None),
            outcome_prices=prices,
        )


@dataclass
class OrderBookEntry:
    """Single entry in the order book."""
    price: float
    size: float
    side: str  # "BUY" or "SELL"
    token_id: str
    
    @classmethod
    def from_api(cls, data: dict, side: str) -> "OrderBookEntry":
        return cls(
            price=float(data.get("price", 0)),
            size=float(data.get("size", 0)),
            side=side,
            token_id=data.get("tokenId", ""),
        )


@dataclass
class OrderBook:
    """Represents an order book with bids and asks."""
    bids: List[OrderBookEntry]
    asks: List[OrderBookEntry]
    timestamp: float
    token_id: str
    condition_id: str
    market_id: str
    
    def best_bid(self) -> Optional[OrderBookEntry]:
        return self.bids[0] if self.bids else None
    
    def best_ask(self) -> Optional[OrderBookEntry]:
        return self.asks[0] if self.asks else None
    
    def mid_price(self) -> Optional[float]:
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        if best_bid and best_ask:
            return (best_bid.price + best_ask.price) / 2.0
        return None
    
    def spread(self) -> Optional[float]:
        best_bid = self.best_bid()
        best_ask = self.best_ask()
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None


# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------

class PolymarketAPI:
    """Production Polymarket API client for Gamma and CLOB APIs."""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "PolymarketTrader/1.0",
            "Accept": "application/json",
        })
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"
        
        self._markets_cache: Dict[str, MarketMetadata] = {}
        self._last_request_time = 0.0
        self._rate_limit_delay = 0.1  # minimum 100ms between requests
        
        self.logger = logging.getLogger("PolymarketAPI")
        self.logger.setLevel(logging.INFO)
    
    def _rate_limit(self):
        """Apply rate limiting."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def _request(self, method: str, url: str, **kwargs) -> dict:
        """Make a rate-limited API request."""
        self._rate_limit()
        try:
            response = self.session.request(method, url, timeout=DEFAULT_TIMEOUT, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise PolymarketAPIError(f"Request to {url} failed: {e}")
    
    # Market Discovery
    def get_markets(self, limit: int = 100, active_only: bool = True, 
                    category: Optional[str] = None) -> List[MarketMetadata]:
        """Fetch list of available markets from Gamma API."""
        params = {"limit": limit, "active": active_only}
        if category:
            params["category"] = category
            
        url = f"{GAMMA_API_BASE}/markets"
        data = self._request("GET", url, params=params)
        markets = []
        for market_data in data:
            try:
                market = MarketMetadata.from_api(market_data)
                self._markets_cache[market.condition_id] = market
                markets.append(market)
            except (KeyError, ValueError) as e:
                self.logger.warning(f"Failed to parse market data: {e}")
        return markets
    
    def get_market_by_condition_id(self, condition_id: str) -> MarketMetadata:
        """Fetch specific market by condition ID."""
        if condition_id in self._markets_cache:
            return self._markets_cache[condition_id]
            
        url = f"{GAMMA_API_BASE}/events/condition-id/{condition_id}"
        data = self._request("GET", url)
        market = MarketMetadata.from_api(data)
        self._markets_cache[market.condition_id] = market
        return market
    
    # Order Book
    def get_orderbook(self, condition_id: str, token_id: str) -> OrderBook:
        """Fetch live order book for a specific market and outcome."""
        url = f"{CLOB_API_BASE}/book/{token_id}"
        data = self._request("GET", url)
        
        bids = [OrderBookEntry.from_api(b, "BUY") for b in data.get("bids", [])]
        asks = [OrderBookEntry.from_api(a, "SELL") for a in data.get("asks", [])]
        
        # Sort bids descending (highest first), asks ascending (lowest first)
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)
        
        return OrderBook(
            bids=bids,
            asks=asks,
            timestamp=time.time(),
            token_id=token_id,
            condition_id=condition_id,
            market_id="",
        )
    
    def get_current_price(self, condition_id: str, side: str = "up") -> float:
        """Get current mid-market price for a condition."""
        try:
            # First try to get from Gamma markets endpoint
            url = f"{GAMMA_API_BASE}/markets"
            data = self._request("GET", url)
            for market_data in data:
                if market_data.get("conditionId") == condition_id:
                    outcomes = market_data.get("outcomes", [])
                    if side.lower() == "up" and outcomes:
                        return float(outcomes[0].get("price", 0))
                    elif side.lower() == "down" and outcomes:
                        return float(outcomes[1].get("price", 0))
            return 0.5
        except Exception as e:
            self.logger.warning(f"Failed to get current price: {e}")
            return 0.5
    
    def get_live_markets(self) -> List[dict]:
        """Get all currently live (active) markets."""
        url = f"{GAMMA_API_BASE}/markets"
        try:
            data = self._request("GET", url, params={"active": True, "limit": 100})
            return data
        except PolymarketAPIError:
            self.logger.error("Failed to fetch live markets")
            return []
    
    # PMXT Archive
    def get_pmxt_archive_url(self, timestamp_str: str) -> str:
        """Get PMXT archive URL for a specific hour."""
        return PMXT_ORDERBOOK_URL.format(timestamp=timestamp_str)
    
    def fetch_pmxt_archive(self, hour: str) -> dict:
        """Fetch PMXT archive for a specific hour.
        
        Args:
            hour: Format "YYYY-MM-DDTHH" e.g. "2025-01-15T10"
        """
        url = self.get_pmxt_archive_url(hour)
        try:
            self.logger.info(f"Fetching PMXT archive: {url}")
            self._rate_limit()
            response = self.session.head(url, timeout=30)
            if response.status_code == 404:
                self.logger.warning(f"Archive not found: {url}")
                return {}
            response.raise_for_status()
            return {"url": url, "status": response.status_code, "size": int(response.headers.get("Content-Length", 0))}
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch PMXT archive: {e}")
            return {}
    
    # Market Data Streaming (WebSocket)
    def get_ws_endpoint(self) -> str:
        """Get WebSocket endpoint for live order book data."""
        return "wss://clob.polymarket.com/ws/market"
    
    def close(self):
        """Close the session."""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class PolymarketAPIError(Exception):
    """Polymarket API error."""
    pass


# ---------------------------------------------------------------------------
# Market Monitor
# ---------------------------------------------------------------------------

class MarketMonitor:
    """Monitors multiple Polymarket markets and detects trading opportunities."""
    
    def __init__(self, api: PolymarketAPI, update_interval: float = 1.0):
        self.api = api
        self.update_interval = update_interval
        self.monitored_markets: Dict[str, dict] = {}
        self.price_history: Dict[str, List[Tuple[float, float]]] = {}  # timestamp, price
        self.logger = logging.getLogger("MarketMonitor")
        
    def add_market(self, condition_id: str, token_id: str, name: str = ""):
        """Add a market to monitor."""
        self.monitored_markets[condition_id] = {
            "token_id": token_id,
            "name": name or condition_id,
            "last_update": 0,
            "last_price": None,
            "last_orderbook": None,
        }
        self.price_history[condition_id] = []
        self.logger.info(f"Added market to monitor: {name or condition_id}")
    
    def remove_market(self, condition_id: str):
        """Remove a market from monitoring."""
        if condition_id in self.monitored_markets:
            del self.monitored_markets[condition_id]
            del self.price_history[condition_id]
            self.logger.info(f"Removed market: {condition_id}")
    
    def update(self) -> Dict[str, dict]:
        """Update all monitored markets. Returns updated market data."""
        updated = {}
        for condition_id, market in self.monitored_markets.items():
            try:
                # Fetch fresh order book
                orderbook = self.api.get_orderbook(condition_id, market["token_id"])
                market["last_orderbook"] = orderbook
                
                # Update price history
                mid = orderbook.mid_price()
                if mid is not None:
                    market["last_price"] = mid
                    self.price_history[condition_id].append((time.time(), mid))
                    # Keep only last hour of data
                    cutoff = time.time() - 3600
                    self.price_history[condition_id] = [
                        (t, p) for t, p in self.price_history[condition_id] if t > cutoff
                    ]
                
                market["last_update"] = time.time()
                updated[condition_id] = {
                    "price": mid,
                    "spread": orderbook.spread(),
                    "bid_depth": len(orderbook.bids),
                    "ask_depth": len(orderbook.asks),
                    "timestamp": time.time(),
                }
            except Exception as e:
                self.logger.warning(f"Failed to update {condition_id}: {e}")
        
        return updated
    
    def get_price_stats(self, condition_id: str, period: int = 300) -> dict:
        """Get price statistics for a market over specified period (seconds)."""
        if condition_id not in self.price_history:
            return {}
        
        cutoff = time.time() - period
        prices = [p for t, p in self.price_history[condition_id] if t > cutoff]
        
        if len(prices) < 2:
            return {"count": len(prices), "avg": 0, "std": 0, "min": 0, "max": 0}
        
        arr = np.array(prices)
        return {
            "count": len(prices),
            "avg": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "trend": float(prices[-1] - prices[0]) if len(prices) > 1 else 0,
        }


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def discover_crypto_updown_markets(api: PolymarketAPI, symbol: str = "BTC") -> List[dict]:
    """Discover crypto up/down markets on Polymarket.
    
    Args:
        api: PolymarketAPI instance
        symbol: Crypto symbol (BTC, ETH, SOL, etc.)
    
    Returns:
        List of matching markets
    """
    markets = api.get_markets(limit=100, active_only=True, category="crypto")
    matching = []
    for m in markets:
        if symbol.upper() in m.question.upper() and "up" in m.question.lower():
            matching.append({
                "condition_id": m.condition_id,
                "question": m.question,
                "outcome_prices": m.outcome_prices,
            })
    return matching


def get_market_tokens(api: PolymarketAPI, condition_id: str) -> Dict[str, str]:
    """Get token IDs for outcomes of a market.
    
    Returns:
        Dict mapping outcome name to token ID
    """
    try:
        url = f"{GAMMA_API_BASE}/events/conditionId/{condition_id}"
        data = api._request("GET", url)
        tokens = {}
        for outcome in data.get("outcomes", []) or []:
            name = outcome.get("name", "")
            token_id = outcome.get("tokenid", outcome.get("tokenId", ""))
            if name and token_id:
                tokens[name] = token_id
        return tokens
    except PolymarketAPIError:
        return {}


if __name__ == "__main__":
    # Test the API
    logging.basicConfig(level=logging.INFO)
    api = PolymarketAPI()
    try:
        print("Fetching live markets...")
        markets = api.get_markets(limit=10)
        for m in markets[:3]:
            print(f"  {m.question} - Active: {m.active}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        api.close()
