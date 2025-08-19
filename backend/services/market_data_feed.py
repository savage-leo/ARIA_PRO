"""
Market Data Feed Service
Simulates live market data and broadcasts via WebSocket.
Now also feeds ticks and aggregated bars into the C++/Python SMC engine
via `backend.services.cpp_integration.cpp_service` so signals can be
generated and broadcast in real time.
"""

import asyncio
import random
import time
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
from backend.services.ws_broadcaster import broadcast_tick, broadcast_bar
from backend.services.cpp_integration import cpp_service

logger = logging.getLogger(__name__)


class MarketDataFeed:
    def __init__(self):
        self.running = False
        self.symbols = [
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "USDCHF",
            "AUDUSD",
            "USDCAD",
            "NZDUSD",
            "EURGBP",
        ]

        # Base prices for realistic simulation
        self.base_prices = {
            "EURUSD": 1.0850,
            "GBPUSD": 1.2650,
            "USDJPY": 148.50,
            "USDCHF": 0.8850,
            "AUDUSD": 0.6650,
            "USDCAD": 1.3550,
            "NZDUSD": 0.6150,
            "EURGBP": 0.8580,
        }

        # Current prices (will be updated)
        self.current_prices = self.base_prices.copy()

        # Simple in-process bar builder state per symbol
        # Aggregates ticks into fixed-size bars (default 5 seconds)
        self.bar_interval = int(os.getenv("ARIA_FEED_BAR_SECONDS", "5"))
        self._bar_state: Dict[str, Dict[str, float]] = {}

    async def start(self):
        """Start the market data feed"""
        if self.running:
            logger.warning("Market data feed is already running")
            return

        self.running = True
        logger.info("Starting market data feed...")

        try:
            await self._run_feed()
        except Exception as e:
            logger.error(f"Error in market data feed: {e}")
            self.running = False

    async def stop(self):
        """Stop the market data feed"""
        self.running = False
        logger.info("Stopping market data feed...")

    async def _run_feed(self):
        """Main feed loop"""
        while self.running:
            try:
                # Generate ticks for all symbols
                for symbol in self.symbols:
                    await self._generate_tick(symbol)

                # Wait before next update
                await asyncio.sleep(1)  # 1 second between updates

            except Exception as e:
                logger.error(f"Error generating tick data: {e}")
                await asyncio.sleep(5)  # Wait longer on error

    async def _generate_tick(self, symbol: str):
        """Generate a single tick for a symbol"""
        try:
            # Get current price
            current_price = self.current_prices[symbol]

            # Generate realistic price movement
            # Small random walk with mean reversion
            movement = random.uniform(-0.0005, 0.0005)  # Max 5 pips movement

            # Add some volatility based on symbol
            volatility_multiplier = {
                "USDJPY": 0.1,  # Lower volatility (pips are smaller)
                "EURUSD": 1.0,  # Standard volatility
                "GBPUSD": 1.2,  # Higher volatility
            }.get(symbol, 1.0)

            movement *= volatility_multiplier

            # Update price
            new_price = current_price + movement

            # Generate bid/ask with realistic spread
            spread = random.uniform(0.0001, 0.0003)  # 1-3 pips spread
            bid = new_price - (spread / 2)
            ask = new_price + (spread / 2)

            # Update current price
            self.current_prices[symbol] = new_price

            # Broadcast the tick
            await broadcast_tick(symbol, bid, ask)

            # Feed tick into C++/Python SMC engine
            ts = time.time()
            try:
                cpp_service.process_tick_data(
                    symbol=symbol,
                    bid=bid,
                    ask=ask,
                    volume=1.0,  # per-tick unit volume for aggregation
                    timestamp=ts,
                )
            except Exception:
                logger.debug("cpp_service.process_tick_data failed", exc_info=True)

            # Update in-process bar state and emit bar if interval elapsed
            try:
                self._update_bar_state_and_maybe_emit(symbol, bid, ask, ts)
            except Exception:
                logger.debug("bar state update failed", exc_info=True)

            logger.debug(f"Tick: {symbol} {bid:.5f}/{ask:.5f}")

        except Exception as e:
            logger.error(f"Error generating tick for {symbol}: {e}")

    def _update_bar_state_and_maybe_emit(
        self, symbol: str, bid: float, ask: float, ts: float
    ):
        """Accumulate ticks into a fixed-size bar and emit to engine + WS when ready."""
        mid = (bid + ask) / 2.0
        st = self._bar_state.get(symbol)
        if not st:
            st = {
                "start_ts": int(ts),
                "open": mid,
                "high": mid,
                "low": mid,
                "close": mid,
                "volume": 0.0,
            }
            self._bar_state[symbol] = st

        # update state
        st["high"] = max(st["high"], mid)
        st["low"] = min(st["low"], mid)
        st["close"] = mid
        st["volume"] += 1.0

        # check interval boundary
        if int(ts) - st["start_ts"] >= self.bar_interval:
            bar_ts = float(st["start_ts"])
            open_p = float(st["open"])
            high_p = float(st["high"])
            low_p = float(st["low"])
            close_p = float(st["close"])
            vol = float(st["volume"])

            # reset for next bar starting now
            self._bar_state[symbol] = {
                "start_ts": int(ts),
                "open": mid,
                "high": mid,
                "low": mid,
                "close": mid,
                "volume": 0.0,
            }

            # emit bar to engine and websocket
            try:
                cpp_service.process_bar_data(
                    symbol=symbol,
                    open_price=open_p,
                    high=high_p,
                    low=low_p,
                    close=close_p,
                    volume=vol,
                    timestamp=bar_ts,
                )
            except Exception:
                logger.debug("cpp_service.process_bar_data failed", exc_info=True)

            try:
                awaitable = broadcast_bar(
                    {
                        "symbol": symbol,
                        "open": open_p,
                        "high": high_p,
                        "low": low_p,
                        "close": close_p,
                        "volume": vol,
                        "timestamp": bar_ts,
                    }
                )
                # we're in async context caller; ensure schedule if not awaited there
                if asyncio.iscoroutine(awaitable):
                    # fire-and-forget, don't block feed loop
                    asyncio.create_task(awaitable)
            except Exception:
                logger.debug("broadcast_bar failed", exc_info=True)

    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current price for a symbol"""
        if symbol not in self.current_prices:
            return None

        price = self.current_prices[symbol]
        spread = random.uniform(0.0001, 0.0003)
        return {
            "bid": price - (spread / 2),
            "ask": price + (spread / 2),
            "spread": spread,
        }


# Global instance
market_data_feed = MarketDataFeed()
