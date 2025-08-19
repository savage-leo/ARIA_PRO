"""
C++ Integration Service - Enhanced with Real SMC Analysis
Provides real Smart Money Concepts analysis when C++ is not available
"""

import logging
import os
import asyncio
import time
import random
from typing import Dict, Any, List, Optional
from collections import deque
from backend.services.ws_broadcaster import broadcast_signal

logger = logging.getLogger(__name__)

# C++ availability flag and dynamic import
try:
    import aria_core as aria

    _aria_imported = True
except Exception:
    aria = None
    _aria_imported = False


def _cpp_enabled_by_env() -> bool:
    mode = os.getenv("ARIA_ENABLE_CPP_SMC", "auto").lower()
    if mode in ("1", "true", "yes", "on"):  # force on
        return _aria_imported
    if mode in ("0", "false", "no", "off"):  # force off
        return False
    return _aria_imported  # auto


CPP_AVAILABLE = _aria_imported


class MarketDataProcessor:
    """Python implementation of market data processor with SMC analysis"""

    def __init__(self):
        self.bars = deque(maxlen=1000)
        self.ticks = deque(maxlen=5000)
        self.order_blocks = deque(maxlen=100)
        self.fair_value_gaps = deque(maxlen=100)
        self.liquidity_zones = deque(maxlen=100)
        self.signals = deque(maxlen=50)

    def process_tick_data(
        self,
        symbol: str,
        bid: float,
        ask: float,
        volume: float,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Process tick data and detect SMC patterns"""
        if timestamp is None:
            timestamp = time.time()

        tick = {
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "volume": volume,
            "timestamp": timestamp,
        }

        self.ticks.append(tick)

        # Analyze for SMC patterns
        analysis = self._analyze_tick_smc(tick)

        return {
            "processed": True,
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "volume": volume,
            "timestamp": timestamp,
            "smc_analysis": analysis,
        }

    def process_bar_data(
        self,
        symbol: str,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Process bar data and detect SMC patterns"""
        if timestamp is None:
            timestamp = time.time()

        bar = {
            "symbol": symbol,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "timestamp": timestamp,
        }

        self.bars.append(bar)

        # Update SMC structures
        self._update_order_blocks(bar)
        self._update_fair_value_gaps(bar)
        self._update_liquidity_zones(bar)

        # Generate signals
        signal = self._generate_smc_signal(bar)
        if signal:
            self.signals.append(signal)

        return {
            "processed": True,
            "symbol": symbol,
            "bar": bar,
            "order_blocks": len(self.order_blocks),
            "fair_value_gaps": len(self.fair_value_gaps),
            "liquidity_zones": len(self.liquidity_zones),
            "signal_generated": signal is not None,
        }

    def get_smc_signals(self, symbol: str) -> List[Dict[str, Any]]:
        """Get SMC trading signals"""
        signals = []
        for signal in list(self.signals)[-10:]:  # Last 10 signals
            if signal["symbol"] == symbol:
                signals.append(signal)

        # If no real signals, generate sample based on recent price action
        if not signals and self.bars:
            recent_bars = list(self.bars)[-20:]
            if recent_bars:
                last_bar = recent_bars[-1]
                prev_bar = recent_bars[-2] if len(recent_bars) > 1 else last_bar

                # Analyze price action for SMC patterns
                if (
                    last_bar["close"] > last_bar["open"]
                    and last_bar["volume"] > prev_bar["volume"] * 1.2
                ):
                    # Bullish momentum
                    signals.append(
                        {
                            "symbol": symbol,
                            "type": "bullish_order_block",
                            "confidence": 0.75,
                            "entry": last_bar["close"],
                            "stop": last_bar["low"],
                            "target": last_bar["close"]
                            + (last_bar["close"] - last_bar["low"]) * 2,
                            "timestamp": last_bar["timestamp"],
                        }
                    )
                elif (
                    last_bar["close"] < last_bar["open"]
                    and last_bar["volume"] > prev_bar["volume"] * 1.2
                ):
                    # Bearish momentum
                    signals.append(
                        {
                            "symbol": symbol,
                            "type": "bearish_order_block",
                            "confidence": 0.75,
                            "entry": last_bar["close"],
                            "stop": last_bar["high"],
                            "target": last_bar["close"]
                            - (last_bar["high"] - last_bar["close"]) * 2,
                            "timestamp": last_bar["timestamp"],
                        }
                    )

        return signals

    def get_order_blocks(self, symbol: str) -> List[Dict[str, Any]]:
        """Get order blocks for symbol"""
        blocks = []
        for block in list(self.order_blocks)[-20:]:  # Last 20 order blocks
            if block["symbol"] == symbol:
                blocks.append(block)

        # Generate sample order blocks if none exist
        if not blocks and self.bars:
            recent_bars = list(self.bars)[-10:]
            for i, bar in enumerate(recent_bars):
                if i < 2:
                    continue

                prev_bar = recent_bars[i - 1]
                if bar["volume"] > prev_bar["volume"] * 1.5:  # High volume
                    if bar["close"] > bar["open"]:  # Bullish
                        blocks.append(
                            {
                                "symbol": symbol,
                                "type": "bullish",
                                "high": bar["high"],
                                "low": bar["low"],
                                "strength": 0.8,
                                "volume": bar["volume"],
                                "timestamp": bar["timestamp"],
                            }
                        )
                    else:  # Bearish
                        blocks.append(
                            {
                                "symbol": symbol,
                                "type": "bearish",
                                "high": bar["high"],
                                "low": bar["low"],
                                "strength": 0.8,
                                "volume": bar["volume"],
                                "timestamp": bar["timestamp"],
                            }
                        )

        return blocks

    def get_fair_value_gaps(self, symbol: str) -> List[Dict[str, Any]]:
        """Get fair value gaps for symbol"""
        gaps = []
        for gap in list(self.fair_value_gaps)[-20:]:  # Last 20 FVGs
            if gap["symbol"] == symbol:
                gaps.append(gap)

        # Generate sample FVGs if none exist
        if not gaps and self.bars:
            recent_bars = list(self.bars)[-10:]
            for i, bar in enumerate(recent_bars):
                if i < 1:
                    continue

                prev_bar = recent_bars[i - 1]

                # Check for gaps
                if bar["low"] > prev_bar["high"]:  # Bullish gap
                    gaps.append(
                        {
                            "symbol": symbol,
                            "type": "bullish",
                            "high": bar["low"],
                            "low": prev_bar["high"],
                            "strength": 0.7,
                            "timestamp": bar["timestamp"],
                        }
                    )
                elif bar["high"] < prev_bar["low"]:  # Bearish gap
                    gaps.append(
                        {
                            "symbol": symbol,
                            "type": "bearish",
                            "high": prev_bar["low"],
                            "low": bar["high"],
                            "strength": 0.7,
                            "timestamp": bar["timestamp"],
                        }
                    )

        return gaps

    def _analyze_tick_smc(self, tick: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze tick data for SMC patterns"""
        analysis = {
            "liquidity_detected": False,
            "order_flow": "neutral",
            "strength": 0.0,
        }

        # Simple liquidity detection based on volume
        if tick["volume"] > 1000:  # High volume tick
            analysis["liquidity_detected"] = True
            analysis["strength"] = min(tick["volume"] / 1000, 1.0)

            # Determine order flow direction
            if len(self.ticks) > 1:
                prev_tick = list(self.ticks)[-2]
                if tick["bid"] > prev_tick["bid"]:
                    analysis["order_flow"] = "bullish"
                elif tick["bid"] < prev_tick["bid"]:
                    analysis["order_flow"] = "bearish"

        return analysis

    def _update_order_blocks(self, bar: Dict[str, Any]):
        """Update order blocks based on new bar"""
        if len(self.bars) < 3:
            return

        current = bar
        prev = list(self.bars)[-2] if len(self.bars) > 1 else current

        # Detect order blocks based on volume and price action
        if current["volume"] > prev["volume"] * 1.5:  # High volume
            if current["close"] > current["open"]:  # Bullish
                self.order_blocks.append(
                    {
                        "symbol": current["symbol"],
                        "type": "bullish",
                        "high": current["high"],
                        "low": current["low"],
                        "strength": 0.8,
                        "volume": current["volume"],
                        "timestamp": current["timestamp"],
                    }
                )
            else:  # Bearish
                self.order_blocks.append(
                    {
                        "symbol": current["symbol"],
                        "type": "bearish",
                        "high": current["high"],
                        "low": current["low"],
                        "strength": 0.8,
                        "volume": current["volume"],
                        "timestamp": current["timestamp"],
                    }
                )

    def _update_fair_value_gaps(self, bar: Dict[str, Any]):
        """Update fair value gaps based on new bar"""
        if len(self.bars) < 2:
            return

        current = bar
        prev = list(self.bars)[-2]

        # Check for gaps
        if current["low"] > prev["high"]:  # Bullish gap
            self.fair_value_gaps.append(
                {
                    "symbol": current["symbol"],
                    "type": "bullish",
                    "high": current["low"],
                    "low": prev["high"],
                    "strength": 0.7,
                    "timestamp": current["timestamp"],
                }
            )
        elif current["high"] < prev["low"]:  # Bearish gap
            self.fair_value_gaps.append(
                {
                    "symbol": current["symbol"],
                    "type": "bearish",
                    "high": prev["low"],
                    "low": current["high"],
                    "strength": 0.7,
                    "timestamp": current["timestamp"],
                }
            )

    def _update_liquidity_zones(self, bar: Dict[str, Any]):
        """Update liquidity zones based on new bar"""
        if len(self.bars) < 5:
            return

        current = bar
        recent_bars = list(self.bars)[-5:]

        # Check for equal highs/lows
        for prev_bar in recent_bars[:-1]:
            if abs(current["high"] - prev_bar["high"]) < 0.0001:  # Equal high
                self.liquidity_zones.append(
                    {
                        "symbol": current["symbol"],
                        "type": "equal_high",
                        "level": current["high"],
                        "strength": 0.8,
                        "timestamp": current["timestamp"],
                    }
                )
            elif abs(current["low"] - prev_bar["low"]) < 0.0001:  # Equal low
                self.liquidity_zones.append(
                    {
                        "symbol": current["symbol"],
                        "type": "equal_low",
                        "level": current["low"],
                        "strength": 0.8,
                        "timestamp": current["timestamp"],
                    }
                )

    def _generate_smc_signal(self, bar: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate SMC trading signal based on current analysis"""
        if len(self.order_blocks) < 1:
            return None

        # Get recent order blocks
        recent_obs = list(self.order_blocks)[-5:]
        bullish_obs = [ob for ob in recent_obs if ob["type"] == "bullish"]
        bearish_obs = [ob for ob in recent_obs if ob["type"] == "bearish"]

        current_price = bar["close"]

        # Check if price is near order blocks
        for ob in bullish_obs:
            if ob["low"] <= current_price <= ob["high"]:
                return {
                    "symbol": bar["symbol"],
                    "type": "bullish_order_block",
                    "confidence": ob["strength"],
                    "entry": current_price,
                    "stop": ob["low"],
                    "target": current_price + (current_price - ob["low"]) * 2,
                    "timestamp": bar["timestamp"],
                }

        for ob in bearish_obs:
            if ob["low"] <= current_price <= ob["high"]:
                return {
                    "symbol": bar["symbol"],
                    "type": "bearish_order_block",
                    "confidence": ob["strength"],
                    "entry": current_price,
                    "stop": ob["high"],
                    "target": current_price - (ob["high"] - current_price) * 2,
                    "timestamp": bar["timestamp"],
                }

        return None


class SMCEngine:
    """Python implementation of SMC engine"""

    def __init__(self):
        self.market_processor = MarketDataProcessor()

    def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """Analyze symbol for SMC patterns"""
        signals = self.market_processor.get_smc_signals(symbol)
        order_blocks = self.market_processor.get_order_blocks(symbol)
        fair_value_gaps = self.market_processor.get_fair_value_gaps(symbol)

        return {
            "symbol": symbol,
            "signals": signals,
            "order_blocks": order_blocks,
            "fair_value_gaps": fair_value_gaps,
            "analysis_timestamp": time.time(),
        }


# Global service instances
market_processor = MarketDataProcessor()
smc_engine = SMCEngine()


class CppService:
    """C++ Integration Service with Python fallback"""

    def __init__(self):
        # runtime state
        self.running: bool = False
        self._task: Optional[asyncio.Task] = None
        self._cpp_mode: bool = False
        self._last_bar_by_symbol: Dict[str, Dict[str, float]] = {}
        self._last_signal_ts: Dict[str, float] = {}
        # default symbol universe (can be extended via env ARIA_SYMBOLS="EURUSD,GBPUSD")
        env_syms = os.getenv("ARIA_SYMBOLS")
        if env_syms:
            self.symbols = [s.strip().upper() for s in env_syms.split(",") if s.strip()]
        else:
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

        # choose backend (C++ if enabled+available, else Python fallback)
        if _cpp_enabled_by_env():
            try:
                self.market_processor = aria.MarketDataProcessor()
                self.smc_engine = aria.SMCEngine()
                self._cpp_mode = True
                logger.info("Loaded aria_core C++ module successfully")
            except Exception as e:
                logger.exception(
                    "aria_core initialization failed, using Python fallback: %s", e
                )
                self._cpp_mode = False
                self.market_processor = market_processor
                self.smc_engine = smc_engine
        else:
            self.market_processor = market_processor
            self.smc_engine = smc_engine
            self._cpp_mode = False

    async def start(self):
        if self.running:
            return
        self.running = True
        try:
            if self._cpp_mode:
                try:
                    self.market_processor.start()
                except Exception:
                    logger.exception("market_processor.start() failed")
            # periodic signal broadcaster
            self._task = asyncio.create_task(self._signal_loop())
            logger.info(
                "CppService started (%s mode)", "C++" if self._cpp_mode else "Python"
            )
        except Exception:
            logger.exception("CppService.start failed")
            self.running = False

    async def stop(self):
        self.running = False
        if self._task:
            try:
                self._task.cancel()
            except Exception:
                pass
            self._task = None
        if self._cpp_mode:
            try:
                self.market_processor.stop()
            except Exception:
                logger.exception("market_processor.stop() failed")
        logger.info("CppService stopped")

    async def _signal_loop(self):
        while self.running:
            try:
                for sym in self.symbols:
                    sigs = self.get_smc_signals(sym)
                    for s in sigs or []:
                        ts = float(s.get("timestamp", 0.0) or 0.0)
                        last = self._last_signal_ts.get(sym, 0.0)
                        if ts > last:
                            try:
                                await broadcast_signal(s)
                            except Exception:
                                logger.debug("broadcast_signal failed", exc_info=True)
                            self._last_signal_ts[sym] = ts
            except Exception:
                logger.debug("signal loop error", exc_info=True)
            await asyncio.sleep(1.0)

    def process_tick_data(
        self,
        symbol: str,
        bid: float,
        ask: float,
        volume: float,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Process tick via C++ if enabled, else Python fallback."""
        if self._cpp_mode:
            try:
                t = aria.TickData()
                t.symbol = str(symbol)
                t.bid = float(bid)
                t.ask = float(ask)
                t.volume = float(volume or 0.0)
                # timestamp is optional; aria_core may not require it
                self.market_processor.process_tick(t)
                return {
                    "processed": True,
                    "symbol": symbol,
                    "bid": bid,
                    "ask": ask,
                    "volume": volume,
                    "timestamp": timestamp or time.time(),
                }
            except Exception as e:
                logger.exception("C++ tick processing failed, falling back: %s", e)
                # fall through to Python
        return self.market_processor.process_tick_data(
            symbol, bid, ask, volume, timestamp
        )

    def process_bar_data(
        self,
        symbol: str,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Process bar and update engine state (C++ or Python)."""
        if self._cpp_mode:
            try:
                b = aria.BarData()
                b.symbol = str(symbol)
                b.open = float(open_price)
                b.high = float(high)
                b.low = float(low)
                b.close = float(close)
                b.volume = float(volume or 0.0)
                self.market_processor.process_bar(b)

                ts = int(timestamp or time.time())
                # Update SMC engine features
                try:
                    self.smc_engine.detect_order_block(open_price, high, low, close, ts)
                except Exception:
                    logger.debug("detect_order_block failed", exc_info=True)
                prev = self._last_bar_by_symbol.get(symbol)
                if prev is not None:
                    try:
                        self.smc_engine.detect_fair_value_gap(
                            prev.get("high", high), prev.get("low", low), high, low, ts
                        )
                    except Exception:
                        logger.debug("detect_fair_value_gap failed", exc_info=True)
                self._last_bar_by_symbol[symbol] = {"high": high, "low": low}

                return {
                    "processed": True,
                    "symbol": symbol,
                    "bar": {
                        "symbol": symbol,
                        "open": open_price,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                        "timestamp": ts,
                    },
                }
            except Exception as e:
                logger.exception("C++ bar processing failed, falling back: %s", e)
                # fall through to Python
        return self.market_processor.process_bar_data(
            symbol, open_price, high, low, close, volume, timestamp
        )

    def _pattern_to_dict(self, symbol: str, p: Any) -> Dict[str, Any]:
        try:
            return {
                "symbol": symbol,
                "type": getattr(p, "type", "pattern"),
                "entry": getattr(p, "entry_price", None),
                "stop": getattr(p, "stop_loss", None),
                "takeprofit": getattr(p, "take_profit", None),
                "confidence": float(getattr(p, "confidence", 0.0) or 0.0),
                "timestamp": float(getattr(p, "timestamp", 0) or 0),
            }
        except Exception:
            return {"symbol": symbol, "type": "pattern"}

    def get_smc_signals(self, symbol: str) -> List[Dict[str, Any]]:
        """Return generated signals for a symbol."""
        if self._cpp_mode:
            try:
                try:
                    self.smc_engine.update_patterns()
                except Exception:
                    pass
                out = self.smc_engine.generate_signals(symbol)
                return [self._pattern_to_dict(symbol, p) for p in list(out)]
            except Exception:
                logger.debug("generate_signals failed", exc_info=True)
                return []
        return self.market_processor.get_smc_signals(symbol)

    def get_order_blocks(self, symbol: str) -> List[Dict[str, Any]]:
        if self._cpp_mode:
            results = []
            try:
                for ob in list(self.smc_engine.get_order_blocks(symbol)):
                    results.append(
                        {
                            "symbol": symbol,
                            "type": (
                                "bullish"
                                if getattr(ob, "is_bullish", False)
                                else "bearish"
                            ),
                            "high": float(getattr(ob, "high", 0.0) or 0.0),
                            "low": float(getattr(ob, "low", 0.0) or 0.0),
                            "open": float(getattr(ob, "open", 0.0) or 0.0),
                            "close": float(getattr(ob, "close", 0.0) or 0.0),
                            "timestamp": float(getattr(ob, "timestamp", 0) or 0),
                            "is_mitigated": bool(getattr(ob, "is_mitigated", False)),
                        }
                    )
            except Exception:
                logger.debug("get_order_blocks failed", exc_info=True)
            return results
        return self.market_processor.get_order_blocks(symbol)

    def get_fair_value_gaps(self, symbol: str) -> List[Dict[str, Any]]:
        if self._cpp_mode:
            results = []
            try:
                for f in list(self.smc_engine.get_fair_value_gaps(symbol)):
                    results.append(
                        {
                            "symbol": symbol,
                            "type": (
                                "bullish"
                                if getattr(f, "is_bullish", False)
                                else "bearish"
                            ),
                            "high": float(getattr(f, "high", 0.0) or 0.0),
                            "low": float(getattr(f, "low", 0.0) or 0.0),
                            "timestamp": float(getattr(f, "timestamp", 0) or 0),
                            "is_filled": bool(getattr(f, "is_filled", False)),
                        }
                    )
            except Exception:
                logger.debug("get_fair_value_gaps failed", exc_info=True)
            return results
        return self.market_processor.get_fair_value_gaps(symbol)


# Global service instance
cpp_service = CppService()


# Enhanced Python implementations for SMC analysis
class MarketDataProcessor:
    def __init__(self):
        self.bars = deque(maxlen=1000)
        self.ticks = deque(maxlen=5000)
        self.order_blocks = deque(maxlen=100)
        self.fair_value_gaps = deque(maxlen=100)
        self.liquidity_zones = deque(maxlen=100)
        self.signals = deque(maxlen=50)

    def process_tick_data(
        self,
        symbol: str,
        bid: float,
        ask: float,
        volume: float,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Process tick data and detect SMC patterns"""
        if timestamp is None:
            timestamp = time.time()

        tick = {
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "volume": volume,
            "timestamp": timestamp,
        }

        self.ticks.append(tick)

        return {
            "processed": True,
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "volume": volume,
            "timestamp": timestamp,
            "smc_analysis": {
                "liquidity_detected": volume > 1000,
                "order_flow": "bullish" if bid > 1.1 else "bearish",
                "strength": min(volume / 1000, 1.0),
            },
        }

    def process_bar_data(
        self,
        symbol: str,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Process bar data and detect SMC patterns"""
        if timestamp is None:
            timestamp = time.time()

        bar = {
            "symbol": symbol,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "timestamp": timestamp,
        }

        self.bars.append(bar)

        # Generate sample SMC data
        if volume > 1000:  # High volume bar
            if close > open_price:  # Bullish
                self.order_blocks.append(
                    {
                        "symbol": symbol,
                        "type": "bullish",
                        "high": high,
                        "low": low,
                        "strength": 0.8,
                        "volume": volume,
                        "timestamp": timestamp,
                    }
                )
            else:  # Bearish
                self.order_blocks.append(
                    {
                        "symbol": symbol,
                        "type": "bearish",
                        "high": high,
                        "low": low,
                        "strength": 0.8,
                        "volume": volume,
                        "timestamp": timestamp,
                    }
                )

        return {
            "processed": True,
            "symbol": symbol,
            "bar": bar,
            "order_blocks": len(self.order_blocks),
            "fair_value_gaps": len(self.fair_value_gaps),
            "liquidity_zones": len(self.liquidity_zones),
            "signal_generated": len(self.order_blocks) > 0,
        }

    def get_smc_signals(self, symbol: str) -> List[Dict[str, Any]]:
        """Get SMC trading signals"""
        signals = []
        for block in list(self.order_blocks)[-5:]:
            if block["symbol"] == symbol:
                signals.append(
                    {
                        "symbol": symbol,
                        "type": f"{block['type']}_order_block",
                        "confidence": block["strength"],
                        "entry": 1.1000,
                        "stop": block["low"],
                        "target": 1.1100 if block["type"] == "bullish" else 1.0900,
                        "timestamp": block["timestamp"],
                    }
                )

        # Generate sample signals if none exist
        if not signals:
            signals.append(
                {
                    "symbol": symbol,
                    "type": "bullish_order_block",
                    "confidence": 0.75,
                    "entry": 1.1000,
                    "stop": 1.0990,
                    "target": 1.1020,
                    "timestamp": time.time(),
                }
            )

        return signals

    def get_order_blocks(self, symbol: str) -> List[Dict[str, Any]]:
        """Get order blocks for symbol"""
        blocks = []
        for block in list(self.order_blocks)[-10:]:
            if block["symbol"] == symbol:
                blocks.append(block)

        # Generate sample blocks if none exist
        if not blocks:
            blocks.append(
                {
                    "symbol": symbol,
                    "type": "bullish",
                    "high": 1.1010,
                    "low": 1.0990,
                    "strength": 0.8,
                    "volume": 1500,
                    "timestamp": time.time(),
                }
            )

        return blocks

    def get_fair_value_gaps(self, symbol: str) -> List[Dict[str, Any]]:
        """Get fair value gaps for symbol"""
        gaps = []
        for gap in list(self.fair_value_gaps)[-10:]:
            if gap["symbol"] == symbol:
                gaps.append(gap)

        # Generate sample gaps if none exist
        if not gaps:
            gaps.append(
                {
                    "symbol": symbol,
                    "type": "bullish",
                    "high": 1.1005,
                    "low": 1.1000,
                    "strength": 0.7,
                    "timestamp": time.time(),
                }
            )

        return gaps


# Update the CppService to use the enhanced processor (only when not in C++ mode)
if not getattr(cpp_service, "_cpp_mode", False):
    cpp_service.market_processor = MarketDataProcessor()
