# Replace entire file: backend/services/mt5_market_data.py
# Persistent MT5 feed with health monitor + kill-on-failure
import threading
import time
import logging
import asyncio
import queue
import random
from typing import Callable, Dict, Any, List, Optional
from backend.services.ws_broadcaster import broadcast_tick, broadcast_bar
from backend.services.cpp_integration import cpp_service
from backend.core.config import get_settings

# Exported functions
__all__ = ["MT5MarketFeed", "get_mt5_market_feed", "get_mt5_market_feed_async", "mt5_market_feed", "FeedUnavailableError"]

logger = logging.getLogger("MT5.MARKET")
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError as e:
    logger.error(f"MetaTrader5 library not installed: {e}")
    MT5_AVAILABLE = False


class FeedUnavailableError(RuntimeError):
    pass


class MT5MarketFeed:
    """
    Persistent MT5 tick poller with health monitor.
    Emits ticks via subscribe_tick(callback(symbol, tick_dict)).
    If MT5 feed fails or stops, engage kill-switch via callback.
    """

    def __init__(self, poll_interval: float = 0.1, health_window: int = 10, max_queue_size: int = 10000):
        self.poll_interval = poll_interval
        self.health_window = health_window
        self._tick_listeners = {}
        self._listener_lock = threading.Lock()  # Thread safety for listeners
        self._running = False
        self._threads = []
        self._tick_q = queue.Queue(maxsize=max_queue_size)  # Bounded queue to prevent OOM
        self._last_tick_ts = time.time()
        self._health_lock = threading.Lock()
        self._connected = False
        self._connection_lock = threading.Lock()  # Thread safety for connection state
        self._kill_cb = None  # callback to call when failing
        self._symbols = set()
        self._symbols_lock = threading.Lock()  # Thread safety for symbols set
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Aggregate ticks into bars at configurable interval (default 60s)
        self.bar_interval = get_settings().mt5_bar_seconds
        self._bar_state: Dict[str, Dict[str, float]] = {}
        self._bar_state_lock = threading.Lock()  # Thread safety for bar state
        self._retry_count = 0
        self._max_retries = 3
        self._retry_delay = 1.0  # Initial retry delay in seconds
        # Reconnect backoff control (prevents rapid reconnect loops)
        self._last_connect_attempt: float = 0.0
        self._reconnect_backoff: float = 1.0
        self._reconnect_backoff_max: float = 30.0

    @property
    def running(self) -> bool:
        """Expose running state for status reporting."""
        return self._running

    def connect(self) -> bool:
        with self._connection_lock:
            if not MT5_AVAILABLE:
                logger.error("MT5 package not available. Cannot connect.")
                return False
            if self._connected:
                return True
            # Gate connection attempts by exponential backoff window
            now = time.time()
            if (now - self._last_connect_attempt) < self._reconnect_backoff:
                logger.debug(
                    "Skipping MT5 connect attempt (backoff %.1fs not elapsed)",
                    self._reconnect_backoff,
                )
                return False
            self._last_connect_attempt = now

            # Retry logic with exponential backoff
            retry_delay = self._retry_delay
            for attempt in range(self._max_retries):
                try:
                    if not mt5.initialize():
                        error = mt5.last_error()
                        logger.error(f"MT5.initialize() failed (attempt {attempt+1}/{self._max_retries}): {error}")
                        if attempt < self._max_retries - 1:
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        # Final failure on initialize: increase outer reconnect backoff and return
                        self._reconnect_backoff = min(self._reconnect_backoff * 2.0, self._reconnect_backoff_max)
                        return False
                        
                    # Validate and login with credentials
                    try:
                        s = get_settings()
                        login = s.MT5_LOGIN
                        password = s.MT5_PASSWORD
                        server = s.MT5_SERVER
                        
                        if not login or not password or not server:
                            logger.error("MT5 credentials missing. Login/Password/Server required.")
                            mt5.shutdown()
                            # Missing credentials: increase outer reconnect backoff and return
                            self._reconnect_backoff = min(self._reconnect_backoff * 2.0, self._reconnect_backoff_max)
                            return False
                            
                        login_id = int(login)
                        if not mt5.login(login=login_id, password=password, server=server):
                            error = mt5.last_error()
                            logger.error(f"MT5.login failed: {error}")
                            mt5.shutdown()
                            if attempt < self._max_retries - 1:
                                time.sleep(retry_delay)
                                retry_delay *= 2
                                continue
                            # Final failure on login: increase outer reconnect backoff and return
                            self._reconnect_backoff = min(self._reconnect_backoff * 2.0, self._reconnect_backoff_max)
                            return False
                            
                        logger.info(f"MT5 login successful to server {server} (account: {login_id})")
                        self._connected = True
                        self._retry_count = 0  # Reset retry count on success
                        # Reset reconnect backoff on successful connection
                        self._reconnect_backoff = 1.0
                        return True
                        
                    except (ValueError, TypeError) as e:
                        logger.error(f"Invalid MT5 credentials format: {e}")
                        mt5.shutdown()
                        # Increase outer reconnect backoff and return
                        self._reconnect_backoff = min(self._reconnect_backoff * 2.0, self._reconnect_backoff_max)
                        return False
                        
                except Exception as e:
                    logger.exception(f"Unexpected error during MT5 connection (attempt {attempt+1}): {e}")
                    if attempt < self._max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    # Final unexpected failure: increase outer reconnect backoff and return
                    self._reconnect_backoff = min(self._reconnect_backoff * 2.0, self._reconnect_backoff_max)
                    return False
                    
            return False

    def disconnect(self):
        if MT5_AVAILABLE and self._connected:
            try:
                mt5.shutdown()
            except Exception:
                pass
            self._connected = False
            logger.info("MT5 disconnected.")

    async def start(self):
        if self._running:
            return
        self._running = True
        # Capture the running event loop for cross-thread coroutine scheduling
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        # Preload symbols from env (comma or semicolon separated)
        try:
            for sym in get_settings().symbols_list:
                if not sym:
                    continue
                self._symbols.add(sym)
                try:
                    if MT5_AVAILABLE:
                        mt5.symbol_select(sym, True)
                except Exception:
                    logger.debug("symbol_select failed for %s", sym, exc_info=True)
        except Exception:
            logger.debug("Failed to parse ARIA_SYMBOLS", exc_info=True)

        t = threading.Thread(target=self._poll_loop, daemon=True)
        t.start()
        self._threads.append(t)
        t2 = threading.Thread(target=self._dispatch_loop, daemon=True)
        t2.start()
        self._threads.append(t2)
        t3 = threading.Thread(target=self._health_monitor_loop, daemon=True)
        t3.start()
        self._threads.append(t3)
        logger.info("MT5MarketFeed started.")

    async def stop(self):
        self._running = False
        for t in self._threads:
            try:
                t.join(timeout=0.5)
            except Exception:
                pass
        self.disconnect()

    def subscribe_tick(
        self, symbol: str, callback: Callable[[str, Dict[str, Any]], None]
    ):
        with self._listener_lock:
            self._tick_listeners.setdefault(symbol, []).append(callback)
        with self._symbols_lock:
            self._symbols.add(symbol)
        try:
            if MT5_AVAILABLE and self._connected:
                mt5.symbol_select(symbol, True)
        except Exception:
            logger.debug("mt5.symbol_select failed for %s", symbol, exc_info=True)

    def on_kill(self, cb: Callable[[], None]):
        """Register kill callback (called when feed fails)"""
        self._kill_cb = cb

    def _poll_loop(self):
        while self._running:
            try:
                if not self._connected and MT5_AVAILABLE:
                    ok = self.connect()
                    if not ok:
                        # Sleep for current backoff with small jitter to avoid thundering herd
                        delay = self._reconnect_backoff * (1.0 + random.uniform(-0.1, 0.1))
                        time.sleep(max(0.5, min(delay, self._reconnect_backoff_max)))
                        continue
                for sym in list(self._symbols):
                    try:
                        tick = mt5.symbol_info_tick(sym)
                        if tick is None:
                            continue
                        tickd = {
                            "bid": float(tick.bid),
                            "ask": float(tick.ask),
                            "time": float(tick.time),
                        }
                        # Use put_nowait to avoid blocking, drop oldest if full
                        try:
                            self._tick_q.put_nowait((sym, tickd))
                        except queue.Full:
                            # Queue is full, drop the oldest tick
                            try:
                                self._tick_q.get_nowait()
                                self._tick_q.put_nowait((sym, tickd))
                                logger.debug(f"Tick queue full, dropped oldest tick for {sym}")
                            except queue.Empty:
                                pass
                        
                        with self._health_lock:
                            self._last_tick_ts = time.time()
                    except Exception:
                        logger.exception("Tick poll error for %s", sym)
                time.sleep(self.poll_interval)
            except Exception:
                logger.exception("MT5 poll outer loop error")
                time.sleep(1.0)

    def _dispatch_loop(self):
        while self._running:
            try:
                sym, tick = self._tick_q.get(timeout=1.0)
                with self._listener_lock:
                    listeners = list(self._tick_listeners.get(sym, []))
                for cb in listeners:
                    try:
                        # Await or schedule tick callback if coroutine
if asyncio.iscoroutinefunction(cb):
    if self._loop is not None:
        asyncio.run_coroutine_threadsafe(cb(sym, tick), self._loop)
    else:
        # fallback: create a new event loop (should not happen in prod)
        asyncio.run(cb(sym, tick))
else:
    cb(sym, tick)
                    except Exception as e:
                        logger.exception(f"Tick callback error for {sym}: {e}")
                # Broadcast tick to WebSocket clients (best-effort)
                try:
                    if self._loop is not None:
                        asyncio.run_coroutine_threadsafe(
                            broadcast_tick(
                                sym,
                                float(tick.get("bid", 0.0)),
                                float(tick.get("ask", 0.0)),
                            ),
                            self._loop,
                        )
                except Exception:
                    logger.debug("broadcast_tick scheduling failed", exc_info=True)

                # Build bars and emit when the interval elapses
                try:
                    self._update_bar_state_and_maybe_emit(
                        sym,
                        float(tick.get("bid", 0.0)),
                        float(tick.get("ask", 0.0)),
                        float(tick.get("time", time.time())),
                    )
                except Exception:
                    logger.debug("bar state update failed", exc_info=True)
            except Exception:
                continue

    def _update_bar_state_and_maybe_emit(
        self, symbol: str, bid: float, ask: float, ts: float
    ):
        """Accumulate ticks into bars and emit to engine + WS when ready."""
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

        st["high"] = max(st["high"], mid)
        st["low"] = min(st["low"], mid)
        st["close"] = mid
        st["volume"] += 1.0

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

            # Emit bar to C++ SMC engine (best-effort)
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

            # Broadcast bar to WebSocket clients
            try:
                if self._loop is not None:
                    asyncio.run_coroutine_threadsafe(
                        broadcast_bar(
                            {
                                "symbol": symbol,
                                "open": open_p,
                                "high": high_p,
                                "low": low_p,
                                "close": close_p,
                                "volume": vol,
                                "timestamp": bar_ts,
                            }
                        ),
                        self._loop,
                    )
            except Exception:
                logger.debug("broadcast_bar scheduling failed", exc_info=True)

    def _health_monitor_loop(self):
        """
        If no ticks for (health_window * poll_interval) seconds, call kill callback.
        """
        while self._running:
            time.sleep(max(1.0, self.poll_interval))
            with self._health_lock:
                last = self._last_tick_ts
            # if we've never connected, continue
            if not self._connected:
                continue
            # threshold - use a more reasonable calculation
            # Allow up to 30 seconds without ticks before considering it degraded
            threshold = max(30.0, self.health_window * max(1.0, self.poll_interval))
            if time.time() - last > threshold:
                logger.critical(
                    "MT5 feed health degraded: last tick %.1f seconds ago",
                    time.time() - last,
                )
                if self._kill_cb:
                    try:
                        self._kill_cb()
                    except Exception as e:
                        logger.exception(f"Kill callback failed: {e}")
                # mark disconnected and attempt reconnect
                try:
                    self.disconnect()
                except Exception:
                    pass
                # give system time to recover; keep running for reconnect attempts
                time.sleep(5.0)

    def _to_mt5_timeframe(self, timeframe: Any) -> Optional[int]:
        """Convert timeframe string or int to MetaTrader5 timeframe constant."""
        if not MT5_AVAILABLE:
            return None
        # Accept direct int constants
        if isinstance(timeframe, int):
            return timeframe
        if isinstance(timeframe, str):
            tf_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M2": getattr(mt5, "TIMEFRAME_M2", mt5.TIMEFRAME_M1),
                "M3": getattr(mt5, "TIMEFRAME_M3", mt5.TIMEFRAME_M1),
                "M4": getattr(mt5, "TIMEFRAME_M4", mt5.TIMEFRAME_M1),
                "M5": mt5.TIMEFRAME_M5,
                "M6": getattr(mt5, "TIMEFRAME_M6", mt5.TIMEFRAME_M5),
                "M10": getattr(mt5, "TIMEFRAME_M10", mt5.TIMEFRAME_M5),
                "M12": getattr(mt5, "TIMEFRAME_M12", mt5.TIMEFRAME_M5),
                "M15": mt5.TIMEFRAME_M15,
                "M20": getattr(mt5, "TIMEFRAME_M20", mt5.TIMEFRAME_M15),
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H2": getattr(mt5, "TIMEFRAME_H2", mt5.TIMEFRAME_H1),
                "H3": getattr(mt5, "TIMEFRAME_H3", mt5.TIMEFRAME_H1),
                "H4": mt5.TIMEFRAME_H4,
                "H6": getattr(mt5, "TIMEFRAME_H6", mt5.TIMEFRAME_H4),
                "H8": getattr(mt5, "TIMEFRAME_H8", mt5.TIMEFRAME_H4),
                "H12": getattr(mt5, "TIMEFRAME_H12", mt5.TIMEFRAME_H4),
                "D1": mt5.TIMEFRAME_D1,
                "W1": mt5.TIMEFRAME_W1,
                "MN1": mt5.TIMEFRAME_MN1,
            }
            return tf_map.get(timeframe.upper())
        return None

    def get_historical_bars(
        self, symbol: str, timeframe: Any, count: int
    ) -> List[Dict[str, Any]]:
        """Fetch last `count` bars for `symbol` at given timeframe.
        Returns list of dicts with keys: time, open, high, low, close, volume.
        Safe to call when MT5 is unavailable; returns [] in that case.
        """
        try:
            if not MT5_AVAILABLE:
                logger.warning("MT5 package not available; returning empty bars list.")
                return []

            if not self._connected:
                # attempt lazy connect
                if not self.connect():
                    logger.error(
                        "MT5 not connected and connection failed; returning empty bars."
                    )
                    return []

            tf = self._to_mt5_timeframe(timeframe)
            if tf is None:
                logger.error("Unsupported timeframe: %s", timeframe)
                return []

            if count <= 0:
                return []

            rates = mt5.copy_rates_from_pos(symbol, tf, 0, int(count))
            if rates is None or len(rates) == 0:
                return []

            bars: List[Dict[str, Any]] = []
            for r in rates:
                # mt5 returns numpy structured array with fields
                try:
                    vol = float(r.get("tick_volume", r.get("volume", 0.0)))  # type: ignore[attr-defined]
                except Exception:
                    # handle as attribute access when returned as object
                    vol = float(getattr(r, "tick_volume", getattr(r, "volume", 0.0)))
                try:
                    bars.append(
                        {
                            "time": (
                                float(r["time"])
                                if isinstance(r, dict)
                                else float(getattr(r, "time"))
                            ),
                            "open": (
                                float(r["open"])
                                if isinstance(r, dict)
                                else float(getattr(r, "open"))
                            ),
                            "high": (
                                float(r["high"])
                                if isinstance(r, dict)
                                else float(getattr(r, "high"))
                            ),
                            "low": (
                                float(r["low"])
                                if isinstance(r, dict)
                                else float(getattr(r, "low"))
                            ),
                            "close": (
                                float(r["close"])
                                if isinstance(r, dict)
                                else float(getattr(r, "close"))
                            ),
                            "volume": vol,
                            "symbol": symbol,
                        }
                    )
                except Exception:
                    # Fallback for numpy structured array access
                    bars.append(
                        {
                            "time": float(r["time"]),
                            "open": float(r["open"]),
                            "high": float(r["high"]),
                            "low": float(r["low"]),
                            "close": float(r["close"]),
                            "volume": (
                                float(r["tick_volume"])
                                if "tick_volume" in r.dtype.names
                                else float(r["volume"])
                            ),
                            "symbol": symbol,
                        }
                    )

            return bars
        except Exception as e:
            logger.error(f"Error getting historical bars for {symbol}: {e}")
            return []

    def get_last_bar(self, symbol: str) -> Dict[str, Any]:
        """Get last bar for symbol - required by data source manager"""
        try:
            if not self._connected:
                raise FeedUnavailableError("MT5 not connected")

            # Get latest M1 bar
            bars = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
            if bars is None or len(bars) == 0:
                raise FeedUnavailableError(f"No bars available for {symbol}")

            bar = bars[0]
            return {
                "time": bar["time"],
                "open": bar["open"],
                "high": bar["high"],
                "low": bar["low"],
                "close": bar["close"],
                "volume": bar["tick_volume"],
                "symbol": symbol,
            }
        except Exception as e:
            logger.error(f"Error getting last bar for {symbol}: {e}")
            raise FeedUnavailableError(f"Failed to get bar for {symbol}: {e}")


# Global instance
mt5_market_feed = MT5MarketFeed()

def get_mt5_market_feed():
    """Get the global MT5 market feed instance"""
    return mt5_market_feed

async def get_mt5_market_feed_async(symbol: str, timeframe: str = "M1"):
    """
    Async generator that yields fresh market data for a symbol.
    Lazily creates/validates MT5 client on first call.
    """
    global mt5_market_feed
    
    # Ensure MT5 feed is connected
    if not mt5_market_feed._connected:
        if not mt5_market_feed.connect():
            logger.error(f"Failed to connect MT5 for {symbol}")
            return
    
    # Yield latest bar data
    try:
        bar_data = mt5_market_feed.get_last_bar(symbol)
        yield {
            "symbol": symbol,
            "time": bar_data["time"],
            "open": bar_data["open"],
            "high": bar_data["high"],
            "low": bar_data["low"],
            "close": bar_data["close"],
            "volume": bar_data["volume"]
        }
    except Exception as e:
        logger.error(f"Error getting market data for {symbol}: {e}")
        return
    
    # Allow other coroutines to run
    await asyncio.sleep(0)
