# mt5_client.py
"""
Persistent MT5 client and bar generator.
Requires MetaTrader5 Python package installed.
Usage:
  client = MT5Client()
  client.connect()
  client.subscribe_tick("EURUSD", on_tick)
  client.start_bar_builder("EURUSD", timeframe_seconds=60, callback=on_bar)
  client.disconnect()
Note: start() runs background threads.
"""

import threading
import time
import queue
import logging
import os
from typing import Callable, Dict, Any, Optional

logger = logging.getLogger("MT5.Client")
try:
    import MetaTrader5 as mt5

    MT5_AVAIL = True
except Exception:
    MT5_AVAIL = False


class MT5Client:
    def __init__(self):
        self._running = False
        self._tick_listeners = {}
        self._tick_queue = queue.Queue()
        self._threads = []
        self._connected = False

    def connect(self):
        if not MT5_AVAIL:
            logger.warning("MetaTrader5 not installed — MT5 client disabled.")
            return False
        if self._connected:
            return True
        if not mt5.initialize():
            logger.error("MT5 initialize failed: %s", mt5.last_error())
            return False
        self._connected = True
        logger.info("MT5 initialized.")
        return True

    def disconnect(self):
        if MT5_AVAIL and self._connected:
            try:
                mt5.shutdown()
            except Exception:
                pass
            self._connected = False
            logger.info("MT5 shutdown complete.")

    def start(self):
        if self._running:
            return
        self._running = True
        t = threading.Thread(target=self._tick_poller, daemon=True)
        t.start()
        self._threads.append(t)
        t2 = threading.Thread(target=self._dispatcher, daemon=True)
        t2.start()
        self._threads.append(t2)
        logger.info("MT5 client background threads started.")

    def stop(self):
        self._running = False
        for t in self._threads:
            t.join(timeout=1.0)
        self.disconnect()

    # Poll ticks from MT5 (low-latency polling)
    def _tick_poller(self):
        while self._running:
            try:
                for sym in list(self._tick_listeners.keys()):
                    tick = mt5.symbol_info_tick(sym)
                    if tick is None:
                        continue
                    self._tick_queue.put(
                        (
                            sym,
                            {
                                "bid": float(tick.bid),
                                "ask": float(tick.ask),
                                "time": float(tick.time),
                            },
                        )
                    )
                time.sleep(0.1)
            except Exception as e:
                logger.exception("tick poller error: %s", e)
                time.sleep(1.0)

    def _dispatcher(self):
        while self._running:
            try:
                sym, data = self._tick_queue.get(timeout=1.0)
                for cb in list(self._tick_listeners.get(sym, [])):
                    try:
                        cb(sym, data)
                    except Exception:
                        logger.exception("tick listener cb error")
            except Exception:
                continue

    def subscribe_tick(self, symbol: str, callback: Callable[[str, Dict], None]):
        self._tick_listeners.setdefault(symbol, []).append(callback)

    # Simple bar builder: aggregates ticks into timeframe_seconds and calls callback on bar completion
    def start_bar_builder(
        self, symbol: str, timeframe_seconds: int, callback: Callable[[str, Dict], None]
    ):
        stop_flag = threading.Event()

        def _builder():
            bucket = []
            start_ts = None
            while not stop_flag.is_set():
                try:
                    tick = self._tick_queue.get(timeout=1.0)
                except Exception:
                    continue
                sym, data = tick
                if sym != symbol:
                    continue
                if start_ts is None:
                    start_ts = int(data["time"])
                bucket.append(data)
                if int(data["time"]) - start_ts >= timeframe_seconds:
                    # build OHLC/volume
                    bids = [x["bid"] for x in bucket]
                    asks = [x["ask"] for x in bucket]
                    price = (bids[-1] + asks[-1]) / 2.0
                    bar = {
                        "time": start_ts,
                        "open": (bids[0] + asks[0]) / 2.0,
                        "high": max((b + a) / 2.0 for b, a in zip(bids, asks)),
                        "low": min((b + a) / 2.0 for b, a in zip(bids, asks)),
                        "close": price,
                        "volume": len(bucket),
                    }
                    try:
                        callback(symbol, bar)
                    except Exception:
                        logger.exception("bar callback error")
                    bucket = []
                    start_ts = None

        t = threading.Thread(target=_builder, daemon=True)
        t.start()
        self._threads.append(t)
        return stop_flag  # caller can set() to stop

    # Execution helpers (uses arbiter instead; keep here for low-level)
    def send_order(self, request: Dict[str, Any]):
        if not MT5_AVAIL:
            raise RuntimeError("MT5 not installed")
        # ensure connected
        if not self._connected:
            self.connect()
        r = mt5.order_send(request)
        return r


if __name__ == "__main__":
    c = MT5Client()
    print("MT5 avail:", MT5_AVAIL)
