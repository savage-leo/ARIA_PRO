"""
Phase 3 Orchestrator — wires EnhancedSMCFusionCore (phase3) to Phase4 models + MT5 + Arbiter
Drop-in runner that:
 - starts MT5 client (MT5Client)
 - builds 1-minute bars per symbol
 - calls ai_signal_generator.get_signals()
 - passes signals + feats into EnhancedSMCFusionCore.ingest_bar()
 - routes execution through the core's execute_trade()
"""

import os
import time
import threading
import logging
import requests
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, Any, List

# Local imports (assumes Phase4 & your Enhanced core exist)
from backend.smc.smc_fusion_core import EnhancedFusionConfig, EnhancedSMCFusionCore
from backend.services.mt5_client import MT5Client
from backend.services.ai_signal_generator import ai_signal_generator
from backend.services.exec_arbiter import TradeArbiter
from backend.core.risk_engine_enhanced import RiskEngine

logger = logging.getLogger("ARIA.PHASE3")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
ch.setFormatter(fmt)
logger.addHandler(ch)

# Debug API integration
# Allow override via env and optional admin key
DEBUG_API_BASE = os.environ.get("ARIA_DEBUG_API_BASE", "http://localhost:8000/debug")
DEBUG_API_ADMIN_KEY = os.environ.get("ARIA_ADMIN_KEY") or os.environ.get(
    "ADMIN_API_KEY", ""
)


def publish_idea_to_debug_api(symbol: str, idea: Any):
    """Publish latest idea to debug API"""
    try:
        if hasattr(idea, "as_dict"):
            idea_data = idea.as_dict()
        else:
            idea_data = idea

        headers = {"X-ARIA-ADMIN": DEBUG_API_ADMIN_KEY} if DEBUG_API_ADMIN_KEY else None
        response = requests.post(
            f"{DEBUG_API_BASE}/idea/{symbol}",
            json=idea_data,
            headers=headers,
            timeout=1.0,
        )
        if response.status_code == 200:
            logger.debug(f"Published idea for {symbol} to debug API")
    except Exception as e:
        logger.debug(f"Failed to publish idea to debug API: {e}")


def publish_orchestrator_status(status: Dict[str, Any]):
    """Publish orchestrator status to debug API"""
    try:
        headers = {"X-ARIA-ADMIN": DEBUG_API_ADMIN_KEY} if DEBUG_API_ADMIN_KEY else None
        response = requests.post(
            f"{DEBUG_API_BASE}/orchestrator/status",
            json=status,
            headers=headers,
            timeout=1.0,
        )
        if response.status_code == 200:
            logger.debug("Published orchestrator status to debug API")
    except Exception as e:
        logger.debug(f"Failed to publish status to debug API: {e}")


class BarBuilder:
    """
    Simple per-symbol bar builder that aggregates MT5 tick dicts into timeframe bars.
    timeframe_seconds: bucket length (default 60 = 1m)
    Emits bar dicts: {'o','h','l','c','v','ts'}
    """

    def __init__(self, timeframe_seconds: int = 60):
        self.tf = timeframe_seconds
        self.buckets = {}  # symbol -> {start_ts, open, high, low, close, vol}
        self.lock = threading.Lock()
        self.listeners = []  # call back: (symbol, bar)

    def attach_listener(self, cb):
        self.listeners.append(cb)

    def on_tick(self, symbol: str, tick: Dict[str, Any]):
        # tick: {"bid","ask","time"}; use mid price for OHLC
        price = (float(tick.get("bid", 0.0)) + float(tick.get("ask", 0.0))) / 2.0
        ts = int(float(tick.get("time", time.time())))
        bucket_start = (ts // self.tf) * self.tf
        with self.lock:
            b = self.buckets.get(symbol)
            if b is None or b["start"] != bucket_start:
                # finalize previous bucket if exists
                if b is not None:
                    bar = {
                        "o": b["open"],
                        "h": b["high"],
                        "l": b["low"],
                        "c": b["last"],
                        "v": b["vol"],
                        "ts": b["start"],
                    }
                    self._emit_bar(symbol, bar)
                # start new bucket
                self.buckets[symbol] = {
                    "start": bucket_start,
                    "open": price,
                    "high": price,
                    "low": price,
                    "last": price,
                    "vol": 1,
                }
            else:
                b["high"] = max(b["high"], price)
                b["low"] = min(b["low"], price)
                b["last"] = price
                b["vol"] += 1

    def _emit_bar(self, symbol: str, bar: Dict[str, Any]):
        for cb in self.listeners:
            try:
                cb(symbol, bar)
            except Exception:
                logger.exception("bar listener failed")

    def flush_all(self):
        with self.lock:
            for symbol, b in list(self.buckets.items()):
                bar = {
                    "o": b["open"],
                    "h": b["high"],
                    "l": b["low"],
                    "c": b["last"],
                    "v": b["vol"],
                    "ts": b["start"],
                }
                self._emit_bar(symbol, bar)
            self.buckets.clear()


class Phase3Orchestrator:
    def __init__(self, symbols: List[str], timeframe: int = 60):
        self.symbols = symbols
        self.tf = timeframe
        self.mt5 = MT5Client()
        self.bar_builder = BarBuilder(timeframe_seconds=self.tf)
        self.cores: Dict[str, EnhancedSMCFusionCore] = {}
        self.running = False
        self.health = {"last_tick_ts": time.time(), "ok": True}
        self._setup_cores_and_components()

    def _setup_cores_and_components(self):
        # create engine per symbol
        for s in self.symbols:
            cfg = EnhancedFusionConfig()
            core = EnhancedSMCFusionCore(
                s, cfg, ["lstm", "cnn", "ppo", "vision", "llm_macro"]
            )
            self.cores[s] = core
        # attach bar builder listener to process completed bars
        self.bar_builder.attach_listener(self._on_bar)
        # register mt5 tick handler
        self.mt5_tick_cb = self._on_tick
        # ensure arbiter/risk client reuse
        self.arbiter = TradeArbiter(self.mt5)
        self.risk = RiskEngine(self.mt5)
        logger.info("Phase3Orchestrator setup complete for symbols: %s", self.symbols)

    def start(self):
        logger.info(
            "Starting Phase3 orchestrator. MT5 enabled: %s",
            os.environ.get("ARIA_ENABLE_MT5", "0"),
        )
        self.running = True

        # Publish initial status
        publish_orchestrator_status(
            {
                "status": "starting",
                "symbols": self.symbols,
                "mt5_enabled": os.environ.get("ARIA_ENABLE_MT5", "0") == "1",
                "execution_enabled": os.environ.get("ARIA_ENABLE_EXEC", "0") == "1",
                "start_time": time.time(),
            }
        )

        if os.environ.get("ARIA_ENABLE_MT5", "0") == "1":
            ok = self.mt5.connect()
            if not ok:
                logger.critical("MT5 client failed to connect")
                self._engage_kill_all()
                return
            self.mt5.start()

            # Publish MT5 connection status
            publish_orchestrator_status(
                {"mt5_connected": True, "mt5_connection_time": time.time()}
            )

        # subscribe per-symbol tick callback
        for s in self.symbols:
            try:
                self.mt5.subscribe_tick(s, self.mt5_tick_cb)
            except Exception:
                logger.exception("Failed subscribe %s", s)
        # start health monitor thread
        t = threading.Thread(target=self._health_monitor_loop, daemon=True)
        t.start()

        # Publish running status
        publish_orchestrator_status({"status": "running", "running_time": time.time()})

        logger.info("Phase3 orchestrator started.")

    def stop(self):
        logger.info("Stopping Phase3 orchestrator.")
        self.running = False
        try:
            self.bar_builder.flush_all()
        except Exception:
            pass
        try:
            self.mt5.stop()
        except Exception:
            pass
        # call save state on cores
        for core in self.cores.values():
            try:
                core._save_state()
            except Exception:
                pass

    # tick callback -> feed bar builder
    def _on_tick(self, symbol: str, tick: Dict[str, Any]):
        self.health["last_tick_ts"] = time.time()
        # forward tick to bar builder
        try:
            self.bar_builder.on_tick(symbol, tick)
        except Exception:
            logger.exception("bar_builder on_tick failed")

    def _on_bar(self, symbol: str, bar: Dict[str, Any]):
        """
        Called on completed bar. Steps:
         1) Build market feats
         2) Build features for models (ohlcv array, image stub)
         3) Call ai_signal_generator.get_signals(...)
         4) Call core.ingest_bar(...)
         5) If enhanced idea returned and execution allowed -> execute_trade
        """
        try:
            logger.debug("Bar completed %s @ %s", symbol, bar["ts"])
            core = self.cores.get(symbol)
            if core is None:
                logger.warning("No core for %s", symbol)
                return

            # Prepare features
            # create ohlcv seq from core.bars (last 64)
            seq = []
            with threading.Lock():
                for b in list(core.bars)[-64:]:
                    seq.append([b["o"], b["h"], b["l"], b["c"], b.get("v", 0)])
            # image data placeholder: if you have actual chart snapshots, wire here
            image = None

            # market features
            market_feats = {
                "price": float(bar["c"]),
                "spread": float(0.0001),  # ideally from tick; placeholder
                "atr": float(core._rolling_atr(core.cfg.atr_lookback) or 0.0),
                "vol": float(core.context.vol_ewma or 0.0),
                "trend_strength": float(0.0),
                "liquidity": float(0.5),
                "session_factor": float(0.9),
            }

            model_features = {
                "ohlcv": seq,
                "image": image,
                "state_vec": [],
                "macro_text": "",
            }

            # 1) Get AI signals
            try:
                raw_signals = ai_signal_generator.get_signals(symbol, model_features)
            except Exception as e:
                logger.exception("ai_signal_generator failed: %s", e)
                raw_signals = {
                    "lstm": 0,
                    "cnn": 0,
                    "ppo": 0,
                    "vision": 0,
                    "llm_macro": 0,
                }

            # 2) Ingest bar into fusion core (this will also update SMC detectors)
            idea = core.ingest_bar(
                bar, raw_signals=raw_signals, market_feats=market_feats
            )
            if idea is None:
                logger.debug("No idea generated for %s at %s", symbol, bar["ts"])
                return

            logger.info("Idea: %s", idea.as_dict())

            # Publish idea to debug API
            publish_idea_to_debug_api(symbol, idea)

            # 3) Pre-execution checks: anomaly, regime, exposure etc.
            if idea.anomaly_score and idea.anomaly_score > core.cfg.anomaly_z:
                logger.warning(
                    "Anomaly score high (%.3f): skipping execution", idea.anomaly_score
                )
                return

            # 4) Execute if allowed
            if core.cfg.enable_execution:
                ok = core.execute_trade(idea, current_price=market_feats["price"])
                if ok:
                    logger.info("Trade executed for %s", symbol)
                else:
                    logger.warning("Trade NOT executed for %s", symbol)
            else:
                logger.info(
                    "[DRY] Idea ready but execution disabled: %s", idea.as_dict()
                )

        except Exception:
            logger.exception("on_bar processing failed")

    def _health_monitor_loop(self):
        """Kill-on-feed-failure: if no ticks for X seconds, engage kill switch."""
        timeout = int(os.environ.get("ARIA_FEED_TIMEOUT_SEC", "30"))
        while self.running:
            last = self.health.get("last_tick_ts", 0)
            if (
                time.time() - last > timeout
                and os.environ.get("ARIA_ENABLE_MT5", "0") == "1"
            ):
                logger.critical(
                    "No ticks received for %s seconds -> engaging kill-switch", timeout
                )
                self._engage_kill_all()
                break
            time.sleep(2.0)

    def _engage_kill_all(self):
        logger.critical("Engaging global kill: stopping orchestrator and cores.")
        for core in self.cores.values():
            try:
                core.kill_switch(True)
            except Exception:
                pass
        # stop mt5 client and orchestrator
        try:
            self.stop()
        except Exception:
            pass


# -------------------- CLI / Entrypoint --------------------


def run_orchestrator(symbols: List[str], timeframe: int = 60):
    orch = Phase3Orchestrator(symbols, timeframe=timeframe)
    try:
        orch.start()
        # keep running until SIGINT/SIGTERM
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received -> shutting down")
    finally:
        orch.stop()


if __name__ == "__main__":
    # Simple CLI: set symbols via ARIA_SYMBOLS env var (comma separated)
    syms = os.environ.get("ARIA_SYMBOLS", "EURUSD").split(",")
    tf = int(os.environ.get("ARIA_BAR_SECONDS", "60"))
    run_orchestrator([s.strip() for s in syms], timeframe=tf)
