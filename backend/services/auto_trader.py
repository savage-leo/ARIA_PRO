import asyncio
import logging
import os
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set

from fastapi import WebSocket
import contextlib
import numpy as np

from backend.services.mt5_market_data import mt5_market_feed
from backend.services.ai_signal_generator import ai_signal_generator
from backend.services.risk_engine import risk_engine
from backend.smc.smc_fusion_core import get_enhanced_engine
from backend.smc.bias_engine import BiasEngine
from backend.services.real_ai_signal_generator import real_ai_signal_generator
from backend.core.fusion import SignalFusion
from backend.core.regime import RegimeDetector, Regime
from backend.core.config import get_settings
from backend.core.risk_budget import RiskBudget
from backend.core.neural_trade_journal import get_neural_journal
from backend.core.meta_rl_selector import get_meta_selector
from backend.core.cross_agent_reasoning import get_cross_agent_reasoning
from backend.core.llm_gatekeeper import get_llm_gatekeeper
from backend.core.performance_monitor import get_performance_monitor, track_performance
from backend.core.microstructure_alpha import get_microstructure_alpha
from backend.strategies.quick_profit_engine import quick_profit_engine
from backend.strategies.arbitrage_detector import arbitrage_detector
from backend.strategies.news_trading import news_trading_system
# Import MT5 executor lazily inside methods to avoid hard fail if MT5 not configured

logger = logging.getLogger("AutoTrader")


class AutoTrader:
    """
    Auto Trader loops over configured symbols, builds real features from MT5-only OHLCV,
    runs active adapters for signals, maps score -> probability, and places trades
    via MT5 when probability >= threshold, with ATR-based SL/TP and risk checks.

    Env vars:
      - AUTO_TRADE_ENABLED=1                  # enable on startup
      - AUTO_TRADE_SYMBOLS=EURUSD,GBPUSD      # symbols to trade
      - AUTO_TRADE_INTERVAL_SEC=60            # polling interval
      - AUTO_TRADE_PROB_THRESHOLD=0.75        # probability threshold
      - AUTO_TRADE_PRIMARY_MODEL=xgb          # prefer this model for decision
      - AUTO_TRADE_DRY_RUN=1                  # simulate orders if 1
      - AUTO_TRADE_ATR_PERIOD=14              # ATR period
      - AUTO_TRADE_ATR_SL_MULT=1.5            # SL = ATR * mult
      - AUTO_TRADE_ATR_TP_MULT=2.0            # TP = ATR * mult
      - AUTO_TRADE_COOLDOWN_SEC=300           # min seconds between trades per symbol
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.running: bool = False
        self.symbols: List[str] = settings.symbols_list or self._parse_symbols(
            os.environ.get("AUTO_TRADE_SYMBOLS", "EURUSD")
        )
        self.interval_sec: int = int(os.environ.get("AUTO_TRADE_INTERVAL_SEC", "60"))
        self.threshold: float = float(
            os.environ.get("AUTO_TRADE_PROB_THRESHOLD", "0.75")
        )
        self.primary_model: str = (
            (settings.AUTO_TRADE_PRIMARY_MODEL or "xgb").strip().lower()
        )
        self.dry_run: bool = bool(settings.AUTO_TRADE_DRY_RUN)
        self.atr_period: int = int(os.environ.get("AUTO_TRADE_ATR_PERIOD", "14"))
        self.atr_sl_mult: float = float(os.environ.get("AUTO_TRADE_ATR_SL_MULT", "1.5"))
        self.atr_tp_mult: float = float(os.environ.get("AUTO_TRADE_ATR_TP_MULT", "2.0"))
        self.cooldown_sec: int = int(os.environ.get("AUTO_TRADE_COOLDOWN_SEC", "300"))
        
        # Circuit breaker configuration
        self.max_losses_per_day: int = int(os.environ.get("AUTO_TRADE_MAX_LOSSES_DAY", "3"))
        self.max_drawdown_pct: float = float(os.environ.get("AUTO_TRADE_MAX_DRAWDOWN_PCT", "0.05"))
        self.losses_today: int = 0
        self.daily_pnl: float = 0.0
        self.circuit_breaker_active: bool = False
        self.circuit_breaker_reason: str = ""
        self.circuit_breaker_cooldown_sec: int = int(os.environ.get("AUTO_TRADE_CB_COOLDOWN", "3600"))  # 1 hour
        self.circuit_breaker_engaged_at: Optional[float] = None
        self.auto_reset_circuit_breaker: bool = bool(int(os.environ.get("AUTO_TRADE_CB_AUTO_RESET", "1")))
        
        # Configurable position limits per symbol
        self.position_limits_per_symbol: Dict[str, int] = self._parse_position_limits(
            os.environ.get("AUTO_TRADE_POS_LIMITS", "")
        )
        self.default_max_positions_per_symbol: int = int(os.environ.get("AUTO_TRADE_MAX_POS_SYMBOL", "1"))
        self.max_total_positions: int = int(os.environ.get("AUTO_TRADE_MAX_TOTAL_POS", "5"))
        self.active_positions: Dict[str, int] = {}
        
        # Adaptive timeout configuration
        self.base_order_timeout: float = float(os.environ.get("AUTO_TRADE_ORDER_TIMEOUT", "5.0"))
        self.max_order_timeout: float = float(os.environ.get("AUTO_TRADE_MAX_ORDER_TIMEOUT", "30.0"))
        self.signal_timeout_sec: float = float(os.environ.get("AUTO_TRADE_SIGNAL_TIMEOUT", "10.0"))
        self.volatility_timeout_multiplier: float = float(os.environ.get("AUTO_TRADE_VOL_TIMEOUT_MULT", "2.0"))
        self.current_timeouts: Dict[str, float] = {}
        # Fusion/regime gating params
        self.min_edge: float = float(os.environ.get("AUTO_TRADE_MIN_EDGE", "0.01"))
        self.max_bucket_exposure: int = int(
            os.environ.get("ARIA_MAX_BUCKET_EXPOSURE", "3")
        )
        
        # Quick profit strategies
        self.enable_quick_profit: bool = bool(settings.ENABLE_QUICK_PROFIT)
        self.enable_arbitrage: bool = bool(settings.ENABLE_ARBITRAGE)
        self.enable_news_trading: bool = bool(settings.ENABLE_NEWS_TRADING)

        # Per-symbol fusion instances (initialized on first use based on observed signal keys)
        self._fusion_by_symbol: Dict[str, Optional[SignalFusion]] = {}
        # Unified RiskBudget engine
        self._risk_budget = RiskBudget()

        self._last_trade_ts: Dict[str, float] = {}
        # Param lock for safe runtime tuning updates
        self._param_lock: asyncio.Lock = asyncio.Lock()

        # Runtime counters/state
        self.signals_today: int = 0
        self.executed_today: int = 0
        self.failed_today: int = 0
        self._counters_day: str = datetime.now().strftime("%Y-%m-%d")
        self._started_at: Optional[float] = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        
        # Health monitoring
        self._last_health_check: float = time.time()
        self._health_check_interval: float = 30.0  # Check health every 30 seconds

        # WebSocket clients
        self._clients: Set[WebSocket] = set()
        self._clients_lock: asyncio.Lock = asyncio.Lock()
        
        # Initialize neural trade journal and microstructure alpha
        self.neural_journal = get_neural_journal()
        self.microstructure_alpha = get_microstructure_alpha()
        self.meta_selector = get_meta_selector()
        self.llm_gatekeeper = get_llm_gatekeeper()
        self.cross_agent_pipeline = get_cross_agent_reasoning()
        self.performance_monitor = get_performance_monitor()

    @staticmethod
    def _parse_symbols(text: str) -> List[str]:
        return [s.strip().upper() for s in text.split(",") if s.strip()]

    def _parse_position_limits(self, text: str) -> Dict[str, int]:
        """Parse position limits in format: EURUSD:2,GBPUSD:3,XAUUSD:1"""
        limits = {}
        if not text:
            return limits
        
        for pair in text.split(","):
            if ":" in pair:
                symbol, limit = pair.strip().split(":", 1)
                try:
                    limits[symbol.strip().upper()] = int(limit.strip())
                except ValueError:
                    logger.warning(f"Invalid position limit for {symbol}: {limit}")
        return limits

    def _get_position_limit(self, symbol: str) -> int:
        """Get position limit for a specific symbol"""
        return self.position_limits_per_symbol.get(symbol, self.default_max_positions_per_symbol)

    def _calculate_adaptive_timeout(self, symbol: str, volatility: Optional[float] = None) -> float:
        """Calculate adaptive timeout based on market volatility"""
        base_timeout = self.base_order_timeout
        
        if volatility is None:
            # Use cached timeout if no volatility provided
            return self.current_timeouts.get(symbol, base_timeout)
        
        # Adjust timeout based on volatility
        if volatility > 0.02:  # High volatility threshold (2%)
            timeout_multiplier = self.volatility_timeout_multiplier
        elif volatility > 0.01:  # Medium volatility (1%)
            timeout_multiplier = 1.5
        else:  # Low volatility
            timeout_multiplier = 1.0
        
        adaptive_timeout = min(base_timeout * timeout_multiplier, self.max_order_timeout)
        self.current_timeouts[symbol] = adaptive_timeout
        return adaptive_timeout

    def _check_circuit_breaker_reset(self) -> bool:
        """Check if circuit breaker should be automatically reset"""
        if not self.circuit_breaker_active or not self.auto_reset_circuit_breaker:
            return False
        
        if self.circuit_breaker_engaged_at is None:
            return False
        
        elapsed = time.time() - self.circuit_breaker_engaged_at
        if elapsed >= self.circuit_breaker_cooldown_sec:
            logger.info(f"Auto-resetting circuit breaker after {elapsed:.0f}s cooldown")
            self.circuit_breaker_active = False
            self.circuit_breaker_reason = ""
            self.circuit_breaker_engaged_at = None
            return True
        
        return False

    def _engage_circuit_breaker(self, reason: str) -> None:
        """Engage circuit breaker with enhanced logging and cooldown tracking"""
        if not self.circuit_breaker_active:
            self.circuit_breaker_active = True
            self.circuit_breaker_reason = reason
            self.circuit_breaker_engaged_at = time.time()
            logger.critical(f"CIRCUIT BREAKER ENGAGED: {reason}")
            logger.info(f"Auto-reset in {self.circuit_breaker_cooldown_sec}s if enabled")

    async def start(self) -> None:
        if self.running:
            logger.warning("AutoTrader already running")
            return
        self.running = True
        self._stop_event.clear()
        self._started_at = time.time()
        self._task = asyncio.current_task()
        logger.info(
            f"AutoTrader starting: symbols={self.symbols}, interval={self.interval_sec}s, "
            f"thresh={self.threshold}, primary={self.primary_model}, dry_run={self.dry_run}"
        )
        try:
            # Warm up adapters once
            try:
                _ = ai_signal_generator.get_signals(
                    "EURUSD", {"series": [1, 1.0001, 1.0002, 1.0003, 1.0004, 1.0005]}
                )
            except Exception:
                logger.exception("Adapter warm-up failed")

            # Initial broadcast
            await self._broadcast_status()

            while self.running and not self._stop_event.is_set():
                tick_start = time.time()
                await self._tick()
                tick_duration = time.time() - tick_start
                
                # Log slow ticks
                if tick_duration > 10:
                    logger.warning(f"Slow tick detected: {tick_duration:.1f}s")
                
                # Sleep with stop awareness
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_sec)
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            logger.info("AutoTrader cancelled")
        except Exception:
            logger.exception("AutoTrader crashed; stopping")
        finally:
            self.running = False
            self._task = None
            try:
                await self._broadcast_status()
            except Exception:
                pass

    async def stop(self) -> None:
        logger.info("AutoTrader stopping gracefully...")
        self.running = False
        self._stop_event.set()
        task = self._task
        if task and task is not asyncio.current_task() and not task.done():
            try:
                await asyncio.wait_for(task, timeout=10)
            except Exception:
                logger.warning("AutoTrader stop wait timed out")
        self._task = None
        try:
            await self._broadcast_status()
        except Exception:
            pass

    async def _tick(self) -> None:
        self._ensure_today_counters()
        
        # Check circuit breaker before processing
        if self._check_circuit_breaker():
            logger.warning(f"Circuit breaker active: {self.circuit_breaker_reason}")
            return
        
        # Health check
        if time.time() - self._last_health_check > self._health_check_interval:
            await self._perform_health_check()
            self._last_health_check = time.time()
        
        # Check total position limits
        total_positions = sum(self.active_positions.values())
        if total_positions >= self.max_total_positions:
            logger.info(f"Max total positions reached ({total_positions}/{self.max_total_positions})")
            return
        
        for symbol in self.symbols:
            try:
                # Check per-symbol position limit
                if self.active_positions.get(symbol, 0) >= self.max_positions_per_symbol:
                    logger.debug(f"Max positions for {symbol} reached")
                    continue
                    
                await asyncio.wait_for(
                    self._process_symbol(symbol),
                    timeout=self.signal_timeout_sec
                )
            except asyncio.TimeoutError:
                logger.error(f"Processing timeout for {symbol} after {self.signal_timeout_sec}s")
                self.failed_today += 1
            except Exception:
                logger.exception(f"AutoTrader symbol loop error for {symbol}")
                self.failed_today += 1

    def _ensure_today_counters(self) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        if self._counters_day != today:
            self._counters_day = today
            self.signals_today = 0
            self.executed_today = 0
            self.failed_today = 0
            self.losses_today = 0
            self.daily_pnl = 0.0
            self.circuit_breaker_active = False
            self.circuit_breaker_reason = ""
            logger.info(f"AutoTrader counters reset for new day: {today}")

    async def register_client(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._clients_lock:
            self._clients.add(websocket)
        # Send initial snapshot
        try:
            await websocket.send_json({
                "type": "auto_trader_status",
                "status": self.get_status(),
            })
        except Exception:
            pass

    def unregister_client(self, websocket: WebSocket) -> None:
        try:
            # No await needed for discard
            if websocket in self._clients:
                self._clients.discard(websocket)
        except Exception:
            pass

    async def _broadcast_status(self) -> None:
        # Snapshot status once per broadcast
        status = self.get_status()
        async with self._clients_lock:
            if not self._clients:
                return
            dead: List[WebSocket] = []
            for ws in list(self._clients):
                try:
                    await ws.send_json({
                        "type": "auto_trader_status",
                        "status": status,
                    })
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)

    def _get_ohlcv_with_fallback(
        self, symbol: str, n: int, use_intraday: bool = True
    ) -> List[List[float]]:
        """MT5-only OHLCV fetch using persistent mt5_market_feed.
        Returns ascending [open, high, low, close, volume] rows.
        """
        min_needed = max(6, self.atr_period + 1)
        timeframe = "M5" if use_intraday else "D1"
        requested = max(60, n)
        fetch_count = requested
        attempts = 3

        for attempt in range(attempts):
            try:
                bars = mt5_market_feed.get_historical_bars(symbol, timeframe, fetch_count)
            except Exception:
                logger.exception("MT5 get_historical_bars failed for %s", symbol)
                bars = []

            if bars and len(bars) >= min_needed:
                try:
                    bars_sorted = sorted(bars, key=lambda b: float(b.get("time", 0.0)))
                except Exception:
                    bars_sorted = bars
                out: List[List[float]] = []
                for b in bars_sorted[-fetch_count:]:
                    try:
                        o = float(b["open"])  # type: ignore[index]
                        h = float(b["high"])  # type: ignore[index]
                        l = float(b["low"])   # type: ignore[index]
                        c = float(b["close"]) # type: ignore[index]
                        v = float(b.get("volume", 0.0))
                    except Exception:
                        # skip malformed bar
                        continue
                    out.append([o, h, l, c, v])
                return out

            logger.warning(
                "MT5 returned %d/%d bars for %s (need >= %d) [attempt %d/%d]",
                len(bars) if bars else 0,
                fetch_count,
                symbol,
                min_needed,
                attempt + 1,
                attempts,
            )
            time.sleep(0.5)
            fetch_count = min(fetch_count * 2, requested * 4)

        return []

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker should be activated"""
        if self.circuit_breaker_active:
            return True
            
        # Check max losses per day
        if self.losses_today >= self.max_losses_per_day:
            self.circuit_breaker_active = True
            self.circuit_breaker_reason = f"Max daily losses reached ({self.losses_today}/{self.max_losses_per_day})"
            logger.error(self.circuit_breaker_reason)
            return True
            
        # Check max drawdown
        if self.daily_pnl < 0 and abs(self.daily_pnl) > self.max_drawdown_pct:
            self.circuit_breaker_active = True
            self.circuit_breaker_reason = f"Max drawdown exceeded ({self.daily_pnl:.2%}/{self.max_drawdown_pct:.2%})"
            logger.error(self.circuit_breaker_reason)
            return True
            
        # Check excessive failures
        if self.failed_today > 10:
            self.circuit_breaker_active = True
            self.circuit_breaker_reason = f"Excessive failures ({self.failed_today})"
            logger.error(self.circuit_breaker_reason)
            return True
            
        return False
    
    async def _perform_health_check(self) -> None:
        """Perform health checks on dependencies"""
        try:
            # Check MT5 connection
            if not mt5_market_feed._connected:
                logger.warning("MT5 feed not connected during health check")
                
            # Check risk engine status
            risk_status = risk_engine.get_status()
            if risk_status.get("kill_switch_engaged"):
                self.circuit_breaker_active = True
                self.circuit_breaker_reason = "Risk engine kill switch engaged"
                logger.error(self.circuit_breaker_reason)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    @track_performance("AutoTrader._process_symbol")
    async def _process_symbol(self, symbol: str) -> None:
        # Enforce cooldown
        now = datetime.now().timestamp()
        last_ts = self._last_trade_ts.get(symbol, 0)
        if now - last_ts < self.cooldown_sec:
            return

        # Fetch OHLCV (intraday 5m) from MT5 (strict)
        ohlcv = self._get_ohlcv_with_fallback(
            symbol, n=max(120, self.atr_period + 20), use_intraday=True
        )
        if not ohlcv or len(ohlcv) < max(6, self.atr_period + 1):
            logger.warning("Insufficient data for %s", symbol)
            return

        # Convert to bars expected by Fusion Core / BiasEngine
        bar_count = len(ohlcv)
        # Approximate timestamps at 5-minute spacing ending now
        base_ts = time.time() - 300.0 * bar_count
        bars: List[Dict[str, float]] = []
        for i, row in enumerate(ohlcv):
            try:
                o, h, l, c, v = [float(x) for x in row[:5]]
            except Exception:
                continue
            ts = base_ts + (i + 1) * 300.0
            bars.append(
                {
                    "ts": ts,
                    "o": o,
                    "h": h,
                    "l": l,
                    "c": c,
                    "v": v,
                    "symbol": symbol,
                }
            )
        if not bars:
            logger.warning("No bars data for %s", symbol)
            return
        logger.debug("Fetched %d bars for %s", len(bars), symbol)

        latest_bar = bars[-1]
        price = float(latest_bar["c"])  # current price from last close

        # Generate raw AI signals and enhanced market features
        try:
            # Use parallel inference for 50ms latency - track performance
            with self.performance_monitor.track_model("AI_Signal_Generation"):
                raw_signals_coro = real_ai_signal_generator._generate_ai_signals_parallel(symbol, bars)
                market_feats_coro = real_ai_signal_generator._build_market_features(symbol, bars)
                
                # Also get microstructure alpha signals
                market_data = {
                    'bid': price - 0.00005,  # Approximate bid
                    'ask': price + 0.00005,  # Approximate ask
                    'price': price,
                    'volume': latest_bar['v'],
                    'timestamp': latest_bar['ts'],
                    'trade_count': 1
                }
                micro_signals_coro = self.microstructure_alpha.extract_alpha(symbol, market_data)
                
                # Gather all signals in parallel
                raw_signals, market_feats, micro_signals = await asyncio.gather(
                    raw_signals_coro, market_feats_coro, micro_signals_coro
                )
                raw_signals = raw_signals or {}
                market_feats = market_feats or {}
            
            # Query similar trades from neural journal for bias adjustment
            trade_bias = await self.neural_journal.get_trade_bias(
                symbol=symbol,
                price=price,
                regime=market_feats.get('regime', 'range'),
                ai_consensus=np.mean(list(raw_signals.values())) if raw_signals else 0.0
            )
            
            # Add microstructure signals to market features
            market_feats['execution_edge'] = micro_signals.execution_edge
            market_feats['flow_toxicity'] = micro_signals.flow_toxicity
            market_feats['liquidity_score'] = micro_signals.liquidity_score
            market_feats['trade_memory_bias'] = trade_bias
        except Exception as e:
            logger.exception(
                "AI signal/market feature generation failed for %s: %s", symbol, str(e)
            )
            raw_signals, market_feats = {}, {}
        
        # Quick Profit Strategy Analysis
        quick_profit_signal = None
        if self.enable_quick_profit:
            try:
                # Prepare market data for quick profit engine
                market_data = {
                    'bid': price,
                    'ask': price + market_feats.get('spread', 0.0001),
                    'atr': market_feats.get('atr', 0.001),
                    'volume': latest_bar['v'],
                    'price_history': [b['c'] for b in bars[-20:]],
                    'sentiment': market_feats.get('sentiment', 0.5)
                }
                
                # Analyze for quick profit opportunities
                opportunity = await quick_profit_engine.analyze_opportunity(symbol, market_data)
                
                if opportunity:
                    quick_profit_signal = opportunity
                    logger.info(f"Quick profit opportunity detected: {opportunity['strategy']} "
                              f"for {symbol} with {opportunity['signal']['expected_roi']:.1f} pips")
            except Exception as e:
                logger.error(f"Quick profit analysis failed for {symbol}: {e}")
        
        # Arbitrage Detection
        arbitrage_signal = None
        if self.enable_arbitrage:
            try:
                # Add current price to arbitrage detector
                await arbitrage_detector.add_price_feed('mt5', symbol, {
                    'bid': price,
                    'ask': price + market_feats.get('spread', 0.0001),
                    'timestamp': time.time(),
                    'volume': latest_bar['v']
                })
                
                # Check for arbitrage opportunities
                arb_opportunity = await arbitrage_detector.detect_latency_arbitrage(symbol)
                if arb_opportunity:
                    arbitrage_signal = arb_opportunity
                    logger.info(f"Arbitrage opportunity: {arb_opportunity['type']} "
                              f"for {symbol} with {arb_opportunity['profit_pips']:.1f} pips")
            except Exception as e:
                logger.error(f"Arbitrage detection failed for {symbol}: {e}")
        
        # News Trading Detection
        news_signal = None
        if self.enable_news_trading:
            try:
                # Check for news trading opportunities
                news_opportunity = await news_trading_system.detect_news_opportunity(
                    symbol, {
                        'bid': price,
                        'atr': market_feats.get('atr', 0.001),
                        'price_history': [b['c'] for b in bars[-20:]],
                        'volume': latest_bar['v']
                    }
                )
                if news_opportunity:
                    news_signal = news_opportunity
                    logger.info(f"News trading opportunity: {news_opportunity['type']} "
                              f"for {symbol}")
            except Exception as e:
                logger.error(f"News trading detection failed for {symbol}: {e}")

        # Regime detection (EWMA vol + 3-state)

        # Regime detection (EWMA vol + 3-state)
        try:
            regime, regime_metrics = RegimeDetector.detect(bars)
        except Exception as e:
            logger.exception("Regime detection failed for %s: %s", symbol, str(e))
            regime, regime_metrics = Regime.RANGE, None

        # Initialize per-symbol fusion if needed and compute fused probabilities
        try:
            fusion = self._fusion_by_symbol.get(symbol)
            if fusion is None and raw_signals:
                fusion = SignalFusion(signal_keys=sorted(list(raw_signals.keys())))
                self._fusion_by_symbol[symbol] = fusion
            fusion_out = (
                fusion.fuse(raw_signals, regime)
                if fusion
                else {
                    "p_long": 0.5,
                    "p_short": 0.5,
                    "direction": "buy",
                    "margin": 0.0,
                    "used_regime": getattr(regime, "value", "range"),
                    "contrib": {},
                }
            )
        except Exception as e:
            logger.exception("Signal fusion failed for %s: %s", symbol, str(e))
            fusion_out = {
                "p_long": 0.5,
                "p_short": 0.5,
                "direction": "buy",
                "margin": 0.0,
                "used_regime": "range",
                "contrib": {},
            }

        # Get or create Fusion Core (reuse generator's engine if available)
        engine = getattr(real_ai_signal_generator, "enhanced_engines", {}).get(symbol)
        if engine is None:
            engine = get_enhanced_engine(symbol)

        # Ensure execution mode respects AutoTrader dry-run
        try:
            engine.cfg.enable_execution = not self.dry_run
        except Exception as e:
            logger.exception("Failed to set execution mode for %s: %s", symbol, str(e))

        # Prime engine with historical bars (no signals), then ingest latest with signals/features
        try:
            for b in bars[:-1]:
                engine.ingest_bar(b)
            idea = engine.ingest_bar(
                latest_bar, raw_signals=raw_signals, market_feats=market_feats
            )
        except Exception as e:
            logger.exception("Fusion ingestion failed for %s: %s", symbol, str(e))
            return

        if not idea:
            logger.debug("No EnhancedTradeIdea produced for %s", symbol)
            return

        # Align SMC idea with fused decision and apply regime-aware gating
        try:
            p_long = float(fusion_out.get("p_long", 0.5))
            p_short = float(fusion_out.get("p_short", 0.5))
            pmax = max(p_long, p_short)
            fusion_dir = str(fusion_out.get("direction", "buy"))
            idea_dir = (
                "buy" if getattr(idea, "bias", "bullish") == "bullish" else "sell"
            )
            margin = float(fusion_out.get("margin", abs(p_long - p_short)))

            # Direction mismatch handling: if strong disagreement, skip
            if fusion_dir != idea_dir and margin >= 0.1:
                logger.info(
                    "Fusion direction disagrees with SMC (%s vs %s) margin=%.3f -> skip",
                    fusion_dir,
                    idea_dir,
                    margin,
                )
                return

            # Pre-trade metrics for risk budgeting (edge, cost)
            atr_val = float(market_feats.get("atr", 0.0) or 0.0)
            if atr_val <= 0:
                atr_val = self._compute_atr(ohlcv, period=self.atr_period)
            spread = float(market_feats.get("spread", 0.0) or 0.0)
            edge_kelly = max(0.0, 2.0 * pmax - 1.0)  # Kelly fraction for binary outcome

            # Edge gate: minimum edge requirement for trade entry
            if edge_kelly < self.min_edge:
                logger.info(
                    "Edge gate: Kelly edge %.3f < minimum %.3f for %s -> skip",
                    edge_kelly,
                    self.min_edge,
                    symbol,
                )
                return

            # Regime-aware scaling into confidence (fractional Kelly shaping)
            try:
                # base scaling from SMC confidence and Kelly edge
                base_conf = float(getattr(idea, "confidence", 0.0) or 0.0)
                new_conf = base_conf * (0.5 + 0.5 * edge_kelly)

                # Regime-specific activation rules
                used_regime = str(
                    fusion_out.get("used_regime", getattr(regime, "value", "range"))
                )

                # Regime-specific multipliers and constraints
                if used_regime == "trend":
                    # Trend regime: slightly higher confidence, stable signals
                    new_conf *= 1.05
                elif used_regime == "range":
                    # Range regime: lower confidence, cautious trading
                    new_conf *= 0.95
                elif used_regime == "breakout":
                    # Breakout regime: higher confidence but requires strong edge
                    if edge_kelly >= 0.1:  # Stronger edge requirement for breakouts
                        new_conf *= 1.1
                    else:
                        new_conf *= (
                            0.8  # Reduce confidence if edge is weak in breakout regime
                        )

                # Apply trade memory bias from neural journal
                trade_memory_bias = market_feats.get('trade_memory_bias', 1.0)
                new_conf *= trade_memory_bias
                
                # Apply microstructure execution edge
                execution_edge = market_feats.get('execution_edge', 0.0)
                if execution_edge > 0.5:
                    new_conf *= 1.05  # Boost confidence with good execution conditions
                elif execution_edge < 0.2:
                    new_conf *= 0.95  # Reduce confidence with poor execution conditions

                # Get Meta-RL strategy selection
                meta_context = {
                    'regime': used_regime,
                    'volatility': float(market_feats.get('volatility', 0.01)),
                    'trend': float(raw_signals.get('lstm', 0.0)),
                    'session': market_feats.get('session', 'unknown'),
                    'symbol_performance': {symbol: float(raw_signals.get('xgboost', 0.0))}
                }
                
                try:
                    meta_action = await self.meta_selector.select_strategies(meta_context)
                    
                    # Apply meta-RL risk multiplier
                    new_conf *= meta_action.risk_multiplier
                    
                    # Log meta-RL decision
                    logger.info(
                        "Meta-RL: %s | Risk mult: %.2f | %s",
                        meta_action.reasoning,
                        meta_action.risk_multiplier,
                        meta_action.time_horizon
                    )
                    
                    # Store meta action in idea for execution reference
                    idea.meta['meta_rl'] = {
                        'strategy_weights': meta_action.strategy_weights,
                        'time_horizon': meta_action.time_horizon,
                        'primary_strategy': max(meta_action.strategy_weights, key=meta_action.strategy_weights.get)
                    }
                except Exception as e:
                    logger.error("Meta-RL selection failed: %s", str(e))
                
                # Cross-Agent Reasoning for collective decision
                try:
                    agent_context = {
                        'symbol': symbol,
                        'price': price,
                        'ai_signals': raw_signals,
                        'regime': used_regime,
                        'volatility': float(market_feats.get('volatility', 0.01)),
                        'volume': float(latest_bar['v']),
                        'atr': atr_val,
                        'spread': spread,
                        'microstructure': {
                            'execution_edge': market_feats.get('execution_edge', 0.0),
                            'flow_toxicity': market_feats.get('flow_toxicity', 0.0),
                            'liquidity_score': market_feats.get('liquidity_score', 0.5)
                        }
                    }
                    
                    # Get consensus from multiple agents
                    consensus = await self.cross_agent_pipeline.get_consensus(
                        signal_data=raw_signals,
                        market_data=agent_context
                    )
                    
                    # Apply consensus to confidence
                    if consensus.confidence > 0.7:
                        new_conf *= (0.9 + 0.2 * consensus.consensus_level)  # Boost for high consensus
                    elif consensus.confidence < 0.3:
                        new_conf *= 0.7  # Reduce for low consensus
                    
                    # Store consensus in metadata
                    idea.meta['cross_agent'] = {
                        'decision': consensus.decision,
                        'confidence': consensus.confidence,
                        'consensus_level': consensus.consensus_level,
                        'reasoning': consensus.reasoning[:200]  # Truncate for storage
                    }
                    
                    logger.info(
                        "Cross-Agent: %s | Consensus: %.2f | Confidence: %.2f",
                        consensus.decision,
                        consensus.consensus_level,
                        consensus.confidence
                    )
                    
                except Exception as e:
                    logger.error("Cross-agent reasoning failed: %s", str(e))
                
                # LLM-powered trade analysis (with security gatekeeper)
                try:
                    # Prepare context for LLM analysis
                    llm_context = {
                        'symbol': symbol,
                        'price': price,
                        'direction': idea_dir,
                        'confidence': new_conf,
                        'regime': used_regime,
                        'ai_consensus': np.mean(list(raw_signals.values())) if raw_signals else 0.0,
                        'edge': edge_kelly,
                        'risk_reward': abs((float(idea.takeprofit) - price) / (price - float(idea.stop))) if idea.stop else 0
                    }
                    
                    # Request LLM analysis through gatekeeper
                    llm_prompt = f"Analyze trade setup for {symbol}: {idea_dir} at {price:.5f} in {used_regime} regime"
                    
                    llm_response = await self.llm_gatekeeper.process_request(
                        prompt=llm_prompt,
                        purpose='analysis',
                        context=llm_context,
                        user_id='auto_trader'
                    )
                    
                    # Adjust confidence based on LLM analysis quality
                    if not llm_response.filtered and llm_response.confidence > 0.7:
                        # High-quality analysis confirms trade
                        idea.meta['llm_analysis'] = llm_response.content[:300]
                        logger.info("LLM Analysis: %s", llm_response.content[:100])
                    elif llm_response.threats_detected:
                        # Security threats detected, be cautious
                        new_conf *= 0.9
                        logger.warning("LLM threats detected: %d", len(llm_response.threats_detected))
                        
                except Exception as e:
                    logger.error("LLM analysis failed: %s", str(e))

                # Ensure confidence stays within valid bounds
                idea.confidence = max(0.0, min(1.0, new_conf))

                # Log regime activation for monitoring
                logger.debug(
                    "Regime activation: %s regime with edge %.3f -> conf %.3f (base %.3f)",
                    used_regime,
                    edge_kelly,
                    idea.confidence,
                    base_conf,
                )
            except Exception:
                pass

            # Annotate idea meta for audit
            try:
                if getattr(idea, "meta_weights", None) is None:
                    idea.meta_weights = {}
                idea.meta_weights.update(
                    {
                        "fusion_p_long": p_long,
                        "fusion_p_short": p_short,
                        "fusion_margin": margin,
                        "fusion_direction": fusion_dir,
                        "used_regime": fusion_out.get("used_regime", "range"),
                        "kelly_edge": edge_kelly,
                    }
                )
            except Exception:
                pass
        except Exception:
            logger.exception("Fusion/regime gating failed for %s", symbol)

        logger.info(
            "Fusion idea %s: bias=%s conf=%.3f entry=%.5f sl=%.5f tp=%.5f",
            symbol,
            getattr(idea, "bias", "neutral"),
            float(getattr(idea, "confidence", 0.0) or 0.0),
            float(getattr(idea, "entry", price) or price),
            float(getattr(idea, "stop", 0.0) or 0.0),
            float(getattr(idea, "takeprofit", 0.0) or 0.0),
        )
        # Count viable signals and notify
        self.signals_today += 1
        try:
            await self._broadcast_status()
        except Exception:
            pass

        # Bias computation for risk/throttle
        bias_engine = BiasEngine()
        bars_deque = deque(bars, maxlen=200)
        market_ctx = {
            "spread": float(market_feats.get("spread", 0.0) or 0.0),
            "slippage": float(market_feats.get("slippage", 0.0) or 0.0),
            "of_imbalance": float(market_feats.get("orderflow_imbalance", 0.0) or 0.0),
        }
        try:
            bias = bias_engine.compute(idea, bars_deque, market_ctx)
        except Exception:
            logger.exception("BiasEngine compute failed for %s", symbol)
            return

        if bias.throttle:
            logger.info(
                "BIAS throttle %s: score=%.3f factor=%.2f reasons=%s",
                symbol,
                float(getattr(bias, "score", 0.0) or 0.0),
                float(getattr(bias, "bias_factor", 1.0) or 1.0),
                getattr(bias, "reasons", {}),
            )
            return

        # Apply bias factor to risk sizing by scaling confidence within [0,1]
        base_conf_pre_bias = float(getattr(idea, "confidence", 0.0) or 0.0)
        try:
            adj_conf = max(
                0.0,
                min(1.0, base_conf_pre_bias * float(bias.bias_factor)),
            )
            if adj_conf != idea.confidence:
                idea.confidence = adj_conf
                # annotate meta for audit
                if getattr(idea, "meta_weights", None) is None:
                    idea.meta_weights = {}
                idea.meta_weights["bias_factor"] = float(bias.bias_factor)
                idea.meta_weights["bias_score"] = float(bias.score)
        except Exception:
            logger.exception("Failed applying bias scaling for %s", symbol)

        # Choose best signal from multiple strategies
        best_signal = None
        signal_source = 'fusion'
        
        # Priority: Arbitrage > News > Quick Profit > Standard Fusion
        if arbitrage_signal and arbitrage_signal.get('confidence', 0) > 0.8:
            best_signal = arbitrage_signal
            signal_source = 'arbitrage'
            # Override idea with arbitrage signal
            idea.bias = 'bullish' if arbitrage_signal['direction'] == 'buy' else 'bearish'
            idea.confidence = arbitrage_signal['confidence']
        elif news_signal and news_signal.get('confidence', 0) > 0.7:
            best_signal = news_signal
            signal_source = 'news'
            # Override idea with news signal
            idea.bias = 'bullish' if news_signal['action'] == 'buy' else 'bearish'
            idea.confidence = news_signal['confidence']
        elif quick_profit_signal and quick_profit_signal.get('signal', {}).get('confidence', 0) > 0.7:
            best_signal = quick_profit_signal['signal']
            signal_source = f"quick_{quick_profit_signal['strategy']}"
            # Override idea with quick profit signal
            idea.bias = 'bullish' if best_signal['action'] == 'buy' else 'bearish'
            idea.confidence = best_signal['confidence']
        
        # Log strategy selection for A/B testing
        if best_signal:
            logger.info(f"Selected {signal_source} strategy for {symbol} with confidence {idea.confidence:.2f}")
        
        # Unified RiskBudget computation to finalize risk_units and throttle
        try:
            try:
                exposure_ok = self._currency_exposure_ok(symbol)
            except Exception:
                exposure_ok = True
            rb = self._risk_budget.compute(
                symbol=symbol,
                base_conf=base_conf_pre_bias,  # pre-bias to avoid double-counting
                kelly_edge=edge_kelly,
                spread=spread,
                atr=atr_val,
                metrics=regime_metrics,
                bias_factor=float(bias.bias_factor),
                bias_throttle=bool(bias.throttle),
                exposure_ok=bool(exposure_ok),
                exposure_ratio=None,
            )
            if rb.throttle:
                logger.info("RiskBudget throttle %s: reasons=%s", symbol, rb.reasons)
                return
            # Overwrite idea.confidence with unified risk_units
            idea.confidence = float(rb.risk_units)
            if getattr(idea, "meta_weights", None) is None:
                idea.meta_weights = {}
            idea.meta_weights["risk_units"] = float(rb.risk_units)
            idea.meta_weights["rb"] = rb.reasons
        except Exception:
            logger.exception("RiskBudget computation failed for %s", symbol)

        # Threshold gate on unified risk_units (after RiskBudget)
        if float(getattr(idea, "confidence", 0.0) or 0.0) < self.threshold:
            logger.info(
                "Below threshold after RiskBudget %s: conf=%.3f < %.3f",
                symbol,
                float(getattr(idea, "confidence", 0.0) or 0.0),
                self.threshold,
            )
            return
        
        # Record trade attempt in neural journal
        trade_id = f"{symbol}_{int(time.time())}"
        await self.neural_journal.record_trade(
            trade_id=trade_id,
            symbol=symbol,
            direction=getattr(idea, 'bias', 'bullish'),
            entry_price=float(getattr(idea, 'entry', price)),
            confidence=float(getattr(idea, 'confidence', 0.0)),
            features={
                'regime': used_regime,
                'ai_consensus': np.mean(list(raw_signals.values())) if raw_signals else 0.0,
                'volatility': float(market_feats.get('volatility', 0.0)),
                'execution_edge': float(market_feats.get('execution_edge', 0.0)),
                'flow_toxicity': float(market_feats.get('flow_toxicity', 0.0)),
                'kelly_edge': edge_kelly
            }
        )
        
        # Execute if threshold reached

        # Currency bucket exposure check (simple per-currency cap)
        try:
            if not self._currency_exposure_ok(symbol):
                logger.info(
                    "Currency bucket exposure cap reached for %s; skipping trade",
                    symbol,
                )
                return
        except Exception:
            pass

        # Execute via Fusion Core with timeout
        try:
            placed = await asyncio.wait_for(
                asyncio.to_thread(engine.execute_trade, idea, current_price=price),
                timeout=self.order_timeout_sec
            )
        except asyncio.TimeoutError:
            logger.error(f"Order execution timeout for {symbol} after {self.order_timeout_sec}s")
            self.failed_today += 1
            placed = False
        except Exception:
            logger.exception("Fusion execute_trade failed for %s", symbol)
            self.failed_today += 1
            placed = False

        if placed:
            self._last_trade_ts[symbol] = now
            # Update position tracking
            self.active_positions[symbol] = self.active_positions.get(symbol, 0) + 1
            logger.info(
                "Trade processed %s: placed=%s dry_run=%s positions=%d",
                symbol,
                placed,
                self.dry_run,
                self.active_positions[symbol]
            )
            self.executed_today += 1
            
            # Track outcome for circuit breaker
            asyncio.create_task(self._monitor_trade_outcome(symbol, idea))
            
            try:
                await self._broadcast_status()
            except Exception:
                pass
        else:
            # Track failed trade
            if not placed:
                logger.warning(f"Trade execution failed for {symbol}")

    # -------------------- Helpers: currency exposure --------------------
    def _parse_fx(self, symbol: str) -> Tuple[str, str]:
        s = symbol.strip().upper()
        if len(s) >= 6:
            return s[:3], s[3:6]
        return s, "USD"

    def _currency_exposure_ok(self, symbol: str) -> bool:
        """Simple per-currency open position cap across base/quote buckets.
        Controlled by ARIA_MAX_BUCKET_EXPOSURE (default 3).
        """
        try:
            from backend.services.mt5_executor import mt5_executor
        except Exception:
            return True  # if we cannot query, do not block
        try:
            positions = mt5_executor.get_positions()
        except Exception:
            return True
        base, quote = self._parse_fx(symbol)
        base_count = 0
        quote_count = 0
        for p in positions:
            sym = str(p.get("symbol", "")).upper()
            if not sym:
                continue
            b, q = self._parse_fx(sym)
            if b == base or q == base:
                base_count += 1
            if b == quote or q == quote:
                quote_count += 1
        return (
            base_count < self.max_bucket_exposure
            and quote_count < self.max_bucket_exposure
        )

    def _select_signal(self, signals: Dict[str, float]) -> Tuple[str, float]:
        # Prefer primary model if present and non-zero; else pick max |score|
        if self.primary_model in signals and abs(signals[self.primary_model]) > 0:
            return self.primary_model, float(signals[self.primary_model])
        # Fallback: choose strongest
        model, score = max(signals.items(), key=lambda kv: abs(kv[1]))
        return model, float(score)

    @staticmethod
    def _score_to_prob(score: float) -> float:
        # Model scores are in [-1, 1]. Use magnitude as probability proxy.
        s = max(-1.0, min(1.0, float(score)))
        return abs(s)

    @staticmethod
    def _compute_atr(ohlcv: List[List[float]], period: int = 14) -> float:
        # ohlcv rows: [open, high, low, close, volume]
        if not ohlcv or len(ohlcv) <= period:
            return 0.0
        trs: List[float] = []
        prev_close = float(ohlcv[0][3])
        for i in range(1, len(ohlcv)):
            o, h, l, c = [float(x) for x in ohlcv[i][:4]]
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
            trs.append(tr)
            prev_close = c
        if len(trs) < period:
            return 0.0
        recent = trs[-period:]
        return sum(recent) / float(period)

    def _calc_sl_tp(self, price: float, atr: float, side: str) -> Tuple[float, float]:
        sl_dist = atr * self.atr_sl_mult
        tp_dist = atr * self.atr_tp_mult
        if side == "buy":
            return price - sl_dist, price + tp_dist
        else:
            return price + sl_dist, price - tp_dist

    async def _try_place_order(
        self,
        symbol: str,
        side: str,
        sl: Optional[float],
        tp: Optional[float],
        entry_price_hint: Optional[float] = None,
    ) -> bool:
        # Require live trading enabled when not dry-run
        settings = get_settings()
        if not self.dry_run and not settings.mt5_enabled:
            logger.warning("AUTO TRADE: MT5 not enabled; skipping real order")
            return False

        try:
            from backend.services.mt5_executor import mt5_executor
        except Exception:
            logger.exception("MT5 executor import failed")
            return False

        # Account & positions
        try:
            account = mt5_executor.get_account_info()
            positions = mt5_executor.get_positions()
        except Exception:
            logger.exception("Failed to get account/positions")
            return False

        # Currency bucket exposure cap
        try:
            if not self._currency_exposure_ok(symbol):
                logger.info(
                    "Currency bucket exposure cap reached for %s; skipping trade",
                    symbol,
                )
                return False
        except Exception:
            pass

        # Current price and symbol trading params
        try:
            sym_info = mt5_executor.get_symbol_info(symbol)
            expected_price = sym_info["ask"] if side == "buy" else sym_info["bid"]
            vol_min = float(sym_info.get("volume_min", 0.01) or 0.01)
            vol_step = float(sym_info.get("volume_step", 0.01) or 0.01)
            vol_max = float(sym_info.get("volume_max", 100.0) or 100.0)
        except Exception:
            expected_price = float(entry_price_hint or 0.0)
            vol_min, vol_step, vol_max = 0.01, 0.01, 100.0

        # Position sizing via risk engine
        volume = risk_engine.calculate_position_size(
            account_balance=account.get("balance", 0.0),
            symbol=symbol,
            entry_price=expected_price or entry_price_hint or 0.0,
            stop_loss=sl,
        )
        # Convert notional units -> lots (approx 100k units per lot for FX majors)
        try:
            volume = float(volume) / 100_000.0
        except Exception:
            volume = 0.0
        # Quantize to symbol constraints
        if volume <= 0:
            logger.warning("Calculated volume <= 0; skipping trade")
            return False
        # clip and round to step
        volume = max(vol_min, min(vol_max, float(volume)))
        # step quantize
        steps = round(volume / vol_step)
        volume = max(vol_min, min(vol_max, steps * vol_step))
        # guard against rounding to zero
        if volume < vol_min:
            volume = vol_min

        # Risk validation
        validation = risk_engine.validate_order(
            symbol=symbol,
            volume=volume,
            order_type=side,
            account_info=account,
            current_positions=positions,
        )
        if not validation.get("approved", False):
            logger.warning(f"Order rejected by risk engine: {validation.get('errors')}")
            return False

        # Emergency stop
        if risk_engine.emergency_stop(account):
            logger.warning("Emergency stop active; skipping trade")
            return False

        # Avoid duplicate symbol exposure (simple: if any open position for symbol)
        if any(p.get("symbol") == symbol for p in positions):
            logger.info(f"Open position exists for {symbol}; skipping new trade")
            return False

        # Execute with timeout
        try:
            if self.dry_run:
                from backend.services.mt5_executor import execute_order

                result = execute_order(
                    symbol,
                    side,
                    float(volume),
                    sl=sl,
                    tp=tp,
                    comment="ARIA AutoTrader",
                    dry_run=True,
                )
                logger.info(
                    f"SIM ORDER: {symbol} {side} vol={volume:.4f} sl={sl:.5f} tp={tp:.5f} -> ticket={result['ticket']}"
                )
            else:
                result = mt5_executor.place_order(
                    symbol=symbol,
                    volume=float(volume),
                    order_type=side,
                    sl=sl,
                    tp=tp,
                    comment="ARIA AutoTrader",
                )
                logger.info(
                    f"LIVE ORDER: {symbol} {side} vol={volume:.4f} sl={sl:.5f} tp={tp:.5f} -> ticket={result['ticket']}"
                )
            return True
        except Exception:
            logger.exception("Order placement failed")
            return False

    async def apply_tuning(
        self,
        *,
        interval_sec: Optional[int] = None,
        threshold: Optional[float] = None,
        atr_period: Optional[int] = None,
        atr_sl_mult: Optional[float] = None,
        atr_tp_mult: Optional[float] = None,
        cooldown_sec: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Apply bounded runtime tuning updates. Returns dict of changes applied."""
        changes: Dict[str, Any] = {}
        async with self._param_lock:
            if interval_sec is not None and interval_sec != self.interval_sec:
                self.interval_sec = int(interval_sec)
                changes["interval_sec"] = self.interval_sec
            if threshold is not None and threshold != self.threshold:
                self.threshold = float(threshold)
                changes["threshold"] = self.threshold
            if atr_period is not None and atr_period != self.atr_period:
                self.atr_period = int(atr_period)
                changes["atr_period"] = self.atr_period
            if atr_sl_mult is not None and atr_sl_mult != self.atr_sl_mult:
                self.atr_sl_mult = float(atr_sl_mult)
                changes["atr_sl_mult"] = self.atr_sl_mult
            if atr_tp_mult is not None and atr_tp_mult != self.atr_tp_mult:
                self.atr_tp_mult = float(atr_tp_mult)
                changes["atr_tp_mult"] = self.atr_tp_mult
            if cooldown_sec is not None and cooldown_sec != self.cooldown_sec:
                self.cooldown_sec = int(cooldown_sec)
                changes["cooldown_sec"] = self.cooldown_sec

        if changes:
            logger.info("AutoTrader parameters updated: %s", changes)
        return changes

    async def _monitor_trade_outcome(self, symbol: str, idea: Any) -> None:
        """Monitor trade outcome for circuit breaker tracking"""
        await asyncio.sleep(60)  # Wait 60s then check
        try:
            from backend.services.mt5_executor import mt5_executor
            positions = mt5_executor.get_positions()
            
            # Check if position still exists
            position = next((p for p in positions if p.get("symbol") == symbol), None)
            
            if position:
                profit = position.get("profit", 0.0)
                if profit < 0:
                    self.losses_today += 1
                    self.daily_pnl += profit
                    logger.info(f"Trade loss recorded for {symbol}: {profit:.2f}")
                else:
                    self.daily_pnl += profit
                    logger.info(f"Trade profit recorded for {symbol}: {profit:.2f}")
            else:
                # Position closed, check history if available
                logger.debug(f"Position for {symbol} no longer active")
                
        except Exception as e:
            logger.error(f"Failed to monitor trade outcome: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        uptime = (
            float(time.time() - self._started_at)
            if (self._started_at is not None and self.running)
            else 0.0
        )
        try:
            ws_clients = len(self._clients)
        except Exception:
            ws_clients = 0
        return {
            "running": self.running,
            "enabled": bool(os.environ.get("AUTO_TRADE_ENABLED") == "1"),
            "symbols": self.symbols,
            "interval_sec": self.interval_sec,
            "threshold": self.threshold,
            "primary_model": self.primary_model,
            "dry_run": self.dry_run,
            "signals_today": self.signals_today,
            "executed_today": self.executed_today,
            "failed_today": self.failed_today,
            "losses_today": self.losses_today,
            "daily_pnl": self.daily_pnl,
            "circuit_breaker_active": self.circuit_breaker_active,
            "circuit_breaker_reason": self.circuit_breaker_reason,
            "active_positions": dict(self.active_positions),
            "max_positions_per_symbol": self.max_positions_per_symbol,
            "max_total_positions": self.max_total_positions,
            "uptime_sec": uptime,
            "ws_clients": ws_clients,
        }


# Global instance
auto_trader = AutoTrader()
