import asyncio
import logging
import os
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from backend.services.mt5_market_data import mt5_market_feed
from backend.services.ai_signal_generator import ai_signal_generator
from backend.services.risk_engine import risk_engine
from backend.smc.smc_fusion_core import get_enhanced_engine
from backend.smc.bias_engine import BiasEngine
from backend.services.real_ai_signal_generator import real_ai_signal_generator
from backend.core.regime import RegimeDetector, Regime
from backend.core.fusion import SignalFusion
from backend.core.risk_budget import RiskBudgetEngine
from backend.strategies.quick_profit_engine import quick_profit_engine
from backend.strategies.arbitrage_detector import arbitrage_detector
from backend.strategies.news_trading import news_trading_system
from backend.core.config import get_settings

# Import MT5 executor lazily inside methods to avoid hard fail if MT5 not configured

logger = logging.getLogger(__name__)


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
        self._fusion_by_symbol: Dict[str, SignalFusion] = {}
        # Unified RiskBudget engine
        self._risk_budget = RiskBudgetEngine()

        self._last_trade_ts: Dict[str, float] = {}
        # Param lock for safe runtime tuning updates
        self._param_lock: asyncio.Lock = asyncio.Lock()

    @staticmethod
    def _parse_symbols(text: str) -> List[str]:
        return [s.strip().upper() for s in text.split(",") if s.strip()]

    async def start(self) -> None:
        if self.running:
            logger.warning("AutoTrader already running")
            return
        self.running = True
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

            while self.running:
                await self._tick()
                await asyncio.sleep(self.interval_sec)
        except asyncio.CancelledError:
            logger.info("AutoTrader cancelled")
        except Exception:
            logger.exception("AutoTrader crashed; stopping")
        finally:
            self.running = False

    async def stop(self) -> None:
        self.running = False
        logger.info("AutoTrader stopping...")

    async def _tick(self) -> None:
        for symbol in self.symbols:
            try:
                await self._process_symbol(symbol)
            except Exception:
                logger.exception(f"AutoTrader symbol loop error for {symbol}")

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
            # _generate_ai_signals is synchronous; offload to a thread to run concurrently
            raw_signals_task = asyncio.to_thread(
                real_ai_signal_generator._generate_ai_signals, symbol, bars
            )
            market_feats_coro = real_ai_signal_generator._build_market_features(
                symbol, bars
            )
            raw_signals, market_feats = await asyncio.gather(
                raw_signals_task, market_feats_coro
            )
            raw_signals = raw_signals or {}
            market_feats = market_feats or {}
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

        # Execute via Fusion Core
        try:
            placed = engine.execute_trade(idea, current_price=price)
        except Exception:
            logger.exception("Fusion execute_trade failed for %s", symbol)
            placed = False

        if placed:
            self._last_trade_ts[symbol] = now
            logger.info(
                "Trade processed %s: placed=%s dry_run=%s",
                symbol,
                placed,
                self.dry_run,
            )

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

        # Execute
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

    def get_status(self) -> Dict[str, object]:
        return {
            "running": self.running,
            "symbols": self.symbols,
            "interval_sec": self.interval_sec,
            "threshold": self.threshold,
            "primary_model": self.primary_model,
            "dry_run": self.dry_run,
            "atr_period": self.atr_period,
            "last_trade_ts": self._last_trade_ts,
        }


# Global instance
auto_trader = AutoTrader()
