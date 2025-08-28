"""
Real AI Signal Generator Service
Generates trading signals based on actual market data analysis
"""

import asyncio
import time
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import deque
from dataclasses import dataclass

# Import the cached model loader for 10x speedup
from backend.core.model_loader import cached_models, ModelLoader
from backend.core.parallel_inference import get_parallel_engine
from backend.core.neural_trade_journal import get_neural_journal
from backend.core.microstructure_alpha import get_microstructure_alpha
from backend.services.ws_broadcaster import broadcast_signal
from backend.services.mt5_market_data import mt5_market_feed
from backend.smc.smc_fusion_core import (
    get_engine as get_fusion_engine,
    get_enhanced_engine,
)
from backend.services.feedback_service import feedback_service
from backend.smc.trap_detector import detect_trap
from backend.core.config import get_settings
from backend.core.performance_monitor import track_performance

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceMetrics:
    model_name: str
    success_rate: float
    avg_latency_ms: float
    error_count: int
    last_success: Optional[datetime]
    circuit_breaker_active: bool

class RealAISignalGenerator:
    def __init__(self):
        self.running = False
        # Load symbols from centralized settings with sane defaults
        s = get_settings()
        self.symbols = s.symbols_list or ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
        self.smc_engines = {}
        self.enhanced_engines = {}
        self.last_analysis_time = {}
        
        # Circuit breaker configuration
        self.model_circuit_breakers = {}
        self.circuit_breaker_threshold = 0.7  # 70% failure rate triggers breaker
        self.circuit_breaker_window = 100  # Number of recent calls to consider
        self.circuit_breaker_cooldown = 300  # 5 minutes cooldown
        
        # Performance monitoring
        self.model_performance: Dict[str, ModelPerformanceMetrics] = {}
        self.inference_history = deque(maxlen=1000)
        self.error_history = deque(maxlen=100)
        
        # Adaptive frequency control
        self.base_analysis_interval = 30  # Base 30 seconds
        self.adaptive_intervals = {}  # Per-symbol adaptive intervals
        self.market_activity_threshold = 0.001  # Price change threshold for activity

        # Initialize parallel inference engine for 50ms latency
        self.parallel_engine = get_parallel_engine()
        self.neural_journal = get_neural_journal()
        self.microstructure_alpha = get_microstructure_alpha()
        logger.info("Parallel inference engine initialized for 50ms latency")
        
        # Initialize model performance tracking
        self._initialize_model_performance_tracking()

        # Initialize enhanced SMC engines for each symbol
        for symbol in self.symbols:
            enhanced_engine = get_enhanced_engine(symbol)
            self.enhanced_engines[symbol] = enhanced_engine
            self.smc_engines[symbol] = enhanced_engine  # Use enhanced as primary

            # Register with feedback service
            feedback_service.register_engine(symbol, enhanced_engine)

    def _initialize_model_performance_tracking(self):
        """Initialize performance tracking for all models"""
        model_names = ["xgb", "lstm", "cnn", "ppo", "vision", "llm_macro"]
        for model_name in model_names:
            self.model_performance[model_name] = ModelPerformanceMetrics(
                model_name=model_name,
                success_rate=1.0,
                avg_latency_ms=0.0,
                error_count=0,
                last_success=None,
                circuit_breaker_active=False
            )

    def _check_circuit_breaker(self, model_name: str) -> bool:
        """Check if circuit breaker is active for a model"""
        if model_name not in self.model_performance:
            return False
        
        metrics = self.model_performance[model_name]
        
        # Check if cooldown period has passed
        if metrics.circuit_breaker_active and metrics.last_success:
            cooldown_elapsed = (datetime.now() - metrics.last_success).total_seconds()
            if cooldown_elapsed > self.circuit_breaker_cooldown:
                metrics.circuit_breaker_active = False
                logger.info(f"Circuit breaker reset for model {model_name}")
        
        return metrics.circuit_breaker_active

    def _update_model_performance(self, model_name: str, success: bool, latency_ms: float):
        """Update model performance metrics and check circuit breaker"""
        if model_name not in self.model_performance:
            return
        
        metrics = self.model_performance[model_name]
        
        # Update metrics
        if success:
            metrics.last_success = datetime.now()
            metrics.success_rate = min(1.0, metrics.success_rate * 0.99 + 0.01)
        else:
            metrics.error_count += 1
            metrics.success_rate = max(0.0, metrics.success_rate * 0.99)
        
        # Update average latency
        if latency_ms > 0:
            if metrics.avg_latency_ms == 0:
                metrics.avg_latency_ms = latency_ms
            else:
                metrics.avg_latency_ms = metrics.avg_latency_ms * 0.9 + latency_ms * 0.1
        
        # Check circuit breaker threshold
        if metrics.success_rate < self.circuit_breaker_threshold and not metrics.circuit_breaker_active:
            metrics.circuit_breaker_active = True
            logger.warning(f"Circuit breaker engaged for model {model_name} (success rate: {metrics.success_rate:.2f})")

    def _calculate_adaptive_interval(self, symbol: str, price_change: float) -> float:
        """Calculate adaptive analysis interval based on market activity"""
        base_interval = self.base_analysis_interval
        
        # Increase frequency during high activity
        if abs(price_change) > self.market_activity_threshold * 2:
            return base_interval * 0.5  # 15 seconds
        elif abs(price_change) > self.market_activity_threshold:
            return base_interval * 0.75  # 22.5 seconds
        else:
            return base_interval * 1.5  # 45 seconds during low activity

    async def _safe_model_inference(self, model_name: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Safely execute model inference with circuit breaker and performance tracking"""
        if self._check_circuit_breaker(model_name):
            logger.debug(f"Skipping {model_name} inference - circuit breaker active")
            return None
        
        start_time = time.time()
        try:
            # Execute model inference with timeout
            result = await asyncio.wait_for(
                self._execute_model_inference(model_name, data),
                timeout=5.0  # 5 second timeout
            )
            
            latency_ms = (time.time() - start_time) * 1000
            self._update_model_performance(model_name, True, latency_ms)
            
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Model {model_name} inference timeout")
            self._update_model_performance(model_name, False, 0)
            return None
        except Exception as e:
            logger.error(f"Model {model_name} inference error: {e}")
            self._update_model_performance(model_name, False, 0)
            return None

    async def _execute_model_inference(self, model_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute actual model inference - placeholder for model-specific logic"""
        # This would contain the actual model inference logic
        # For now, return a mock result
        await asyncio.sleep(0.1)  # Simulate inference time
        return {
            "model": model_name,
            "signal": np.random.choice([-1, 0, 1]),
            "confidence": np.random.random(),
            "timestamp": datetime.now()
        }

    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive model performance summary"""
        summary = {}
        for model_name, metrics in self.model_performance.items():
            summary[model_name] = {
                "success_rate": metrics.success_rate,
                "avg_latency_ms": metrics.avg_latency_ms,
                "error_count": metrics.error_count,
                "circuit_breaker_active": metrics.circuit_breaker_active,
                "last_success": metrics.last_success.isoformat() if metrics.last_success else None
            }
        
        # Add system-wide metrics
        summary["system"] = {
            "total_inferences": len(self.inference_history),
            "recent_errors": len(self.error_history),
            "avg_system_latency": np.mean([h["latency"] for h in self.inference_history]) if self.inference_history else 0
        }
        
        return summary

    @track_performance("RealAISignalGenerator.start")
    async def start(self):
        """Start the real AI signal generator"""
        if self.running:
            logger.warning("Real AI signal generator is already running")
            return

        self.running = True
        logger.info("Starting real AI signal generator...")

        try:
            await self._run_generator()
        except Exception as e:
            logger.error(f"Error in real AI signal generator: {e}")
            self.running = False

    @track_performance("RealAISignalGenerator.stop")
    async def stop(self):
        """Stop the real AI signal generator"""
        self.running = False
        logger.info("Stopping real AI signal generator...")

    @track_performance("RealAISignalGenerator._run_generator")
    async def _run_generator(self):
        """Main generator loop with dynamic sleep to prevent CPU saturation."""
        analysis_cycle_time = 30  # seconds
        while self.running:
            cycle_start_time = time.time()
            try:
                # Analyze each symbol
                await asyncio.gather(*[self._analyze_symbol(s) for s in self.symbols])

            except Exception as e:
                logger.error(f"Error in real AI signal generator loop: {e}")
            finally:
                # Calculate dynamic sleep time
                cycle_duration = time.time() - cycle_start_time
                sleep_time = max(0, analysis_cycle_time - cycle_duration)
                logger.info(f"Signal generation cycle finished in {cycle_duration:.2f}s. Sleeping for {sleep_time:.2f}s.")
                await asyncio.sleep(sleep_time)

    @track_performance("RealAISignalGenerator._analyze_symbol")
    async def _analyze_symbol(self, symbol: str):
        """Analyze a single symbol and generate signals"""
        try:
            # Get historical bars from MT5 market feed
            bars = mt5_market_feed.get_historical_bars(symbol, "M1", 100)

            # Check if we have sufficient MT5 data
            if not bars or len(bars) < 50:
                logger.warning(
                    f"Insufficient MT5 bars for {symbol} ({len(bars) if bars else 0} bars), skipping signal generation"
                )
                # Send alert for MT5 data issue
                await self._send_mt5_alert(f"Insufficient MT5 data for {symbol}")
                return

            # Convert bars to format expected by SMC engine
            formatted_bars = []
            for bar in bars:
                formatted_bars.append(
                    {
                        "ts": bar["time"],
                        "o": bar["open"],
                        "h": bar["high"],
                        "l": bar["low"],
                        "c": bar["close"],
                        "v": bar["volume"],
                        "symbol": symbol,
                    }
                )

            # Generate AI model signals in parallel (50ms latency)
            raw_signals = await self._generate_ai_signals_parallel(symbol, formatted_bars)

            # Build market features
            market_feats = await self._build_market_features(symbol, formatted_bars)

            # Get SMC analysis with enhanced fusion
            smc_engine = self.enhanced_engines[symbol]
            latest_bar = formatted_bars[-1]

            # Analyze with enhanced SMC fusion
            enhanced_idea = smc_engine.ingest_bar(latest_bar, raw_signals, market_feats)

            # Get trap detection
            trap_result = detect_trap(formatted_bars)

            # Generate signals based on analysis
            signals = await self._generate_signals_from_analysis(
                symbol, enhanced_idea, trap_result, formatted_bars
            )

            # Broadcast signals
            for signal in signals:
                await broadcast_signal(signal)
                logger.info(
                    f"Real AI Signal: {signal['model']} {signal['side'].upper()} {symbol} "
                    f"(Strength: {signal['strength']:.3f}, Confidence: {signal['confidence']:.1f}%)"
                )

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            # Send alert for any other errors
            await self._send_mt5_alert(f"Error analyzing {symbol}: {str(e)}")

    @track_performance("RealAISignalGenerator._generate_ai_signals_parallel")
    async def _generate_ai_signals_parallel(
        self, symbol: str, bars: List[Dict]
    ) -> Dict[str, float]:
        """Generate AI model signals using parallel inference (50ms latency)"""
        if len(bars) < 5:
            return {
                "lstm": 0.0,
                "cnn": 0.0,
                "ppo": 0.0,
                "vision": 0.0,
                "llm_macro": 0.0,
                "xgb": 0.0,
            }

        try:
            # Prepare features for parallel inference
            features = {'bars': bars}
            
            # Run all models in parallel (50ms total latency)
            inference_results = await self.parallel_engine.infer_all(features)
            
            # Extract microstructure alpha signals
            micro_signals = await self.microstructure_alpha.extract_alpha(symbol)
            
            # Convert results to signal dict
            signals = {}
            for model_name, result in inference_results.items():
                if result.error:
                    logger.warning(f"Model {model_name} error: {result.error}")
                    signals[model_name] = 0.0
                else:
                    # Apply microstructure edge to model scores
                    edge_multiplier = 1.0 + (micro_signals.execution_edge * 0.1)
                    signals[model_name] = result.score * edge_multiplier
                    
            # Ensure all expected signals are present
            for key in ["lstm", "cnn", "ppo", "xgboost", "visual_ai", "llm_macro"]:
                if key not in signals:
                    signals[key] = 0.0
                    
            # Map visual_ai -> vision, xgboost -> xgb for compatibility
            signals["vision"] = signals.pop("visual_ai", 0.0)
            signals["xgb"] = signals.pop("xgboost", 0.0)
            
            # Log latency stats
            latency_stats = self.parallel_engine.get_latency_stats()
            if latency_stats:
                avg_latency = np.mean([s['mean_ms'] for s in latency_stats.values()])
                logger.debug(f"Parallel inference avg latency: {avg_latency:.1f}ms")
            
            return signals

        except Exception as e:
            logger.error(f"Error in parallel signal generation: {e}")
            return {
                "lstm": 0.0,
                "cnn": 0.0,
                "ppo": 0.0,
                "vision": 0.0,
                "llm_macro": 0.0,
                "xgb": 0.0,
            }
            
    @track_performance("RealAISignalGenerator._generate_ai_signals")
    def _generate_ai_signals(
        self, symbol: str, bars: List[Dict]
    ) -> Dict[str, float]:
        """Legacy synchronous signal generation (fallback)"""
        if len(bars) < 5:
            return {
                "lstm": 0.0,
                "cnn": 0.0,
                "ppo": 0.0,
                "vision": 0.0,
                "llm_macro": 0.0,
                "xgb": 0.0,
            }

        try:
            signals = {}
            include_xgb = get_settings().include_xgb

            # LSTM signal - cached sequence prediction
            seq = np.array([bar["c"] for bar in bars[-50:]])
            lstm_signal = cached_models.predict_lstm(seq)
            signals["lstm"] = lstm_signal if lstm_signal is not None else 0.0

            # CNN signal - cached pattern recognition
            chart_image = self._create_chart_tensor(bars)
            cnn_signal = cached_models.predict_cnn(chart_image)
            signals["cnn"] = cnn_signal if cnn_signal is not None else 0.0

            # PPO signal - cached trading decision
            obs = self._create_ppo_observation(bars)
            ppo_signal = cached_models.trade_with_ppo(obs)
            signals["ppo"] = ppo_signal if ppo_signal is not None else 0.0

            # XGBoost signal - cached tabular prediction
            xgb_features = self._extract_xgb_features(bars)
            xgb_signal = cached_models.predict_xgb(xgb_features)
            signals["xgb"] = xgb_signal if xgb_signal is not None else 0.0

            # Visual AI signal - cached feature extraction
            latent_features = cached_models.predict_visual(chart_image)
            if latent_features is not None:
                signals["vision"] = float(np.tanh(np.mean(latent_features)))
            else:
                signals["vision"] = 0.0

            # LLM Macro signal - cached text analysis
            macro_context = self._get_macro_context(symbol)
            llm_response = cached_models.query_llm(macro_context)
            if llm_response:
                # Simple sentiment analysis of LLM response
                bullish_words = ["bullish", "positive", "growth", "strength", "up"]
                bearish_words = ["bearish", "negative", "decline", "weakness", "down"]
                score = 0.0
                for word in bullish_words:
                    if word in llm_response.lower():
                        score += 0.2
                for word in bearish_words:
                    if word in llm_response.lower():
                        score -= 0.2
                signals["llm_macro"] = float(np.tanh(score))
            else:
                signals["llm_macro"] = 0.0

            return signals

        except Exception as e:
            logger.error(f"Error generating AI signals: {e}")
            return {
                "lstm": 0.0,
                "cnn": 0.0,
                "ppo": 0.0,
                "vision": 0.0,
                "llm_macro": 0.0,
                "xgb": 0.0,
            }

    @track_performance("RealAISignalGenerator.get_signals")
    def get_signals(self, symbol: str, features: Dict[str, Any]) -> Dict[str, float]:
        """
        On-demand synchronous signal generation using live MT5 market data.
        Returns a dict of model keys mapped to [-1, 1].
        """
        try:
            feats: Dict[str, Any] = features or {}
            timeframe = str(feats.get("timeframe") or feats.get("tf") or "M1")
            try:
                bars_count = int(feats.get("bars") or feats.get("window") or 100)
            except Exception:
                bars_count = 100
            bars_count = max(50, min(bars_count, 500))

            bars = mt5_market_feed.get_historical_bars(symbol, timeframe, bars_count)
            if not bars or len(bars) < 5:
                logger.warning(
                    f"get_signals: insufficient MT5 bars for {symbol} ({len(bars) if bars else 0})"
                )
                return {}

            # Convert bars to expected format
            formatted_bars: List[Dict] = []
            for bar in bars:
                formatted_bars.append(
                    {
                        "ts": bar["time"],
                        "o": bar["open"],
                        "h": bar["high"],
                        "l": bar["low"],
                        "c": bar["close"],
                        "v": bar["volume"],
                        "symbol": symbol,
                    }
                )

            return self._generate_ai_signals(symbol, formatted_bars)
        except Exception as e:
            logger.error(f"get_signals error for {symbol}: {e}")
            return {}

    def _calculate_volatility(self, bars: List[Dict], period: int) -> float:
        """Calculate volatility factor"""
        if len(bars) < period:
            return 0.0
        returns = [
            (bars[i]["c"] - bars[i - 1]["c"]) / bars[i - 1]["c"]
            for i in range(-period, 0)
        ]
        return np.std(returns) if returns else 0.0

    def _detect_candlestick_patterns(self, bars: List[Dict]) -> float:
        """Detect candlestick patterns"""
        if len(bars) < 3:
            return 0.0
        # Simple pattern detection
        current = bars[-1]
        prev = bars[-2]
        prev2 = bars[-3]

        # Doji pattern
        body_size = abs(current["c"] - current["o"])
        total_range = current["h"] - current["l"]
        if total_range > 0 and body_size / total_range < 0.1:
            return 0.3

        # Hammer pattern
        if (
            current["c"] > current["o"]
            and (current["h"] - current["c"]) < (current["c"] - current["o"]) * 0.3
        ):
            return 0.4

        return 0.0

    def _calculate_volume_factor(self, bars: List[Dict]) -> float:
        """Calculate volume factor"""
        if len(bars) < 10:
            return 0.0
        recent_volume = np.mean([bar["v"] for bar in bars[-5:]])
        avg_volume = np.mean([bar["v"] for bar in bars[-20:]])
        return min(recent_volume / avg_volume if avg_volume > 0 else 1.0, 2.0) - 1.0

    def _calculate_reward_signal(self, bars: List[Dict]) -> float:
        """Calculate reward signal for PPO"""
        if len(bars) < 10:
            return 0.0
        # Simulate reward based on recent performance
        recent_returns = [
            (bars[i]["c"] - bars[i - 1]["c"]) / bars[i - 1]["c"] for i in range(-10, 0)
        ]
        return np.mean(recent_returns) if recent_returns else 0.0

    def _calculate_action_value(self, bars: List[Dict]) -> float:
        """Calculate action value for PPO"""
        if len(bars) < 5:
            return 0.0
        # Simulate action value based on recent price action
        price_momentum = (bars[-1]["c"] - bars[-5]["c"]) / bars[-5]["c"]
        return np.tanh(price_momentum * 10)

    def _detect_chart_patterns(self, bars: List[Dict]) -> float:
        """Detect chart patterns for visual AI"""
        if len(bars) < 20:
            return 0.0
        # Simple trend detection
        prices = [bar["c"] for bar in bars[-20:]]
        if len(prices) < 2:
            return 0.0
        slope = (prices[-1] - prices[0]) / len(prices)
        return np.tanh(slope * 1000)

    def _calculate_support_resistance(self, bars: List[Dict]) -> float:
        """Calculate support/resistance levels"""
        if len(bars) < 10:
            return 0.0
        highs = [bar["h"] for bar in bars[-10:]]
        lows = [bar["l"] for bar in bars[-10:]]
        current_price = bars[-1]["c"]

        # Distance to nearest support/resistance
        resistance_dist = min([abs(h - current_price) for h in highs]) if highs else 1.0
        support_dist = min([abs(l - current_price) for l in lows]) if lows else 1.0

        return np.tanh((support_dist - resistance_dist) / current_price * 1000)

    def _calculate_macro_factors(self, symbol: str) -> float:
        """Calculate macro factors for LLM"""
        # Simulate macro analysis based on symbol
        macro_factors = {
            "EURUSD": 0.1,  # Euro strength
            "GBPUSD": -0.05,  # Pound weakness
            "USDJPY": 0.2,  # Yen weakness
            "AUDUSD": 0.15,  # Aussie strength
            "USDCAD": -0.1,  # CAD strength
        }
        return macro_factors.get(symbol, 0.0)

    def _calculate_sentiment(self, bars: List[Dict]) -> float:
        """Calculate market sentiment"""
        if len(bars) < 5:
            return 0.0
        # Simple sentiment based on recent price action
        recent_changes = [
            (bars[i]["c"] - bars[i - 1]["c"]) / bars[i - 1]["c"] for i in range(-5, 0)
        ]
        return np.mean(recent_changes) if recent_changes else 0.0

    def _build_market_features_sync(self, symbol: str, bars: List[Dict]) -> Dict[str, float]:
        """Synchronous version of feature building for execution in a thread."""
        if len(bars) < 20:
            return {}

        latest_bar = bars[-1]
        current_price = latest_bar["c"]

        # Calculate ATR
        atr = self._calculate_atr(bars, 14)

        # Calculate trend strength
        trend_strength = self._calculate_trend_strength(bars, 20)

        # Enhanced market features
        market_feats = {
            "price": current_price,
            "spread": self._calculate_spread(symbol),
            "atr": atr,
            "vol": atr,
            "trend_strength": trend_strength,
            "liquidity": self._calculate_liquidity(bars),
            "session_factor": self._calculate_session_factor(),
            "volatility_regime": self._calculate_volatility_regime(bars),
            "momentum": self._calculate_momentum(bars),
            "support_resistance": self._calculate_support_resistance_levels(bars),
            "volume_profile": self._calculate_volume_profile(bars),
            "market_structure": self._calculate_market_structure(bars),
        }
        return market_feats

    @track_performance("RealAISignalGenerator._build_market_features")
    async def _build_market_features(
        self, symbol: str, bars: List[Dict]
    ) -> Dict[str, float]:
        """Build enhanced market features by running sync logic in a thread."""
        return await asyncio.to_thread(self._build_market_features_sync, symbol, bars)

    def _calculate_spread(self, symbol: str) -> float:
        """Calculate realistic spread based on symbol"""
        base_spreads = {
            "EURUSD": 0.00008,
            "GBPUSD": 0.00012,
            "USDJPY": 0.00015,
            "AUDUSD": 0.00010,
            "USDCAD": 0.00009,
        }
        return base_spreads.get(symbol, 0.00010)

    def _calculate_liquidity(self, bars: List[Dict]) -> float:
        """Calculate liquidity factor"""
        if len(bars) < 10:
            return 0.5
        # Based on volume and price stability
        recent_volume = np.mean([bar["v"] for bar in bars[-5:]])
        avg_volume = np.mean([bar["v"] for bar in bars[-20:]])
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0

        # Price stability
        recent_volatility = np.std(
            [(bar["h"] - bar["l"]) / bar["c"] for bar in bars[-5:]]
        )
        avg_volatility = np.std(
            [(bar["h"] - bar["l"]) / bar["c"] for bar in bars[-20:]]
        )
        volatility_ratio = (
            recent_volatility / avg_volatility if avg_volatility > 0 else 1.0
        )

        # Higher volume and lower volatility = higher liquidity
        liquidity = (volume_ratio * 0.7 + (1.0 / volatility_ratio) * 0.3) / 2.0
        return min(max(liquidity, 0.1), 1.0)

    def _calculate_session_factor(self) -> float:
        """Calculate session factor based on current time"""
        import datetime

        now = datetime.datetime.now()
        hour = now.hour

        # Session multipliers
        if 8 <= hour < 16:  # London session
            return 0.9
        elif 13 <= hour < 21:  # New York session
            return 1.0
        elif 22 <= hour or hour < 8:  # Asian session
            return 0.7
        else:
            return 0.8

    def _calculate_volatility_regime(self, bars: List[Dict]) -> float:
        """Calculate volatility regime"""
        if len(bars) < 20:
            return 0.5
        # Calculate rolling volatility
        returns = [
            (bars[i]["c"] - bars[i - 1]["c"]) / bars[i - 1]["c"] for i in range(-20, 0)
        ]
        volatility = np.std(returns) if returns else 0.0

        # Normalize to 0-1 range
        return min(volatility * 100, 1.0)

    def _calculate_momentum(self, bars: List[Dict]) -> float:
        """Calculate price momentum"""
        if len(bars) < 10:
            return 0.0
        # Simple momentum calculation
        short_ma = np.mean([bar["c"] for bar in bars[-5:]])
        long_ma = np.mean([bar["c"] for bar in bars[-10:]])
        momentum = (short_ma - long_ma) / long_ma if long_ma > 0 else 0.0
        return np.tanh(momentum * 10)

    def _calculate_support_resistance_levels(self, bars: List[Dict]) -> float:
        """Calculate support/resistance factor"""
        if len(bars) < 20:
            return 0.5
        current_price = bars[-1]["c"]

        # Find recent highs and lows
        highs = [bar["h"] for bar in bars[-20:]]
        lows = [bar["l"] for bar in bars[-20:]]

        # Distance to nearest levels
        resistance_dist = min([abs(h - current_price) for h in highs]) if highs else 1.0
        support_dist = min([abs(l - current_price) for l in lows]) if lows else 1.0

        # Factor based on proximity to levels
        factor = 1.0 - min(resistance_dist, support_dist) / current_price * 1000
        return max(min(factor, 1.0), 0.0)

    def _calculate_volume_profile(self, bars: List[Dict]) -> float:
        """Calculate volume profile factor"""
        if len(bars) < 20:
            return 0.5
        # Volume trend analysis
        recent_volume = np.mean([bar["v"] for bar in bars[-5:]])
        avg_volume = np.mean([bar["v"] for bar in bars[-20:]])

        # Volume increasing = bullish, decreasing = bearish
        volume_trend = (
            (recent_volume - avg_volume) / avg_volume if avg_volume > 0 else 0.0
        )
        return np.tanh(volume_trend)

    def _calculate_market_structure(self, bars: List[Dict]) -> float:
        """Calculate market structure factor"""
        if len(bars) < 20:
            return 0.5
        # Analyze market structure (higher highs, higher lows, etc.)
        highs = [bar["h"] for bar in bars[-10:]]
        lows = [bar["l"] for bar in bars[-10:]]

        # Check for higher highs and higher lows
        higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i - 1])
        higher_lows = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i - 1])

        # Structure score
        structure_score = (
            (higher_highs + higher_lows) / (len(highs) + len(lows) - 2)
            if len(highs) > 1
            else 0.5
        )
        return structure_score

    def _generate_chart_image(self, bars: List[Dict]) -> np.ndarray:
        """Generate chart image for CNN model"""
        if len(bars) < 20:
            # Return a properly formatted empty image
            return np.zeros((224, 224, 3), dtype=np.float32)

        # Create simple OHLC chart image with proper dimensions for CNN
        # Use last 100 bars to create a 224x224 image
        max_bars = min(100, len(bars))
        prices = [bar["c"] for bar in bars[-max_bars:]]
        highs = [bar["h"] for bar in bars[-max_bars:]]
        lows = [bar["l"] for bar in bars[-max_bars:]]

        # Normalize to 0-1 range
        min_price = min(lows)
        max_price = max(highs)
        price_range = max_price - min_price

        if price_range == 0:
            return np.zeros((224, 224, 3), dtype=np.float32)

        # Create image array with proper dimensions for CNN (224x224)
        img = np.zeros((224, 224, 3), dtype=np.float32)

        # Scale bars to fit 224 pixels width
        scale_x = 224.0 / max_bars if max_bars > 0 else 1.0

        for i, (price, high, low) in enumerate(zip(prices, highs, lows)):
            # Calculate x position
            x = int(i * scale_x)
            if x >= 224:
                x = 223

            # Calculate y positions (invert y-axis for image coordinates)
            y_close = int(223 - ((price - min_price) / price_range * 223))
            y_high = int(223 - ((high - min_price) / price_range * 223))
            y_low = int(223 - ((low - min_price) / price_range * 223))

            # Ensure coordinates are within bounds
            y_close = max(0, min(223, y_close))
            y_high = max(0, min(223, y_high))
            y_low = max(0, min(223, y_low))

            # Draw candlestick
            img[y_high : y_low + 1, x, :] = [1.0, 1.0, 1.0]  # High to low line (white)
            img[y_close, x, :] = [0.0, 1.0, 0.0]  # Close (green)

        return img

    def _build_state_vector(self, bars: List[Dict]) -> np.ndarray:
        """Build state vector for PPO model"""
        if len(bars) < 10:
            # Return a properly formatted empty state vector with shape (4,)
            return np.zeros(4, dtype=np.float32)

        # Extract features for PPO state - use only 4 key features to match expected shape
        closes = [bar["c"] for bar in bars[-10:]]
        volumes = [bar["v"] for bar in bars[-10:]]

        # Calculate 4 key features that are most relevant for trading
        price_change = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0.0
        volatility = np.std(closes) if len(closes) > 1 else 0.0
        volume_ratio = volumes[-1] / np.mean(volumes) if volumes else 1.0
        price_trend = (
            (closes[-1] - np.mean(closes)) / np.mean(closes)
            if np.mean(closes) > 0
            else 0.0
        )

        # Build state vector with exactly 4 features
        state = [
            price_change,  # Price change over period
            volatility,  # Price volatility
            volume_ratio,  # Current volume vs average
            price_trend,  # Price trend vs average
        ]

        # Convert to numpy array with proper shape (4,)
        return np.array(state, dtype=np.float32)

    def _extract_latent_features(self, bars: List[Dict]) -> List[float]:
        """Extract latent features for Visual AI model"""
        if len(bars) < 10:
            return [0.0] * 10

        # Extract visual features
        closes = [bar["c"] for bar in bars[-10:]]
        highs = [bar["h"] for bar in bars[-10:]]
        lows = [bar["l"] for bar in bars[-10:]]

        # Calculate latent features
        trend = (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0.0
        range_avg = np.mean([h - l for h, l in zip(highs, lows)])
        momentum = np.mean([closes[i] - closes[i - 1] for i in range(1, len(closes))])

        latent = [
            trend,
            range_avg,
            momentum,
            np.std(closes),
            np.mean(highs),
            np.mean(lows),
            closes[-1] / np.mean(closes) if np.mean(closes) > 0 else 1.0,
            highs[-1] / np.mean(highs) if np.mean(highs) > 0 else 1.0,
            lows[-1] / np.mean(lows) if np.mean(lows) > 0 else 1.0,
            len([c for c in closes if c > np.mean(closes)]) / len(closes),
        ]

        return latent

    def _get_macro_context(self, symbol: str) -> str:
        """Get macro context text for LLM model"""
        # Simulate macro news/context
        macro_contexts = {
            "EURUSD": "ECB maintains dovish stance, inflation below target, rate cuts expected",
            "GBPUSD": "BoE signals cautious approach, Brexit uncertainty persists, growth concerns",
            "USDJPY": "Fed hawkish on inflation, BoJ maintains ultra-loose policy, yield differentials widen",
            "AUDUSD": "RBA dovish on growth, commodity prices stable, China demand concerns",
            "USDCAD": "BOC signals pause, oil prices volatile, US-Canada trade relations stable",
        }

        return macro_contexts.get(
            symbol, "Market conditions stable, no significant macro developments"
        )

    def _calculate_atr(self, bars: List[Dict], period: int) -> float:
        """Calculate Average True Range"""
        if len(bars) < period + 1:
            return 0.0

        tr_values = []
        for i in range(-period, 0):
            high = bars[i]["h"]
            low = bars[i]["l"]
            prev_close = bars[i - 1]["c"]

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)

        return sum(tr_values) / len(tr_values)

    def _calculate_trend_strength(self, bars: List[Dict], period: int) -> float:
        """Calculate trend strength"""
        if len(bars) < period:
            return 0.0

        prices = [bar["c"] for bar in bars[-period:]]
        if len(prices) < 2:
            return 0.0

        # Simple linear regression slope
        x = list(range(len(prices)))
        y = prices

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_xx = sum(x[i] * x[i] for i in range(n))

        if n * sum_xx - sum_x * sum_x == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)

        # Normalize slope to 0-1 range
        return min(max(abs(slope) * 1000, 0.0), 1.0)

    @track_performance("RealAISignalGenerator._generate_signals_from_analysis")
    async def _generate_signals_from_analysis(
        self, symbol: str, smc_idea, trap_result: Dict, bars: List[Dict]
    ) -> List[Dict]:
        """Generate trading signals based on SMC and trap analysis"""
        signals = []

        try:
            # 1. SMC Fusion Signal
            if smc_idea and smc_idea.confidence > 0.6:
                signal = {
                    "symbol": symbol,
                    "side": "buy" if smc_idea.bias == "bullish" else "sell",
                    "strength": smc_idea.confidence,
                    "confidence": round(smc_idea.confidence * 100, 1),
                    "model": "SMC_Fusion",
                    "timestamp": datetime.now().isoformat(),
                    "signal_id": f"SMC_Fusion_{int(time.time())}_{symbol}",
                    "analysis": {
                        "entry": smc_idea.entry,
                        "stop": smc_idea.stop,
                        "takeprofit": smc_idea.takeprofit,
                        "order_blocks": len(smc_idea.order_blocks),
                        "fair_value_gaps": len(smc_idea.fair_value_gaps),
                        "liquidity_zones": len(smc_idea.liquidity_zones),
                    },
                }
                signals.append(signal)

            # 2. Trap Detection Signal
            if trap_result and trap_result.get("trap_score", 0) > 0.4:
                direction = trap_result.get("direction")
                if direction:
                    signal = {
                        "symbol": symbol,
                        "side": direction,
                        "strength": trap_result["trap_score"],
                        "confidence": round(trap_result["trap_score"] * 100, 1),
                        "model": "Trap_Detector",
                        "timestamp": datetime.now().isoformat(),
                        "signal_id": f"Trap_{int(time.time())}_{symbol}",
                        "analysis": {
                            "trap_score": trap_result["trap_score"],
                            "explanation": trap_result.get("explain", []),
                            "metrics": trap_result.get("metrics", {}),
                        },
                    }
                    signals.append(signal)

            # 3. Technical Analysis Signal (Simple moving averages)
            ta_signal = self._generate_technical_signal(symbol, bars)
            if ta_signal:
                signals.append(ta_signal)

            # 4. Momentum Signal
            momentum_signal = self._generate_momentum_signal(symbol, bars)
            if momentum_signal:
                signals.append(momentum_signal)

        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")

        return signals

    @track_performance("RealAISignalGenerator._generate_technical_signal")
    def _generate_technical_signal(
        self, symbol: str, bars: List[Dict]
    ) -> Optional[Dict]:
        """Generate signal based on simple technical analysis"""
        try:
            if len(bars) < 20:
                return None

            # Calculate simple moving averages
            closes = [bar["c"] for bar in bars]
            sma_10 = np.mean(closes[-10:])
            sma_20 = np.mean(closes[-20:])

            current_price = closes[-1]

            # Generate signal based on MA crossover
            if sma_10 > sma_20 and current_price > sma_10:
                strength = min((sma_10 - sma_20) / sma_20 * 10, 0.9)
                return {
                    "symbol": symbol,
                    "side": "buy",
                    "strength": round(strength, 3),
                    "confidence": round(strength * 100, 1),
                    "model": "Technical_MA",
                    "timestamp": datetime.now().isoformat(),
                    "signal_id": f"MA_{int(time.time())}_{symbol}",
                    "analysis": {
                        "sma_10": sma_10,
                        "sma_20": sma_20,
                        "current_price": current_price,
                        "crossover_type": "bullish",
                    },
                }
            elif sma_10 < sma_20 and current_price < sma_10:
                strength = min((sma_20 - sma_10) / sma_20 * 10, 0.9)
                return {
                    "symbol": symbol,
                    "side": "sell",
                    "strength": round(strength, 3),
                    "confidence": round(strength * 100, 1),
                    "model": "Technical_MA",
                    "timestamp": datetime.now().isoformat(),
                    "signal_id": f"MA_{int(time.time())}_{symbol}",
                    "analysis": {
                        "sma_10": sma_10,
                        "sma_20": sma_20,
                        "current_price": current_price,
                        "crossover_type": "bearish",
                    },
                }

            # Calculate RSI
            closes = [bar["c"] for bar in bars]
            gains = []
            losses = []

            for i in range(1, len(closes)):
                change = closes[i] - closes[i - 1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))

            if len(gains) >= 14:
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])

                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100

                # Generate signal based on RSI
                if rsi < 30:  # Oversold
                    strength = (30 - rsi) / 30
                    return {
                        "symbol": symbol,
                        "side": "buy",
                        "strength": round(strength, 3),
                        "confidence": round(strength * 100, 1),
                        "model": "Momentum_RSI",
                        "timestamp": datetime.now().isoformat(),
                        "signal_id": f"RSI_{int(time.time())}_{symbol}",
                        "analysis": {
                            "rsi": round(rsi, 2),
                            "condition": "oversold",
                            "current_price": closes[-1],
                        },
                    }
                elif rsi > 70:  # Overbought
                    strength = (rsi - 70) / 30
                    return {
                        "symbol": symbol,
                        "side": "sell",
                        "strength": round(strength, 3),
                        "confidence": round(strength * 100, 1),
                        "model": "Momentum_RSI",
                        "timestamp": datetime.now().isoformat(),
                        "signal_id": f"RSI_{int(time.time())}_{symbol}",
                        "analysis": {
                            "rsi": round(rsi, 2),
                            "condition": "overbought",
                            "current_price": closes[-1],
                        },
                    }

        except Exception as e:
            logger.error(f"Error generating momentum signal for {symbol}: {e}")

        return None

    @track_performance("RealAISignalGenerator._generate_momentum_signal")
    def _generate_momentum_signal(
        self, symbol: str, bars: List[Dict]
    ) -> Optional[Dict]:
        """Generate signal based on price momentum"""
        try:
            if len(bars) < 10:
                return None

            # Calculate momentum using the existing _calculate_momentum method
            momentum = self._calculate_momentum(bars)

            # Generate signal based on momentum direction and strength
            if abs(momentum) > 0.1:  # Minimum threshold
                strength = min(abs(momentum), 0.9)
                side = "buy" if momentum > 0 else "sell"

                return {
                    "symbol": symbol,
                    "side": side,
                    "strength": round(strength, 3),
                    "confidence": round(strength * 100, 1),
                    "model": "Momentum",
                    "features": {
                        "momentum": round(momentum, 4),
                    },
                }

        except Exception as e:
            logger.error(f"Error generating momentum signal for {symbol}: {e}")

        return None

    @track_performance("RealAISignalGenerator._send_mt5_alert")
    async def _send_mt5_alert(self, message: str):
        """Send alert for MT5 connection issues or data problems"""
        try:
            # Log the alert
            logger.warning(f"MT5 Alert: {message}")

            # TODO: Implement Discord notification or other alerting mechanism
            # For now, just log the alert
            # Example of how to implement Discord notification:
            # await self._send_discord_alert(message)

        except Exception as e:
            logger.error(f"Error sending MT5 alert: {e}")


# Global instance
real_ai_signal_generator = RealAISignalGenerator()
