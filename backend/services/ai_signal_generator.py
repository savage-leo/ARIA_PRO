"""
AI Signal Generator Service
Simulates trading signals from AI models and broadcasts via WebSocket
"""

import asyncio
import random
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
from backend.services.ws_broadcaster import broadcast_signal
from backend.services.models_interface import score_and_calibrate

logger = logging.getLogger(__name__)


class AISignalGenerator:
    def __init__(self):
        self.running = False
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
        self.models = [
            "LSTM",
            "SMC_Fusion",
            "Trap_Detector",
            "Neural_Network",
            "Random_Forest",
        ]

        # Signal generation probabilities
        self.signal_probabilities = {
            "LSTM": 0.25,  # 25% chance of signal
            "SMC_Fusion": 0.20,  # 20% chance of signal
            "Trap_Detector": 0.15,  # 15% chance of signal
            "Neural_Network": 0.30,  # 30% chance of signal
            "Random_Forest": 0.35,  # 35% chance of signal
        }

        # Model-specific characteristics
        self.model_characteristics = {
            "LSTM": {
                "avg_confidence": 0.82,
                "confidence_range": 0.15,
                "avg_strength": 0.75,
                "strength_range": 0.20,
            },
            "SMC_Fusion": {
                "avg_confidence": 0.88,
                "confidence_range": 0.10,
                "avg_strength": 0.85,
                "strength_range": 0.12,
            },
            "Trap_Detector": {
                "avg_confidence": 0.78,
                "confidence_range": 0.18,
                "avg_strength": 0.70,
                "strength_range": 0.25,
            },
            "Neural_Network": {
                "avg_confidence": 0.80,
                "confidence_range": 0.16,
                "avg_strength": 0.78,
                "strength_range": 0.18,
            },
            "Random_Forest": {
                "avg_confidence": 0.85,
                "confidence_range": 0.12,
                "avg_strength": 0.80,
                "strength_range": 0.15,
            },
        }
        # Initialize adapters mapping (may remain empty if no adapters configured)
        self._adapters = {}

    def get_signals(self, symbol: str, model_features: Dict) -> Dict[str, float]:
        """Synchronous scoring using real adapters selected via env.
        Runs only active adapters from `build_default_adapters()` (default xgb+lstm).
        Returns a dict of active model keys mapped to [-1, 1].
        """
        try:
            # Lazy-init adapters once per process
            if not hasattr(self, "_adapters") or self._adapters is None:
                # No default builder available; ensure dict exists and skip load if empty
                self._adapters = {}
            # Attempt to load adapters if they define a load() method
            for a in self._adapters.values():
                try:
                    if hasattr(a, "load"):
                        a.load()
                except Exception:
                    logger.exception("Adapter load failed")

            # Normalize features for adapters
            feats: Dict = dict(model_features or {})
            # Derive price series from OHLCV if needed (close column index 3)
            if "series" not in feats:
                ohlcv = feats.get("ohlcv")
                if (
                    isinstance(ohlcv, list)
                    and ohlcv
                    and isinstance(ohlcv[-1], (list, tuple))
                ):
                    try:
                        feats["series"] = [
                            float(r[3]) for r in ohlcv if isinstance(r, (list, tuple))
                        ]
                    except Exception:
                        pass

            out: Dict[str, float] = {}
            for key, adapter in self._adapters.items():
                try:
                    out[key] = float(adapter.predict(feats))
                except Exception:
                    logger.exception(f"Adapter {key} predict failed")
                    out[key] = 0.0
            return out
        except Exception:
            logger.exception("get_signals failed; returning empty dict")
            return {}

    async def start(self):
        """Start the AI signal generator"""
        if self.running:
            logger.warning("AI signal generator is already running")
            return

        self.running = True
        logger.info("Starting AI signal generator...")

        try:
            await self._run_generator()
        except Exception as e:
            logger.error(f"Error in AI signal generator: {e}")
            self.running = False

    async def stop(self):
        """Stop the AI signal generator"""
        self.running = False
        logger.info("Stopping AI signal generator...")

    async def _run_generator(self):
        """Main generator loop"""
        while self.running:
            try:
                # Generate signals for each model
                for model in self.models:
                    await self._generate_model_signals(model)

                # Wait before next cycle
                await asyncio.sleep(5)  # Check for signals every 5 seconds

            except Exception as e:
                logger.error(f"Error generating AI signals: {e}")
                await asyncio.sleep(10)  # Wait longer on error

    async def _generate_model_signals(self, model: str):
        """Generate signals for a specific model"""
        try:
            # Check if this model should generate a signal
            if random.random() > self.signal_probabilities[model]:
                return

            # Select a random symbol
            symbol = random.choice(self.symbols)

            # Generate signal characteristics
            signal = await self._create_signal(symbol, model)

            # Broadcast the signal
            await broadcast_signal(signal)

            logger.info(
                f"Signal: {model} {signal['side'].upper()} {symbol} "
                f"(Strength: {signal['strength']:.3f}, Confidence: {signal['confidence']:.1f}%)"
            )

        except Exception as e:
            logger.error(f"Error generating signal for {model}: {e}")

    async def _create_signal(self, symbol: str, model: str) -> Dict:
        """Create a trading signal with realistic characteristics"""
        # Get model characteristics
        chars = self.model_characteristics[model]

        # Generate confidence and strength based on model characteristics
        confidence = random.uniform(
            chars["avg_confidence"] - chars["confidence_range"] / 2,
            chars["avg_confidence"] + chars["confidence_range"] / 2,
        )
        confidence = max(0.5, min(0.98, confidence))  # Clamp between 50% and 98%

        strength = random.uniform(
            chars["avg_strength"] - chars["strength_range"] / 2,
            chars["avg_strength"] + chars["strength_range"] / 2,
        )
        strength = max(0.3, min(0.95, strength))  # Clamp between 30% and 95%

        # Determine signal side (buy/sell)
        side = random.choice(["buy", "sell"])

        # Add some market context
        market_context = self._get_market_context(symbol)

        # Create signal data
        signal = {
            "symbol": symbol,
            "side": side,
            "strength": round(strength, 3),
            "confidence": round(confidence * 100, 1),  # Convert to percentage
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "market_context": market_context,
            "signal_id": f"{model}_{int(time.time())}_{random.randint(1000, 9999)}",
        }

        return signal

    def _get_market_context(self, symbol: str) -> Dict:
        """Generate market context for the signal"""
        contexts = [
            "trend_following",
            "mean_reversion",
            "breakout",
            "support_resistance",
            "momentum",
            "volatility_expansion",
        ]

        return {
            "context": random.choice(contexts),
            "timeframe": random.choice(["M1", "M5", "M15", "H1", "H4", "D1"]),
            "volatility": random.uniform(0.1, 0.8),
            "trend_strength": random.uniform(0.2, 0.9),
        }

    def get_model_stats(self) -> Dict:
        """Get statistics about signal generation"""
        return {
            "models": self.models,
            "probabilities": self.signal_probabilities,
            "characteristics": self.model_characteristics,
        }


# Global instance
ai_signal_generator = AISignalGenerator()
