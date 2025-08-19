# -*- coding: utf-8 -*-
"""
T470 Optimized Trading Pipeline: Complete hedge fund brain within 8GB RAM
Lightweight ensemble with extended models optimized for ThinkPad T470
"""
from __future__ import annotations
import time, gc, threading, math
from typing import Dict, Any, Optional
import numpy as np

from backend.core.regime_online import regime_manager
from backend.services.models_interface import score_and_calibrate
from backend.services.lightweight_ensemble import (
    t470_ensemble,
    MiniTransformer,
    TinyAutoencoder,
    BayesianLite,
    MicroRL,
)
from backend.core.risk_budget_enhanced import position_sizer


class T470TradingPipeline:
    """
    Complete institutional trading pipeline optimized for T470 constraints:
    - 8GB RAM: Max 6GB usage (2GB OS buffer)
    - 256GB SSD: <100MB model storage
    - CPU-only: Sub-20ms decisions
    - Multi-strategy: 9 models + ensemble meta-learner
    """

    def __init__(self):
        # Core models (already optimized)
        self.core_models = {
            "LSTM": None,  # Will use existing LSTM adapter
            "CNN": None,  # Will use existing CNN adapter
            "PPO": None,  # Will use existing PPO adapter
            "XGB": None,  # Will use existing XGB adapter
        }

        # Extended lightweight models
        self.extended_models = {
            "MiniTransformer": MiniTransformer(),
            "TinyAutoencoder": TinyAutoencoder(),
            "BayesianLite": BayesianLite(),
            "MicroRL": MicroRL(),
        }

        # Memory management
        self.max_memory_mb = 6 * 1024  # 6GB limit
        self.gc_interval = 100  # Run GC every 100 decisions
        self.decision_count = 0

        # Performance optimization
        self.feature_cache = {}
        self.cache_ttl = 60  # 1 minute cache

        self.lock = threading.Lock()

    def process_tick_optimized(
        self,
        symbol: str,
        price: float,
        timestamp: Optional[float] = None,
        account_balance: float = 100000.0,
        atr: float = 0.001,
        spread_pips: float = 1.0,
    ) -> Dict[str, Any]:
        """
        T470-optimized tick processing with multi-strategy ensemble
        Target: <20ms total latency, <50MB RAM per decision
        """
        t0 = time.perf_counter_ns()

        with self.lock:
            self.decision_count += 1

            # Memory management
            if self.decision_count % self.gc_interval == 0:
                self._optimize_memory()

        # 1. Regime Detection (already optimized)
        regime_info = regime_manager.update_symbol(symbol, price, timestamp)
        state = regime_info["state"]
        vol_bucket = regime_info["vol_bucket"]

        # 2. Prepare market data for all models
        market_data = self._prepare_market_data(symbol, price, regime_info, spread_pips)

        # 3. Model Predictions (core + extended)
        model_predictions = {}
        model_confidences = {}

        # Core model predictions (mock for now - integrate with existing adapters)
        core_preds = self._get_core_predictions(market_data)
        model_predictions.update(core_preds)

        # Extended model predictions
        extended_preds = self._get_extended_predictions(market_data)
        model_predictions.update(extended_preds)

        # Default confidences
        for model_name in model_predictions:
            model_confidences[model_name] = (
                abs(model_predictions[model_name]) * 0.8 + 0.2
            )

        # 4. Ensemble Meta-Learning
        ensemble_pred, ensemble_conf, ensemble_meta = t470_ensemble.ensemble_predict(
            model_predictions, state, vol_bucket
        )

        # 5. Calibration (lightweight)
        calibrated_prob = self._lightweight_calibration(ensemble_pred, state)

        # 6. Position Sizing
        sizing = position_sizer.calculate_position_size(
            symbol=symbol,
            p_star=calibrated_prob,
            vol_bucket=vol_bucket,
            account_balance=account_balance,
            atr=atr,
            spread_pips=spread_pips,
        )

        # 7. Decision Logic
        action = "FLAT"
        if calibrated_prob >= 0.6:
            action = "LONG" if ensemble_pred > 0 else "SHORT"

        total_latency_ms = (time.perf_counter_ns() - t0) / 1e6

        # 8. Register predictions for meta-learning
        for model_name, prediction in model_predictions.items():
            t470_ensemble.register_prediction(model_name, prediction, state)

        # 9. Build optimized result
        result = {
            "symbol": symbol,
            "timestamp": timestamp or time.time(),
            "price": price,
            # Regime (minimal)
            "regime": {
                "state": state,
                "vol_bucket": vol_bucket,
                "volatility": regime_info["volatility"],
            },
            # Models (summary only)
            "models": {
                "count": len(model_predictions),
                "ensemble_pred": round(ensemble_pred, 4),
                "top_contributors": self._get_top_contributors(
                    model_predictions, ensemble_meta["weights"]
                ),
            },
            # Decision
            "decision": {
                "action": action,
                "confidence": round(calibrated_prob, 4),
                "ensemble_confidence": round(ensemble_conf, 4),
            },
            # Position Sizing (essential only)
            "sizing": {
                "position_size": sizing.get("position_size", 0),
                "risk_pct": sizing.get("risk_pct", 0),
                "kill_switch": sizing.get("kill_switch", False),
            },
            # Performance
            "performance": {
                "latency_ms": round(total_latency_ms, 2),
                "memory_mb": self._estimate_memory_usage(),
                "models_used": ensemble_meta["models_used"],
                "memory_pressure": ensemble_meta["memory_pressure"],
            },
        }

        return result

    def _prepare_market_data(
        self, symbol: str, price: float, regime_info: Dict, spread_pips: float
    ) -> Dict[str, Any]:
        """Prepare market data for all models (cached for efficiency)"""
        cache_key = f"{symbol}_{int(time.time() / self.cache_ttl)}"

        if cache_key in self.feature_cache:
            cached_data = self.feature_cache[cache_key].copy()
            cached_data["price"] = price  # Update current price
            return cached_data

        # Build market data structure
        market_data = {
            "symbol": symbol,
            "price": price,
            "spread": spread_pips,
            "regime": regime_info["state"],
            "vol_bucket": regime_info["vol_bucket"],
            "volatility": regime_info["volatility"],
            "regime_probs": regime_info["state_probs"],
            "session": self._get_session(),
        }

        # Add price series (lightweight)
        if hasattr(regime_manager, "get_detector"):
            detector = regime_manager.get_detector(symbol)
            if hasattr(detector, "returns_buffer") and len(detector.returns_buffer) > 0:
                # Use last 50 prices for models
                recent_prices = list(detector.returns_buffer)[-50:]
                market_data["series"] = recent_prices

                # Simple OHLCV construction for models that need it
                if len(recent_prices) >= 4:
                    market_data["ohlcv"] = self._construct_ohlcv(recent_prices)

        # Cache for efficiency (memory conscious)
        if len(self.feature_cache) > 10:  # Limit cache size
            oldest_key = min(self.feature_cache.keys())
            del self.feature_cache[oldest_key]

        self.feature_cache[cache_key] = market_data.copy()
        return market_data

    def _get_core_predictions(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Get predictions from core models (integrate with existing adapters)"""
        # Mock predictions for now - replace with actual model calls
        predictions = {
            "LSTM": self._mock_model_prediction("LSTM", market_data),
            "CNN": self._mock_model_prediction("CNN", market_data),
            "PPO": self._mock_model_prediction("PPO", market_data),
            "XGB": self._mock_model_prediction("XGB", market_data),
        }
        return predictions

    def _get_extended_predictions(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Get predictions from extended lightweight models"""
        predictions = {}

        for model_name, model in self.extended_models.items():
            try:
                pred = model.predict(market_data)
                predictions[model_name] = pred
            except Exception:
                predictions[model_name] = 0.0  # Graceful fallback

        return predictions

    def _mock_model_prediction(
        self, model_name: str, market_data: Dict[str, Any]
    ) -> float:
        """Mock prediction for core models (replace with actual integration)"""
        if "series" not in market_data or len(market_data["series"]) < 2:
            return 0.0

        series = market_data["series"]
        momentum = (series[-1] - series[-2]) / (abs(series[-2]) + 1e-8)

        # Different model characteristics
        if model_name == "LSTM":
            return float(np.tanh(momentum * 0.8))
        elif model_name == "CNN":
            return float(np.tanh(momentum * 0.6 + 0.1))
        elif model_name == "PPO":
            return float(np.tanh(momentum * 1.2))
        elif model_name == "XGB":
            volatility = np.std(series[-10:]) if len(series) >= 10 else 0.01
            return float(np.tanh(momentum * (1.0 / (1.0 + volatility * 10))))

        return 0.0

    def _lightweight_calibration(self, prediction: float, state: str) -> float:
        """Lightweight calibration without heavy Platt scaling"""
        # Simple state-based calibration
        state_bias = {"T": 0.02, "R": -0.01, "B": 0.01}.get(state, 0.0)

        # Convert to probability space
        prob = 1.0 / (1.0 + math.exp(-prediction)) + state_bias

        # Clamp to valid range
        return max(0.01, min(0.99, prob))

    def _get_session(self) -> str:
        """Simple session detection"""
        import datetime

        hour = datetime.datetime.utcnow().hour
        if 0 <= hour < 7:
            return "ASIA"
        elif 7 <= hour < 15:
            return "EU"
        else:
            return "US"

    def _construct_ohlcv(self, prices: List[float]) -> List[List[float]]:
        """Construct simple OHLCV from price series"""
        ohlcv = []
        for i in range(0, len(prices), 4):
            chunk = prices[i : i + 4]
            if len(chunk) >= 4:
                ohlcv.append(
                    [
                        chunk[0],  # Open
                        max(chunk),  # High
                        min(chunk),  # Low
                        chunk[-1],  # Close
                        1000.0,  # Mock volume
                    ]
                )
        return ohlcv[-20:]  # Last 20 bars

    def _get_top_contributors(
        self, predictions: Dict[str, float], weights: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Get top contributing models"""
        contributions = []
        for model, pred in predictions.items():
            weight = weights.get(model, 0.0)
            contribution = abs(pred * weight)
            contributions.append(
                {
                    "model": model,
                    "prediction": round(pred, 3),
                    "weight": round(weight, 3),
                    "contribution": round(contribution, 3),
                }
            )

        return sorted(contributions, key=lambda x: x["contribution"], reverse=True)[:3]

    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB"""
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return round(memory_mb, 1)
        except:
            return 0.0

    def _optimize_memory(self):
        """T470 memory optimization"""
        # Clear feature cache
        current_time = time.time()
        self.feature_cache = {
            k: v
            for k, v in self.feature_cache.items()
            if current_time - v.get("timestamp", 0) < self.cache_ttl
        }

        # Optimize ensemble
        t470_ensemble.optimize_for_t470()

        # Force garbage collection
        gc.collect()

    def get_system_status(self) -> Dict[str, Any]:
        """Get T470-optimized system status"""
        ensemble_status = t470_ensemble.get_performance_summary()

        return {
            "timestamp": time.time(),
            "t470_optimized": True,
            "decisions_processed": self.decision_count,
            "memory_usage_mb": self._estimate_memory_usage(),
            "memory_limit_mb": self.max_memory_mb,
            "models_available": len(self.core_models) + len(self.extended_models),
            "cache_size": len(self.feature_cache),
            "ensemble_status": ensemble_status,
            "gc_runs": self.decision_count // self.gc_interval,
        }


# Global instance optimized for T470
t470_pipeline = T470TradingPipeline()
