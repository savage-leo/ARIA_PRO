# -*- coding: utf-8 -*-
"""
Lightweight Ensemble Meta-Learner: T470 Optimized
8GB RAM, 256GB SSD constraints - Maximum alpha with minimal footprint
"""
from __future__ import annotations
import os, json, time, math, threading, pathlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


@dataclass
class LightModelPerf:
    """Ultra-lightweight model performance tracker"""

    accuracy: float = 0.5
    recent_perf: float = 0.5  # Rolling performance (last 100 predictions)
    prediction_count: int = 0
    regime_scores: Tuple[float, float, float] = (0.5, 0.5, 0.5)  # T, R, B
    memory_kb: int = 0


class T470EnsembleBrain:
    """
    T470-Optimized Ensemble Meta-Learner

    Design Constraints:
    - Max 50MB RAM footprint for ensemble logic
    - CPU-only inference (no GPU)
    - Models ≤5MB each (ONNX quantized)
    - Total model storage ≤100MB
    - Sub-10ms ensemble decisions
    """

    def __init__(self):
        # Ultra-lightweight state (target: <5MB RAM)
        self.models = {
            # Core models (already implemented)
            "LSTM": LightModelPerf(memory_kb=512),  # 0.5MB ONNX
            "CNN": LightModelPerf(memory_kb=2048),  # 2MB ONNX
            "PPO": LightModelPerf(memory_kb=1024),  # 1MB ONNX
            "XGB": LightModelPerf(memory_kb=512),  # 0.5MB ONNX
            # Extended models (lightweight versions)
            "LightGBM": LightModelPerf(memory_kb=256),  # 0.25MB ONNX
            "MiniTransformer": LightModelPerf(
                memory_kb=3072
            ),  # 3MB ONNX (mini version)
            "TinyAutoencoder": LightModelPerf(memory_kb=1024),  # 1MB ONNX
            "BayesianLite": LightModelPerf(memory_kb=64),  # 64KB params
            "MicroRL": LightModelPerf(memory_kb=512),  # 0.5MB ONNX
        }

        # Minimal state tracking (target: <1MB RAM)
        self.recent_decisions = deque(maxlen=200)  # ~40KB
        self.weight_cache = {}  # ~10KB
        self.performance_window = 100  # Rolling window size

        # T470-optimized parameters
        self.max_ensemble_size = 4  # Never use more than 4 models simultaneously
        self.weight_update_interval = 10  # Update weights every 10 decisions
        self.memory_pressure_threshold = 6 * 1024 * 1024 * 1024  # 6GB RAM threshold

        self.lock = threading.Lock()
        self._decision_count = 0

    def register_prediction(
        self,
        model_name: str,
        prediction: float,
        regime: str,
        actual_outcome: Optional[float] = None,
    ):
        """Lightweight prediction registration"""
        if model_name not in self.models:
            return

        model = self.models[model_name]
        model.prediction_count += 1

        # Update accuracy if outcome available
        if actual_outcome is not None:
            pred_direction = 1 if prediction > 0 else -1
            actual_direction = 1 if actual_outcome > 0 else -1
            correct = pred_direction == actual_direction

            # Exponential moving average update (memory efficient)
            alpha = min(0.1, 2.0 / model.prediction_count)
            model.accuracy = (
                alpha * (1.0 if correct else 0.0) + (1 - alpha) * model.accuracy
            )

            # Update regime-specific performance
            regime_idx = {"T": 0, "R": 1, "B": 2}.get(regime, 0)
            regime_scores = list(model.regime_scores)
            regime_scores[regime_idx] = (
                alpha * (1.0 if correct else 0.0)
                + (1 - alpha) * regime_scores[regime_idx]
            )
            model.regime_scores = tuple(regime_scores)

        # Minimal logging (memory conscious)
        self.recent_decisions.append(
            {
                "model": model_name,
                "pred": round(prediction, 3),
                "regime": regime,
                "ts": time.time(),
            }
        )

        self._decision_count += 1

    def compute_lightweight_weights(
        self, regime: str, vol_bucket: str, available_models: List[str]
    ) -> Dict[str, float]:
        """Ultra-fast weight computation optimized for T470"""

        # Cache key for performance
        cache_key = f"{regime}_{vol_bucket}_{len(available_models)}"

        # Use cached weights if recent
        if (
            cache_key in self.weight_cache
            and time.time() - self.weight_cache[cache_key]["timestamp"] < 60
        ):
            return self.weight_cache[cache_key]["weights"]

        weights = {}

        # Regime expertise lookup (pre-computed for speed)
        regime_expertise = {
            "T": {
                "LSTM": 0.9,
                "MiniTransformer": 0.95,
                "XGB": 0.85,
                "LightGBM": 0.9,
                "CNN": 0.6,
                "PPO": 0.7,
                "TinyAutoencoder": 0.7,
                "BayesianLite": 0.75,
                "MicroRL": 0.8,
            },
            "R": {
                "CNN": 0.95,
                "TinyAutoencoder": 0.9,
                "XGB": 0.8,
                "LightGBM": 0.85,
                "BayesianLite": 0.9,
                "LSTM": 0.6,
                "MiniTransformer": 0.7,
                "PPO": 0.5,
                "MicroRL": 0.6,
            },
            "B": {
                "CNN": 0.85,
                "PPO": 0.9,
                "MicroRL": 0.95,
                "TinyAutoencoder": 0.85,
                "MiniTransformer": 0.8,
                "LSTM": 0.7,
                "XGB": 0.6,
                "LightGBM": 0.7,
                "BayesianLite": 0.8,
            },
        }

        # Volatility adjustments (memory efficient lookup)
        vol_multipliers = {
            "Low": 1.0,
            "Med": 1.0,
            "High": 0.7,  # Reduce all weights in high vol
        }
        vol_mult = vol_multipliers.get(vol_bucket, 1.0)

        # Compute base weights
        for model_name in available_models:
            if model_name in self.models:
                model = self.models[model_name]

                # Base score: regime expertise + recent accuracy
                expertise = regime_expertise.get(regime, {}).get(model_name, 0.5)
                performance = model.accuracy

                base_weight = (expertise * 0.6 + performance * 0.4) * vol_mult
                weights[model_name] = max(0.05, base_weight)

        # Memory pressure adjustment: reduce ensemble size if needed
        if self._check_memory_pressure():
            # Keep only top 3 models
            sorted_models = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            weights = dict(sorted_models[:3])

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        # Cache result
        self.weight_cache[cache_key] = {"weights": weights, "timestamp": time.time()}

        # Prevent cache bloat (T470 memory management)
        if len(self.weight_cache) > 20:
            oldest_key = min(
                self.weight_cache.keys(),
                key=lambda k: self.weight_cache[k]["timestamp"],
            )
            del self.weight_cache[oldest_key]

        return weights

    def _check_memory_pressure(self) -> bool:
        """Check if we're approaching memory limits"""
        try:
            import psutil

            memory = psutil.virtual_memory()
            return memory.available < self.memory_pressure_threshold
        except:
            # Conservative fallback
            return len(self.recent_decisions) > 150

    def ensemble_predict(
        self, model_predictions: Dict[str, float], regime: str, vol_bucket: str
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        T470-optimized ensemble prediction
        Target: <5ms execution time
        """
        start_time = time.perf_counter_ns()

        # Get available models (filter by memory constraints)
        available_models = list(model_predictions.keys())

        # Adaptive ensemble size based on system resources
        max_models = 3 if self._check_memory_pressure() else 4
        if len(available_models) > max_models:
            # Select top performing models
            model_scores = [
                (name, self.models[name].accuracy)
                for name in available_models
                if name in self.models
            ]
            model_scores.sort(key=lambda x: x[1], reverse=True)
            available_models = [name for name, _ in model_scores[:max_models]]

        # Get lightweight weights
        weights = self.compute_lightweight_weights(regime, vol_bucket, available_models)

        # Fast ensemble computation
        ensemble_pred = 0.0
        total_weight = 0.0

        for model_name in available_models:
            if model_name in weights and model_name in model_predictions:
                weight = weights[model_name]
                prediction = model_predictions[model_name]
                ensemble_pred += weight * prediction
                total_weight += weight

        if total_weight > 0:
            ensemble_pred /= total_weight

        # Simple confidence estimation (fast)
        pred_values = [
            model_predictions[m] for m in available_models if m in model_predictions
        ]
        if len(pred_values) > 1:
            agreement = 1.0 - min(1.0, np.std(pred_values))
            confidence = 0.5 + 0.3 * agreement
        else:
            confidence = 0.6

        # Minimal metadata
        meta = {
            "models_used": len(available_models),
            "weights": {k: round(v, 3) for k, v in weights.items()},
            "memory_pressure": self._check_memory_pressure(),
            "latency_ns": time.perf_counter_ns() - start_time,
        }

        return float(ensemble_pred), float(confidence), meta

    def get_memory_footprint(self) -> Dict[str, int]:
        """Estimate memory usage (T470 monitoring)"""
        footprint = {
            "model_registry_kb": len(self.models) * 0.1,  # ~100 bytes per model
            "recent_decisions_kb": len(self.recent_decisions)
            * 0.2,  # ~200 bytes per decision
            "weight_cache_kb": len(self.weight_cache)
            * 0.5,  # ~500 bytes per cache entry
            "total_estimated_kb": 0,
        }

        # Add model memory usage
        total_model_kb = sum(model.memory_kb for model in self.models.values())
        footprint["models_total_kb"] = total_model_kb

        footprint["total_estimated_kb"] = (
            footprint["model_registry_kb"]
            + footprint["recent_decisions_kb"]
            + footprint["weight_cache_kb"]
        )

        return footprint

    def optimize_for_t470(self):
        """Runtime optimization for T470 constraints"""
        # Clear old cache entries
        current_time = time.time()
        self.weight_cache = {
            k: v
            for k, v in self.weight_cache.items()
            if current_time - v["timestamp"] < 300  # 5 minute cache
        }

        # Trim decision history if memory pressure
        if self._check_memory_pressure() and len(self.recent_decisions) > 100:
            # Keep only recent decisions
            while len(self.recent_decisions) > 100:
                self.recent_decisions.popleft()

        # Reset models with too few predictions (prevent overfitting)
        for model in self.models.values():
            if model.prediction_count > 10000:  # Reset after 10k predictions
                model.accuracy = 0.5
                model.prediction_count = 0
                model.regime_scores = (0.5, 0.5, 0.5)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Lightweight performance summary for T470"""
        memory_info = self.get_memory_footprint()

        # Top 3 models by performance
        model_rankings = [(name, model.accuracy) for name, model in self.models.items()]
        model_rankings.sort(key=lambda x: x[1], reverse=True)

        return {
            "timestamp": time.time(),
            "total_decisions": self._decision_count,
            "memory_footprint_kb": memory_info["total_estimated_kb"],
            "model_memory_kb": memory_info["models_total_kb"],
            "memory_pressure": self._check_memory_pressure(),
            "top_models": model_rankings[:3],
            "cache_entries": len(self.weight_cache),
            "recent_decisions": len(self.recent_decisions),
        }


# ---- Lightweight Extended Models for T470 ----
class MiniTransformer:
    """Tiny transformer optimized for T470 (≤3MB)"""

    def __init__(self):
        self.name = "MiniTransformer"
        self.context_length = 64  # Reduced from 512
        self.embed_dim = 32  # Reduced from 256

    def predict(self, market_data: Dict[str, Any]) -> float:
        """Mini transformer prediction"""
        if "series" not in market_data:
            return 0.0

        series = np.array(market_data["series"])
        if len(series) < 2:
            return 0.0

        # Ultra-simple attention mechanism
        context = series[-min(self.context_length, len(series)) :]

        # Position weighting (recent = more important)
        weights = np.exp(np.arange(len(context)) * 0.1)
        weights = weights / np.sum(weights)

        # Weighted prediction
        weighted_signal = np.dot(context, weights)
        prediction = (weighted_signal - np.mean(context)) / (np.std(context) + 1e-6)

        return float(np.tanh(prediction * 0.1))


class TinyAutoencoder:
    """Tiny autoencoder for T470 (≤1MB)"""

    def __init__(self):
        self.name = "TinyAutoencoder"
        self.input_dim = 32  # Reduced from 256
        self.latent_dim = 8  # Reduced from 32

    def predict(self, market_data: Dict[str, Any]) -> float:
        """Tiny autoencoder prediction"""
        # Simple PCA-like compression
        features = []

        if "ohlcv" in market_data and len(market_data["ohlcv"]) >= 10:
            ohlcv = np.array(market_data["ohlcv"][-10:])  # Last 10 bars only
            features.extend(ohlcv.flatten()[: self.input_dim])

        # Pad to input_dim
        while len(features) < self.input_dim:
            features.append(0.0)
        features = features[: self.input_dim]

        # Simple compression: group means
        compressed = []
        group_size = self.input_dim // self.latent_dim
        for i in range(self.latent_dim):
            start = i * group_size
            end = min((i + 1) * group_size, len(features))
            group_mean = np.mean(features[start:end])
            compressed.append(group_mean)

        prediction = np.mean(compressed) * 0.05
        return float(np.tanh(prediction))


class BayesianLite:
    """Ultra-lightweight Bayesian model (≤64KB)"""

    def __init__(self):
        self.name = "BayesianLite"
        self.mean = 0.0
        self.variance = 1.0
        self.count = 0

    def predict(self, market_data: Dict[str, Any]) -> float:
        """Bayesian lite prediction with online updates"""
        signal = 0.0
        if "series" in market_data and len(market_data["series"]) >= 2:
            series = market_data["series"]
            signal = (series[-1] - series[-2]) / (abs(series[-2]) + 1e-8)

        # Online Bayesian update
        self.count += 1
        alpha = 1.0 / self.count
        self.mean = (1 - alpha) * self.mean + alpha * signal

        # Conservative prediction with uncertainty scaling
        uncertainty = 1.0 / (1.0 + math.sqrt(self.count))
        prediction = self.mean * (1.0 - uncertainty)

        return float(np.tanh(prediction))


class MicroRL:
    """Micro RL policy for T470 (≤512KB)"""

    def __init__(self):
        self.name = "MicroRL"
        self.state_dim = 8  # Minimal state
        self.action_history = deque(maxlen=10)

    def predict(self, market_data: Dict[str, Any]) -> float:
        """Micro RL prediction"""
        # Minimal state construction
        state = [0.0] * self.state_dim

        if "series" in market_data and len(market_data["series"]) >= 5:
            series = np.array(market_data["series"][-5:])
            returns = np.diff(series) / (series[:-1] + 1e-8)

            state[0] = returns[-1]  # Last return
            state[1] = np.mean(returns)  # Average return
            state[2] = np.std(returns)  # Volatility
            state[3] = len(self.action_history) / 10.0  # Action count

        # Simple policy: momentum with volatility adjustment
        action = state[0] * (1.0 / (1.0 + state[2] * 5.0))  # Vol-adjusted momentum

        self.action_history.append(action)
        return float(np.tanh(action))


# Global instance optimized for T470
t470_ensemble = T470EnsembleBrain()
