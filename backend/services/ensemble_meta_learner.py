# -*- coding: utf-8 -*-
"""
Ensemble Meta-Learner: Self-managing multi-strategy hedge fund brain
Dynamically allocates between models, learns fusion logic, adapts to market regimes
"""
from __future__ import annotations
import os, json, time, math, threading, pathlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.getenv("ARIA_DATA_ROOT", PROJECT_ROOT / "data"))


@dataclass
class ModelPerformance:
    """Track model performance metrics"""

    name: str
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.5
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    recent_returns: deque = None
    confidence_calibration: float = 1.0
    regime_expertise: Dict[str, float] = None

    def __post_init__(self):
        if self.recent_returns is None:
            self.recent_returns = deque(maxlen=1000)
        if self.regime_expertise is None:
            self.regime_expertise = {"T": 0.5, "R": 0.5, "B": 0.5}


class EnsembleMetaLearner:
    """
    Multi-strategy hedge fund brain that self-manages model allocation
    Features:
    - Dynamic weight learning based on recent performance
    - Regime-specific model expertise tracking
    - Uncertainty-aware ensemble decisions
    - Online adaptation to market conditions
    """

    def __init__(self):
        self.models = {}  # ModelPerformance instances
        self.ensemble_history = deque(maxlen=10000)
        self.weight_history = deque(maxlen=1000)

        # Meta-learning parameters
        self.learning_rate = float(os.getenv("ARIA_META_LR", 0.01))
        self.decay_factor = float(os.getenv("ARIA_META_DECAY", 0.99))
        self.min_weight = float(os.getenv("ARIA_META_MIN_WEIGHT", 0.05))
        self.uncertainty_threshold = float(
            os.getenv("ARIA_META_UNCERTAINTY_THRESH", 0.8)
        )

        # Performance tracking
        self.regime_transition_detector = RegimeTransitionDetector()
        self.uncertainty_estimator = UncertaintyEstimator()

        self.lock = threading.Lock()

        # Initialize model registry
        self._initialize_model_registry()

    def _initialize_model_registry(self):
        """Initialize all available models in ARIA ecosystem"""
        model_configs = {
            # Core models (already implemented)
            "LSTM": {"type": "sequence", "expertise": {"T": 0.8, "R": 0.6, "B": 0.7}},
            "CNN": {"type": "pattern", "expertise": {"T": 0.6, "R": 0.9, "B": 0.8}},
            "PPO": {"type": "policy", "expertise": {"T": 0.7, "R": 0.5, "B": 0.9}},
            "XGB": {"type": "tabular", "expertise": {"T": 0.9, "R": 0.8, "B": 0.6}},
            # Extended models (to be implemented)
            "LightGBM": {
                "type": "tabular",
                "expertise": {"T": 0.9, "R": 0.8, "B": 0.7},
            },
            "Transformer": {
                "type": "sequence",
                "expertise": {"T": 0.9, "R": 0.7, "B": 0.8},
            },
            "Autoencoder": {
                "type": "latent",
                "expertise": {"T": 0.7, "R": 0.7, "B": 0.9},
            },
            "VAE": {"type": "latent", "expertise": {"T": 0.6, "R": 0.8, "B": 0.9}},
            "Bayesian": {
                "type": "probabilistic",
                "expertise": {"T": 0.7, "R": 0.9, "B": 0.8},
            },
            "AdaptiveRL": {
                "type": "policy",
                "expertise": {"T": 0.8, "R": 0.6, "B": 0.9},
            },
        }

        for name, config in model_configs.items():
            self.models[name] = ModelPerformance(
                name=name, regime_expertise=config["expertise"]
            )

    def register_model_prediction(
        self,
        model_name: str,
        prediction: float,
        confidence: float,
        regime: str,
        timestamp: float,
        meta: Optional[Dict] = None,
    ):
        """Register a model's prediction for meta-learning"""
        with self.lock:
            if model_name not in self.models:
                self.models[model_name] = ModelPerformance(name=model_name)

            model_perf = self.models[model_name]
            model_perf.total_predictions += 1

            # Store prediction for later evaluation
            pred_record = {
                "model": model_name,
                "prediction": prediction,
                "confidence": confidence,
                "regime": regime,
                "timestamp": timestamp,
                "meta": meta or {},
            }

            self.ensemble_history.append(pred_record)

    def update_model_performance(
        self, model_name: str, actual_return: float, prediction_timestamp: float
    ):
        """Update model performance based on actual outcomes"""
        with self.lock:
            if model_name not in self.models:
                return

            model_perf = self.models[model_name]

            # Find matching prediction
            for record in reversed(self.ensemble_history):
                if (
                    record["model"] == model_name
                    and abs(record["timestamp"] - prediction_timestamp) < 300
                ):  # 5 min window

                    # Update accuracy
                    predicted_direction = 1 if record["prediction"] > 0 else -1
                    actual_direction = 1 if actual_return > 0 else -1

                    if predicted_direction == actual_direction:
                        model_perf.correct_predictions += 1

                    model_perf.accuracy = model_perf.correct_predictions / max(
                        1, model_perf.total_predictions
                    )

                    # Update returns tracking
                    model_perf.recent_returns.append(actual_return)

                    # Update Sharpe ratio
                    if len(model_perf.recent_returns) >= 30:
                        returns = np.array(model_perf.recent_returns)
                        model_perf.sharpe_ratio = np.mean(returns) / (
                            np.std(returns) + 1e-8
                        )

                    # Update regime expertise
                    regime = record.get("regime", "T")
                    if regime in model_perf.regime_expertise:
                        # Exponential moving average update
                        alpha = 0.1
                        performance_signal = (
                            1.0 if predicted_direction == actual_direction else 0.0
                        )
                        model_perf.regime_expertise[regime] = (
                            alpha * performance_signal
                            + (1 - alpha) * model_perf.regime_expertise[regime]
                        )

                    break

    def compute_ensemble_weights(
        self, regime: str, vol_bucket: str, market_conditions: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compute dynamic ensemble weights based on:
        - Recent model performance
        - Regime-specific expertise
        - Market uncertainty
        - Model diversity
        """
        with self.lock:
            weights = {}

            # Get available models
            available_models = [
                name for name in self.models.keys() if self._is_model_available(name)
            ]

            if not available_models:
                return {"LSTM": 1.0}  # Fallback

            # Base weights from regime expertise
            for model_name in available_models:
                model_perf = self.models[model_name]

                # Regime expertise score
                expertise_score = model_perf.regime_expertise.get(regime, 0.5)

                # Recent performance score
                perf_score = model_perf.accuracy

                # Sharpe ratio contribution
                sharpe_score = np.tanh(model_perf.sharpe_ratio) * 0.5 + 0.5

                # Combine scores
                base_weight = (
                    expertise_score * 0.4 + perf_score * 0.4 + sharpe_score * 0.2
                )

                weights[model_name] = max(self.min_weight, base_weight)

            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}

            # Apply uncertainty adjustments
            uncertainty = self.uncertainty_estimator.estimate_uncertainty(
                regime, vol_bucket, market_conditions
            )

            if uncertainty > self.uncertainty_threshold:
                # High uncertainty: favor robust models
                weights = self._apply_uncertainty_adjustment(weights, uncertainty)

            # Apply diversity bonus
            weights = self._apply_diversity_bonus(weights)

            # Store weight history for analysis
            self.weight_history.append(
                {
                    "timestamp": time.time(),
                    "regime": regime,
                    "vol_bucket": vol_bucket,
                    "weights": weights.copy(),
                    "uncertainty": uncertainty,
                }
            )

            return weights

    def _is_model_available(self, model_name: str) -> bool:
        """Check if model is available for inference"""
        # For now, assume core models are always available
        core_models = {"LSTM", "CNN", "PPO", "XGB"}
        return model_name in core_models

    def _apply_uncertainty_adjustment(
        self, weights: Dict[str, float], uncertainty: float
    ) -> Dict[str, float]:
        """Adjust weights during high uncertainty periods"""
        # Favor models with better calibration during uncertainty
        adjusted_weights = {}

        for model_name, weight in weights.items():
            model_perf = self.models[model_name]

            # Boost conservative/robust models during uncertainty
            if model_name in ["Bayesian", "XGB", "LightGBM"]:
                uncertainty_bonus = 1.2
            elif model_name in ["PPO", "AdaptiveRL"]:
                uncertainty_bonus = 0.8  # Reduce reactive models
            else:
                uncertainty_bonus = 1.0

            adjusted_weights[model_name] = weight * uncertainty_bonus

        # Normalize
        total = sum(adjusted_weights.values())
        return {k: v / total for k, v in adjusted_weights.items()}

    def _apply_diversity_bonus(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply diversity bonus to encourage model variety"""
        model_types = {
            "LSTM": "sequence",
            "Transformer": "sequence",
            "CNN": "pattern",
            "Autoencoder": "latent",
            "VAE": "latent",
            "PPO": "policy",
            "AdaptiveRL": "policy",
            "XGB": "tabular",
            "LightGBM": "tabular",
            "Bayesian": "probabilistic",
        }

        # Count models per type
        type_counts = {}
        for model_name in weights.keys():
            model_type = model_types.get(model_name, "unknown")
            type_counts[model_type] = type_counts.get(model_type, 0) + 1

        # Apply diversity bonus (favor underrepresented types)
        adjusted_weights = {}
        for model_name, weight in weights.items():
            model_type = model_types.get(model_name, "unknown")
            diversity_bonus = 1.0 / max(1, type_counts[model_type])
            adjusted_weights[model_name] = weight * (1 + 0.1 * diversity_bonus)

        # Normalize
        total = sum(adjusted_weights.values())
        return {k: v / total for k, v in adjusted_weights.items()}

    def ensemble_predict(
        self,
        model_predictions: Dict[str, float],
        model_confidences: Dict[str, float],
        regime: str,
        vol_bucket: str,
        market_conditions: Dict[str, Any],
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Generate ensemble prediction with meta-learned weights

        Returns:
            (ensemble_prediction, ensemble_confidence, meta_info)
        """
        # Get dynamic weights
        weights = self.compute_ensemble_weights(regime, vol_bucket, market_conditions)

        # Compute weighted ensemble prediction
        ensemble_pred = 0.0
        total_weight = 0.0

        for model_name, prediction in model_predictions.items():
            if model_name in weights:
                weight = weights[model_name]
                ensemble_pred += weight * prediction
                total_weight += weight

        if total_weight > 0:
            ensemble_pred /= total_weight

        # Compute ensemble confidence
        ensemble_confidence = self._compute_ensemble_confidence(
            model_predictions, model_confidences, weights
        )

        # Meta information
        meta_info = {
            "weights_used": weights,
            "uncertainty": self.uncertainty_estimator.estimate_uncertainty(
                regime, vol_bucket, market_conditions
            ),
            "regime_transition_prob": self.regime_transition_detector.transition_probability(),
            "ensemble_diversity": self._compute_diversity_score(model_predictions),
            "meta_learning_confidence": self._compute_meta_confidence(),
        }

        return ensemble_pred, ensemble_confidence, meta_info

    def _compute_ensemble_confidence(
        self,
        predictions: Dict[str, float],
        confidences: Dict[str, float],
        weights: Dict[str, float],
    ) -> float:
        """Compute ensemble confidence considering prediction agreement"""
        if not predictions:
            return 0.5

        # Weighted average confidence
        weighted_conf = 0.0
        total_weight = 0.0

        for model_name, conf in confidences.items():
            if model_name in weights:
                weight = weights[model_name]
                weighted_conf += weight * conf
                total_weight += weight

        if total_weight > 0:
            weighted_conf /= total_weight

        # Agreement bonus (models agreeing increases confidence)
        pred_values = list(predictions.values())
        if len(pred_values) > 1:
            agreement = 1.0 - np.std(pred_values) / (
                np.mean(np.abs(pred_values)) + 1e-8
            )
            agreement_bonus = max(0, min(0.2, agreement * 0.2))
        else:
            agreement_bonus = 0.0

        ensemble_confidence = min(1.0, weighted_conf + agreement_bonus)
        return ensemble_confidence

    def _compute_diversity_score(self, predictions: Dict[str, float]) -> float:
        """Compute diversity score of ensemble predictions"""
        if len(predictions) < 2:
            return 0.0

        pred_values = np.array(list(predictions.values()))
        return float(np.std(pred_values))

    def _compute_meta_confidence(self) -> float:
        """Compute confidence in meta-learning decisions"""
        if len(self.weight_history) < 10:
            return 0.5

        # Stability of recent weight decisions
        recent_weights = list(self.weight_history)[-10:]
        weight_stability = self._compute_weight_stability(recent_weights)

        return weight_stability

    def _compute_weight_stability(self, weight_history: List[Dict]) -> float:
        """Compute stability of weight allocations"""
        if len(weight_history) < 2:
            return 0.5

        # Compute variance in weight allocations
        all_models = set()
        for w_dict in weight_history:
            all_models.update(w_dict["weights"].keys())

        stability_scores = []
        for model in all_models:
            weights = [w_dict["weights"].get(model, 0) for w_dict in weight_history]
            if len(weights) > 1:
                stability = 1.0 / (1.0 + np.var(weights))
                stability_scores.append(stability)

        return np.mean(stability_scores) if stability_scores else 0.5

    def get_model_rankings(self) -> List[Tuple[str, float]]:
        """Get current model rankings by performance"""
        rankings = []

        for model_name, model_perf in self.models.items():
            # Composite score
            score = (
                model_perf.accuracy * 0.4
                + (np.tanh(model_perf.sharpe_ratio) * 0.5 + 0.5) * 0.4
                + np.mean(list(model_perf.regime_expertise.values())) * 0.2
            )

            rankings.append((model_name, score))

        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        with self.lock:
            report = {
                "timestamp": time.time(),
                "ensemble_stats": {
                    "total_predictions": len(self.ensemble_history),
                    "models_tracked": len(self.models),
                    "weight_decisions": len(self.weight_history),
                },
                "model_performance": {},
                "recent_weight_allocation": (
                    self.weight_history[-1] if self.weight_history else {}
                ),
                "model_rankings": self.get_model_rankings(),
            }

            # Individual model stats
            for model_name, model_perf in self.models.items():
                report["model_performance"][model_name] = {
                    "accuracy": model_perf.accuracy,
                    "sharpe_ratio": model_perf.sharpe_ratio,
                    "total_predictions": model_perf.total_predictions,
                    "regime_expertise": model_perf.regime_expertise,
                    "available": self._is_model_available(model_name),
                }

            return report


class RegimeTransitionDetector:
    """Detect regime transitions for meta-learning adjustments"""

    def __init__(self):
        self.recent_regimes = deque(maxlen=50)

    def update(self, regime: str):
        self.recent_regimes.append(regime)

    def transition_probability(self) -> float:
        """Estimate probability of regime transition"""
        if len(self.recent_regimes) < 10:
            return 0.5

        # Count transitions in recent history
        transitions = 0
        for i in range(1, len(self.recent_regimes)):
            if self.recent_regimes[i] != self.recent_regimes[i - 1]:
                transitions += 1

        return transitions / max(1, len(self.recent_regimes) - 1)


class UncertaintyEstimator:
    """Estimate market uncertainty for ensemble adjustments"""

    def estimate_uncertainty(
        self, regime: str, vol_bucket: str, market_conditions: Dict[str, Any]
    ) -> float:
        """Estimate current market uncertainty"""
        uncertainty_factors = []

        # Volatility factor
        vol_uncertainty = {"Low": 0.2, "Med": 0.5, "High": 0.8}.get(vol_bucket, 0.5)
        uncertainty_factors.append(vol_uncertainty)

        # Regime factor (Breakout = more uncertain)
        regime_uncertainty = {"T": 0.3, "R": 0.2, "B": 0.7}.get(regime, 0.5)
        uncertainty_factors.append(regime_uncertainty)

        # News/event factor (if available)
        if "news_importance" in market_conditions:
            news_uncertainty = min(1.0, market_conditions["news_importance"] / 10.0)
            uncertainty_factors.append(news_uncertainty)

        # Spread factor (wider spreads = more uncertainty)
        if "spread_z" in market_conditions:
            spread_uncertainty = min(1.0, abs(market_conditions["spread_z"]) / 3.0)
            uncertainty_factors.append(spread_uncertainty)

        return np.mean(uncertainty_factors)


# Global instance
ensemble_meta_learner = EnsembleMetaLearner()
