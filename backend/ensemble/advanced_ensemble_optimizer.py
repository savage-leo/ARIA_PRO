"""
Advanced Ensemble Layer Optimization
Combines LSTM + CNN + PPO + Visual + LLM signals with adaptive meta-learning
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
import time
import threading
from collections import deque

from backend.core.model_loader import cached_models
# Avoid circular import - import at runtime when needed

logger = logging.getLogger(__name__)

@dataclass
class EnsembleWeights:
    """Dynamic ensemble weights with confidence scores"""
    lstm: float = 0.2
    cnn: float = 0.2
    ppo: float = 0.2
    xgb: float = 0.2
    visual: float = 0.1
    llm_macro: float = 0.1
    confidence: float = 0.5
    timestamp: float = 0.0
    market_regime: str = "normal"


@dataclass
class SignalMetrics:
    """Performance metrics for individual signals"""
    accuracy: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    avg_return: float = 0.0
    volatility: float = 0.0
    correlation_with_market: float = 0.0


class AdaptiveEnsembleOptimizer:
    """Advanced ensemble optimizer with meta-learning and regime adaptation"""
    
    def __init__(self, lookback_window: int = 1000, adaptation_rate: float = 0.1):
        self.lookback_window = lookback_window
        self.adaptation_rate = adaptation_rate
        
        # Signal history for optimization
        self.signal_history = deque(maxlen=lookback_window)
        self.performance_history = deque(maxlen=lookback_window)
        self.market_returns = deque(maxlen=lookback_window)
        
        # Current ensemble weights
        self.current_weights = EnsembleWeights()
        self.weight_history = deque(maxlen=100)
        
        # Meta-learning models
        self.meta_models = {
            "regime_detector": None,
            "weight_optimizer": None,
            "confidence_estimator": None
        }
        
        # Performance tracking
        self.signal_metrics = {
            "lstm": SignalMetrics(),
            "cnn": SignalMetrics(),
            "ppo": SignalMetrics(),
            "xgb": SignalMetrics(),
            "visual": SignalMetrics(),
            "llm_macro": SignalMetrics()
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Regime detection parameters
        self.regime_features = ["volatility", "trend", "momentum", "volume"]
        self.regime_states = ["trending_up", "trending_down", "sideways", "volatile", "low_volatility"]
        
        # Initialize meta-models
        self._initialize_meta_models()
    
    def _initialize_meta_models(self):
        """Initialize meta-learning models"""
        try:
            # Regime detector - Random Forest for market regime classification
            self.meta_models["regime_detector"] = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=1  # CPU-only
            )
            
            # Weight optimizer - Ridge regression for optimal weight prediction
            self.meta_models["weight_optimizer"] = Ridge(
                alpha=1.0,
                random_state=42
            )
            
            # Confidence estimator - Elastic Net for confidence scoring
            self.meta_models["confidence_estimator"] = ElasticNet(
                alpha=0.5,
                l1_ratio=0.5,
                random_state=42
            )
            
            logger.info("Meta-learning models initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize meta-models: {e}")
    
    def add_signal_observation(self, signals: Dict[str, float], market_return: float = 0.0, 
                             market_features: Dict[str, float] = None):
        """Add new signal observation for learning"""
        with self.lock:
            timestamp = time.time()
            
            # Store signal observation
            signal_obs = {
                "timestamp": timestamp,
                "signals": signals.copy(),
                "market_return": market_return,
                "features": market_features or {}
            }
            self.signal_history.append(signal_obs)
            self.market_returns.append(market_return)
            
            # Update signal metrics
            self._update_signal_metrics(signals, market_return)
            
            # Trigger adaptation if enough data
            if len(self.signal_history) >= 50:
                self._adapt_weights()
    
    def _update_signal_metrics(self, signals: Dict[str, float], market_return: float):
        """Update performance metrics for individual signals"""
        try:
            for signal_name, signal_value in signals.items():
                if signal_name not in self.signal_metrics:
                    continue
                
                metrics = self.signal_metrics[signal_name]
                
                # Simple performance tracking
                if abs(signal_value) > 0.1:  # Only count significant signals
                    signal_return = signal_value * market_return
                    
                    # Update running averages
                    alpha = 0.1  # Learning rate
                    metrics.avg_return = (1 - alpha) * metrics.avg_return + alpha * signal_return
                    metrics.hit_rate = (1 - alpha) * metrics.hit_rate + alpha * (1.0 if signal_return > 0 else 0.0)
                    
                    # Update volatility
                    metrics.volatility = (1 - alpha) * metrics.volatility + alpha * abs(signal_return)
                    
                    # Update correlation with market
                    if market_return != 0:
                        correlation = np.sign(signal_value) == np.sign(market_return)
                        metrics.correlation_with_market = (1 - alpha) * metrics.correlation_with_market + alpha * (1.0 if correlation else 0.0)
        
        except Exception as e:
            logger.warning(f"Error updating signal metrics: {e}")
    
    def _extract_market_features(self, recent_observations: List[Dict]) -> np.ndarray:
        """Extract market regime features from recent observations"""
        try:
            if len(recent_observations) < 20:
                return np.zeros(len(self.regime_features))
            
            returns = [obs["market_return"] for obs in recent_observations[-20:]]
            
            # Calculate regime features
            volatility = np.std(returns) if len(returns) > 1 else 0.0
            trend = np.mean(returns) if returns else 0.0
            momentum = returns[-1] - returns[-5] if len(returns) >= 5 else 0.0
            volume = np.mean([obs["features"].get("volume", 1000) for obs in recent_observations[-10:]])
            
            return np.array([volatility, trend, momentum, volume])
            
        except Exception as e:
            logger.warning(f"Error extracting market features: {e}")
            return np.zeros(len(self.regime_features))
    
    def _detect_market_regime(self, market_features: np.ndarray) -> str:
        """Detect current market regime"""
        try:
            if self.meta_models["regime_detector"] is None:
                return "normal"
            
            # Simple rule-based regime detection
            volatility, trend, momentum, volume = market_features
            
            if volatility > 0.02:
                return "volatile"
            elif volatility < 0.005:
                return "low_volatility"
            elif abs(trend) > 0.001:
                return "trending_up" if trend > 0 else "trending_down"
            else:
                return "sideways"
                
        except Exception as e:
            logger.warning(f"Error detecting market regime: {e}")
            return "normal"
    
    def _adapt_weights(self):
        """Adapt ensemble weights based on recent performance"""
        try:
            if len(self.signal_history) < 50:
                return
            
            recent_obs = list(self.signal_history)[-50:]
            
            # Extract features and targets for meta-learning
            X_features = []
            y_performance = []
            
            for i in range(10, len(recent_obs)):
                # Market features
                market_features = self._extract_market_features(recent_obs[i-10:i])
                
                # Signal features
                signals = recent_obs[i]["signals"]
                signal_features = [signals.get(name, 0.0) for name in ["lstm", "cnn", "ppo", "xgb", "visual", "llm_macro"]]
                
                # Combined features
                features = np.concatenate([market_features, signal_features])
                X_features.append(features)
                
                # Performance target (next period return)
                if i < len(recent_obs) - 1:
                    next_return = recent_obs[i + 1]["market_return"]
                    y_performance.append(next_return)
            
            if len(X_features) < 20:
                return
            
            X = np.array(X_features[:-1])  # Exclude last observation
            y = np.array(y_performance)
            
            # Detect current regime
            current_features = self._extract_market_features(recent_obs[-10:])
            current_regime = self._detect_market_regime(current_features)
            
            # Optimize weights based on regime and performance
            new_weights = self._optimize_weights_for_regime(X, y, current_regime)
            
            # Update current weights with adaptation rate
            self._update_weights(new_weights, current_regime)
            
            logger.info(f"Adapted weights for regime: {current_regime}")
            
        except Exception as e:
            logger.error(f"Error adapting weights: {e}")
    
    def _optimize_weights_for_regime(self, X: np.ndarray, y: np.ndarray, regime: str) -> Dict[str, float]:
        """Optimize ensemble weights for specific market regime"""
        try:
            # Calculate individual signal performance
            signal_names = ["lstm", "cnn", "ppo", "xgb", "visual", "llm_macro"]
            signal_performance = {}
            
            for i, name in enumerate(signal_names):
                if len(X) > 0:
                    signal_values = X[:, 4 + i]  # Signal features start at index 4
                    
                    # Calculate correlation with future returns
                    if len(signal_values) > 1 and np.std(signal_values) > 0:
                        correlation = np.corrcoef(signal_values, y)[0, 1]
                        signal_performance[name] = abs(correlation) if not np.isnan(correlation) else 0.0
                    else:
                        signal_performance[name] = 0.0
                else:
                    signal_performance[name] = 0.0
            
            # Regime-specific adjustments
            regime_adjustments = self._get_regime_adjustments(regime)
            
            # Calculate optimized weights
            total_performance = sum(signal_performance.values()) + 1e-8
            optimized_weights = {}
            
            for name in signal_names:
                base_weight = signal_performance[name] / total_performance
                adjusted_weight = base_weight * regime_adjustments.get(name, 1.0)
                optimized_weights[name] = max(0.01, min(0.5, adjusted_weight))  # Clamp weights
            
            # Normalize weights
            total_weight = sum(optimized_weights.values())
            if total_weight > 0:
                for name in optimized_weights:
                    optimized_weights[name] /= total_weight
            
            return optimized_weights
            
        except Exception as e:
            logger.error(f"Error optimizing weights: {e}")
            return {"lstm": 0.2, "cnn": 0.2, "ppo": 0.2, "xgb": 0.2, "visual": 0.1, "llm_macro": 0.1}
    
    def _get_regime_adjustments(self, regime: str) -> Dict[str, float]:
        """Get regime-specific weight adjustments"""
        adjustments = {
            "trending_up": {
                "lstm": 1.2,  # LSTM good at trend following
                "cnn": 1.1,   # CNN good at pattern recognition
                "ppo": 1.3,   # PPO good at trend trading
                "xgb": 1.0,
                "visual": 0.9,
                "llm_macro": 1.1
            },
            "trending_down": {
                "lstm": 1.2,
                "cnn": 1.1,
                "ppo": 1.3,
                "xgb": 1.0,
                "visual": 0.9,
                "llm_macro": 1.2  # Macro analysis important in downtrends
            },
            "sideways": {
                "lstm": 0.9,
                "cnn": 1.2,   # CNN good at range patterns
                "ppo": 0.8,   # PPO less effective in ranges
                "xgb": 1.3,   # XGBoost good at complex patterns
                "visual": 1.1,
                "llm_macro": 0.9
            },
            "volatile": {
                "lstm": 0.8,
                "cnn": 1.1,
                "ppo": 0.7,   # PPO can struggle with volatility
                "xgb": 1.2,
                "visual": 1.3, # Visual patterns important in volatility
                "llm_macro": 1.1
            },
            "low_volatility": {
                "lstm": 1.1,
                "cnn": 1.0,
                "ppo": 1.2,
                "xgb": 1.0,
                "visual": 0.8,
                "llm_macro": 1.0
            }
        }
        
        return adjustments.get(regime, {name: 1.0 for name in ["lstm", "cnn", "ppo", "xgb", "visual", "llm_macro"]})
    
    def _update_weights(self, new_weights: Dict[str, float], regime: str):
        """Update current weights with adaptation rate"""
        with self.lock:
            # Apply adaptation rate
            for name in ["lstm", "cnn", "ppo", "xgb", "visual", "llm_macro"]:
                current_weight = getattr(self.current_weights, name, 0.0)
                new_weight = new_weights.get(name, current_weight)
                
                # Smooth adaptation
                adapted_weight = (1 - self.adaptation_rate) * current_weight + self.adaptation_rate * new_weight
                setattr(self.current_weights, name, adapted_weight)
            
            # Update metadata
            self.current_weights.timestamp = time.time()
            self.current_weights.market_regime = regime
            self.current_weights.confidence = self._calculate_confidence()
            
            # Store in history
            self.weight_history.append(self.current_weights.__dict__.copy())
    
    def _calculate_confidence(self) -> float:
        """Calculate confidence in current ensemble weights"""
        try:
            if len(self.signal_history) < 20:
                return 0.5
            
            # Calculate recent performance consistency
            recent_returns = list(self.market_returns)[-20:]
            if len(recent_returns) < 2:
                return 0.5
            
            volatility = np.std(recent_returns)
            consistency = 1.0 / (1.0 + volatility * 10)  # Lower volatility = higher confidence
            
            # Factor in signal agreement
            recent_signals = [obs["signals"] for obs in list(self.signal_history)[-10:]]
            if recent_signals:
                signal_agreement = self._calculate_signal_agreement(recent_signals)
                confidence = 0.7 * consistency + 0.3 * signal_agreement
            else:
                confidence = consistency
            
            return max(0.1, min(0.9, confidence))
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.5
    
    def _calculate_signal_agreement(self, recent_signals: List[Dict[str, float]]) -> float:
        """Calculate agreement between different signals"""
        try:
            if not recent_signals:
                return 0.5
            
            # Calculate correlation matrix between signals
            signal_names = ["lstm", "cnn", "ppo", "xgb", "visual", "llm_macro"]
            signal_matrix = []
            
            for signals in recent_signals:
                row = [signals.get(name, 0.0) for name in signal_names]
                signal_matrix.append(row)
            
            if len(signal_matrix) < 2:
                return 0.5
            
            signal_matrix = np.array(signal_matrix)
            
            # Calculate average correlation
            correlations = []
            for i in range(len(signal_names)):
                for j in range(i + 1, len(signal_names)):
                    col_i = signal_matrix[:, i]
                    col_j = signal_matrix[:, j]
                    
                    if np.std(col_i) > 0 and np.std(col_j) > 0:
                        corr = np.corrcoef(col_i, col_j)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
            
            if correlations:
                avg_correlation = np.mean(correlations)
                return max(0.1, min(0.9, avg_correlation))
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"Error calculating signal agreement: {e}")
            return 0.5
    
    def get_optimized_signal(self, raw_signals: Dict[str, float], 
                           market_features: Dict[str, float] = None) -> Tuple[float, float]:
        """Get optimized ensemble signal with confidence"""
        with self.lock:
            try:
                # Apply current weights
                weighted_signal = 0.0
                total_weight = 0.0
                
                for signal_name, signal_value in raw_signals.items():
                    if signal_value is not None and hasattr(self.current_weights, signal_name):
                        weight = getattr(self.current_weights, signal_name)
                        weighted_signal += weight * signal_value
                        total_weight += weight
                
                # Normalize
                if total_weight > 0:
                    weighted_signal /= total_weight
                
                # Apply confidence scaling
                confidence = self.current_weights.confidence
                final_signal = weighted_signal * confidence
                
                return final_signal, confidence
                
            except Exception as e:
                logger.error(f"Error getting optimized signal: {e}")
                return 0.0, 0.5
    
    def get_current_weights(self) -> Dict[str, Any]:
        """Get current ensemble weights and metadata"""
        with self.lock:
            return {
                "weights": {
                    "lstm": self.current_weights.lstm,
                    "cnn": self.current_weights.cnn,
                    "ppo": self.current_weights.ppo,
                    "xgb": self.current_weights.xgb,
                    "visual": self.current_weights.visual,
                    "llm_macro": self.current_weights.llm_macro
                },
                "confidence": self.current_weights.confidence,
                "market_regime": self.current_weights.market_regime,
                "timestamp": self.current_weights.timestamp,
                "signal_metrics": {
                    name: {
                        "avg_return": metrics.avg_return,
                        "hit_rate": metrics.hit_rate,
                        "volatility": metrics.volatility,
                        "correlation": metrics.correlation_with_market
                    }
                    for name, metrics in self.signal_metrics.items()
                }
            }
    
    def save_model(self, filepath: str):
        """Save ensemble optimizer state"""
        try:
            state = {
                "current_weights": self.current_weights.__dict__,
                "signal_metrics": {name: metrics.__dict__ for name, metrics in self.signal_metrics.items()},
                "weight_history": list(self.weight_history),
                "lookback_window": self.lookback_window,
                "adaptation_rate": self.adaptation_rate
            }
            
            joblib.dump(state, filepath)
            logger.info(f"Ensemble optimizer saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving ensemble optimizer: {e}")
    
    def load_model(self, filepath: str):
        """Load ensemble optimizer state"""
        try:
            state = joblib.load(filepath)
            
            # Restore weights
            weights_dict = state["current_weights"]
            self.current_weights = EnsembleWeights(**weights_dict)
            
            # Restore metrics
            for name, metrics_dict in state["signal_metrics"].items():
                if name in self.signal_metrics:
                    self.signal_metrics[name] = SignalMetrics(**metrics_dict)
            
            # Restore history
            self.weight_history = deque(state["weight_history"], maxlen=100)
            
            logger.info(f"Ensemble optimizer loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading ensemble optimizer: {e}")


class EnhancedSignalGenerator:
    """Enhanced signal generator with optimized ensemble"""
    
    def __init__(self):
        # Import at runtime to avoid circular import
        from backend.services.real_ai_signal_generator import RealAISignalGenerator
        self.base_generator = RealAISignalGenerator()
        self.ensemble_optimizer = AdaptiveEnsembleOptimizer()
        
    def generate_optimized_signals(self, symbol: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimized ensemble signals"""
        try:
            # Get raw signals from base generator
            raw_signals = self.base_generator.get_signals(symbol, features)
            
            if not raw_signals:
                return {"signal": 0.0, "confidence": 0.0, "raw_signals": {}}
            
            # Get optimized ensemble signal
            optimized_signal, confidence = self.ensemble_optimizer.get_optimized_signal(raw_signals)
            
            # Get current weights for transparency
            weights_info = self.ensemble_optimizer.get_current_weights()
            
            return {
                "signal": optimized_signal,
                "confidence": confidence,
                "raw_signals": raw_signals,
                "weights": weights_info["weights"],
                "market_regime": weights_info["market_regime"],
                "ensemble_metadata": {
                    "adaptation_timestamp": weights_info["timestamp"],
                    "signal_metrics": weights_info["signal_metrics"]
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating optimized signals: {e}")
            return {"signal": 0.0, "confidence": 0.0, "raw_signals": {}}
    
    def update_performance(self, symbol: str, market_return: float, market_features: Dict[str, float] = None):
        """Update ensemble performance with market feedback"""
        try:
            # Get recent signals for this symbol
            recent_signals = self.base_generator.get_signals(symbol, {})
            
            if recent_signals:
                self.ensemble_optimizer.add_signal_observation(recent_signals, market_return, market_features)
                
        except Exception as e:
            logger.error(f"Error updating performance: {e}")


# Global enhanced signal generator
enhanced_signal_generator = EnhancedSignalGenerator()
