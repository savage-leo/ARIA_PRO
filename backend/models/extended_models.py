# -*- coding: utf-8 -*-
"""
Extended Models: Missing quant desk essentials for multi-strategy alpha
XGBoost/LightGBM, Transformers, Autoencoders, Bayesian, Advanced RL
"""
from __future__ import annotations
import os, time, math, pathlib
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


# ---- XGBoost/LightGBM Enhanced ----
class XGBoostEnhanced:
    """Enhanced XGBoost with automatic feature engineering"""

    def __init__(self, model_path: Optional[str] = None):
        self.name = "XGBoost_Enhanced"
        self.model_path = model_path or "backend/models/xgboost_enhanced.onnx"
        self.model = None
        self.feature_importance = {}

    def engineer_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Advanced feature engineering for tabular data"""
        features = []

        # Price-based features
        if "ohlcv" in market_data and len(market_data["ohlcv"]) >= 20:
            ohlcv = np.array(market_data["ohlcv"])
            closes = ohlcv[:, 3]  # Close prices
            highs = ohlcv[:, 1]  # High prices
            lows = ohlcv[:, 2]  # Low prices
            volumes = ohlcv[:, 4]  # Volumes

            # Technical indicators
            features.extend(
                [
                    # Returns and momentum
                    (closes[-1] - closes[-2]) / closes[-2],  # 1-period return
                    (closes[-1] - closes[-5]) / closes[-5],  # 5-period return
                    (closes[-1] - closes[-20]) / closes[-20],  # 20-period return
                    # Volatility measures
                    np.std(closes[-20:]) / np.mean(closes[-20:]),  # CV
                    (highs[-20:] - lows[-20:]).mean()
                    / closes[-20:].mean(),  # ATR ratio
                    # Volume features
                    (
                        volumes[-1] / volumes[-20:].mean()
                        if volumes[-20:].mean() > 0
                        else 1.0
                    ),
                    (
                        np.corrcoef(closes[-20:], volumes[-20:])[0, 1]
                        if len(closes) >= 20
                        else 0.0
                    ),
                    # Price position
                    (closes[-1] - lows[-20:].min())
                    / (highs[-20:].max() - lows[-20:].min() + 1e-8),
                    # Trend strength
                    np.polyfit(range(20), closes[-20:], 1)[0] / closes[-1],
                ]
            )

        # Regime features
        if "regime_probs" in market_data:
            regime_probs = market_data["regime_probs"]
            features.extend(
                [
                    regime_probs.get("T", 0.33),
                    regime_probs.get("R", 0.33),
                    regime_probs.get("B", 0.33),
                    max(regime_probs.values()),  # Regime confidence
                ]
            )

        # Market microstructure
        if "spread" in market_data:
            features.append(market_data["spread"])
        if "session" in market_data:
            # One-hot encode session
            sessions = ["ASIA", "EU", "US"]
            for session in sessions:
                features.append(1.0 if market_data["session"] == session else 0.0)

        # Cross-asset features (if available)
        if "correlations" in market_data:
            corr_data = market_data["correlations"]
            features.extend(list(corr_data.values())[:5])  # Top 5 correlations

        # Ensure consistent feature count
        target_features = 25
        if len(features) < target_features:
            features.extend([0.0] * (target_features - len(features)))
        elif len(features) > target_features:
            features = features[:target_features]

        return np.array(features, dtype=np.float32)

    def predict(self, market_data: Dict[str, Any]) -> float:
        """Predict using enhanced XGBoost with feature engineering"""
        features = self.engineer_features(market_data)

        # Mock prediction for now (replace with actual ONNX inference)
        # Real implementation would load ONNX model and run inference

        # Simple weighted combination based on feature importance
        weights = np.array([0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05] + [0.037] * 18)
        prediction = np.dot(features, weights)

        return float(np.tanh(prediction))


class LightGBMAdapter:
    """LightGBM adapter optimized for low-latency inference"""

    def __init__(self, model_path: Optional[str] = None):
        self.name = "LightGBM"
        self.model_path = model_path or "backend/models/lightgbm.onnx"
        self.model = None

    def predict(self, market_data: Dict[str, Any]) -> float:
        """Fast LightGBM prediction"""
        # Similar feature engineering to XGBoost but optimized for speed
        xgb_enhanced = XGBoostEnhanced()
        features = xgb_enhanced.engineer_features(market_data)

        # LightGBM tends to be more conservative
        weights = np.array([0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04] + [0.04] * 18)
        prediction = np.dot(features, weights) * 0.8  # More conservative

        return float(np.tanh(prediction))


# ---- Transformer Models ----
class TransformerAdapter:
    """Transformer model for longer sequence contexts"""

    def __init__(self, model_path: Optional[str] = None):
        self.name = "Transformer"
        self.model_path = model_path or "backend/models/transformer.onnx"
        self.context_length = 512
        self.model = None

    def prepare_sequence(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Prepare sequence data for transformer"""
        if "series" not in market_data:
            return np.zeros((self.context_length, 1), dtype=np.float32)

        series = np.array(market_data["series"])

        # Normalize and pad/truncate to context length
        if len(series) > self.context_length:
            series = series[-self.context_length :]
        else:
            series = np.pad(series, (self.context_length - len(series), 0), "edge")

        # Add positional encoding (simplified)
        positions = np.arange(self.context_length) / self.context_length
        sequence = np.column_stack([series, positions])

        return sequence.astype(np.float32)

    def predict(self, market_data: Dict[str, Any]) -> float:
        """Transformer prediction with attention over long sequences"""
        sequence = self.prepare_sequence(market_data)

        # Mock transformer attention (replace with actual model)
        # Real implementation: attention weights over sequence
        recent_weight = 0.7
        historical_weight = 0.3

        recent_signal = np.mean(sequence[-20:, 0])  # Recent prices
        historical_signal = np.mean(sequence[:-20, 0])  # Historical context

        prediction = (
            recent_weight * recent_signal + historical_weight * historical_signal
        )

        return float(np.tanh(prediction * 0.5))


# ---- Autoencoder/VAE ----
class AutoencoderAdapter:
    """Autoencoder for latent feature extraction and denoising"""

    def __init__(self, model_path: Optional[str] = None):
        self.name = "Autoencoder"
        self.model_path = model_path or "backend/models/autoencoder.onnx"
        self.latent_dim = 32
        self.model = None

    def encode_market_state(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Encode market data into latent representation"""
        # Combine multiple data sources into single representation
        features = []

        # Price data
        if "ohlcv" in market_data:
            ohlcv = np.array(market_data["ohlcv"])
            if len(ohlcv) >= 50:
                # Recent OHLCV patterns
                recent_ohlcv = ohlcv[-50:].flatten()[:200]  # Limit size
                features.extend(recent_ohlcv.tolist())

        # Pad to fixed size for autoencoder
        target_size = 256
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]

        # Mock autoencoder encoding (replace with actual model)
        input_data = np.array(features)

        # Simulated latent encoding: PCA-like dimensionality reduction
        latent_features = []
        chunk_size = len(input_data) // self.latent_dim

        for i in range(self.latent_dim):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(input_data))
            if start_idx < len(input_data):
                chunk_mean = np.mean(input_data[start_idx:end_idx])
                latent_features.append(chunk_mean)
            else:
                latent_features.append(0.0)

        return np.array(latent_features, dtype=np.float32)

    def predict(self, market_data: Dict[str, Any]) -> float:
        """Predict using latent features from autoencoder"""
        latent_features = self.encode_market_state(market_data)

        # Use latent features for prediction
        prediction = np.mean(latent_features) * 0.1  # Conservative scaling

        return float(np.tanh(prediction))


class VAEAdapter:
    """Variational Autoencoder for probabilistic latent features"""

    def __init__(self, model_path: Optional[str] = None):
        self.name = "VAE"
        self.model_path = model_path or "backend/models/vae.onnx"
        self.latent_dim = 16
        self.model = None

    def encode_probabilistic(
        self, market_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode to probabilistic latent space (mean, logvar)"""
        # Similar to autoencoder but returns mean and variance
        ae_adapter = AutoencoderAdapter()
        latent_mean = ae_adapter.encode_market_state(market_data)[: self.latent_dim]

        # Mock variance (replace with actual VAE)
        latent_logvar = np.random.normal(0, 0.1, self.latent_dim).astype(np.float32)

        return latent_mean, latent_logvar

    def predict(self, market_data: Dict[str, Any]) -> float:
        """VAE prediction with uncertainty estimation"""
        latent_mean, latent_logvar = self.encode_probabilistic(market_data)

        # Sample from latent distribution
        latent_std = np.exp(0.5 * latent_logvar)
        latent_sample = latent_mean + latent_std * np.random.normal(
            0, 1, self.latent_dim
        )

        prediction = np.mean(latent_sample) * 0.05

        return float(np.tanh(prediction))


# ---- Bayesian Models ----
class BayesianAdapter:
    """Bayesian model for uncertainty-aware predictions"""

    def __init__(self):
        self.name = "Bayesian"
        self.prior_mean = 0.0
        self.prior_variance = 1.0
        self.observations = []
        self.model = None

    def update_posterior(self, observation: float, likelihood_precision: float = 1.0):
        """Update Bayesian posterior with new observation"""
        # Bayesian update
        prior_precision = 1.0 / self.prior_variance
        posterior_precision = prior_precision + likelihood_precision

        posterior_variance = 1.0 / posterior_precision
        posterior_mean = (
            prior_precision * self.prior_mean + likelihood_precision * observation
        ) / posterior_precision

        self.prior_mean = posterior_mean
        self.prior_variance = posterior_variance

        self.observations.append(observation)
        if len(self.observations) > 1000:
            self.observations.pop(0)

    def predict_with_uncertainty(
        self, market_data: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Predict with uncertainty bounds"""
        # Extract signal from market data
        signal = 0.0
        if "series" in market_data and len(market_data["series"]) >= 2:
            series = market_data["series"]
            signal = (series[-1] - series[-2]) / (abs(series[-2]) + 1e-8)

        # Bayesian prediction
        prediction_mean = self.prior_mean + 0.1 * signal
        prediction_variance = self.prior_variance + 0.01  # Model uncertainty

        return prediction_mean, math.sqrt(prediction_variance)

    def predict(self, market_data: Dict[str, Any]) -> float:
        """Standard prediction interface"""
        pred_mean, pred_std = self.predict_with_uncertainty(market_data)

        # Conservative prediction scaled by uncertainty
        confidence = 1.0 / (1.0 + pred_std)

        return float(np.tanh(pred_mean * confidence))


# ---- Advanced RL ----
class AdaptiveRLAdapter:
    """Advanced RL for adaptive execution policy"""

    def __init__(self, model_path: Optional[str] = None):
        self.name = "AdaptiveRL"
        self.model_path = model_path or "backend/models/adaptive_rl.onnx"
        self.model = None
        self.state_history = []
        self.action_history = []
        self.reward_history = []

    def construct_state(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Construct RL state from market data"""
        state_features = []

        # Market features
        if "series" in market_data and len(market_data["series"]) >= 10:
            series = np.array(market_data["series"][-10:])
            returns = np.diff(series) / (series[:-1] + 1e-8)
            state_features.extend(
                [
                    np.mean(returns),
                    np.std(returns),
                    returns[-1],  # Last return
                    (series[-1] - series[0]) / (series[0] + 1e-8),  # Total return
                ]
            )

        # Position information (if available)
        current_position = market_data.get("current_position", 0.0)
        position_pnl = market_data.get("position_pnl", 0.0)

        state_features.extend(
            [
                current_position,
                position_pnl,
                len(self.state_history),  # Time since start
            ]
        )

        # Recent action history
        if len(self.action_history) >= 3:
            state_features.extend(self.action_history[-3:])
        else:
            state_features.extend([0.0] * 3)

        # Ensure fixed state size
        target_size = 15
        if len(state_features) < target_size:
            state_features.extend([0.0] * (target_size - len(state_features)))
        else:
            state_features = state_features[:target_size]

        return np.array(state_features, dtype=np.float32)

    def predict(self, market_data: Dict[str, Any]) -> float:
        """RL policy prediction"""
        state = self.construct_state(market_data)
        self.state_history.append(state)

        # Mock RL policy (replace with actual trained policy)
        # Real implementation would use trained neural network policy

        # Simple policy based on state features
        if len(state) >= 4:
            momentum = state[2]  # Last return
            volatility = state[1]  # Volatility

            # Adaptive policy: more aggressive in low vol, conservative in high vol
            vol_adjustment = 1.0 / (1.0 + volatility * 10)
            action = momentum * vol_adjustment
        else:
            action = 0.0

        # Store action for state construction
        self.action_history.append(action)
        if len(self.action_history) > 100:
            self.action_history.pop(0)

        return float(np.tanh(action))

    def update_reward(self, reward: float):
        """Update RL agent with reward signal"""
        self.reward_history.append(reward)
        if len(self.reward_history) > 1000:
            self.reward_history.pop(0)

        # In real implementation, this would trigger policy updates


# ---- Factory Functions ----
def build_extended_adapters() -> Dict[str, Any]:
    """Build all extended model adapters"""
    return {
        "XGBoost_Enhanced": XGBoostEnhanced(),
        "LightGBM": LightGBMAdapter(),
        "Transformer": TransformerAdapter(),
        "Autoencoder": AutoencoderAdapter(),
        "VAE": VAEAdapter(),
        "Bayesian": BayesianAdapter(),
        "AdaptiveRL": AdaptiveRLAdapter(),
    }
