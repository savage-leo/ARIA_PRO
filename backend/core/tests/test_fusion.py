"""
Unit tests for signal fusion functionality.
"""

import os
import json
import tempfile
import math
from typing import Dict, List

import numpy as np
import pytest

from backend.core.fusion import SignalFusion, FusionParams
from backend.core.calibration import ScoreCalibrator
from backend.core.regime import Regime


class TestSignalFusion:
    """Test signal fusion functionality."""

    def test_fusion_params(self):
        """Test FusionParams dataclass."""
        params = FusionParams(b=0.1, w={"model1": 0.5, "model2": 0.5})
        assert params.b == 0.1
        assert params.w["model1"] == 0.5

    def test_signal_fusion_init(self):
        """Test SignalFusion initialization."""
        # Initialize with calibrator
        calibrator = ScoreCalibrator({})
        fusion = SignalFusion(
            signal_keys=["lstm", "cnn", "xgb", "ppo"], calibrator=calibrator
        )

        assert fusion.signal_keys == ["lstm", "cnn", "xgb", "ppo"]
        assert fusion.calibrator is calibrator

    def test_sigmoid_logit(self):
        """Test sigmoid and logit functions."""
        from backend.core.fusion import _sigmoid, _logit

        # Test sigmoid
        assert abs(_sigmoid(0) - 0.5) < 1e-6
        assert _sigmoid(100) > 0.99
        assert _sigmoid(-100) < 0.01

        # Test logit
        assert abs(_logit(0.5)) < 1e-6
        assert _logit(0.99) > 0
        assert _logit(0.01) < 0

        # Test inverse relationship
        for x in [0.1, 0.25, 0.5, 0.75, 0.9]:
            y = _sigmoid(_logit(x))
            assert abs(y - x) < 1e-6

    def test_weighted_logit(self):
        """Test weighted logit computation."""
        from backend.core.fusion import _weighted_logit

        # Equal weights
        weights = {"a": 0.5, "b": 0.5}
        logits = {"a": 0.5, "b": 0.5}  # 50% probabilities
        result, _ = _weighted_logit(logits, weights, 0.0)
        expected = 0.0  # logit(0.5) = 0
        assert abs(result - expected) < 1e-6

        # One-sided weights
        weights = {"a": 1.0, "b": 0.0}
        logits = {"a": 0.73, "b": 0.27}  # 73% vs 27%
        result, _ = _weighted_logit(logits, weights, 0.0)
        expected = 1.0  # logit(0.73) ≈ 1.0
        assert abs(result - expected) < 0.1

    def test_fuse_with_equal_weights(self):
        """Test fusion with equal weights fallback."""
        calibrator = ScoreCalibrator({})
        fusion = SignalFusion(signal_keys=["model1", "model2"], calibrator=calibrator)

        # Equal signals
        raw_signals = {"model1": 0.0, "model2": 0.0}  # Both neutral
        result = fusion.fuse(raw_signals, Regime.RANGE)

        assert "p_long" in result
        assert "p_short" in result
        assert "direction" in result
        assert "margin" in result

        # Check probabilities are valid
        assert 0 <= result["p_long"] <= 1
        assert 0 <= result["p_short"] <= 1

        # Neutral signals should give ~50% probabilities
        assert abs(result["p_long"] - 0.5) < 0.1

    def test_fuse_with_extreme_signals(self):
        """Test fusion with extreme signals."""
        calibrator = ScoreCalibrator({})
        fusion = SignalFusion(signal_keys=["model1", "model2"], calibrator=calibrator)

        # Strong buy signals - with identity calibration, 0.9 and 0.8 map to 0.95 and 0.9 probabilities
        # With equal weights, this should give a strong long signal
        raw_signals = {"model1": 0.9, "model2": 0.8}
        result = fusion.fuse(raw_signals, Regime.TREND)

        assert result["p_long"] > 0.5  # Should be long-biased
        assert result["direction"] in ["buy", "sell"]

        # Strong sell signals
        raw_signals = {"model1": -0.9, "model2": -0.8}
        result = fusion.fuse(raw_signals, Regime.TREND)

        assert result["p_short"] > 0.5  # Should be short-biased

    def test_fuse_with_calibrated_scores(self):
        """Test fusion with calibrated scores."""
        # Create a calibrator with some calibration data
        calibrator = ScoreCalibrator({})

        # Add some dummy calibration (identity mapping)
        fusion = SignalFusion(signal_keys=["model1", "model2"], calibrator=calibrator)

        raw_signals = {"model1": 0.3, "model2": -0.2}
        result = fusion.fuse(raw_signals, Regime.RANGE)

        assert "p_long" in result
        assert "p_short" in result
        assert 0 <= result["p_long"] <= 1
        assert 0 <= result["p_short"] <= 1

    def test_fuse_with_context(self):
        """Test fusion with context information."""
        calibrator = ScoreCalibrator({})
        fusion = SignalFusion(signal_keys=["model1", "model2"], calibrator=calibrator)

        raw_signals = {"model1": 0.5, "model2": 0.3}
        context = {"timestamp": 1609459200, "atr_pct": 0.01, "spread_pct": 0.001}

        result = fusion.fuse(raw_signals, Regime.BREAKOUT, context)

        assert "p_long" in result
        assert "p_short" in result
        assert "direction" in result

    def test_fuse_missing_signals(self):
        """Test fusion with missing signals."""
        calibrator = ScoreCalibrator({})
        fusion = SignalFusion(
            signal_keys=["model1", "model2", "model3"], calibrator=calibrator
        )

        # Only provide some signals
        raw_signals = {"model1": 0.5, "model2": -0.3}
        result = fusion.fuse(raw_signals, Regime.RANGE)

        # Should still produce valid result
        assert "p_long" in result
        assert "p_short" in result
        assert 0 <= result["p_long"] <= 1
        assert 0 <= result["p_short"] <= 1

    def test_fuse_invalid_signals(self):
        """Test fusion with invalid signals."""
        calibrator = ScoreCalibrator({})
        fusion = SignalFusion(signal_keys=["model1", "model2"], calibrator=calibrator)

        # Provide out-of-range signals
        raw_signals = {"model1": 1.5, "model2": -1.2}  # Outside [-1,1]
        result = fusion.fuse(raw_signals, Regime.RANGE)

        # Should clamp to valid range
        assert "p_long" in result
        assert "p_short" in result
        assert 0 <= result["p_long"] <= 1
        assert 0 <= result["p_short"] <= 1
