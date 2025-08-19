"""
Unit tests for score calibration functionality.
"""

import os
import json
import tempfile
import math

import numpy as np
import pytest

from backend.core.calibration import ScoreCalibrator, _isotonic_fit, _isotonic_eval


class TestScoreCalibrator:
    """Test score calibration functionality."""

    def test_identity_calibration(self):
        """Test identity calibration (no transformation)."""
        calibrator = ScoreCalibrator({})

        # Test identity mapping
        assert (
            abs(calibrator._apply_calibration({"method": "identity"}, 0.0) - 0.5) < 1e-6
        )
        assert (
            abs(calibrator._apply_calibration({"method": "identity"}, -1.0) - 0.0)
            < 1e-6
        )
        assert (
            abs(calibrator._apply_calibration({"method": "identity"}, 1.0) - 1.0) < 1e-6
        )

    def test_platt_scaling(self):
        """Test Platt scaling calibration."""
        calibrator = ScoreCalibrator({})

        # Simple linear case: A*score + B
        spec = {"method": "platt", "A": 1.0, "B": 0.0}

        # Test mapping
        assert abs(calibrator._apply_calibration(spec, 0.0) - 0.5) < 1e-6
        assert abs(calibrator._apply_calibration(spec, -1.0) - 0.27) < 0.01
        assert abs(calibrator._apply_calibration(spec, 1.0) - 0.73) < 0.01

    def test_isotonic_fitting(self):
        """Test isotonic regression fitting."""
        # Create test data with clear monotonic relationship
        scores = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        # Fit isotonic regression
        result = _isotonic_fit(scores, probs)

        assert len(result.thresholds) == len(result.values)
        assert len(result.thresholds) > 0

        # Test that fitted curve is monotonic
        for i in range(1, len(result.values)):
            assert result.values[i] >= result.values[i - 1]

    def test_isotonic_evaluation(self):
        """Test isotonic regression evaluation."""
        # Simple step function
        thresholds = [-1.0, 0.0, 1.0]
        values = [0.1, 0.5, 0.9]

        # Test interpolation
        assert abs(_isotonic_eval(-1.0, thresholds, values) - 0.1) < 1e-6
        assert abs(_isotonic_eval(0.0, thresholds, values) - 0.5) < 1e-6
        assert abs(_isotonic_eval(1.0, thresholds, values) - 0.9) < 1e-6

        # Test extrapolation
        assert abs(_isotonic_eval(-2.0, thresholds, values) - 0.1) < 1e-6  # Clamped
        assert abs(_isotonic_eval(2.0, thresholds, values) - 0.9) < 1e-6  # Clamped

    def test_calibrate_with_identity_fallback(self):
        """Test calibration with identity fallback."""
        calibrator = ScoreCalibrator({})

        # No calibration file -> identity fallback
        prob = calibrator.calibrate("nonexistent_model", 0.5)
        expected = (0.5 + 1.0) / 2.0  # Linear map from [-1,1] to [0,1]
        assert abs(prob - expected) < 1e-6

    def test_calibrate_with_model_specific(self):
        """Test calibration with model-specific parameters."""
        # Create temporary calibration file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            cal_file = f.name
            json.dump({"test_model": {"type": "platt", "a": 2.0, "b": 0.0}}, f)

        try:
            # Load calibrator with temp file
            with open(cal_file, "r") as f:
                model_params = json.load(f)
            calibrator = ScoreCalibrator(model_params)

            # Test calibrated score
            prob = calibrator.calibrate("test_model", 0.0)
            expected = 1.0 / (1.0 + math.exp(-2.0 * 0.0))  # Sigmoid(0) = 0.5
            assert abs(prob - expected) < 1e-6

        finally:
            os.unlink(cal_file)

    def test_calibrate_with_regime_specific(self):
        """Test calibration with regime-specific parameters."""
        # For regime-specific calibration, the system looks for files in
        # backend/calibration/{model}/{symbol}/{regime}.json
        # So we'll test the default calibration only in this unit test
        model_params = {"test_model": {"method": "identity"}}
        calibrator = ScoreCalibrator(model_params)

        # Test default calibration
        prob_default = calibrator.calibrate("test_model", 0.0)
        expected_default = 0.5  # Identity
        assert abs(prob_default - expected_default) < 1e-6

        # Test regime-specific calibration falls back to default
        prob_trend = calibrator.calibrate("test_model", 0.0, regime="trend")
        expected_trend = 0.5  # Identity fallback
        assert abs(prob_trend - expected_trend) < 1e-6

    def test_calibrate_out_of_range_scores(self):
        """Test calibration with out-of-range scores."""
        calibrator = ScoreCalibrator({})

        # Test clamping of extreme scores
        prob_low = calibrator.calibrate("test", -2.0)  # Below -1
        prob_high = calibrator.calibrate("test", 2.0)  # Above 1

        # Should be clamped to valid probability range
        assert 0.0 <= prob_low <= 1.0
        assert 0.0 <= prob_high <= 1.0
