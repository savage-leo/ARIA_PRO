"""
Unit tests for regime detection functionality.
"""

import math
import os
import tempfile
import json
from typing import List, Dict

import numpy as np
import pytest

from backend.core.regime import (
    RegimeDetector,
    Regime,
    RegimeMetrics,
    _session_from_ts,
    _m15_return_from_bars,
    fit_hmm_on_m15_returns,
)


def make_test_bars(
    n: int = 100, trend: float = 0.0, volatility: float = 0.001
) -> List[Dict]:
    """Create test bars with specified characteristics."""
    bars = []
    price = 1.1000
    base_time = 1609459200  # 2021-01-01 00:00:00 UTC

    for i in range(n):
        # Add trend and noise
        drift = trend + np.random.normal(0, volatility)
        price = price * math.exp(drift)

        # Create OHLC
        o = price
        c = price * math.exp(np.random.normal(0, volatility * 0.1))
        h = max(o, c) * (1 + abs(np.random.normal(0, volatility * 0.2)))
        l = min(o, c) * (1 - abs(np.random.normal(0, volatility * 0.2)))
        v = 1000 + np.random.poisson(500)

        bars.append(
            {
                "ts": base_time + i * 60,  # 1-minute bars
                "o": o,
                "h": h,
                "l": l,
                "c": c,
                "v": v,
                "symbol": "EURUSD",
            }
        )

    return bars


class TestRegimeDetector:
    """Test regime detection functionality."""

    def test_returns_from_bars(self):
        """Test return computation from bars."""
        bars = make_test_bars(10)
        rets = RegimeDetector._returns_from_bars(bars)
        assert len(rets) == 9  # n-1 returns
        assert isinstance(rets, np.ndarray)

    def test_ewma_volatility(self):
        """Test EWMA volatility computation."""
        # Constant returns should give predictable volatility
        rets = np.array([0.001] * 100)
        vol = RegimeDetector._ewma_volatility(rets, lam=0.94)
        expected = math.sqrt(0.001 * 0.001)  # Should converge to this
        # Allow for some numerical error due to windowing
        assert abs(vol - expected) < 1e-5

    def test_trend_strength(self):
        """Test trend strength computation."""
        # Strong uptrend
        closes = np.array([1.0 + i * 0.001 for i in range(100)])
        strength = RegimeDetector._trend_strength(closes, lookback=50)
        assert strength > 0.0  # Should be positive for uptrend

        # Strong downtrend
        closes = np.array([1.0 - i * 0.001 for i in range(100)])
        strength = RegimeDetector._trend_strength(closes, lookback=50)
        assert strength < 0.0  # Should be negative for downtrend

        # Sideways/flat
        closes = np.array([1.0 + 0.001 * math.sin(i * 0.1) for i in range(100)])
        strength = RegimeDetector._trend_strength(closes, lookback=50)
        assert abs(strength) < 0.3

    def test_atr(self):
        """Test ATR computation."""
        bars = make_test_bars(20)
        atr = RegimeDetector._atr(bars, period=14)
        assert atr >= 0

    def test_breakout_score(self):
        """Test breakout score computation."""
        bars = make_test_bars(30)
        atr = RegimeDetector._atr(bars, period=14)
        score = RegimeDetector._breakout_score(bars, atr, lookback=20)
        assert score >= 0

    def test_detect_trend_regime(self):
        """Test detection of trend regime."""
        # Create strong trend
        bars = make_test_bars(100, trend=0.0005, volatility=0.0005)
        regime, metrics = RegimeDetector.detect(bars)

        assert isinstance(regime, Regime)
        assert isinstance(metrics, RegimeMetrics)

    def test_detect_range_regime(self):
        """Test detection of range regime."""
        # Create sideways movement
        bars = make_test_bars(100, trend=0.0, volatility=0.0002)
        regime, metrics = RegimeDetector.detect(bars)

        assert isinstance(regime, Regime)
        assert isinstance(metrics, RegimeMetrics)

    def test_session_from_ts(self):
        """Test session classification from timestamp."""
        # Asia session (00:00-07:00 UTC)
        asia_ts = 1609459200  # 2021-01-01 00:00:00 UTC
        assert _session_from_ts(asia_ts) == "asia"

        # London session (07:00-13:00 UTC)
        london_ts = 1609484400  # 2021-01-01 07:00:00 UTC
        assert _session_from_ts(london_ts) == "london"

        # NY session (13:00-22:00 UTC)
        ny_ts = 1609506000  # 2021-01-01 13:00:00 UTC
        assert _session_from_ts(ny_ts) == "ny"

        # Off session (22:00-00:00 UTC)
        off_ts = 1609538400  # 2021-01-01 22:00:00 UTC
        assert _session_from_ts(off_ts) == "off"

    def test_m15_return_from_bars(self):
        """Test 15-minute return approximation."""
        bars = make_test_bars(20)  # 20 1-minute bars
        r15 = _m15_return_from_bars(bars)
        if r15 is not None:
            assert isinstance(r15, float)

    def test_fit_hmm_on_m15_returns(self):
        """Test HMM fitting on M15 returns."""
        # Generate synthetic M15 returns with 3 regimes
        np.random.seed(42)

        # Regime 1: Low volatility (RANGE)
        rets1 = np.random.normal(0, 0.0005, 1000)

        # Regime 2: Medium volatility with trend (TREND)
        rets2 = np.random.normal(0.0002, 0.001, 1000)

        # Regime 3: High volatility (BREAKOUT)
        rets3 = np.random.normal(0, 0.002, 1000)

        # Combine all returns
        all_rets = np.concatenate([rets1, rets2, rets3])

        # Fit HMM
        result = fit_hmm_on_m15_returns("TEST", all_rets.tolist())

        # Check result structure
        assert result is not None
        assert "mu" in result
        assert "sigma" in result
        assert "A" in result
        assert len(result["mu"]) == 3
        assert len(result["sigma"]) == 3
        assert len(result["A"]) == 3

        # Check parameter values are reasonable
        mu = np.array(result["mu"])
        sigma = np.array(result["sigma"])

        # All variances should be positive
        assert np.all(sigma > 0)

        # All transition probabilities should be valid
        A = np.array(result["A"])
        assert np.all(A >= 0)
        assert np.all(A <= 1)
        assert np.allclose(np.sum(A, axis=1), 1.0)
