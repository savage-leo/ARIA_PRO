#!/usr/bin/env python3
"""
Unit tests for critical ARIA Pro components
"""
import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from backend.smc.smc_fusion_core import pips
from backend.core.risk_budget import RiskBudgetEngine
from backend.core.regime import RegimeDetector, RegimeMetrics
from backend.smc.bias_engine import BiasEngine


class TestPipsCalculation(unittest.TestCase):
    """Test pips calculation for different symbol types"""
    
    def test_major_pairs(self):
        """Test major currency pairs use 0.0001 point value"""
        result = pips('EURUSD', 0.0001, {})
        self.assertEqual(result, 1.0)
        
        result = pips('GBPUSD', 0.0002, {})
        self.assertEqual(result, 2.0)
    
    def test_jpy_pairs(self):
        """Test JPY pairs use 0.01 point value"""
        result = pips('USDJPY', 0.01, {})
        self.assertEqual(result, 1.0)
        
        result = pips('GBPJPY', 0.02, {})
        self.assertEqual(result, 2.0)
    
    def test_custom_point_map(self):
        """Test custom point value mapping"""
        custom_map = {'EURUSD': 0.00001}
        result = pips('EURUSD', 0.0001, custom_map)
        self.assertEqual(result, 10.0)
    
    def test_unknown_symbol(self):
        """Test unknown symbol defaults to 0.0001"""
        result = pips('UNKNOWN', 0.0001, {})
        self.assertEqual(result, 1.0)


class TestRiskBudgetEngine(unittest.TestCase):
    """Test risk budget calculations"""
    
    def setUp(self):
        self.engine = RiskBudgetEngine()
    
    def test_normal_scenario(self):
        """Test normal risk calculation"""
        metrics = RegimeMetrics(0.002, 'medium', 0.3, 0.5)
        result = self.engine.compute(
            symbol='EURUSD',
            base_conf=0.8,
            kelly_edge=0.6,
            spread=0.0001,
            atr=0.005,
            metrics=metrics,
            bias_factor=1.2,
            bias_throttle=False,
            exposure_ok=True
        )
        
        self.assertFalse(result.throttle)
        self.assertGreater(result.risk_units, 0.0)
        self.assertLessEqual(result.risk_units, 1.0)
    
    def test_throttle_conditions(self):
        """Test throttling scenarios"""
        metrics = RegimeMetrics(0.002, 'medium', 0.3, 0.5)
        
        # Test bias throttle
        result = self.engine.compute(
            symbol='EURUSD',
            base_conf=0.8,
            kelly_edge=0.6,
            spread=0.0001,
            atr=0.005,
            metrics=metrics,
            bias_factor=1.2,
            bias_throttle=True,  # Should throttle
            exposure_ok=True
        )
        self.assertTrue(result.throttle)
        self.assertEqual(result.risk_units, 0.0)
        
        # Test exposure not ok
        result = self.engine.compute(
            symbol='EURUSD',
            base_conf=0.8,
            kelly_edge=0.6,
            spread=0.0001,
            atr=0.005,
            metrics=metrics,
            bias_factor=1.2,
            bias_throttle=False,
            exposure_ok=False  # Should throttle
        )
        self.assertTrue(result.throttle)
        self.assertEqual(result.risk_units, 0.0)


class TestRegimeDetection(unittest.TestCase):
    """Test regime detection functionality"""
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data"""
        bars = [{'o': 1.1, 'h': 1.1, 'l': 1.1, 'c': 1.1, 'v': 100, 'ts': 1692000000}]
        regime, metrics = RegimeDetector.detect(bars)
        
        self.assertEqual(regime.value, 'range')  # Default fallback
        self.assertEqual(metrics.vol_bucket, 'low')
    
    def test_trending_market(self):
        """Test detection of trending market"""
        # Create uptrending bars
        bars = []
        base_price = 1.1000
        for i in range(100):
            price = base_price + (i * 0.0001)  # Steady uptrend
            bars.append({
                'o': price - 0.00005,
                'h': price + 0.00010,
                'l': price - 0.00010,
                'c': price,
                'v': 1000,
                'ts': 1692000000 + i * 60
            })
        
        regime, metrics = RegimeDetector.detect(bars)
        self.assertGreater(metrics.trend_strength, 0.5)  # Should detect uptrend


class TestBiasEngine(unittest.TestCase):
    """Test bias engine calculations"""
    
    def setUp(self):
        self.engine = BiasEngine()
    
    def test_high_confidence_idea(self):
        """Test bias calculation for high confidence trade idea"""
        # Mock trade idea
        mock_idea = MagicMock()
        mock_idea.confidence = 0.9
        mock_idea.bias = 'bullish'
        mock_idea.order_blocks = []
        mock_idea.fair_value_gaps = []
        mock_idea.liquidity_zones = []
        
        # Mock bars
        bars = [
            {'o': 1.1, 'h': 1.11, 'l': 1.09, 'c': 1.105, 'v': 1000, 'ts': 1692000000}
            for _ in range(50)
        ]
        
        result = self.engine.compute(mock_idea, bars)
        
        self.assertGreater(result.bias_factor, 0.5)
        self.assertLessEqual(result.bias_factor, 2.0)
        self.assertIsInstance(result.throttle, bool)
    
    def test_low_confidence_throttle(self):
        """Test throttling for low confidence scenarios"""
        mock_idea = MagicMock()
        mock_idea.confidence = 0.3  # Very low confidence
        mock_idea.bias = 'bullish'
        mock_idea.order_blocks = []
        mock_idea.fair_value_gaps = []
        mock_idea.liquidity_zones = []
        
        bars = [
            {'o': 1.1, 'h': 1.11, 'l': 1.09, 'c': 1.105, 'v': 1000, 'ts': 1692000000}
            for _ in range(50)
        ]
        
        result = self.engine.compute(mock_idea, bars)
        self.assertTrue(result.throttle)  # Should throttle low confidence


if __name__ == '__main__':
    unittest.main()
