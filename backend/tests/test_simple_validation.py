"""
Simple validation tests for enhanced ARIA components
Tests core functionality without complex dependencies
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


class TestEnhancedComponents:
    """Simple validation of enhanced components"""
    
    def test_auto_trader_imports(self):
        """Test AutoTrader imports and basic initialization"""
        with patch.dict(os.environ, {
            'AUTO_TRADE_SYMBOLS': 'EURUSD,GBPUSD',
            'AUTO_TRADE_DRY_RUN': '1',
            'AUTO_TRADE_CB_COOLDOWN': '60',
            'AUTO_TRADE_POS_LIMITS': 'EURUSD:2,GBPUSD:1'
        }):
            from backend.services.auto_trader import AutoTrader
            trader = AutoTrader()
            
            # Test position limits parsing
            limits = trader._parse_position_limits("EURUSD:2,GBPUSD:3")
            assert limits['EURUSD'] == 2
            assert limits['GBPUSD'] == 3
            
            # Test adaptive timeout calculation
            timeout = trader._calculate_adaptive_timeout('EURUSD', 0.005)
            assert timeout >= trader.base_order_timeout
    
    def test_risk_engine_imports(self):
        """Test RiskEngine imports and basic functionality"""
        from backend.services.risk_engine import RiskEngine
        
        engine = RiskEngine()
        
        # Test basic risk calculations with sample data
        returns = [-0.02, -0.01, 0.01, 0.02, -0.015] * 10
        engine.returns_history.extend(returns)
        
        var = engine.calculate_var(0.95)
        assert isinstance(var, float)
        assert var > 0
        
        kelly = engine.calculate_kelly_optimal_size(0.6, 0.02, 0.01)
        assert 0 <= kelly <= 0.25
    
    def test_ai_signal_generator_imports(self):
        """Test AI signal generator imports and basic functionality"""
        with patch('backend.services.real_ai_signal_generator.get_settings') as mock_settings:
            mock_settings.return_value.symbols_list = ['EURUSD']
            
            from backend.services.real_ai_signal_generator import RealAISignalGenerator
            
            generator = RealAISignalGenerator()
            
            # Test model performance tracking
            assert 'xgb' in generator.model_performance
            assert generator.model_performance['xgb'].success_rate == 1.0
            
            # Test circuit breaker check
            assert not generator._check_circuit_breaker('xgb')
    
    def test_enhanced_rate_limiter_imports(self):
        """Test enhanced rate limiter imports"""
        from backend.middleware.enhanced_rate_limit import EnhancedRateLimiter
        
        rate_limiter = EnhancedRateLimiter()
        assert rate_limiter is not None
    
    def test_jwt_rotation_imports(self):
        """Test JWT rotation manager imports"""
        from backend.core.jwt_rotation import JWTRotationManager
        
        manager = JWTRotationManager()
        assert manager is not None
        
        # Test secret generation
        active_secret = manager.get_active_secret()
        assert active_secret is not None
        assert len(active_secret.secret) >= 32
    
    def test_distributed_lock_imports(self):
        """Test distributed lock imports"""
        from backend.core.distributed_lock import LockManager, DistributedLock, LockConfig
        
        config = LockConfig(default_timeout=5)
        assert config.default_timeout == 5
        
        # Test with mock Redis
        mock_redis = Mock()
        lock = DistributedLock(mock_redis, "test_lock", config)
        assert lock.key == "test_lock"
    
    def test_comprehensive_monitoring_imports(self):
        """Test comprehensive monitoring imports"""
        from backend.core.comprehensive_monitoring import (
            MonitoringSystem, MetricCollector, AnomalyDetector, AlertManager
        )
        
        monitoring = MonitoringSystem()
        assert monitoring.metric_collector is not None
        
        # Test metric collection
        monitoring.metric_collector.record("test.metric", 100.0)
        latest = monitoring.metric_collector.get_latest_value("test.metric")
        assert latest.value == 100.0
        
        # Test anomaly detection
        detector = AnomalyDetector(sensitivity=2.0)
        historical = [100.0] * 50
        assert not detector.detect_anomaly("test", 101.0, historical)
    
    def test_redis_cache_imports(self):
        """Test Redis cache imports"""
        from backend.core.redis_cache import RedisCache
        
        cache = RedisCache()
        assert cache is not None


class TestSystemIntegration:
    """Test system integration points"""
    
    def test_environment_configuration(self):
        """Test environment configuration handling"""
        test_env = {
            'AUTO_TRADE_SYMBOLS': 'EURUSD,GBPUSD',
            'AUTO_TRADE_DRY_RUN': '1',
            'RISK_MAX_POSITION_SIZE': '0.01',
            'REDIS_HOST': 'localhost'
        }
        
        with patch.dict(os.environ, test_env):
            # Test that components can read configuration
            assert os.getenv('AUTO_TRADE_SYMBOLS') == 'EURUSD,GBPUSD'
            assert os.getenv('AUTO_TRADE_DRY_RUN') == '1'
    
    def test_component_initialization_order(self):
        """Test that components can be initialized in correct order"""
        with patch.dict(os.environ, {
            'AUTO_TRADE_SYMBOLS': 'EURUSD',
            'AUTO_TRADE_DRY_RUN': '1'
        }):
            # Initialize components in typical startup order
            from backend.core.redis_cache import RedisCache
            from backend.core.distributed_lock import LockManager
            from backend.services.risk_engine import RiskEngine
            
            cache = RedisCache()
            lock_manager = LockManager()
            risk_engine = RiskEngine()
            
            # All should initialize without errors
            assert cache is not None
            assert lock_manager is not None
            assert risk_engine is not None


def test_performance_metrics():
    """Test basic performance characteristics"""
    import time
    from backend.core.comprehensive_monitoring import MetricCollector, MetricType
    
    collector = MetricCollector()
    
    # Test metric recording performance
    start_time = time.time()
    for i in range(100):
        collector.record(f"test.metric.{i % 10}", float(i), {}, MetricType.GAUGE)
    end_time = time.time()
    
    duration = end_time - start_time
    assert duration < 0.1  # Should complete in < 100ms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
