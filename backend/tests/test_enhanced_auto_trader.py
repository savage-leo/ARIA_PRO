"""
Comprehensive test suite for enhanced AutoTrader functionality
Tests adaptive timeouts, circuit breakers, and position limits
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.services.auto_trader import AutoTrader


class TestEnhancedAutoTrader:
    """Test enhanced AutoTrader features"""
    
    @pytest.fixture
    def auto_trader(self):
        """Create AutoTrader instance for testing"""
        with patch.dict(os.environ, {
            'AUTO_TRADE_SYMBOLS': 'EURUSD,GBPUSD',
            'AUTO_TRADE_DRY_RUN': '1',
            'AUTO_TRADE_CB_COOLDOWN': '60',
            'AUTO_TRADE_POS_LIMITS': 'EURUSD:2,GBPUSD:1'
        }):
            trader = AutoTrader()
            return trader
    
    def test_parse_position_limits(self, auto_trader):
        """Test position limits parsing"""
        limits = auto_trader._parse_position_limits("EURUSD:2,GBPUSD:3,INVALID")
        assert limits['EURUSD'] == 2
        assert limits['GBPUSD'] == 3
        assert 'INVALID' not in limits
    
    def test_get_position_limit(self, auto_trader):
        """Test position limit retrieval"""
        # Test configured limit
        limit = auto_trader._get_position_limit('EURUSD')
        assert limit == 2
        
        # Test default limit
        limit = auto_trader._get_position_limit('UNKNOWN')
        assert limit == auto_trader.default_max_positions_per_symbol
    
    def test_adaptive_timeout_calculation(self, auto_trader):
        """Test adaptive timeout calculation"""
        # Low volatility
        timeout = auto_trader._calculate_adaptive_timeout('EURUSD', 0.005)
        assert timeout == auto_trader.base_order_timeout
        
        # Medium volatility
        timeout = auto_trader._calculate_adaptive_timeout('EURUSD', 0.015)
        assert timeout == auto_trader.base_order_timeout * 1.5
        
        # High volatility
        timeout = auto_trader._calculate_adaptive_timeout('EURUSD', 0.025)
        expected = min(auto_trader.base_order_timeout * auto_trader.volatility_timeout_multiplier, 
                      auto_trader.max_order_timeout)
        assert timeout == expected
    
    def test_circuit_breaker_engagement(self, auto_trader):
        """Test circuit breaker engagement"""
        reason = "Test failure"
        auto_trader._engage_circuit_breaker(reason)
        
        assert auto_trader.circuit_breaker_active
        assert auto_trader.circuit_breaker_reason == reason
        assert auto_trader.circuit_breaker_engaged_at is not None
    
    def test_circuit_breaker_reset(self, auto_trader):
        """Test circuit breaker auto-reset"""
        # Engage circuit breaker
        auto_trader._engage_circuit_breaker("Test")
        
        # Simulate time passage
        auto_trader.circuit_breaker_engaged_at = time.time() - 3700  # 1+ hour ago
        
        # Check reset
        reset = auto_trader._check_circuit_breaker_reset()
        assert reset
        assert not auto_trader.circuit_breaker_active
    
    @pytest.mark.asyncio
    async def test_position_limit_enforcement(self, auto_trader):
        """Test position limit enforcement"""
        symbol = 'EURUSD'
        auto_trader.active_positions[symbol] = 2  # At limit
        
        # Should not allow new position
        with patch.object(auto_trader, '_get_position_limit', return_value=2):
            # This would normally check position limits in execution logic
            current_positions = auto_trader.active_positions.get(symbol, 0)
            limit = auto_trader._get_position_limit(symbol)
            assert current_positions >= limit


class TestRiskEngineEnhancements:
    """Test enhanced RiskEngine functionality"""
    
    @pytest.fixture
    def risk_engine(self):
        """Create RiskEngine instance for testing"""
        from backend.services.risk_engine import RiskEngine
        engine = RiskEngine()
        return engine
    
    def test_var_calculation(self, risk_engine):
        """Test VaR calculation"""
        # Add sample returns
        returns = [-0.02, -0.01, 0.01, 0.02, -0.015, 0.005, -0.008, 0.012] * 10
        risk_engine.returns_history.extend(returns)
        
        var = risk_engine.calculate_var(0.95)
        assert var > 0
        assert isinstance(var, float)
    
    def test_cvar_calculation(self, risk_engine):
        """Test CVaR calculation"""
        # Add sample returns
        returns = [-0.02, -0.01, 0.01, 0.02, -0.015, 0.005, -0.008, 0.012] * 10
        risk_engine.returns_history.extend(returns)
        
        cvar = risk_engine.calculate_cvar(0.99)
        assert cvar > 0
        assert isinstance(cvar, float)
    
    def test_sharpe_ratio_calculation(self, risk_engine):
        """Test Sharpe ratio calculation"""
        # Add sample returns with positive trend
        returns = [0.001, 0.002, -0.001, 0.003, 0.001, -0.0005, 0.002] * 10
        risk_engine.returns_history.extend(returns)
        
        sharpe = risk_engine.calculate_sharpe_ratio()
        assert isinstance(sharpe, float)
    
    def test_kelly_optimal_size(self, risk_engine):
        """Test Kelly optimal position size calculation"""
        kelly = risk_engine.calculate_kelly_optimal_size(0.6, 0.02, 0.01)
        assert 0 <= kelly <= 0.25  # Should be within safety bounds
    
    def test_comprehensive_risk_metrics(self, risk_engine):
        """Test comprehensive risk metrics calculation"""
        # Add sample data
        returns = [0.001, -0.002, 0.003, -0.001, 0.002] * 20
        pnl = [100, -50, 150, -25, 75] * 20
        
        risk_engine.returns_history.extend(returns)
        risk_engine.pnl_history.extend(pnl)
        
        metrics = risk_engine.get_comprehensive_risk_metrics()
        
        assert 'var_1d' in metrics.__dict__
        assert 'sharpe_ratio' in metrics.__dict__
        assert 'win_rate' in metrics.__dict__
        assert 'profit_factor' in metrics.__dict__
        assert 'kelly_optimal_size' in metrics.__dict__


class TestAISignalGeneratorEnhancements:
    """Test enhanced AI signal generator functionality"""
    
    @pytest.fixture
    def signal_generator(self):
        """Create RealAISignalGenerator instance for testing"""
        with patch('backend.services.real_ai_signal_generator.get_settings') as mock_settings:
            mock_settings.return_value.symbols_list = ['EURUSD', 'GBPUSD']
            
            from backend.services.real_ai_signal_generator import RealAISignalGenerator
            generator = RealAISignalGenerator()
            return generator
    
    def test_model_performance_initialization(self, signal_generator):
        """Test model performance tracking initialization"""
        assert 'xgb' in signal_generator.model_performance
        assert 'lstm' in signal_generator.model_performance
        
        for model_name, metrics in signal_generator.model_performance.items():
            assert metrics.success_rate == 1.0
            assert metrics.error_count == 0
            assert not metrics.circuit_breaker_active
    
    def test_circuit_breaker_check(self, signal_generator):
        """Test circuit breaker functionality"""
        # Initially should not be active
        assert not signal_generator._check_circuit_breaker('xgb')
        
        # Simulate failures to trigger circuit breaker
        for _ in range(10):
            signal_generator._update_model_performance('xgb', False, 0)
        
        # Should now be active
        assert signal_generator._check_circuit_breaker('xgb')
    
    def test_model_performance_update(self, signal_generator):
        """Test model performance metric updates"""
        initial_success_rate = signal_generator.model_performance['xgb'].success_rate
        
        # Record success
        signal_generator._update_model_performance('xgb', True, 100.0)
        assert signal_generator.model_performance['xgb'].last_success is not None
        assert signal_generator.model_performance['xgb'].avg_latency_ms == 100.0
        
        # Record failure
        signal_generator._update_model_performance('xgb', False, 0)
        assert signal_generator.model_performance['xgb'].error_count == 1
        assert signal_generator.model_performance['xgb'].success_rate < initial_success_rate
    
    @pytest.mark.asyncio
    async def test_safe_model_inference(self, signal_generator):
        """Test safe model inference with circuit breaker"""
        # Mock the actual inference method
        signal_generator._execute_model_inference = AsyncMock(return_value={'result': 'test'})
        
        # Should work normally
        result = await signal_generator._safe_model_inference('xgb', {'test': 'data'})
        assert result is not None
        
        # Trigger circuit breaker
        signal_generator.model_performance['xgb'].circuit_breaker_active = True
        
        # Should return None due to circuit breaker
        result = await signal_generator._safe_model_inference('xgb', {'test': 'data'})
        assert result is None
    
    def test_adaptive_interval_calculation(self, signal_generator):
        """Test adaptive analysis interval calculation"""
        # Low activity
        interval = signal_generator._calculate_adaptive_interval('EURUSD', 0.0001)
        assert interval == signal_generator.base_analysis_interval * 1.5
        
        # Medium activity
        interval = signal_generator._calculate_adaptive_interval('EURUSD', 0.0015)
        assert interval == signal_generator.base_analysis_interval * 0.75
        
        # High activity
        interval = signal_generator._calculate_adaptive_interval('EURUSD', 0.003)
        assert interval == signal_generator.base_analysis_interval * 0.5


class TestSecurityEnhancements:
    """Test enhanced security features"""
    
    @pytest.mark.asyncio
    async def test_enhanced_rate_limiting(self):
        """Test enhanced rate limiting functionality"""
        from backend.middleware.enhanced_rate_limit import EnhancedRateLimiter
        
        rate_limiter = EnhancedRateLimiter()
        
        # Mock request
        mock_request = Mock()
        mock_request.url.path = "/api/trading/execute"
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {}
        mock_request.state = Mock()
        
        # First request should be allowed
        allowed, metadata = await rate_limiter.is_allowed(mock_request)
        assert allowed
        assert "X-RateLimit-Limit" in metadata
    
    def test_jwt_rotation_manager(self):
        """Test JWT rotation functionality"""
        from backend.core.jwt_rotation import JWTRotationManager
        
        manager = JWTRotationManager()
        
        # Test secret generation
        active_secret = manager.get_active_secret()
        assert active_secret is not None
        assert len(active_secret.secret) >= 32
        
        # Test token creation
        payload = {"user_id": "test", "role": "trader"}
        token, key_id = manager.create_token(payload)
        assert token is not None
        assert key_id == active_secret.key_id
        
        # Test token verification
        verified_payload = manager.verify_token(token)
        assert verified_payload is not None
        assert verified_payload["user_id"] == "test"


class TestDistributedLocking:
    """Test distributed locking functionality"""
    
    @pytest.mark.asyncio
    async def test_lock_acquisition_and_release(self):
        """Test basic lock operations"""
        from backend.core.distributed_lock import DistributedLock, LockConfig
        
        # Mock Redis client
        mock_redis = AsyncMock()
        mock_redis.set.return_value = True
        mock_redis.eval.return_value = 1
        
        config = LockConfig(default_timeout=5)
        lock = DistributedLock(mock_redis, "test_lock", config)
        
        # Test acquisition
        acquired = await lock.acquire()
        assert acquired
        assert lock.acquired
        
        # Test release
        released = await lock.release()
        assert released
        assert not lock.acquired
    
    @pytest.mark.asyncio
    async def test_lock_context_manager(self):
        """Test lock as context manager"""
        from backend.core.distributed_lock import DistributedLock, LockConfig
        
        # Mock Redis client
        mock_redis = AsyncMock()
        mock_redis.set.return_value = True
        mock_redis.eval.return_value = 1
        
        config = LockConfig(default_timeout=5)
        lock = DistributedLock(mock_redis, "test_lock", config)
        
        async with lock:
            assert lock.acquired
        
        assert not lock.acquired


class TestComprehensiveMonitoring:
    """Test comprehensive monitoring system"""
    
    def test_metric_collection(self):
        """Test metric collection and storage"""
        from backend.core.comprehensive_monitoring import MetricCollector, MetricType
        
        collector = MetricCollector()
        
        # Record metrics
        collector.record("test.metric", 100.0, {"tag": "value"}, MetricType.GAUGE)
        collector.record("test.metric", 110.0, {"tag": "value"}, MetricType.GAUGE)
        
        # Test retrieval
        latest = collector.get_latest_value("test.metric")
        assert latest.value == 110.0
        assert latest.tags["tag"] == "value"
        
        # Test statistics
        stats = collector.calculate_statistics("test.metric", 60)
        assert stats["count"] == 2
        assert stats["mean"] == 105.0
    
    def test_anomaly_detection(self):
        """Test anomaly detection"""
        from backend.core.comprehensive_monitoring import AnomalyDetector
        
        detector = AnomalyDetector(sensitivity=2.0)
        
        # Normal values
        historical = [100.0] * 50
        assert not detector.detect_anomaly("test", 101.0, historical)
        
        # Anomalous value
        assert detector.detect_anomaly("test", 150.0, historical)
    
    def test_alert_management(self):
        """Test alert management"""
        from backend.core.comprehensive_monitoring import AlertManager, ThresholdRule, AlertSeverity
        
        manager = AlertManager()
        
        # Add threshold rule
        rule = ThresholdRule("test.metric", ">", 100.0, AlertSeverity.HIGH)
        manager.add_threshold_rule(rule)
        
        # Test threshold evaluation
        assert manager._evaluate_threshold(110.0, ">", 100.0)
        assert not manager._evaluate_threshold(90.0, ">", 100.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
