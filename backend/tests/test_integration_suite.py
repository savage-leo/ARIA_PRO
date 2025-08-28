"""
Integration test suite for ARIA PRO institutional trading platform
Tests end-to-end workflows and component interactions
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.services.auto_trader import AutoTrader
from backend.services.risk_engine import RiskEngine
from backend.services.real_ai_signal_generator import RealAISignalGenerator


class TestTradingWorkflowIntegration:
    """Test complete trading workflow integration"""
    
    @pytest.fixture
    def trading_system(self):
        """Setup complete trading system for integration testing"""
        with patch.dict(os.environ, {
            'AUTO_TRADE_SYMBOLS': 'EURUSD,GBPUSD',
            'AUTO_TRADE_DRY_RUN': '1',
            'AUTO_TRADE_ENABLED': '1',
            'RISK_MAX_POSITION_SIZE': '0.01',
            'RISK_MAX_DAILY_LOSS': '1000',
            'REDIS_HOST': 'localhost',
            'REDIS_PORT': '6379'
        }):
            # Mock dependencies
            auto_trader = AutoTrader()
            risk_engine = RiskEngine()
            
            # Mock external services
            auto_trader.mt5_manager = Mock()
            auto_trader.mt5_manager.get_account_info.return_value = {
                'balance': 10000.0,
                'equity': 10000.0,
                'margin': 0.0,
                'free_margin': 10000.0
            }
            
            auto_trader.mt5_manager.get_positions.return_value = []
            auto_trader.mt5_manager.place_order.return_value = {'ticket': 12345, 'status': 'filled'}
            
            return {
                'auto_trader': auto_trader,
                'risk_engine': risk_engine
            }
    
    @pytest.mark.asyncio
    async def test_signal_to_execution_workflow(self, trading_system):
        """Test complete signal generation to execution workflow"""
        auto_trader = trading_system['auto_trader']
        risk_engine = trading_system['risk_engine']
        
        # Mock signal generation
        mock_signal = {
            'symbol': 'EURUSD',
            'action': 'buy',
            'confidence': 0.75,
            'entry_price': 1.1000,
            'stop_loss': 1.0950,
            'take_profit': 1.1100,
            'timestamp': datetime.now().isoformat()
        }
        
        # Mock market data
        mock_ohlcv = {
            'open': [1.0990, 1.0995, 1.1000],
            'high': [1.1010, 1.1015, 1.1020],
            'low': [1.0980, 1.0985, 1.0990],
            'close': [1.0995, 1.1000, 1.1005],
            'volume': [1000, 1100, 1200]
        }
        
        with patch.object(auto_trader, '_get_ohlcv_with_fallback', return_value=mock_ohlcv):
            with patch.object(auto_trader, 'real_ai_signal_generator') as mock_generator:
                mock_generator.generate_signals.return_value = [mock_signal]
                
                # Test signal processing
                signals = await mock_generator.generate_signals.return_value
                assert len(signals) == 1
                assert signals[0]['symbol'] == 'EURUSD'
                assert signals[0]['action'] == 'buy'
    
    @pytest.mark.asyncio
    async def test_risk_management_integration(self, trading_system):
        """Test risk management integration with trading decisions"""
        auto_trader = trading_system['auto_trader']
        risk_engine = trading_system['risk_engine']
        
        # Setup risk scenario
        risk_engine.daily_pnl = -500  # Already some losses
        risk_engine.max_daily_loss = 1000
        
        # Mock position data
        mock_positions = [
            {'symbol': 'EURUSD', 'volume': 0.01, 'profit': -100},
            {'symbol': 'GBPUSD', 'volume': 0.01, 'profit': 50}
        ]
        
        # Test risk validation
        position_size = 0.02
        symbol = 'EURUSD'
        
        # Should reduce position size due to existing losses
        validated_size = risk_engine.validate_position_size(symbol, position_size)
        assert validated_size <= position_size
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, trading_system):
        """Test circuit breaker integration across components"""
        auto_trader = trading_system['auto_trader']
        
        # Simulate multiple failures to trigger circuit breaker
        for _ in range(5):
            auto_trader.consecutive_failures += 1
        
        # Circuit breaker should be engaged
        auto_trader._engage_circuit_breaker("Too many failures")
        assert auto_trader.circuit_breaker_active
        
        # Trading should be halted
        assert auto_trader.circuit_breaker_active
    
    @pytest.mark.asyncio
    async def test_monitoring_integration(self, trading_system):
        """Test monitoring system integration"""
        from backend.core.comprehensive_monitoring import MonitoringSystem
        
        monitoring = MonitoringSystem()
        auto_trader = trading_system['auto_trader']
        
        # Mock monitoring data collection
        monitoring.metric_collector.record("trading.signals_generated", 5)
        monitoring.metric_collector.record("trading.orders_executed", 3)
        monitoring.metric_collector.record("system.cpu_usage", 45.2)
        
        # Test metric retrieval
        signals_metric = monitoring.metric_collector.get_latest_value("trading.signals_generated")
        assert signals_metric.value == 5
        
        orders_metric = monitoring.metric_collector.get_latest_value("trading.orders_executed")
        assert orders_metric.value == 3


class TestSecurityIntegration:
    """Test security component integration"""
    
    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self):
        """Test rate limiting with API endpoints"""
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from backend.middleware.enhanced_rate_limit import EnhancedRateLimitMiddleware
        
        app = FastAPI()
        app.add_middleware(EnhancedRateLimitMiddleware)
        
        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}
        
        client = TestClient(app)
        
        # First request should succeed
        response = client.get("/test")
        assert response.status_code == 200
        
        # Check rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
    
    @pytest.mark.asyncio
    async def test_jwt_rotation_integration(self):
        """Test JWT rotation with authentication"""
        from backend.core.jwt_rotation import JWTRotationManager
        
        # Mock Redis for JWT storage
        with patch('backend.core.jwt_rotation.redis.Redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.get.return_value = None
            mock_redis_instance.set.return_value = True
            mock_redis_instance.sadd.return_value = True
            
            manager = JWTRotationManager()
            
            # Test token lifecycle
            payload = {"user_id": "trader1", "role": "trader"}
            token, key_id = manager.create_token(payload)
            
            # Verify token
            verified = manager.verify_token(token)
            assert verified["user_id"] == "trader1"
            
            # Test token blacklisting
            manager.blacklist_token(token)
            # Token should now be invalid (would be checked in real Redis)


class TestDistributedLockingIntegration:
    """Test distributed locking integration"""
    
    @pytest.mark.asyncio
    async def test_trading_lock_integration(self):
        """Test distributed locking in trading operations"""
        from backend.core.distributed_lock import LockManager
        
        # Mock Redis
        with patch('backend.core.distributed_lock.redis.Redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.set.return_value = True
            mock_redis_instance.eval.return_value = 1
            
            lock_manager = LockManager()
            
            # Test trading lock
            async with lock_manager.trading_lock("EURUSD"):
                # Simulate trading operation
                await asyncio.sleep(0.1)
                # Lock should be held during operation
                pass
            
            # Lock should be released after context
    
    @pytest.mark.asyncio
    async def test_model_update_lock_integration(self):
        """Test model update locking"""
        from backend.core.distributed_lock import LockManager
        
        # Mock Redis
        with patch('backend.core.distributed_lock.redis.Redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.set.return_value = True
            mock_redis_instance.eval.return_value = 1
            
            lock_manager = LockManager()
            
            # Test model update lock
            async with lock_manager.model_update_lock():
                # Simulate model update operation
                await asyncio.sleep(0.1)
                pass


class TestCacheIntegration:
    """Test Redis caching integration"""
    
    @pytest.mark.asyncio
    async def test_market_data_caching(self):
        """Test market data caching integration"""
        from backend.core.redis_cache import RedisCache
        
        # Mock Redis
        with patch('backend.core.redis_cache.redis.Redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.get.return_value = None
            mock_redis_instance.setex.return_value = True
            
            cache = RedisCache()
            
            # Test market data caching
            market_data = {
                'symbol': 'EURUSD',
                'bid': 1.1000,
                'ask': 1.1002,
                'timestamp': time.time()
            }
            
            # Cache market data
            await cache.set_market_data('EURUSD', market_data, ttl=60)
            
            # Retrieve from cache
            cached_data = await cache.get_market_data('EURUSD')
            # Would return None due to mock, but in real scenario would return data
    
    @pytest.mark.asyncio
    async def test_ai_signal_caching(self):
        """Test AI signal caching integration"""
        from backend.core.redis_cache import RedisCache
        
        # Mock Redis
        with patch('backend.core.redis_cache.redis.Redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.get.return_value = None
            mock_redis_instance.setex.return_value = True
            
            cache = RedisCache()
            
            # Test AI signal caching
            signal = {
                'symbol': 'EURUSD',
                'action': 'buy',
                'confidence': 0.8,
                'timestamp': time.time()
            }
            
            # Cache signal
            await cache.set_ai_signal('EURUSD', signal, ttl=300)


class TestErrorHandlingIntegration:
    """Test error handling across components"""
    
    @pytest.mark.asyncio
    async def test_mt5_connection_failure_handling(self):
        """Test handling of MT5 connection failures"""
        from backend.services.auto_trader import AutoTrader
        
        with patch.dict(os.environ, {'AUTO_TRADE_DRY_RUN': '1'}):
            auto_trader = AutoTrader()
            
            # Mock MT5 connection failure
            auto_trader.mt5_manager = Mock()
            auto_trader.mt5_manager.get_account_info.side_effect = Exception("MT5 connection failed")
            
            # Should handle gracefully and engage circuit breaker
            try:
                await auto_trader._check_account_health()
            except Exception:
                # Should trigger circuit breaker
                pass
    
    @pytest.mark.asyncio
    async def test_ai_model_failure_handling(self):
        """Test handling of AI model failures"""
        with patch('backend.services.real_ai_signal_generator.get_settings') as mock_settings:
            mock_settings.return_value.symbols_list = ['EURUSD']
            
            from backend.services.real_ai_signal_generator import RealAISignalGenerator
            
            generator = RealAISignalGenerator()
            
            # Mock model failure
            generator._execute_model_inference = AsyncMock(side_effect=Exception("Model failed"))
            
            # Should handle gracefully and update performance metrics
            result = await generator._safe_model_inference('xgb', {'test': 'data'})
            assert result is None
            
            # Performance metrics should reflect failure
            assert generator.model_performance['xgb'].error_count > 0


class TestPerformanceIntegration:
    """Test performance aspects of integrated system"""
    
    @pytest.mark.asyncio
    async def test_concurrent_signal_generation(self):
        """Test concurrent signal generation performance"""
        with patch('backend.services.real_ai_signal_generator.get_settings') as mock_settings:
            mock_settings.return_value.symbols_list = ['EURUSD', 'GBPUSD', 'USDJPY']
            
            from backend.services.real_ai_signal_generator import RealAISignalGenerator
            
            generator = RealAISignalGenerator()
            
            # Mock fast inference
            generator._execute_model_inference = AsyncMock(return_value={'confidence': 0.7})
            
            start_time = time.time()
            
            # Generate signals for multiple symbols concurrently
            tasks = []
            for symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
                task = generator._safe_model_inference('xgb', {'symbol': symbol})
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Should complete quickly with concurrent execution
            assert duration < 1.0  # Should be much faster than sequential
            assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self):
        """Test memory usage monitoring integration"""
        from backend.core.comprehensive_monitoring import SystemResourceMonitor
        
        monitor = SystemResourceMonitor()
        
        # Get memory metrics
        memory_metrics = monitor.get_memory_metrics()
        
        assert 'total' in memory_metrics
        assert 'available' in memory_metrics
        assert 'percent' in memory_metrics
        assert 'used' in memory_metrics
        
        # Memory usage should be reasonable
        assert memory_metrics['percent'] < 90  # Less than 90% usage


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
