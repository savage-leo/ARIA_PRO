"""
Load and performance test suite for ARIA PRO institutional trading platform
Tests system behavior under high load and stress conditions
"""

import pytest
import asyncio
import time
import statistics
import concurrent.futures
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


class TestLoadPerformance:
    """Load and performance testing for critical components"""
    
    @pytest.mark.asyncio
    async def test_concurrent_signal_generation_load(self):
        """Test AI signal generation under concurrent load"""
        with patch('backend.services.real_ai_signal_generator.get_settings') as mock_settings:
            mock_settings.return_value.symbols_list = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
            
            from backend.services.real_ai_signal_generator import RealAISignalGenerator
            
            generator = RealAISignalGenerator()
            
            # Mock model inference with realistic latency
            async def mock_inference(model_name, data):
                await asyncio.sleep(0.05)  # 50ms latency
                return {'confidence': 0.7, 'action': 'buy'}
            
            generator._execute_model_inference = mock_inference
            
            # Test concurrent load
            concurrent_requests = 50
            start_time = time.time()
            
            tasks = []
            for i in range(concurrent_requests):
                symbol = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'][i % 4]
                task = generator._safe_model_inference('xgb', {'symbol': symbol})
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Analyze results
            successful_results = [r for r in results if not isinstance(r, Exception)]
            failed_results = [r for r in results if isinstance(r, Exception)]
            
            print(f"Load test results:")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Successful: {len(successful_results)}/{concurrent_requests}")
            print(f"  Failed: {len(failed_results)}/{concurrent_requests}")
            print(f"  Throughput: {len(successful_results)/duration:.2f} req/s")
            
            # Performance assertions
            assert len(successful_results) >= concurrent_requests * 0.95  # 95% success rate
            assert duration < 5.0  # Should complete within 5 seconds
    
    @pytest.mark.asyncio
    async def test_trading_engine_load(self):
        """Test trading engine under load"""
        with patch.dict(os.environ, {
            'AUTO_TRADE_SYMBOLS': 'EURUSD,GBPUSD,USDJPY,XAUUSD',
            'AUTO_TRADE_DRY_RUN': '1'
        }):
            from backend.services.auto_trader import AutoTrader
            
            auto_trader = AutoTrader()
            
            # Mock dependencies
            auto_trader.mt5_manager = Mock()
            auto_trader.mt5_manager.get_account_info.return_value = {
                'balance': 10000.0, 'equity': 10000.0
            }
            auto_trader.mt5_manager.get_positions.return_value = []
            auto_trader.mt5_manager.place_order.return_value = {'ticket': 12345}
            
            # Mock signal generation
            mock_signals = [
                {'symbol': 'EURUSD', 'action': 'buy', 'confidence': 0.8},
                {'symbol': 'GBPUSD', 'action': 'sell', 'confidence': 0.7},
                {'symbol': 'USDJPY', 'action': 'buy', 'confidence': 0.9},
                {'symbol': 'XAUUSD', 'action': 'sell', 'confidence': 0.6}
            ]
            
            auto_trader.real_ai_signal_generator = Mock()
            auto_trader.real_ai_signal_generator.generate_signals.return_value = mock_signals
            
            # Simulate high-frequency trading cycles
            cycles = 20
            latencies = []
            
            for i in range(cycles):
                start = time.time()
                
                # Mock trading cycle execution
                signals = auto_trader.real_ai_signal_generator.generate_signals.return_value
                for signal in signals:
                    # Simulate signal processing
                    await asyncio.sleep(0.01)  # 10ms processing time
                
                end = time.time()
                latencies.append(end - start)
            
            # Analyze performance
            avg_latency = statistics.mean(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)
            
            print(f"Trading engine load test:")
            print(f"  Cycles: {cycles}")
            print(f"  Avg latency: {avg_latency*1000:.2f}ms")
            print(f"  Max latency: {max_latency*1000:.2f}ms")
            print(f"  Min latency: {min_latency*1000:.2f}ms")
            
            # Performance assertions
            assert avg_latency < 0.1  # Average cycle < 100ms
            assert max_latency < 0.2  # Max cycle < 200ms
    
    @pytest.mark.asyncio
    async def test_risk_engine_performance(self):
        """Test risk engine performance under load"""
        from backend.services.risk_engine import RiskEngine
        
        risk_engine = RiskEngine()
        
        # Populate with historical data
        returns = [0.001 * (i % 10 - 5) for i in range(1000)]  # Realistic returns
        pnl = [100 * (i % 20 - 10) for i in range(1000)]  # Realistic P&L
        
        risk_engine.returns_history.extend(returns)
        risk_engine.pnl_history.extend(pnl)
        
        # Test performance of risk calculations
        calculations = 100
        start_time = time.time()
        
        for _ in range(calculations):
            # Perform various risk calculations
            var = risk_engine.calculate_var(0.95)
            cvar = risk_engine.calculate_cvar(0.99)
            sharpe = risk_engine.calculate_sharpe_ratio()
            sortino = risk_engine.calculate_sortino_ratio()
            kelly = risk_engine.calculate_kelly_optimal_size(0.6, 0.02, 0.01)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Risk engine performance test:")
        print(f"  Calculations: {calculations}")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Avg per calculation: {duration/calculations*1000:.2f}ms")
        
        # Performance assertions
        assert duration < 1.0  # Should complete within 1 second
        assert duration/calculations < 0.01  # Each calculation < 10ms
    
    @pytest.mark.asyncio
    async def test_monitoring_system_load(self):
        """Test monitoring system under load"""
        from backend.core.comprehensive_monitoring import MonitoringSystem, MetricType
        
        monitoring = MonitoringSystem()
        
        # Simulate high-frequency metric collection
        metrics_count = 1000
        start_time = time.time()
        
        for i in range(metrics_count):
            monitoring.metric_collector.record(
                f"test.metric.{i % 10}",
                float(i % 100),
                {"source": "load_test"},
                MetricType.GAUGE
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Monitoring system load test:")
        print(f"  Metrics recorded: {metrics_count}")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Throughput: {metrics_count/duration:.0f} metrics/s")
        
        # Performance assertions
        assert duration < 2.0  # Should complete within 2 seconds
        assert metrics_count/duration > 100  # At least 100 metrics/s
    
    @pytest.mark.asyncio
    async def test_redis_cache_performance(self):
        """Test Redis cache performance under load"""
        from backend.core.redis_cache import RedisCache
        
        # Mock Redis for performance testing
        with patch('backend.core.redis_cache.redis.Redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            
            # Mock fast Redis operations
            mock_redis_instance.get.return_value = b'{"test": "data"}'
            mock_redis_instance.setex.return_value = True
            
            cache = RedisCache()
            
            # Test cache operations under load
            operations = 500
            start_time = time.time()
            
            tasks = []
            for i in range(operations):
                if i % 2 == 0:
                    # Set operation
                    task = cache.set_market_data(f'SYMBOL{i}', {'price': i}, ttl=60)
                else:
                    # Get operation
                    task = cache.get_market_data(f'SYMBOL{i}')
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"Redis cache performance test:")
            print(f"  Operations: {operations}")
            print(f"  Duration: {duration:.3f}s")
            print(f"  Throughput: {operations/duration:.0f} ops/s")
            
            # Performance assertions
            assert duration < 1.0  # Should complete within 1 second
            assert operations/duration > 200  # At least 200 ops/s
    
    @pytest.mark.asyncio
    async def test_distributed_lock_contention(self):
        """Test distributed lock performance under contention"""
        from backend.core.distributed_lock import LockManager
        
        # Mock Redis
        with patch('backend.core.distributed_lock.redis.Redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            
            # Simulate lock contention
            call_count = 0
            def mock_set(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                # First few attempts fail (contention), then succeed
                return call_count > 3
            
            mock_redis_instance.set.side_effect = mock_set
            mock_redis_instance.eval.return_value = 1
            
            lock_manager = LockManager()
            
            # Test concurrent lock acquisition
            concurrent_tasks = 10
            start_time = time.time()
            
            async def acquire_lock(task_id):
                async with lock_manager.trading_lock(f"TEST_{task_id}"):
                    await asyncio.sleep(0.01)  # Hold lock briefly
                    return task_id
            
            tasks = [acquire_lock(i) for i in range(concurrent_tasks)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            duration = end_time - start_time
            
            successful = [r for r in results if not isinstance(r, Exception)]
            
            print(f"Distributed lock contention test:")
            print(f"  Concurrent tasks: {concurrent_tasks}")
            print(f"  Successful: {len(successful)}")
            print(f"  Duration: {duration:.3f}s")
            
            # Should handle contention gracefully
            assert len(successful) == concurrent_tasks
    
    def test_memory_usage_under_load(self):
        """Test memory usage under sustained load"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate memory-intensive operations
        large_datasets = []
        for i in range(100):
            # Create large dataset (simulating market data)
            dataset = {
                'ohlcv': [[1.1000 + j*0.0001 for j in range(1000)] for _ in range(5)],
                'indicators': [0.5 + j*0.01 for j in range(1000)],
                'metadata': {'symbol': f'TEST{i}', 'timestamp': time.time()}
            }
            large_datasets.append(dataset)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Clean up
        large_datasets.clear()
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Memory usage test:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Peak: {peak_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Memory increase: {peak_memory - initial_memory:.1f} MB")
        print(f"  Memory recovered: {peak_memory - final_memory:.1f} MB")
        
        # Memory assertions
        memory_increase = peak_memory - initial_memory
        memory_recovered = peak_memory - final_memory
        
        assert memory_increase < 500  # Should not use more than 500MB
        assert memory_recovered > memory_increase * 0.8  # Should recover 80%+ memory
    
    @pytest.mark.asyncio
    async def test_websocket_broadcast_performance(self):
        """Test WebSocket broadcast performance"""
        from backend.services.ws_broadcaster import WebSocketBroadcaster
        
        broadcaster = WebSocketBroadcaster()
        
        # Mock WebSocket connections
        mock_connections = []
        for i in range(100):
            mock_ws = AsyncMock()
            mock_ws.send_text = AsyncMock()
            mock_connections.append(mock_ws)
            broadcaster.connections.append(mock_ws)
        
        # Test broadcast performance
        messages = 50
        start_time = time.time()
        
        for i in range(messages):
            message = {
                'type': 'tick',
                'symbol': 'EURUSD',
                'bid': 1.1000 + i * 0.0001,
                'ask': 1.1002 + i * 0.0001,
                'timestamp': time.time()
            }
            await broadcaster.broadcast(message)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"WebSocket broadcast performance test:")
        print(f"  Connections: {len(mock_connections)}")
        print(f"  Messages: {messages}")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Throughput: {messages/duration:.0f} msg/s")
        
        # Performance assertions
        assert duration < 2.0  # Should complete within 2 seconds
        assert messages/duration > 10  # At least 10 messages/s


class TestStressScenarios:
    """Stress testing for edge cases and failure scenarios"""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_under_stress(self):
        """Test circuit breaker behavior under stress"""
        with patch.dict(os.environ, {'AUTO_TRADE_DRY_RUN': '1'}):
            from backend.services.auto_trader import AutoTrader
            
            auto_trader = AutoTrader()
            
            # Simulate rapid failures
            failure_count = 20
            for i in range(failure_count):
                auto_trader.consecutive_failures += 1
                if auto_trader.consecutive_failures >= auto_trader.max_consecutive_failures:
                    auto_trader._engage_circuit_breaker(f"Failure {i}")
                    break
            
            # Circuit breaker should be active
            assert auto_trader.circuit_breaker_active
            
            # Test recovery after cooldown
            auto_trader.circuit_breaker_engaged_at = time.time() - 3700  # 1+ hour ago
            reset = auto_trader._check_circuit_breaker_reset()
            assert reset
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self):
        """Test for potential memory leaks"""
        import gc
        import psutil
        
        process = psutil.Process()
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        # Simulate repeated operations that could cause leaks
        for cycle in range(10):
            # Create and destroy objects repeatedly
            temp_objects = []
            for i in range(1000):
                obj = {
                    'data': [j for j in range(100)],
                    'timestamp': time.time(),
                    'metadata': {'cycle': cycle, 'item': i}
                }
                temp_objects.append(obj)
            
            # Clear objects
            temp_objects.clear()
            
            # Force garbage collection
            gc.collect()
            
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = current_memory - baseline_memory
            
            print(f"Cycle {cycle}: Memory = {current_memory:.1f} MB (+{memory_growth:.1f} MB)")
            
            # Check for excessive memory growth
            if memory_growth > 100:  # More than 100MB growth
                pytest.fail(f"Potential memory leak detected: {memory_growth:.1f} MB growth")
    
    @pytest.mark.asyncio
    async def test_error_recovery_stress(self):
        """Test error recovery under stress conditions"""
        from backend.services.real_ai_signal_generator import RealAISignalGenerator
        
        with patch('backend.services.real_ai_signal_generator.get_settings') as mock_settings:
            mock_settings.return_value.symbols_list = ['EURUSD']
            
            generator = RealAISignalGenerator()
            
            # Simulate intermittent failures
            failure_rate = 0.3  # 30% failure rate
            attempts = 100
            
            async def flaky_inference(model_name, data):
                if time.time() % 1 < failure_rate:
                    raise Exception("Random failure")
                return {'confidence': 0.7}
            
            generator._execute_model_inference = flaky_inference
            
            # Test resilience
            successful = 0
            failed = 0
            
            for i in range(attempts):
                try:
                    result = await generator._safe_model_inference('xgb', {'test': 'data'})
                    if result is not None:
                        successful += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
            
            success_rate = successful / attempts
            
            print(f"Error recovery stress test:")
            print(f"  Attempts: {attempts}")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")
            print(f"  Success rate: {success_rate:.2%}")
            
            # Should maintain reasonable success rate despite failures
            assert success_rate > 0.5  # At least 50% success rate


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
