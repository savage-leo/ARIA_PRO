"""
Pytest configuration and shared fixtures for ARIA PRO test suite
"""

import pytest
import asyncio
import os
import sys
from unittest.mock import Mock, patch, AsyncMock

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_environment():
    """Mock environment variables for testing"""
    test_env = {
        'AUTO_TRADE_SYMBOLS': 'EURUSD,GBPUSD,USDJPY',
        'AUTO_TRADE_DRY_RUN': '1',
        'AUTO_TRADE_ENABLED': '1',
        'AUTO_TRADE_CB_COOLDOWN': '60',
        'AUTO_TRADE_POS_LIMITS': 'EURUSD:2,GBPUSD:1',
        'RISK_MAX_POSITION_SIZE': '0.01',
        'RISK_MAX_DAILY_LOSS': '1000',
        'REDIS_HOST': 'localhost',
        'REDIS_PORT': '6379',
        'REDIS_DB': '0',
        'JWT_SECRET_KEY': 'test_secret_key_for_testing_only_32_chars',
        'ADMIN_API_KEY': 'test_admin_key_16ch'
    }
    
    with patch.dict(os.environ, test_env):
        yield test_env


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing"""
    mock_redis_instance = AsyncMock()
    mock_redis_instance.get.return_value = None
    mock_redis_instance.set.return_value = True
    mock_redis_instance.setex.return_value = True
    mock_redis_instance.delete.return_value = True
    mock_redis_instance.exists.return_value = False
    mock_redis_instance.eval.return_value = 1
    mock_redis_instance.sadd.return_value = True
    mock_redis_instance.srem.return_value = True
    mock_redis_instance.smembers.return_value = set()
    
    with patch('redis.Redis', return_value=mock_redis_instance):
        yield mock_redis_instance


@pytest.fixture
def mock_mt5_manager():
    """Mock MT5 manager for testing"""
    mock_manager = Mock()
    mock_manager.get_account_info.return_value = {
        'balance': 10000.0,
        'equity': 10000.0,
        'margin': 0.0,
        'free_margin': 10000.0,
        'profit': 0.0
    }
    mock_manager.get_positions.return_value = []
    mock_manager.place_order.return_value = {
        'ticket': 12345,
        'status': 'filled',
        'volume': 0.01,
        'price': 1.1000
    }
    mock_manager.get_symbol_info.return_value = {
        'bid': 1.1000,
        'ask': 1.1002,
        'spread': 2,
        'digits': 5
    }
    return mock_manager


@pytest.fixture
def mock_market_data():
    """Mock market data for testing"""
    return {
        'open': [1.0990, 1.0995, 1.1000, 1.1005, 1.1010],
        'high': [1.1010, 1.1015, 1.1020, 1.1025, 1.1030],
        'low': [1.0980, 1.0985, 1.0990, 1.0995, 1.1000],
        'close': [1.0995, 1.1000, 1.1005, 1.1010, 1.1015],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }


@pytest.fixture
def mock_ai_signals():
    """Mock AI signals for testing"""
    return [
        {
            'symbol': 'EURUSD',
            'action': 'buy',
            'confidence': 0.75,
            'entry_price': 1.1000,
            'stop_loss': 1.0950,
            'take_profit': 1.1100,
            'timestamp': '2024-01-01T12:00:00Z'
        },
        {
            'symbol': 'GBPUSD',
            'action': 'sell',
            'confidence': 0.68,
            'entry_price': 1.2500,
            'stop_loss': 1.2550,
            'take_profit': 1.2400,
            'timestamp': '2024-01-01T12:00:00Z'
        }
    ]


@pytest.fixture
def sample_returns():
    """Sample return data for risk calculations"""
    return [
        0.001, -0.002, 0.003, -0.001, 0.002,
        -0.003, 0.001, 0.004, -0.002, 0.001,
        0.002, -0.001, 0.003, -0.004, 0.002
    ] * 10  # 150 data points


@pytest.fixture
def sample_pnl():
    """Sample P&L data for risk calculations"""
    return [
        100, -50, 150, -25, 75,
        -100, 50, 200, -75, 25,
        125, -30, 180, -120, 90
    ] * 10  # 150 data points


# Test markers
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "load: mark test as a load/performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Skip tests if dependencies not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle missing dependencies"""
    skip_redis = pytest.mark.skip(reason="Redis not available")
    skip_mt5 = pytest.mark.skip(reason="MT5 not available")
    
    for item in items:
        if "redis" in item.keywords and not _redis_available():
            item.add_marker(skip_redis)
        if "mt5" in item.keywords and not _mt5_available():
            item.add_marker(skip_mt5)


def _redis_available():
    """Check if Redis is available for testing"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        return True
    except:
        return False


def _mt5_available():
    """Check if MT5 is available for testing"""
    try:
        import MetaTrader5 as mt5
        return mt5.initialize()
    except:
        return False
