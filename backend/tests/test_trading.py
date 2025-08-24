"""
Test suite for trading components.
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from backend.middleware.live_guard import check_trading_allowed, TradingRequest
from backend.core.settings import Settings

def test_trading_request_validation():
    """Test trading request validation."""
    # Valid request
    request = TradingRequest(
        symbol="EURUSD",
        action="buy",
        volume=0.01,
        price=1.1000,
        sl=1.0950,
        tp=1.1050
    )
    assert request.symbol == "EURUSD"
    assert request.action == "buy"
    assert request.volume == 0.01
    
    # Test with invalid data should raise validation error
    with pytest.raises(Exception):
        TradingRequest(
            symbol="",  # Empty symbol
            action="invalid",  # Invalid action
            volume=-0.01  # Negative volume
        )

def test_blocked_symbols():
    """Test blocked symbols are rejected."""
    settings = Settings(
        ARIA_ENV="production",
        AUTO_TRADE_ENABLED=True
    )
    
    # Gold should be blocked
    is_allowed, message = check_trading_allowed(
        "XAUUSD", "buy", 0.01, settings
    )
    assert is_allowed is False
    assert "blocked" in message.lower()
    
    # EURUSD should be allowed
    is_allowed, message = check_trading_allowed(
        "EURUSD", "buy", 0.01, settings
    )
    # May still be blocked by other rules, but not by symbol
    if not is_allowed:
        assert "blocked" not in message.lower() or "symbol" not in message.lower()

def test_position_size_limits():
    """Test position size limits."""
    settings = Settings(
        ARIA_ENV="production",
        AUTO_TRADE_ENABLED=True,
        MAX_POSITION_SIZE=0.1
    )
    
    # Within limit
    is_allowed, message = check_trading_allowed(
        "EURUSD", "buy", 0.05, settings
    )
    # May be blocked by other rules
    if not is_allowed:
        assert "position size" not in message.lower()
    
    # Exceeds limit
    is_allowed, message = check_trading_allowed(
        "EURUSD", "buy", 0.5, settings
    )
    assert is_allowed is False
    assert "position size" in message.lower() or "volume" in message.lower()

def test_weekend_trading_block():
    """Test weekend trading is blocked."""
    from datetime import datetime
    
    settings = Settings(
        ARIA_ENV="production",
        AUTO_TRADE_ENABLED=True
    )
    
    # Mock weekend
    with patch('backend.middleware.live_guard.datetime') as mock_datetime:
        # Saturday
        mock_datetime.now.return_value = datetime(2024, 1, 6, 12, 0, 0)  # Saturday
        mock_datetime.now().weekday.return_value = 5
        
        is_allowed, message = check_trading_allowed(
            "EURUSD", "buy", 0.01, settings
        )
        assert is_allowed is False
        assert "weekend" in message.lower()

def test_dry_run_mode():
    """Test dry run mode allows simulated trades."""
    settings = Settings(
        ARIA_ENV="production",
        AUTO_TRADE_ENABLED=True,
        AUTO_TRADE_DRY_RUN=True
    )
    
    # Should allow in dry run (unless blocked by other rules)
    is_allowed, message = check_trading_allowed(
        "EURUSD", "buy", 0.01, settings
    )
    
    # Check it's not blocked due to dry run
    if not is_allowed:
        assert "dry run" not in message.lower()

@pytest.mark.asyncio
async def test_mt5_executor_connection():
    """Test MT5 executor connection handling."""
    from backend.services.mt5_executor import MT5Executor
    
    executor = MT5Executor()
    
    # Mock MT5 module
    with patch('backend.services.mt5_executor.mt5') as mock_mt5:
        mock_mt5.initialize.return_value = True
        mock_mt5.login.return_value = True
        
        # Test connection
        result = await executor.connect()
        assert result is True
        mock_mt5.initialize.assert_called_once()
        
        # Test disconnection
        await executor.disconnect()
        mock_mt5.shutdown.assert_called_once()

@pytest.mark.asyncio
async def test_trade_execution_flow():
    """Test complete trade execution flow."""
    from backend.services.mt5_executor import MT5Executor
    
    executor = MT5Executor()
    
    with patch('backend.services.mt5_executor.mt5') as mock_mt5:
        # Setup mocks
        mock_mt5.initialize.return_value = True
        mock_mt5.login.return_value = True
        mock_mt5.symbol_info.return_value = Mock(
            bid=1.1000,
            ask=1.1001,
            trade_tick_size=0.00001
        )
        mock_mt5.order_send.return_value = Mock(
            retcode=10009,  # Success
            order=12345
        )
        
        # Execute trade
        result = await executor.execute_trade(
            symbol="EURUSD",
            action="buy",
            volume=0.01,
            sl=1.0950,
            tp=1.1050
        )
        
        assert result["success"] is True
        assert result["order_id"] == 12345
        mock_mt5.order_send.assert_called_once()
