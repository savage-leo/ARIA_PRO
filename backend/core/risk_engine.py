# backend/core/risk_engine.py
"""
Enhanced Risk Engine for position sizing and validation
"""

import os
import logging
from typing import Optional

# Optional MT5 binding (graceful degrade)
try:
    import MetaTrader5 as mt5

    MT5_ENABLED = True
except Exception:
    MT5_ENABLED = False

logger = logging.getLogger("aria.core.risk")


def validate_and_size_order(
    symbol: str,
    side: str,
    risk_percent: float,
    sl_price: float,
    account_balance: Optional[float] = None,
    current_price: Optional[float] = None,
) -> float:
    """
    Enhanced position sizing based on risk percentage and account balance
    Returns lot size
    """
    try:
        # Get account balance if not provided
        if account_balance is None:
            account_balance = get_account_balance()

        # Get current price if not provided
        if current_price is None:
            current_price = get_current_price(symbol)

        if account_balance <= 0 or current_price <= 0:
            logger.warning(
                f"Invalid account balance ({account_balance}) or price ({current_price})"
            )
            return 0.01  # Minimum lot size

        # Calculate stop loss distance in pips
        sl_distance = abs(current_price - sl_price)
        if sl_distance == 0:
            logger.warning(f"Stop loss distance is zero for {symbol}")
            return 0.01

        # Convert to pips based on symbol
        pip_value = get_pip_value(symbol)
        sl_distance_pips = sl_distance / pip_value

        # Calculate risk amount in currency
        risk_amount = account_balance * (risk_percent / 100.0)

        # Calculate lot size based on risk
        # Risk amount = (SL distance in pips) * (pip value) * (lot size)
        # Therefore: lot size = risk_amount / (SL distance in pips * pip value)
        lot_size = risk_amount / (sl_distance_pips * pip_value)

        # Apply minimum and maximum lot size constraints
        min_lot = get_min_lot_size(symbol)
        max_lot = get_max_lot_size(symbol)

        lot_size = max(min_lot, min(lot_size, max_lot))

        # Apply Kelly criterion cap if enabled
        kelly_cap = float(os.environ.get("ARIA_KELLY_CAP", 0.25))
        if kelly_cap > 0:
            max_kelly_lot = account_balance * kelly_cap / (sl_distance_pips * pip_value)
            lot_size = min(lot_size, max_kelly_lot)

        logger.info(
            f"Enhanced sizing: {symbol} {side} {lot_size:.4f} lots (risk: {risk_percent}%, balance: {account_balance:.2f})"
        )
        return round(lot_size, 4)

    except Exception as e:
        logger.exception(f"Error in position sizing: {e}")
        return 0.01  # Safe fallback


def get_account_balance() -> float:
    """Get current account balance from MT5"""
    try:
        if MT5_ENABLED:
            account_info = mt5.account_info()
            if account_info:
                return float(account_info.balance)
        # Fallback to environment variable or default
        return float(os.environ.get("ARIA_ACCOUNT_BALANCE", 10000.0))
    except Exception as e:
        logger.error(f"Error getting account balance: {e}")
        return 10000.0  # Default fallback


def get_current_price(symbol: str) -> float:
    """Get current price for symbol"""
    try:
        if MT5_ENABLED:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                return (tick.bid + tick.ask) / 2
        # Fallback to environment variable or default
        default_prices = {
            "EURUSD": 1.0850,
            "GBPUSD": 1.2650,
            "USDJPY": 148.50,
            "AUDUSD": 0.6650,
            "USDCAD": 1.3550,
        }
        return default_prices.get(symbol, 1.0)
    except Exception as e:
        logger.error(f"Error getting current price for {symbol}: {e}")
        return 1.0


def get_pip_value(symbol: str) -> float:
    """Get pip value for symbol"""
    pip_values = {
        "EURUSD": 0.0001,
        "GBPUSD": 0.0001,
        "USDJPY": 0.01,
        "AUDUSD": 0.0001,
        "USDCAD": 0.0001,
        "XAUUSD": 0.01,
    }
    return pip_values.get(symbol, 0.0001)


def get_min_lot_size(symbol: str) -> float:
    """Get minimum lot size for symbol"""
    try:
        if MT5_ENABLED:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                return float(symbol_info.volume_min)
        return 0.01  # Default minimum
    except Exception as e:
        logger.error(f"Error getting min lot size for {symbol}: {e}")
        return 0.01


def get_max_lot_size(symbol: str) -> float:
    """Get maximum lot size for symbol"""
    try:
        if MT5_ENABLED:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                return float(symbol_info.volume_max)
        return 100.0  # Default maximum
    except Exception as e:
        logger.error(f"Error getting max lot size for {symbol}: {e}")
        return 100.0
