"""
Lightweight MetaTrader5 stub for non-MT5 environments.
Provides a minimal surface so imports succeed and code can guard on availability.
This module never performs real trading; all operations return safe defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

# --- Constants (approximate) ---
TRADE_ACTION_DEAL = 1
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
ORDER_TIME_GTC = 0
ORDER_FILLING_IOC = 1
POSITION_TYPE_BUY = 0
POSITION_TYPE_SELL = 1
TRADE_RETCODE_DONE = 1  # Arbitrary non-zero code for successful trade in real API

# Timeframe constants
TIMEFRAME_M1 = 1
TIMEFRAME_M2 = 2
TIMEFRAME_M3 = 3
TIMEFRAME_M4 = 4
TIMEFRAME_M5 = 5
TIMEFRAME_M6 = 6
TIMEFRAME_M10 = 10
TIMEFRAME_M12 = 12
TIMEFRAME_M15 = 15
TIMEFRAME_M20 = 20
TIMEFRAME_M30 = 30
TIMEFRAME_H1 = 60
TIMEFRAME_H2 = 120
TIMEFRAME_H3 = 180
TIMEFRAME_H4 = 240
TIMEFRAME_H6 = 360
TIMEFRAME_H8 = 480
TIMEFRAME_H12 = 720
TIMEFRAME_D1 = 1440
TIMEFRAME_W1 = 10080
TIMEFRAME_MN1 = 43200

# --- Data classes to mimic structures ---
@dataclass
class AccountInfo:
    login: int = 0
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    margin_free: float = 0.0
    profit: float = 0.0
    currency: str = "USD"
    leverage: int = 1

@dataclass
class Position:
    ticket: int
    symbol: str
    type: int  # POSITION_TYPE_BUY/SELL
    volume: float
    price_open: float
    price_current: float
    profit: float
    sl: float
    tp: float
    time: float
    comment: str = ""

@dataclass
class SymbolInfo:
    name: str
    bid: float = 0.0
    ask: float = 0.0
    point: float = 0.0001
    digits: int = 5
    spread: int = 0
    trade_mode: int = 0
    volume_min: float = 0.01
    volume_max: float = 100.0
    volume_step: float = 0.01

@dataclass
class OrderResult:
    retcode: int = 0
    comment: str = "MT5 stub not available"
    order: int = 0
    volume: float = 0.0
    price: float = 0.0

# --- Functions ---

def initialize(path: Optional[str] = None) -> bool:  # noqa: ARG001
    return False


def login(login: int, password: str, server: str) -> bool:  # noqa: ARG001
    return False


def shutdown() -> None:
    return None


def last_error():
    return (0, "MT5 stub - not available")


def account_info() -> Optional[AccountInfo]:
    return None


def positions_get(ticket: Optional[int] = None) -> List[Position]:  # noqa: ARG001
    return []


def order_send(request: Dict[str, Any]) -> OrderResult:  # noqa: ARG001
    return OrderResult(retcode=0)


def symbol_info(symbol: str) -> Optional[SymbolInfo]:  # noqa: ARG001
    return None


def symbol_select(symbol: str, enable: bool) -> bool:  # noqa: ARG001
    return False


def symbol_info_tick(symbol: str) -> Optional[SimpleNamespace]:  # noqa: ARG001
    return None


def copy_rates_from_pos(symbol: str, timeframe: int, start_pos: int, count: int):  # noqa: ARG001
    return []

__all__ = [
    # constants
    "TRADE_ACTION_DEAL",
    "ORDER_TYPE_BUY",
    "ORDER_TYPE_SELL",
    "ORDER_TIME_GTC",
    "ORDER_FILLING_IOC",
    "POSITION_TYPE_BUY",
    "POSITION_TYPE_SELL",
    "TRADE_RETCODE_DONE",
    # timeframes
    "TIMEFRAME_M1",
    "TIMEFRAME_M2",
    "TIMEFRAME_M3",
    "TIMEFRAME_M4",
    "TIMEFRAME_M5",
    "TIMEFRAME_M6",
    "TIMEFRAME_M10",
    "TIMEFRAME_M12",
    "TIMEFRAME_M15",
    "TIMEFRAME_M20",
    "TIMEFRAME_M30",
    "TIMEFRAME_H1",
    "TIMEFRAME_H2",
    "TIMEFRAME_H3",
    "TIMEFRAME_H4",
    "TIMEFRAME_H6",
    "TIMEFRAME_H8",
    "TIMEFRAME_H12",
    "TIMEFRAME_D1",
    "TIMEFRAME_W1",
    "TIMEFRAME_MN1",
    # structures
    "AccountInfo",
    "Position",
    "SymbolInfo",
    "OrderResult",
    # functions
    "initialize",
    "login",
    "shutdown",
    "last_error",
    "account_info",
    "positions_get",
    "order_send",
    "symbol_info",
    "symbol_select",
    "symbol_info_tick",
    "copy_rates_from_pos",
]
