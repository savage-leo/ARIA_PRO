try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except Exception:
    try:
        # Fallback to internal stub for non-MT5 environments
        from backend import mt5 as mt5  # type: ignore
        MT5_AVAILABLE = False
    except Exception:
        mt5 = None  # type: ignore
        MT5_AVAILABLE = False
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import subprocess
import time
import os
from pathlib import Path
from backend.services.mt5_connection_pool import get_connection_pool
from backend.core.config import get_settings

settings = get_settings()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MT5_LOGIN = int(settings.MT5_LOGIN) if settings.MT5_LOGIN is not None else 0
MT5_PASSWORD = settings.MT5_PASSWORD or ""
MT5_SERVER = settings.MT5_SERVER or ""


class MT5Executor:
    def __init__(self):
        self.initialized = False
        self.connected = False
        self.connection_pool = get_connection_pool()

    def _discover_terminal_path(self) -> Optional[str]:
        """Return a plausible MetaTrader 5 terminal path if available.

        Priority:
        1) MT5_TERMINAL_PATH env var
        2) Common install locations on Windows
        """
        # 1) Explicit env override
        env_path = os.getenv("MT5_TERMINAL_PATH", "").strip()
        if env_path:
            p = Path(env_path)
            if p.exists():
                return str(p)

        # 2) Common defaults
        candidates = [
            r"C:\\Program Files\\MetaTrader 5\\terminal64.exe",
            r"C:\\Program Files (x86)\\MetaTrader 5\\terminal64.exe",
            r"C:\\Program Files\\MetaTrader 5\\terminal.exe",
            r"C:\\Program Files (x86)\\MetaTrader 5\\terminal.exe",
        ]
        for c in candidates:
            if Path(c).exists():
                return c
        return None

    def _attempt_launch_terminal(self) -> bool:
        """Try to start the MetaTrader 5 terminal if not running.

        Returns True if a launch was attempted (and possibly succeeded), False otherwise.
        """
        terminal_path = self._discover_terminal_path()
        if not terminal_path:
            logger.error(
                "MT5 terminal path not found. Set MT5_TERMINAL_PATH or install MetaTrader 5."
            )
            return False

        try:
            logger.info(f"Attempting to launch MT5 terminal: {terminal_path}")
            # Start the terminal minimized and detached
            subprocess.Popen(
                [terminal_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            # Give terminal some time to start up
            time.sleep(5)
            return True
        except Exception as e:
            logger.error(f"Failed to launch MT5 terminal: {e}")
            return False

    def connect(self) -> bool:
        """Connect to MT5 - check existing or establish new connection"""
        if self.initialized and self.connected:
            # Already connected, verify it's still valid
            try:
                account = mt5.account_info()
                if account is not None:
                    return True
                # Connection lost, try to reconnect
                self.initialized = False
                self.connected = False
            except Exception:
                self.initialized = False
                self.connected = False
        
        # Try to establish connection
        try:
            return self.init_mt5()
        except Exception as e:
            logger.error(f"Failed to connect to MT5: {e}")
            return False

    def disconnect(self):
        """Disconnect from MT5 - alias for shutdown for compatibility"""
        try:
            self.shutdown()
        except Exception:
            pass

    def init_mt5(self) -> bool:
        """Initialize MT5 connection.

        - Try direct initialize/login first
        - If that fails, attempt to launch terminal and retry initialize/login
        """

        # Helper to try initialize + login with optional path
        def _try_init_and_login(init_path: Optional[str] = None) -> bool:
            ok = mt5.initialize(path=init_path) if init_path else mt5.initialize()
            if not ok:
                return False
            if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
                # ensure shutdown to avoid stale state
                mt5.shutdown()
                return False
            return True

        try:
            # 1) Direct attempt
            if _try_init_and_login():
                self.initialized = True
                self.connected = True
                logger.info("MT5 connection established successfully")
                return True

            # 2) Try launching terminal, then retry with discovered path
            launched = self._attempt_launch_terminal()
            # Give terminal time to fully initialize
            if launched:
                for delay_s in (5, 10, 15):
                    time.sleep(delay_s)
                    term_path = self._discover_terminal_path()
                    if _try_init_and_login(init_path=term_path):
                        self.initialized = True
                        self.connected = True
                        logger.info(
                            "MT5 connection established after launching terminal"
                        )
                        return True

            # 3) Final failure
            error = mt5.last_error()
            logger.error(f"MT5 initialization/login failed. Last error: {error}")
            raise RuntimeError(f"MT5 init/login failed: {error}")

        except Exception as e:
            logger.error(f"MT5 connection error: {str(e)}")
            raise

    def get_account_info(self) -> Dict[str, Any]:
        """Get MT5 account information"""
        if not self.initialized:
            self.init_mt5()

        account_info = mt5.account_info()
        if account_info is None:
            raise RuntimeError("Failed to get account info")

        return {
            "login": account_info.login,
            "balance": account_info.balance,
            "equity": account_info.equity,
            "margin": account_info.margin,
            "margin_free": account_info.margin_free,
            "profit": account_info.profit,
            "currency": account_info.currency,
            "leverage": account_info.leverage,
        }

    def place_order(
        self,
        symbol: str,
        volume: float,
        order_type: str,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "ARIA AI Order",
    ) -> Dict[str, Any]:
        """Place a real order in MT5 using connection pool"""
        with self.connection_pool.get_connection() as conn:
            if not conn:
                raise RuntimeError("No MT5 connection available")

        # Validate order type
        if order_type.lower() not in ["buy", "sell"]:
            raise ValueError("Order type must be 'buy' or 'sell'")

        # Prepare order request
        order_dict = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": (
                mt5.ORDER_TYPE_BUY
                if order_type.lower() == "buy"
                else mt5.ORDER_TYPE_SELL
            ),
            "deviation": 10,
            "magic": 234000,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Add stop loss and take profit if provided
        if sl is not None:
            order_dict["sl"] = sl
        if tp is not None:
            order_dict["tp"] = tp

        # Send order
        result = mt5.order_send(order_dict)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_msg = f"Order failed: {result.retcode} - {result.comment}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info(f"Order placed successfully: Ticket {result.order}")

        return {
            "ticket": result.order,
            "status": "success",
            "volume": result.volume,
            "price": result.price,
            "retcode": result.retcode,
            "comment": result.comment,
            "timestamp": datetime.now().isoformat(),
        }

    def get_positions(self) -> list:
        """Get all open positions"""
        if not self.initialized:
            self.init_mt5()

        positions = mt5.positions_get()
        if positions is None:
            return []

        return [
            {
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": "buy" if pos.type == mt5.POSITION_TYPE_BUY else "sell",
                "volume": pos.volume,
                "price_open": pos.price_open,
                "price_current": pos.price_current,
                "profit": pos.profit,
                "sl": pos.sl,
                "tp": pos.tp,
                "time": pos.time,
                "comment": pos.comment,
            }
            for pos in positions
        ]

    def close_position(
        self, ticket: int, volume: Optional[float] = None
    ) -> Dict[str, Any]:
        """Close a specific position"""
        if not self.initialized:
            self.init_mt5()

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            raise ValueError(f"Position with ticket {ticket} not found")

        pos = positions[0]
        close_volume = volume if volume is not None else pos.volume

        close_dict = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": close_volume,
            "type": (
                mt5.ORDER_TYPE_SELL
                if pos.type == mt5.POSITION_TYPE_BUY
                else mt5.ORDER_TYPE_BUY
            ),
            "position": ticket,
            "deviation": 10,
            "magic": 234000,
            "comment": "ARIA AI Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(close_dict)

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_msg = f"Position close failed: {result.retcode} - {result.comment}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        return {
            "ticket": result.order,
            "status": "closed",
            "volume": result.volume,
            "price": result.price,
            "timestamp": datetime.now().isoformat(),
        }

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get symbol information"""
        if not self.initialized:
            self.init_mt5()

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise ValueError(f"Symbol {symbol} not found")

        return {
            "symbol": symbol_info.name,
            "bid": symbol_info.bid,
            "ask": symbol_info.ask,
            "point": symbol_info.point,
            "digits": symbol_info.digits,
            "spread": symbol_info.spread,
            "trade_mode": symbol_info.trade_mode,
            "volume_min": symbol_info.volume_min,
            "volume_max": symbol_info.volume_max,
            "volume_step": symbol_info.volume_step,
        }

    def shutdown(self):
        """Shutdown MT5 connection"""
        if self.initialized:
            mt5.shutdown()
            self.initialized = False
            self.connected = False
            logger.info("MT5 connection closed")


def execute_order(
    symbol: str,
    side: str,
    volume: float,
    sl: Optional[float] = None,
    tp: Optional[float] = None,
    comment: str = "ARIA SMC Order",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Execute an order (real or simulated)
    Args:
        symbol: Trading symbol
        side: 'buy' or 'sell'
        volume: Order volume
        sl: Stop loss price
        tp: Take profit price
        comment: Order comment
        dry_run: If True, simulate execution
    """
    if dry_run:
        # Simulate execution
        import time

        return {
            "ticket": int(time.time() * 1000),  # Simulated ticket
            "status": "simulated",
            "volume": volume,
            "price": 1.1000,  # Simulated price
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "side": side,
            "sl": sl,
            "tp": tp,
            "comment": comment,
            "dry_run": True,
        }
    else:
        # Real execution using MT5
        return mt5_executor.place_order(
            symbol=symbol, volume=volume, order_type=side, sl=sl, tp=tp, comment=comment
        )


# Global instance
mt5_executor = MT5Executor()
