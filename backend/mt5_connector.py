# backend/mt5_connector.py
"""
Simplified MT5 connector for CPU-friendly operations
"""
import logging
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

class MT5Connector:
    """Lightweight MT5 connection manager"""
    
    def __init__(self):
        self.connected = False
        self.account_info = None
        
    def connect(self) -> bool:
        """Connect to MT5"""
        if not MT5_AVAILABLE:
            logger.warning("MT5 not available, running in simulation mode")
            self.connected = False
            return False
            
        if not mt5.initialize():
            logger.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
            
        self.connected = True
        self.account_info = mt5.account_info()._asdict() if mt5.account_info() else {}
        logger.info(f"Connected to MT5: {self.account_info.get('server', 'Unknown')}")
        return True
    
    def disconnect(self):
        """Disconnect from MT5"""
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")
    
    def get_rates(self, symbol: str = "EURUSD", timeframe: int = None, count: int = 100) -> pd.DataFrame:
        """Get market rates"""
        if not self.connected or not MT5_AVAILABLE:
            return self._simulate_rates(symbol, count)
            
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_M1
            
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        
        if rates is None:
            logger.warning(f"Failed to get rates for {symbol}")
            return self._simulate_rates(symbol, count)
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df
    
    def _simulate_rates(self, symbol: str, count: int) -> pd.DataFrame:
        """Generate simulated rates for testing"""
        now = datetime.now()
        dates = pd.date_range(end=now, periods=count, freq='1min')
        
        # Base prices for different symbols
        base_prices = {
            "EURUSD": 1.1000,
            "GBPUSD": 1.2500,
            "USDJPY": 110.00,
            "AUDUSD": 0.7500,
            "USDCHF": 0.9200
        }
        
        base = base_prices.get(symbol, 1.0)
        
        # Generate realistic OHLC data
        np.random.seed(42)  # For consistency
        returns = np.random.randn(count) * 0.0001  # Small returns
        close = base * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'open': np.roll(close, 1),
            'high': close + np.abs(np.random.randn(count) * 0.0001),
            'low': close - np.abs(np.random.randn(count) * 0.0001),
            'close': close,
            'tick_volume': np.random.randint(100, 1000, count),
            'spread': np.random.randint(1, 5, count),
            'real_volume': np.random.randint(1000000, 10000000, count)
        }, index=dates)
        
        df['open'].iloc[0] = base
        return df
    
    def send_order(self, order_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Send order to MT5"""
        if not self.connected or not MT5_AVAILABLE:
            # Simulate order execution
            return {
                'retcode': 10009,  # TRADE_RETCODE_DONE
                'order': np.random.randint(1000000, 9999999),
                'price': order_dict.get('price', 1.1000),
                'volume': order_dict.get('volume', 0.01),
                'comment': 'simulated'
            }
        
        result = mt5.order_send(order_dict)
        if result is None:
            return {'retcode': -1, 'comment': 'Order send failed'}
            
        return result._asdict()
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get open positions"""
        if not self.connected or not MT5_AVAILABLE:
            return []
            
        positions = mt5.positions_get()
        if positions is None:
            return []
            
        return [pos._asdict() for pos in positions]
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information"""
        if not self.connected or not MT5_AVAILABLE:
            # Return simulated symbol info
            return {
                'symbol': symbol,
                'point': 0.00001 if 'JPY' not in symbol else 0.001,
                'digits': 5 if 'JPY' not in symbol else 3,
                'spread': 2,
                'volume_min': 0.01,
                'volume_max': 100.0,
                'volume_step': 0.01
            }
        
        info = mt5.symbol_info(symbol)
        return info._asdict() if info else None

# Global instance
mt5_connector = MT5Connector()
