"""
MT5 Stub Module for ARIA PRO
Provides fallback functionality when MetaTrader5 is not available
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class MT5Stub:
    """Stub implementation for MT5 when not available"""
    
    def __init__(self):
        self.connected = False
        logger.warning("Using MT5 stub - no actual MT5 connection")
    
    def initialize(self, *args, **kwargs) -> bool:
        """Stub initialize"""
        logger.info("MT5 stub initialized")
        return True
    
    def login(self, login: int, password: str, server: str) -> bool:
        """Stub login"""
        logger.info(f"MT5 stub login attempt for {login}@{server}")
        self.connected = True
        return True
    
    def shutdown(self):
        """Stub shutdown"""
        logger.info("MT5 stub shutdown")
        self.connected = False
    
    def account_info(self) -> Optional[Dict[str, Any]]:
        """Stub account info"""
        if not self.connected:
            return None
        
        return {
            'login': 12345678,
            'trade_mode': 0,
            'leverage': 100,
            'limit_orders': 200,
            'margin_so_mode': 0,
            'trade_allowed': True,
            'trade_expert': True,
            'margin_mode': 0,
            'currency_digits': 2,
            'fifo_close': False,
            'balance': 10000.0,
            'credit': 0.0,
            'profit': 0.0,
            'equity': 10000.0,
            'margin': 0.0,
            'margin_free': 10000.0,
            'margin_level': 0.0,
            'margin_so_call': 50.0,
            'margin_so_so': 30.0,
            'margin_initial': 0.0,
            'margin_maintenance': 0.0,
            'assets': 10000.0,
            'liabilities': 0.0,
            'commission_blocked': 0.0,
            'name': 'ARIA PRO Demo',
            'server': 'Demo-Server',
            'currency': 'USD',
            'company': 'ARIA Trading'
        }
    
    def copy_rates_from_pos(self, symbol: str, timeframe: int, start_pos: int, count: int):
        """Stub rates data"""
        if not self.connected:
            return None
        
        # Return mock OHLCV data
        import numpy as np
        base_time = int(datetime.now().timestamp())
        
        rates = []
        for i in range(count):
            rates.append({
                'time': base_time - (count - i) * 60,
                'open': 1.1000 + np.random.uniform(-0.001, 0.001),
                'high': 1.1005 + np.random.uniform(-0.001, 0.001),
                'low': 1.0995 + np.random.uniform(-0.001, 0.001),
                'close': 1.1000 + np.random.uniform(-0.001, 0.001),
                'tick_volume': np.random.randint(100, 1000),
                'spread': 2,
                'real_volume': 0
            })
        
        return rates
    
    def symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Stub symbol info"""
        if not self.connected:
            return None
        
        return {
            'custom': False,
            'chart_mode': 0,
            'select': True,
            'visible': True,
            'session_deals': 0,
            'session_buy_orders': 0,
            'session_sell_orders': 0,
            'volume': 0,
            'volumehigh': 0,
            'volumelow': 0,
            'time': int(datetime.now().timestamp()),
            'digits': 5,
            'spread': 2,
            'spread_float': True,
            'ticks_bookdepth': 10,
            'trade_calc_mode': 0,
            'trade_mode': 4,
            'start_time': 0,
            'expiration_time': 0,
            'trade_stops_level': 0,
            'trade_freeze_level': 0,
            'trade_exemode': 2,
            'swap_mode': 1,
            'swap_rollover3days': 3,
            'margin_hedged_use_leg': False,
            'expiration_mode': 7,
            'filling_mode': 1,
            'order_mode': 127,
            'order_gtc_mode': 0,
            'option_mode': 0,
            'option_right': 0,
            'bid': 1.1000,
            'bidhigh': 1.1010,
            'bidlow': 1.0990,
            'ask': 1.1002,
            'askhigh': 1.1012,
            'asklow': 1.0992,
            'last': 1.1001,
            'lasthigh': 1.1011,
            'lastlow': 1.0991,
            'volume_real': 0.0,
            'volumehigh_real': 0.0,
            'volumelow_real': 0.0,
            'option_strike': 0.0,
            'point': 0.00001,
            'trade_tick_value': 1.0,
            'trade_tick_value_profit': 1.0,
            'trade_tick_value_loss': 1.0,
            'trade_tick_size': 0.00001,
            'trade_contract_size': 100000.0,
            'trade_accrued_interest': 0.0,
            'trade_face_value': 0.0,
            'trade_liquidity_rate': 0.0,
            'volume_min': 0.01,
            'volume_max': 500.0,
            'volume_step': 0.01,
            'volume_limit': 0.0,
            'swap_long': -0.76,
            'swap_short': 0.24,
            'margin_initial': 0.0,
            'margin_maintenance': 0.0,
            'session_volume': 0.0,
            'session_turnover': 0.0,
            'session_interest': 0.0,
            'session_buy_orders_volume': 0.0,
            'session_sell_orders_volume': 0.0,
            'session_open': 1.1000,
            'session_close': 1.1001,
            'session_aw': 0.0,
            'session_price_settlement': 0.0,
            'session_price_limit_min': 0.0,
            'session_price_limit_max': 0.0,
            'margin_hedged': 50000.0,
            'price_change': 0.0001,
            'price_volatility': 0.0,
            'price_theoretical': 0.0,
            'price_greeks_delta': 0.0,
            'price_greeks_theta': 0.0,
            'price_greeks_gamma': 0.0,
            'price_greeks_vega': 0.0,
            'price_greeks_rho': 0.0,
            'price_greeks_omega': 0.0,
            'price_sensitivity': 0.0,
            'basis': '',
            'category': '',
            'currency_base': 'EUR',
            'currency_profit': 'USD',
            'currency_margin': 'EUR',
            'bank': '',
            'description': f'{symbol} stub data',
            'exchange': '',
            'formula': '',
            'isin': '',
            'name': symbol,
            'page': '',
            'path': f'Forex\\{symbol}'
        }
    
    def positions_get(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Stub positions"""
        if not self.connected:
            return []
        
        # Return empty positions list for stub
        return []
    
    def orders_get(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Stub orders"""
        if not self.connected:
            return []
        
        # Return empty orders list for stub
        return []

# Try to import real MT5, fallback to stub
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    logger.info("MetaTrader5 module imported successfully")
except ImportError:
    mt5 = MT5Stub()
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 not available, using stub implementation")

# Export the mt5 instance
__all__ = ['mt5', 'MT5_AVAILABLE', 'MT5Stub']
