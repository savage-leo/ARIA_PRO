"""
MT5 Executor - Lean & Fast
Direct MT5 execution without overhead
"""

import MetaTrader5 as mt5
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger("MT5_EXECUTOR")


class MT5Executor:
    """Lean MT5 executor for instant trades"""
    
    def __init__(self):
        self.connected = False
        self.account_info = None
        self.symbol_info = {}
        
    async def initialize(self) -> bool:
        """Initialize MT5 connection"""
        try:
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
                
            # Login from environment
            import os
            login = int(os.getenv("MT5_LOGIN", "0"))
            password = os.getenv("MT5_PASSWORD", "")
            server = os.getenv("MT5_SERVER", "")
            
            if login and password and server:
                if not mt5.login(login, password=password, server=server):
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
                    return False
                    
            self.connected = True
            self.account_info = mt5.account_info()._asdict()
            logger.info(f"MT5 connected: Balance={self.account_info['balance']}")
            return True
            
        except Exception as e:
            logger.error(f"MT5 init error: {e}")
            return False
            
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data for symbol"""
        if not self.connected:
            await self.initialize()
            
        # Get tick
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return {}
            
        # Get bars for ATR calculation
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 14)
        
        atr = 0
        if rates is not None and len(rates) > 0:
            high_low = [r['high'] - r['low'] for r in rates]
            atr = sum(high_low) / len(high_low)
            
        return {
            "symbol": symbol,
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": tick.ask - tick.bid,
            "spread_pct": (tick.ask - tick.bid) / tick.ask,
            "volume": tick.volume,
            "timestamp": datetime.fromtimestamp(tick.time),
            "atr": atr,
            "atr_pct": atr / tick.ask if tick.ask > 0 else 0
        }
        
    async def get_tick(self, symbol: str) -> Dict[str, Any]:
        """Get current tick"""
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return {}
        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": tick.ask - tick.bid,
            "time": tick.time
        }
        
    async def place_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        sl_distance: float = 0,
        tp_distance: float = 0,
        magic: int = 8100,
        comment: str = "ARIA"
    ) -> Dict[str, Any]:
        """Place market order"""
        try:
            if not self.connected:
                await self.initialize()
                
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return {"status": "error", "message": "Symbol not found"}
                
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            
            # Determine order type and price
            if order_type.upper() == "BUY":
                order_type_mt5 = mt5.ORDER_TYPE_BUY
                price = tick.ask
                sl = price - sl_distance if sl_distance > 0 else 0
                tp = price + tp_distance if tp_distance > 0 else 0
            else:
                order_type_mt5 = mt5.ORDER_TYPE_SELL
                price = tick.bid
                sl = price + sl_distance if sl_distance > 0 else 0
                tp = price - tp_distance if tp_distance > 0 else 0
                
            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type_mt5,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,  # Max slippage in points
                "magic": magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Order placed: {result.order}")
                return {
                    "status": "success",
                    "ticket": result.order,
                    "price": result.price,
                    "volume": result.volume,
                    "symbol": symbol,
                    "type": order_type
                }
            else:
                logger.error(f"Order failed: {result.comment}")
                return {
                    "status": "error",
                    "message": result.comment,
                    "retcode": result.retcode
                }
                
        except Exception as e:
            logger.error(f"Order error: {e}")
            return {"status": "error", "message": str(e)}
            
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions"""
        positions = mt5.positions_get()
        if not positions:
            return []
            
        result = []
        for pos in positions:
            result.append({
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                "volume": pos.volume,
                "price": pos.price_open,
                "current": pos.price_current,
                "profit": pos.profit,
                "sl": pos.sl,
                "tp": pos.tp,
                "magic": pos.magic,
                "comment": pos.comment
            })
        return result
        
    async def close_position(self, ticket: int) -> Dict[str, Any]:
        """Close specific position"""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return {"status": "error", "message": "Position not found"}
            
        pos = position[0]
        
        # Determine close type
        if pos.type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(pos.symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(pos.symbol).ask
            
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": pos.magic,
            "comment": f"Close {ticket}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return {
                "status": "success",
                "closed_ticket": ticket,
                "close_price": result.price,
                "profit": pos.profit
            }
        else:
            return {
                "status": "error",
                "message": result.comment
            }
            
    async def close_all_positions(self) -> Dict[str, Any]:
        """Close all open positions"""
        positions = await self.get_positions()
        results = []
        
        for pos in positions:
            result = await self.close_position(pos['ticket'])
            results.append(result)
            
        total_profit = sum(
            r.get('profit', 0) for r in results 
            if r.get('status') == 'success'
        )
        
        return {
            "closed": len([r for r in results if r.get('status') == 'success']),
            "failed": len([r for r in results if r.get('status') != 'success']),
            "total_profit": total_profit,
            "details": results
        }
        
    async def get_account_balance(self) -> float:
        """Get current account balance"""
        if not self.connected:
            await self.initialize()
        info = mt5.account_info()
        return info.balance if info else 0
        
    async def close(self):
        """Close MT5 connection"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 disconnected")
