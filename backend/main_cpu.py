# backend/main_cpu.py
"""
ARIA CPU-Friendly War Machine
Institutional-grade FX trading backend optimized for T470
Port: 8100
"""
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
import json
import logging
import os
import sys
import secrets
import uvicorn

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, os.pardir))

# Security settings - PRODUCTION CRITICAL
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("TOKEN_EXPIRE_MINUTES", "30"))

# Import CPU module routes
from routes.cpu_modules import router as cpu_router

from orchestrator.orchestrator_cpu_friendly import orchestrator
from tools.strategy_accounting import strategy_accounting
from tools.coach import explain_trade, explain_performance

# Simple logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("ARIA")

# Initialize FastAPI
app = FastAPI(
    title="ARIA CPU War Machine",
    version="2.0.0",
    description="Lightweight institutional FX trading system"
)

# CORS - simple open configuration for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
active_connections: List[WebSocket] = []
mt5_connected = False
market_data_task = None

# Request Models
class SignalRequest(BaseModel):
    symbol: str = "EURUSD"
    timeframe: str = "M1"
    lookback: int = 100

class ExecuteRequest(BaseModel):
    symbol: str = "EURUSD"
    action: str  # buy/sell
    volume: float = 0.01
    sl_pips: Optional[float] = 20
    tp_pips: Optional[float] = 40
    comment: str = "aria_cpu"

class AutoTradeRequest(BaseModel):
    symbol: str = "EURUSD"
    timeframe: str = "M1"
    confidence_threshold: float = 0.55
    max_volume: float = 0.10

# MT5 Connection
def init_mt5():
    """Initialize MT5 connection"""
    global mt5_connected
    
    if not mt5.initialize():
        logger.error("MT5 initialization failed")
    else:
        account = mt5.account_info()
        if account:
            logger.info(f"MT5 connected: Balance={account.balance:.2f}, Equity={account.equity:.2f}")
            logger.info(f"Broker={account.company}, Server={account.server}, Leverage=1:{account.leverage}")
        
        mt5_connected = True
    
    return mt5_connected

def get_market_data(symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    """Get market data from MT5 or generate dummy data"""
    
    if mt5_connected:
        # Map timeframe string to MT5 constant
        tf_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }
        
        tf = tf_map.get(timeframe, mt5.TIMEFRAME_M1)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        
        if rates is not None:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            return df
    
    # Generate dummy data for testing
    logger.warning("Using simulated market data")
    dates = pd.date_range(end=datetime.now(), periods=count, freq='1min')
    base_price = 1.1000
    df = pd.DataFrame({
        'open': np.random.randn(count) * 0.001 + base_price,
        'high': np.random.randn(count) * 0.001 + base_price + 0.0005,
        'low': np.random.randn(count) * 0.001 + base_price - 0.0005,
        'close': np.random.randn(count) * 0.001 + base_price,
        'tick_volume': np.random.randint(100, 1000, count),
        'spread': np.random.randint(1, 5, count),
        'real_volume': np.random.randint(1000000, 10000000, count)
    }, index=dates)
    
    # Make it more realistic
    df['close'] = df['close'].cumsum() / 100 + base_price
    df['open'] = df['close'].shift(1).fillna(base_price)
    df['high'] = df[['open', 'close']].max(axis=1) + abs(np.random.randn(count) * 0.0001)
    df['low'] = df[['open', 'close']].min(axis=1) - abs(np.random.randn(count) * 0.0001)
    
    return df

# Startup and Shutdown
@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    logger.info("ARIA CPU WAR MACHINE INITIALIZING...")
    
    # Skip heavy scans for faster startup
    logger.info("Skipping preflight checks for development mode...")
    
    # Initialize MT5
    init_mt5()
    
    # Start market data streaming task
    global market_data_task
    market_data_task = asyncio.create_task(stream_market_data())
    
    logger.info("System ready on port 8100")

@app.on_event("shutdown")
async def shutdown():
    """Clean shutdown"""
    logger.info("Shutting down...")
    
    # Cancel market data task
    if market_data_task:
        market_data_task.cancel()
        try:
            await market_data_task
        except asyncio.CancelledError:
            pass
    
    # Disconnect MT5
    if mt5_connected and mt5:
        mt5.shutdown()
    
    # Close WebSocket connections
    for connection in active_connections:
        await connection.close()
    
    logger.info("Shutdown complete")

# WebSocket handler
async def stream_market_data():
    """Stream market data to connected clients"""
    while True:
        try:
            if active_connections:
                # Get latest market data
                df = get_market_data("EURUSD", "M1", 100)
                
                # Process through orchestrator
                intent = await orchestrator.process_market_data(df)
                
                # Broadcast to all connected clients
                data = {
                    "type": "market_update",
                    "timestamp": time.time(),
                    "price": float(df['close'].iloc[-1]),
                    "signal": intent if intent else None,
                    "metrics": strategy_accounting.get_metrics()
                }
                
                for connection in active_connections:
                    try:
                        await connection.send_json(data)
                    except:
                        active_connections.remove(connection)
            
            await asyncio.sleep(1)  # Stream every second
            
        except Exception as e:
            logger.error(f"Market data stream error: {e}")
            await asyncio.sleep(5)

# Include CPU module routes
app.include_router(cpu_router)

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "ARIA CPU War Machine",
        "version": "2.0.0",
        "status": "running",
        "mt5_connected": mt5_connected,
        "active_connections": len(active_connections),
        "port": 8100
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    metrics = strategy_accounting.get_metrics()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mt5_connected": mt5_connected,
        "total_trades": metrics['total_trades'],
        "win_rate": metrics['win_rate'],
        "sharpe": metrics['sharpe']
    }

@app.post("/signal")
async def generate_signal(request: SignalRequest):
    """Generate trading signal"""
    try:
        # Get market data
        df = get_market_data(request.symbol, request.timeframe, request.lookback)
        
        # Process through orchestrator
        intent = await orchestrator.process_market_data(df)
        
        if intent:
            return JSONResponse(content={
                "status": "success",
                "signal": intent,
                "explanation": explain_trade(intent)
            })
        else:
            return JSONResponse(content={
                "status": "no_signal",
                "message": "No trading opportunity detected"
            })
            
    except Exception as e:
        logger.error(f"Signal generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute")
async def execute_trade(request: ExecuteRequest):
    """Execute a trade"""
    try:
        if not mt5_connected:
            raise HTTPException(status_code=503, detail="MT5 not connected")
        
        # Prepare order request
        symbol_info = mt5.symbol_info(request.symbol)
        if symbol_info is None:
            raise HTTPException(status_code=404, detail=f"Symbol {request.symbol} not found")
        
        point = symbol_info.point
        price = mt5.symbol_info_tick(request.symbol).ask if request.action == "buy" else mt5.symbol_info_tick(request.symbol).bid
        
        order = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": request.symbol,
            "volume": request.volume,
            "type": mt5.ORDER_TYPE_BUY if request.action == "buy" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": request.comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Add SL/TP if provided
        if request.sl_pips:
            order["sl"] = price - request.sl_pips * point if request.action == "buy" else price + request.sl_pips * point
        
        if request.tp_pips:
            order["tp"] = price + request.tp_pips * point if request.action == "buy" else price - request.tp_pips * point
        
        # Send order
        result = mt5.order_send(order)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise HTTPException(status_code=400, detail=f"Trade failed: {result.comment}")
        
        # Record trade
        pnl = 0  # Will be updated when trade closes
        strategy_accounting.add_trade(pnl, request.symbol, request.volume)
        
        return JSONResponse(content={
            "status": "success",
            "order": result.order,
            "price": result.price,
            "volume": result.volume,
            "comment": result.comment
        })
        
    except Exception as e:
        logger.error(f"Trade execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auto_trade")
async def auto_trade(request: AutoTradeRequest):
    """Auto trade based on signals"""
    try:
        # Get market data
        df = get_market_data(request.symbol, request.timeframe, 100)
        
        # Process through orchestrator
        intent = await orchestrator.process_market_data(df)
        
        if intent and intent['confidence'] >= request.confidence_threshold:
            # Execute the trade
            execute_req = ExecuteRequest(
                symbol=request.symbol,
                action=intent['action'],
                volume=min(intent['volume'], request.max_volume),
                comment=f"auto_{intent['strategy']}"
            )
            
            result = await execute_trade(execute_req)
            
            return JSONResponse(content={
                "status": "traded",
                "signal": intent,
                "execution": result
            })
        else:
            return JSONResponse(content={
                "status": "no_trade",
                "message": f"No signal above confidence threshold {request.confidence_threshold}"
            })
            
    except Exception as e:
        logger.error(f"Auto trade error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/positions")
async def get_positions():
    """Get open positions"""
    if not mt5_connected:
        return JSONResponse(content={"positions": [], "total": 0})
    
    positions = mt5.positions_get()
    if positions is None:
        return JSONResponse(content={"positions": [], "total": 0})
    
    pos_list = []
    for pos in positions:
        pos_list.append({
            "ticket": pos.ticket,
            "symbol": pos.symbol,
            "type": "buy" if pos.type == 0 else "sell",
            "volume": pos.volume,
            "price": pos.price_open,
            "current_price": pos.price_current,
            "profit": pos.profit,
            "comment": pos.comment
        })
    
    return JSONResponse(content={
        "positions": pos_list,
        "total": len(pos_list),
        "total_profit": sum(p["profit"] for p in pos_list)
    })

@app.get("/pnl")
async def get_pnl():
    """Get P&L and performance metrics"""
    metrics = strategy_accounting.get_metrics()
    
    return JSONResponse(content={
        "metrics": metrics,
        "explanation": explain_performance(
            metrics['win_rate'],
            metrics['sharpe'],
            metrics['total_trades']
        )
    })

@app.post("/close_all")
async def close_all_positions():
    """Emergency close all positions"""
    if not mt5_connected:
        raise HTTPException(status_code=503, detail="MT5 not connected")
    
    positions = mt5.positions_get()
    if not positions:
        return JSONResponse(content={"message": "No open positions", "closed": 0})
    
    closed = 0
    errors = []
    
    for pos in positions:
        # Prepare close request
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": pos.ticket,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
            "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == 0 else mt5.symbol_info_tick(pos.symbol).ask,
            "deviation": 20,
            "magic": 234000,
            "comment": "emergency_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(close_request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            closed += 1
            # Record the P&L
            strategy_accounting.add_trade(pos.profit, pos.symbol, pos.volume)
        else:
            errors.append(f"Failed to close {pos.ticket}: {result.comment}")
    
    return JSONResponse(content={
        "message": "Emergency close completed",
        "closed": closed,
        "errors": errors
    })

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Receive and echo messages
            data = await websocket.receive_text()
            
            # Process commands
            msg = json.loads(data)
            
            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong", "timestamp": time.time()})
            
            elif msg.get("type") == "subscribe":
                await websocket.send_json({
                    "type": "subscribed",
                    "channel": msg.get("channel", "market_data")
                })
            
            elif msg.get("type") == "get_metrics":
                metrics = strategy_accounting.get_metrics()
                await websocket.send_json({
                    "type": "metrics",
                    "data": metrics
                })
                
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

# Run the application
if __name__ == "__main__":
    # Institutional-grade uvicorn configuration
    uvicorn.run(
        "main_cpu:app",
        host="0.0.0.0",
        port=8100,
        reload=False,
        log_level="info",
        workers=1,  # Single worker for institutional trading consistency
        access_log=True,
        use_colors=True,
        server_header=False,  # Security: hide server info
        date_header=False,    # Security: hide date header
        timeout_keep_alive=30,
        timeout_graceful_shutdown=10,
        limit_concurrency=1000,
        limit_max_requests=10000,
        backlog=2048
    )
