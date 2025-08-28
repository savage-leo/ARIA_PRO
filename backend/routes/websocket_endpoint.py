"""
WebSocket Endpoint for Real-time Data Streaming
High-performance WebSocket implementation for ARIA Pro
"""

from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.routing import APIRouter
import json
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from backend.core.websocket_pool import get_websocket_pool, MessageType
from backend.core.redis_cache import get_redis_cache
from backend.services.mt5_market_data import MT5MarketFeed
from backend.services.real_ai_signal_generator import RealAISignalGenerator

router = APIRouter()
logger = logging.getLogger(__name__)

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time data streaming"""
    pool = get_websocket_pool()
    connection_id = None
    
    try:
        # Accept WebSocket connection
        await websocket.accept()
        logger.info("🔗 New WebSocket client connected: %s", websocket.client)
        
        # Get client info
        client_ip = websocket.client.host if websocket.client else "unknown"
        user_agent = websocket.headers.get("user-agent", "")
        
        # Add to connection pool
        connection_id = await pool.add_connection(websocket, client_ip, user_agent)
        if not connection_id:
            await websocket.close(code=1013, reason="Connection limit reached")
            return
        
        logger.info(f"WebSocket connection established: {connection_id}")
        
        # Send welcome message
        welcome_message = {
            "type": MessageType.SYSTEM_STATUS.value,
            "data": {
                "connection_id": connection_id,
                "status": "connected",
                "timestamp": datetime.now().isoformat(),
                "available_topics": [
                    "market_data",
                    "ai_signals", 
                    "trade_updates",
                    "system_status",
                    "alerts",
                    "performance"
                ]
            }
        }
        await pool.send_to_connection(connection_id, welcome_message)
        
        # Message handling loop
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                await handle_client_message(connection_id, message, pool)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket client {connection_id} disconnected")
                break
            except json.JSONDecodeError:
                error_msg = {
                    "type": MessageType.ERROR.value,
                    "data": {"error": "Invalid JSON format"}
                }
                await pool.send_to_connection(connection_id, error_msg)
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                error_msg = {
                    "type": MessageType.ERROR.value,
                    "data": {"error": str(e)}
                }
                await pool.send_to_connection(connection_id, error_msg)
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        # Clean up connection
        if connection_id:
            await pool.remove_connection(connection_id)
            logger.info(f"WebSocket connection {connection_id} cleaned up")

async def handle_client_message(connection_id: str, message: Dict[str, Any], pool):
    """Handle incoming client messages"""
    try:
        msg_type = message.get("type")
        data = message.get("data", {})
        
        if msg_type == "subscribe":
            # Subscribe to topics
            topics = data.get("topics", [])
            for topic in topics:
                success = pool.subscribe(connection_id, topic)
                response = {
                    "type": "subscription_response",
                    "data": {
                        "topic": topic,
                        "status": "subscribed" if success else "failed"
                    }
                }
                await pool.send_to_connection(connection_id, response)
        
        elif msg_type == "unsubscribe":
            # Unsubscribe from topics
            topics = data.get("topics", [])
            for topic in topics:
                success = pool.unsubscribe(connection_id, topic)
                response = {
                    "type": "subscription_response",
                    "data": {
                        "topic": topic,
                        "status": "unsubscribed" if success else "failed"
                    }
                }
                await pool.send_to_connection(connection_id, response)
        
        elif msg_type == "get_market_data":
            # Request specific market data
            symbol = data.get("symbol", "EURUSD")
            await send_market_data(connection_id, symbol, pool)
        
        elif msg_type == "get_ai_signals":
            # Request AI signals
            symbol = data.get("symbol", "EURUSD")
            await send_ai_signals(connection_id, symbol, pool)
        
        elif msg_type == "ping":
            # Respond to ping
            pong_message = {
                "type": "pong",
                "data": {
                    "timestamp": datetime.now().isoformat(),
                    "connection_id": connection_id
                }
            }
            await pool.send_to_connection(connection_id, pong_message)
        
        else:
            # Unknown message type
            error_msg = {
                "type": MessageType.ERROR.value,
                "data": {"error": f"Unknown message type: {msg_type}"}
            }
            await pool.send_to_connection(connection_id, error_msg)
            
    except Exception as e:
        logger.error(f"Error handling client message: {e}")
        error_msg = {
            "type": MessageType.ERROR.value,
            "data": {"error": str(e)}
        }
        await pool.send_to_connection(connection_id, error_msg)

async def send_market_data(connection_id: str, symbol: str, pool):
    """Send market data to specific connection"""
    try:
        # Try cache first
        cache = get_redis_cache()
        cached_data = await cache.get_market_data(symbol)
        
        if cached_data:
            market_data = cached_data
        else:
            # Get live data from MT5MarketFeed using async generator
            from backend.services.mt5_market_data import get_mt5_market_feed_async
            try:
                feed = get_mt5_market_feed_async(symbol)
                market_data = await anext(feed, None)
            except Exception as e:
                logger.error(f"Error getting market data for {symbol}: {e}")
                market_data = None
            
            # Cache the data
            if market_data:
                await cache.set_market_data(symbol, market_data)
        
        if market_data:
            message = {
                "type": MessageType.MARKET_DATA.value,
                "data": {
                    "symbol": symbol,
                    "market_data": market_data,
                    "timestamp": datetime.now().isoformat()
                }
            }
            await pool.send_to_connection(connection_id, message)
        else:
            error_msg = {
                "type": MessageType.ERROR.value,
                "data": {"error": f"No market data available for {symbol}"}
            }
            await pool.send_to_connection(connection_id, error_msg)
            
    except Exception as e:
        logger.error(f"Error sending market data: {e}")
        error_msg = {
            "type": MessageType.ERROR.value,
            "data": {"error": str(e)}
        }
        await pool.send_to_connection(connection_id, error_msg)

async def send_ai_signals(connection_id: str, symbol: str, pool):
    """Send AI signals to specific connection"""
    try:
        # Try cache first
        cache = get_redis_cache()
        cached_signals = await cache.get_ai_signals(symbol)
        
        if cached_signals:
            ai_signals = cached_signals
        else:
            # Generate fresh signals
            signal_generator = RealAISignalGenerator()
            ai_signals = await signal_generator.generate_signals(symbol)
            
            # Cache the signals
            if ai_signals:
                await cache.set_ai_signals(symbol, ai_signals)
        
        if ai_signals:
            message = {
                "type": MessageType.AI_SIGNALS.value,
                "data": {
                    "symbol": symbol,
                    "signals": ai_signals,
                    "timestamp": datetime.now().isoformat()
                }
            }
            await pool.send_to_connection(connection_id, message)
        else:
            error_msg = {
                "type": MessageType.ERROR.value,
                "data": {"error": f"No AI signals available for {symbol}"}
            }
            await pool.send_to_connection(connection_id, error_msg)
            
    except Exception as e:
        logger.error(f"Error sending AI signals: {e}")
        error_msg = {
            "type": MessageType.ERROR.value,
            "data": {"error": str(e)}
        }
        await pool.send_to_connection(connection_id, error_msg)

# Background task for periodic data broadcasting
async def periodic_data_broadcast():
    """Broadcast market data and AI signals periodically"""
    pool = get_websocket_pool()
    
    while pool.running:
        try:
            # Get symbols from subscriptions
            symbols = set()
            for topic in pool.subscriptions.keys():
                if topic.startswith("market_data_"):
                    symbol = topic.replace("market_data_", "")
                    symbols.add(symbol)
                elif topic.startswith("ai_signals_"):
                    symbol = topic.replace("ai_signals_", "")
                    symbols.add(symbol)
            
            # Default symbols if no specific subscriptions
            if not symbols:
                symbols = {"EURUSD", "GBPUSD", "USDJPY", "XAUUSD"}
            
            # Broadcast market data
            for symbol in symbols:
                try:
                    # Get live data from MT5MarketFeed using async generator
                    from backend.services.mt5_market_data import get_mt5_market_feed_async
                    feed = get_mt5_market_feed_async(symbol)
                    market_data = await anext(feed, None)
                    
                    if market_data:
                        message = {
                            "type": MessageType.MARKET_DATA.value,
                            "data": {
                                "symbol": symbol,
                                "market_data": market_data,
                                "timestamp": datetime.now().isoformat()
                            }
                        }
                        await pool.broadcast_to_topic(f"market_data_{symbol}", message)
                        await pool.broadcast_to_topic("market_data", message)
                        
                except Exception as e:
                    logger.error(f"Error broadcasting market data for {symbol}: {e}")
            
            # Wait before next broadcast
            await asyncio.sleep(5)  # 5-second intervals
            
        except Exception as e:
            logger.error(f"Error in periodic data broadcast: {e}")
            await asyncio.sleep(10)

# Start periodic broadcasting when module is imported
_broadcast_task = None

async def start_periodic_broadcast():
    """Start periodic data broadcasting"""
    global _broadcast_task
    if _broadcast_task is None or _broadcast_task.done():
        _broadcast_task = asyncio.create_task(periodic_data_broadcast())
        logger.info("Periodic WebSocket data broadcast started")

async def stop_periodic_broadcast():
    """Stop periodic data broadcasting"""
    global _broadcast_task
    if _broadcast_task and not _broadcast_task.done():
        _broadcast_task.cancel()
        try:
            await _broadcast_task
        except asyncio.CancelledError:
            pass
        logger.info("Periodic WebSocket data broadcast stopped")
