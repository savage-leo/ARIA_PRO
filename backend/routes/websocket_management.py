"""
WebSocket Management API Endpoints
Administrative control over WebSocket connections and pools
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import logging
from backend.core.websocket_pool import get_websocket_pool, MessageType

router = APIRouter(prefix="/websocket", tags=["WebSocket Management"])
logger = logging.getLogger(__name__)

@router.get("/stats")
async def get_websocket_stats() -> Dict[str, Any]:
    """Get WebSocket pool statistics"""
    try:
        pool = get_websocket_pool()
        stats = pool.get_stats()
        return {"ok": True, "stats": stats}
    except Exception as e:
        logger.error(f"Failed to get WebSocket stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/connections")
async def get_active_connections() -> Dict[str, Any]:
    """Get list of active WebSocket connections"""
    try:
        pool = get_websocket_pool()
        
        # Collect connection info (without sensitive data)
        connections = []
        for pool_dict in pool.pools:
            for conn_id, conn_info in pool_dict.items():
                connections.append({
                    "id": conn_id,
                    "client_ip": conn_info.client_ip,
                    "connected_at": conn_info.connected_at.isoformat(),
                    "last_activity": conn_info.last_activity.isoformat(),
                    "state": conn_info.state.value,
                    "subscriptions": list(conn_info.subscriptions),
                    "message_count": conn_info.message_count,
                    "bytes_sent": conn_info.bytes_sent,
                    "error_count": conn_info.error_count
                })
        
        # Add main connections dict
        for conn_id, conn_info in pool.connections.items():
            connections.append({
                "id": conn_id,
                "client_ip": conn_info.client_ip,
                "connected_at": conn_info.connected_at.isoformat(),
                "last_activity": conn_info.last_activity.isoformat(),
                "state": conn_info.state.value,
                "subscriptions": list(conn_info.subscriptions),
                "message_count": conn_info.message_count,
                "bytes_sent": conn_info.bytes_sent,
                "error_count": conn_info.error_count
            })
        
        return {
            "ok": True,
            "connections": connections,
            "total_count": len(connections)
        }
    except Exception as e:
        logger.error(f"Failed to get active connections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/subscriptions")
async def get_subscriptions() -> Dict[str, Any]:
    """Get all topic subscriptions"""
    try:
        pool = get_websocket_pool()
        
        subscriptions = {}
        for topic, connection_ids in pool.subscriptions.items():
            subscriptions[topic] = {
                "subscriber_count": len(connection_ids),
                "connection_ids": list(connection_ids)
            }
        
        return {
            "ok": True,
            "subscriptions": subscriptions,
            "total_topics": len(subscriptions)
        }
    except Exception as e:
        logger.error(f"Failed to get subscriptions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/broadcast")
async def broadcast_message(
    topic: str = None,
    message_type: str = "system_status",
    data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Broadcast message to WebSocket connections"""
    try:
        pool = get_websocket_pool()
        
        message = {
            "type": message_type,
            "data": data or {},
            "timestamp": pool.stats.get("timestamp", "")
        }
        
        if topic:
            # Broadcast to specific topic
            sent_count = await pool.broadcast_to_topic(topic, message)
        else:
            # Broadcast to all connections
            sent_count = await pool.broadcast_to_all(message)
        
        return {
            "ok": True,
            "message": "Broadcast sent",
            "sent_count": sent_count,
            "topic": topic
        }
    except Exception as e:
        logger.error(f"Failed to broadcast message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/connection/{connection_id}")
async def disconnect_connection(connection_id: str) -> Dict[str, Any]:
    """Forcefully disconnect a WebSocket connection"""
    try:
        pool = get_websocket_pool()
        await pool.remove_connection(connection_id)
        
        return {
            "ok": True,
            "message": f"Connection {connection_id} disconnected",
            "connection_id": connection_id
        }
    except Exception as e:
        logger.error(f"Failed to disconnect connection {connection_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start")
async def start_websocket_pool() -> Dict[str, Any]:
    """Start the WebSocket pool"""
    try:
        pool = get_websocket_pool()
        await pool.start()
        
        return {
            "ok": True,
            "message": "WebSocket pool started",
            "status": "running"
        }
    except Exception as e:
        logger.error(f"Failed to start WebSocket pool: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_websocket_pool() -> Dict[str, Any]:
    """Stop the WebSocket pool"""
    try:
        pool = get_websocket_pool()
        await pool.stop()
        
        return {
            "ok": True,
            "message": "WebSocket pool stopped",
            "status": "stopped"
        }
    except Exception as e:
        logger.error(f"Failed to stop WebSocket pool: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def websocket_health() -> Dict[str, Any]:
    """Check WebSocket pool health"""
    try:
        pool = get_websocket_pool()
        stats = pool.get_stats()
        
        # Determine health status
        active_connections = stats.get("active_connections", 0)
        max_connections = stats.get("config", {}).get("max_connections", 1000)
        error_rate = stats.get("errors", 0) / max(stats.get("messages_sent", 1), 1)
        
        if error_rate > 0.1:  # More than 10% error rate
            status = "unhealthy"
        elif active_connections > max_connections * 0.9:  # More than 90% capacity
            status = "degraded"
        else:
            status = "healthy"
        
        return {
            "ok": True,
            "status": status,
            "pool_running": pool.running,
            "active_connections": active_connections,
            "max_connections": max_connections,
            "error_rate": round(error_rate * 100, 2),
            "capacity_usage": round((active_connections / max_connections) * 100, 2)
        }
    except Exception as e:
        logger.error(f"WebSocket health check failed: {e}")
        return {
            "ok": False,
            "status": "unhealthy",
            "error": str(e)
        }
