"""
WebSocket Connection Pooling and Load Balancing
High-performance WebSocket management for ARIA Pro
"""

import asyncio
import logging
import json
import time
import weakref
from typing import Dict, Set, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
from collections import defaultdict, deque
import uuid

try:
    from fastapi import WebSocket, WebSocketDisconnect
    from starlette.websockets import WebSocketState
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    """WebSocket connection states"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"

class MessageType(Enum):
    """WebSocket message types"""
    MARKET_DATA = "market_data"
    AI_SIGNALS = "ai_signals"
    TRADE_UPDATES = "trade_updates"
    SYSTEM_STATUS = "system_status"
    ALERTS = "alerts"
    PERFORMANCE = "performance"
    HEARTBEAT = "heartbeat"
    ERROR = "error"

@dataclass
class ConnectionInfo:
    """WebSocket connection information"""
    id: str
    websocket: Any  # WebSocket instance
    client_ip: str
    user_agent: str
    connected_at: datetime
    last_activity: datetime
    subscriptions: Set[str] = field(default_factory=set)
    state: ConnectionState = ConnectionState.CONNECTING
    message_count: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    error_count: int = 0

@dataclass
class PoolConfig:
    """WebSocket pool configuration"""
    max_connections: int = 1000
    max_connections_per_ip: int = 10
    heartbeat_interval: int = 30  # seconds
    connection_timeout: int = 300  # seconds
    message_rate_limit: int = 100  # messages per minute
    max_message_size: int = 1024 * 1024  # 1MB
    enable_compression: bool = True
    enable_load_balancing: bool = True
    pool_size: int = 4  # Number of connection pools

class WebSocketPool:
    """High-performance WebSocket connection pool"""
    
    def __init__(self, config: PoolConfig = None):
        self.config = config or PoolConfig()
        self.connections: Dict[str, ConnectionInfo] = {}
        self.ip_connections: Dict[str, Set[str]] = defaultdict(set)
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)  # topic -> connection_ids
        self.message_queues: Dict[str, deque] = defaultdict(deque)
        self.rate_limits: Dict[str, List[float]] = defaultdict(list)  # connection_id -> timestamps
        
        # Load balancing
        self.pools: List[Dict[str, ConnectionInfo]] = [
            {} for _ in range(self.config.pool_size)
        ]
        self.current_pool = 0
        
        # Background tasks
        self.running = False
        self.heartbeat_task = None
        self.cleanup_task = None
        self.stats_task = None
        
        # Statistics
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "errors": 0,
            "disconnections": 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("WebSocket pool initialized with config: %s", self.config)
    
    async def start(self):
        """Start the WebSocket pool"""
        if self.running:
            return
        
        self.running = True
        
        # Start background tasks
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.stats_task = asyncio.create_task(self._stats_loop())
        
        logger.info("WebSocket pool started")
    
    async def stop(self):
        """Stop the WebSocket pool"""
        self.running = False
        
        # Cancel background tasks
        for task in [self.heartbeat_task, self.cleanup_task, self.stats_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close all connections
        await self._close_all_connections()
        
        logger.info("WebSocket pool stopped")
    
    def _get_next_pool(self) -> Dict[str, ConnectionInfo]:
        """Get next pool for load balancing"""
        if not self.config.enable_load_balancing:
            return self.connections
        
        pool = self.pools[self.current_pool]
        self.current_pool = (self.current_pool + 1) % self.config.pool_size
        return pool
    
    def _check_rate_limit(self, connection_id: str) -> bool:
        """Check if connection exceeds rate limit"""
        now = time.time()
        timestamps = self.rate_limits[connection_id]
        
        # Remove old timestamps
        cutoff = now - 60  # 1 minute window
        while timestamps and timestamps[0] < cutoff:
            timestamps.pop(0)
        
        # Check rate limit
        if len(timestamps) >= self.config.message_rate_limit:
            return False
        
        timestamps.append(now)
        return True
    
    async def add_connection(
        self, 
        websocket: Any, 
        client_ip: str, 
        user_agent: str = ""
    ) -> Optional[str]:
        """Add new WebSocket connection to pool"""
        with self._lock:
            # Check global connection limit
            if len(self.connections) >= self.config.max_connections:
                logger.warning("Connection limit reached, rejecting new connection")
                return None
            
            # Check per-IP connection limit
            if len(self.ip_connections[client_ip]) >= self.config.max_connections_per_ip:
                logger.warning(f"IP connection limit reached for {client_ip}")
                return None
            
            # Create connection info
            connection_id = str(uuid.uuid4())
            now = datetime.now()
            
            connection_info = ConnectionInfo(
                id=connection_id,
                websocket=websocket,
                client_ip=client_ip,
                user_agent=user_agent,
                connected_at=now,
                last_activity=now,
                state=ConnectionState.CONNECTED
            )
            
            # Add to appropriate pool
            if self.config.enable_load_balancing:
                pool = self._get_next_pool()
                pool[connection_id] = connection_info
            else:
                self.connections[connection_id] = connection_info
            
            # Update tracking
            self.ip_connections[client_ip].add(connection_id)
            
            # Update stats
            self.stats["total_connections"] += 1
            self.stats["active_connections"] += 1
            
            logger.info(f"Added WebSocket connection {connection_id} from {client_ip}")
            return connection_id
    
    async def remove_connection(self, connection_id: str):
        """Remove WebSocket connection from pool"""
        with self._lock:
            connection_info = self._get_connection(connection_id)
            if not connection_info:
                return
            
            # Remove from subscriptions
            for topic in list(connection_info.subscriptions):
                self.unsubscribe(connection_id, topic)
            
            # Remove from pools
            for pool in self.pools:
                if connection_id in pool:
                    del pool[connection_id]
                    break
            
            if connection_id in self.connections:
                del self.connections[connection_id]
            
            # Update IP tracking
            self.ip_connections[connection_info.client_ip].discard(connection_id)
            if not self.ip_connections[connection_info.client_ip]:
                del self.ip_connections[connection_info.client_ip]
            
            # Cleanup
            if connection_id in self.rate_limits:
                del self.rate_limits[connection_id]
            if connection_id in self.message_queues:
                del self.message_queues[connection_id]
            
            # Update stats
            self.stats["active_connections"] -= 1
            self.stats["disconnections"] += 1
            
            logger.info(f"Removed WebSocket connection {connection_id}")
    
    def _get_connection(self, connection_id: str) -> Optional[ConnectionInfo]:
        """Get connection info by ID"""
        # Check main connections dict first
        if connection_id in self.connections:
            return self.connections[connection_id]
        
        # Check load-balanced pools
        for pool in self.pools:
            if connection_id in pool:
                return pool[connection_id]
        
        return None
    
    def subscribe(self, connection_id: str, topic: str) -> bool:
        """Subscribe connection to topic"""
        with self._lock:
            connection_info = self._get_connection(connection_id)
            if not connection_info:
                return False
            
            connection_info.subscriptions.add(topic)
            self.subscriptions[topic].add(connection_id)
            
            logger.debug(f"Connection {connection_id} subscribed to {topic}")
            return True
    
    def unsubscribe(self, connection_id: str, topic: str) -> bool:
        """Unsubscribe connection from topic"""
        with self._lock:
            connection_info = self._get_connection(connection_id)
            if not connection_info:
                return False
            
            connection_info.subscriptions.discard(topic)
            self.subscriptions[topic].discard(connection_id)
            
            # Clean up empty topic
            if not self.subscriptions[topic]:
                del self.subscriptions[topic]
            
            logger.debug(f"Connection {connection_id} unsubscribed from {topic}")
            return True
    
    async def send_to_connection(
        self, 
        connection_id: str, 
        message: Dict[str, Any]
    ) -> bool:
        """Send message to specific connection"""
        connection_info = self._get_connection(connection_id)
        if not connection_info:
            return False
        
        # Check rate limit
        if not self._check_rate_limit(connection_id):
            logger.warning(f"Rate limit exceeded for connection {connection_id}")
            return False
        
        try:
            # Serialize message
            message_data = json.dumps(message)
            message_size = len(message_data.encode('utf-8'))
            
            # Check message size
            if message_size > self.config.max_message_size:
                logger.warning(f"Message too large for connection {connection_id}: {message_size} bytes")
                return False
            
            # Send message
            await connection_info.websocket.send_text(message_data)
            
            # Update stats
            connection_info.message_count += 1
            connection_info.bytes_sent += message_size
            connection_info.last_activity = datetime.now()
            
            self.stats["messages_sent"] += 1
            self.stats["bytes_sent"] += message_size
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to connection {connection_id}: {e}")
            connection_info.error_count += 1
            connection_info.state = ConnectionState.ERROR
            self.stats["errors"] += 1
            
            # Queue for removal
            asyncio.create_task(self.remove_connection(connection_id))
            return False
    
    async def broadcast_to_topic(
        self, 
        topic: str, 
        message: Dict[str, Any]
    ) -> int:
        """Broadcast message to all connections subscribed to topic"""
        if topic not in self.subscriptions:
            return 0
        
        connection_ids = list(self.subscriptions[topic])
        sent_count = 0
        
        # Send to all subscribers concurrently
        tasks = []
        for connection_id in connection_ids:
            task = asyncio.create_task(self.send_to_connection(connection_id, message))
            tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            sent_count = sum(1 for result in results if result is True)
        
        logger.debug(f"Broadcast to topic {topic}: {sent_count}/{len(connection_ids)} delivered")
        return sent_count
    
    async def broadcast_to_all(self, message: Dict[str, Any]) -> int:
        """Broadcast message to all active connections"""
        all_connections = []
        
        # Collect all connections from pools
        for pool in self.pools:
            all_connections.extend(pool.keys())
        all_connections.extend(self.connections.keys())
        
        # Remove duplicates
        all_connections = list(set(all_connections))
        
        sent_count = 0
        tasks = []
        
        for connection_id in all_connections:
            task = asyncio.create_task(self.send_to_connection(connection_id, message))
            tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            sent_count = sum(1 for result in results if result is True)
        
        logger.debug(f"Broadcast to all: {sent_count}/{len(all_connections)} delivered")
        return sent_count
    
    async def _heartbeat_loop(self):
        """Send heartbeat messages to maintain connections"""
        while self.running:
            try:
                heartbeat_message = {
                    "type": MessageType.HEARTBEAT.value,
                    "timestamp": datetime.now().isoformat(),
                    "server_time": time.time()
                }
                
                await self.broadcast_to_all(heartbeat_message)
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_loop(self):
        """Clean up stale connections"""
        while self.running:
            try:
                now = datetime.now()
                timeout_threshold = now - timedelta(seconds=self.config.connection_timeout)
                
                stale_connections = []
                
                # Check all pools for stale connections
                for pool in self.pools:
                    for connection_id, connection_info in pool.items():
                        if connection_info.last_activity < timeout_threshold:
                            stale_connections.append(connection_id)
                
                # Check main connections dict
                for connection_id, connection_info in self.connections.items():
                    if connection_info.last_activity < timeout_threshold:
                        stale_connections.append(connection_id)
                
                # Remove stale connections
                for connection_id in stale_connections:
                    await self.remove_connection(connection_id)
                
                if stale_connections:
                    logger.info(f"Cleaned up {len(stale_connections)} stale connections")
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(10)
    
    async def _stats_loop(self):
        """Log connection statistics"""
        while self.running:
            try:
                logger.info(f"WebSocket Pool Stats: {self.stats}")
                await asyncio.sleep(300)  # Log every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stats loop: {e}")
                await asyncio.sleep(30)
    
    async def _close_all_connections(self):
        """Close all active connections"""
        all_connections = []
        
        # Collect all connections
        for pool in self.pools:
            all_connections.extend(pool.values())
        all_connections.extend(self.connections.values())
        
        # Close connections
        for connection_info in all_connections:
            try:
                if hasattr(connection_info.websocket, 'close'):
                    await connection_info.websocket.close()
            except Exception as e:
                logger.error(f"Error closing connection {connection_info.id}: {e}")
        
        # Clear all data structures
        self.connections.clear()
        self.ip_connections.clear()
        self.subscriptions.clear()
        self.message_queues.clear()
        self.rate_limits.clear()
        
        for pool in self.pools:
            pool.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        with self._lock:
            active_by_pool = [len(pool) for pool in self.pools]
            
            return {
                **self.stats,
                "config": {
                    "max_connections": self.config.max_connections,
                    "max_connections_per_ip": self.config.max_connections_per_ip,
                    "pool_size": self.config.pool_size,
                    "load_balancing_enabled": self.config.enable_load_balancing
                },
                "pools": {
                    "active_by_pool": active_by_pool,
                    "total_pools": len(self.pools)
                },
                "subscriptions": {
                    "total_topics": len(self.subscriptions),
                    "topics": list(self.subscriptions.keys())
                },
                "connections_by_ip": {
                    ip: len(conn_ids) for ip, conn_ids in self.ip_connections.items()
                }
            }

# Global WebSocket pool instance
_websocket_pool: Optional[WebSocketPool] = None

def get_websocket_pool() -> WebSocketPool:
    """Get or create singleton WebSocket pool"""
    global _websocket_pool
    if _websocket_pool is None:
        from backend.core.config import get_settings
        settings = get_settings()
        
        config = PoolConfig(
            max_connections=getattr(settings, 'WS_MAX_CONNECTIONS', 1000),
            max_connections_per_ip=getattr(settings, 'WS_MAX_CONNECTIONS_PER_IP', 10),
            heartbeat_interval=getattr(settings, 'WS_HEARTBEAT_INTERVAL', 30),
            connection_timeout=getattr(settings, 'WS_CONNECTION_TIMEOUT', 300),
            message_rate_limit=getattr(settings, 'WS_MESSAGE_RATE_LIMIT', 100),
            enable_load_balancing=getattr(settings, 'WS_ENABLE_LOAD_BALANCING', True),
            pool_size=getattr(settings, 'WS_POOL_SIZE', 4)
        )
        _websocket_pool = WebSocketPool(config)
    return _websocket_pool
