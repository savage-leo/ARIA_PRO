import json
import logging
import asyncio
import os
import time
import hmac
import hashlib
from typing import Dict, Set, Any, Optional, Tuple
from datetime import datetime
from collections import deque
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketBroadcaster:
    def __init__(self):
        self.clients: Set[WebSocket] = set()
        # Per-client channel subscriptions. Default subscription is {"all"}
        self.subscriptions: Dict[WebSocket, Set[str]] = {}
        # Client authentication state
        self.authenticated_clients: Dict[WebSocket, Dict[str, Any]] = {}
        # Configurable maximum number of clients (default 100)
        self.max_clients: int = int(os.environ.get("ARIA_WS_MAX_CLIENTS", "100"))
        # Message queue per client with bounded size
        self.client_queues: Dict[WebSocket, deque] = {}
        self.max_queue_size: int = int(os.environ.get("ARIA_WS_MAX_QUEUE", "1000"))
        # Simple metrics
        self.sent_count: int = 0
        self.fail_count: int = 0
        self.drop_count: int = 0
        self.auth_fail_count: int = 0
        self._lock = asyncio.Lock()
        # Authentication secret from environment
        self.auth_secret = os.environ.get("ARIA_WS_AUTH_SECRET", "")
        self.auth_required = os.environ.get("ARIA_WS_AUTH_REQUIRED", "1") == "1"

    async def add_client(self, websocket: WebSocket, auth_token: Optional[str] = None) -> Tuple[bool, str]:
        """Add a new WebSocket client with optional authentication"""
        try:
            # Verify authentication if required
            if self.auth_required:
                if not auth_token:
                    self.auth_fail_count += 1
                    return False, "Authentication required"
                
                if not self._verify_auth_token(auth_token):
                    self.auth_fail_count += 1
                    logger.warning(f"WebSocket authentication failed from {websocket.client}")
                    return False, "Invalid authentication token"
            
            async with self._lock:
                # Limit maximum connections to prevent resource exhaustion
                if len(self.clients) >= self.max_clients:
                    logger.warning(
                        f"Maximum WebSocket connections reached ({self.max_clients}). Rejecting new connection."
                    )
                    return False, "Maximum connections reached"
                
                self.clients.add(websocket)
                # Initialize bounded message queue for this client
                self.client_queues[websocket] = deque(maxlen=self.max_queue_size)
                
                if auth_token:
                    self.authenticated_clients[websocket] = {
                        "authenticated": True,
                        "timestamp": time.time(),
                        "token_hash": hashlib.sha256(auth_token.encode()).hexdigest()
                    }
                
                total = len(self.clients)
            
            logger.info(f"WebSocket client added successfully. Total clients: {total}")
            return True, "Connected"
        except Exception as e:
            logger.error(f"Error adding WebSocket client: {e}")
            return False, f"Connection error: {e}"

    async def remove_client(self, websocket: WebSocket):
        """Remove a WebSocket client"""
        try:
            async with self._lock:
                self.clients.discard(websocket)
                self.subscriptions.pop(websocket, None)
                self.authenticated_clients.pop(websocket, None)
                self.client_queues.pop(websocket, None)
                total = len(self.clients)
            logger.info(f"WebSocket client removed. Total clients: {total}")
        except Exception as e:
            logger.error(f"Error removing WebSocket client: {e}")

    async def get_client_count(self) -> int:
        """Return the number of connected WebSocket clients (thread-safe)."""
        async with self._lock:
            return len(self.clients)

    def _verify_auth_token(self, token: str) -> bool:
        """Verify authentication token using HMAC"""
        if not self.auth_secret:
            # If no secret configured, accept any non-empty token in dev mode
            return bool(token)
        
        try:
            # Token format: timestamp:signature
            parts = token.split(":")
            if len(parts) != 2:
                return False
            
            timestamp, signature = parts
            
            # Check token age (max 1 hour)
            token_age = time.time() - float(timestamp)
            if token_age > 3600:
                return False
            
            # Verify HMAC signature
            expected_sig = hmac.new(
                self.auth_secret.encode(),
                timestamp.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_sig)
        except Exception as e:
            logger.debug(f"Auth token verification failed: {e}")
            return False
    
    async def broadcast(
        self,
        message_type: str,
        data: Dict[str, Any],
        channels: Optional[Set[str]] = None,
        timeout: float = 1.0,
        require_auth: bool = False,
    ):
        """Broadcast a message to connected clients with bounded queuing.

        If `channels` is provided, only clients whose subscriptions intersect with
        `channels` (or that have the implicit "all") will receive the message.
        """
        async with self._lock:
            if not self.clients:
                return
            clients_snapshot = list(self.clients)

        payload = {
            "type": message_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data": data,
        }

        message = json.dumps(payload)
        disconnected_clients = set()

        for client in clients_snapshot:
            # Check authentication if required
            if require_auth and client not in self.authenticated_clients:
                continue
            
            # Subscription filter
            try:
                subs = self.subscriptions.get(client, {"all"})
                if channels and ("all" not in subs) and subs.isdisjoint(channels):
                    continue
            except Exception:
                # On any unexpected error, fall back to sending
                pass
            
            # Add to client's queue (bounded)
            if client in self.client_queues:
                self.client_queues[client].append(message)
            
            # Try to send immediately
            try:
                await asyncio.wait_for(client.send_text(message), timeout=timeout)
                self.sent_count += 1
            except asyncio.TimeoutError:
                logger.debug(f"WebSocket send timeout for client")
                self.fail_count += 1
            except Exception as e:
                logger.warning(f"Failed to send to client: {e}")
                disconnected_clients.add(client)
                self.fail_count += 1

        # Remove disconnected clients
        if disconnected_clients:
            async with self._lock:
                for client in disconnected_clients:
                    self.clients.discard(client)
                    self.subscriptions.pop(client, None)
                    self.authenticated_clients.pop(client, None)
                    self.client_queues.pop(client, None)
                    self.drop_count += 1

    async def subscribe(
        self, websocket: WebSocket, channels: Set[str], replace: bool = False
    ) -> Set[str]:
        """Subscribe a client to specific channels.

        If replace=True, overwrite existing subscriptions. Otherwise, union.
        Returns the effective subscription set.
        """
        async with self._lock:
            current = self.subscriptions.get(websocket, {"all"})
            if replace:
                new_set = set(channels) if channels else set()
            else:
                new_set = set(current)
                new_set.update(channels or set())
            if not new_set:
                # Never allow empty set to avoid surprising silences; default to {"all"}
                new_set = {"all"}
            self.subscriptions[websocket] = new_set
            return new_set

    async def unsubscribe(self, websocket: WebSocket, channels: Set[str]) -> Set[str]:
        """Unsubscribe a client from channels. Returns the effective set."""
        async with self._lock:
            current = self.subscriptions.get(websocket, {"all"})
            # If currently "all", and unsubscribe comes, switch to empty then remove listed
            if "all" in current:
                current = set()
            current.difference_update(channels or set())
            if not current:
                current = {"all"}
            self.subscriptions[websocket] = current
            return current

    async def broadcast_tick(self, symbol: str, bid: float, ask: float):
        data = {"symbol": symbol, "bid": bid, "ask": ask, "spread": ask - bid}
        chans: Set[str] = {"tick", f"symbol:{symbol}", f"tick:{symbol}"}
        await self.broadcast("tick", data, channels=chans)

    async def broadcast_bar(self, bar_data: Dict[str, Any]):
        symbol = str(bar_data.get("symbol") or bar_data.get("pair") or "").upper()
        chans: Set[str] = {"bar"}
        if symbol:
            chans.update({f"symbol:{symbol}", f"bar:{symbol}"})
        await self.broadcast("bar", bar_data, channels=chans)

    async def broadcast_signal(self, signal_data: Dict[str, Any]):
        symbol = str(signal_data.get("symbol") or signal_data.get("pair") or "").upper()
        chans: Set[str] = {"signal"}
        if symbol:
            chans.update({f"symbol:{symbol}", f"signal:{symbol}"})
        await self.broadcast("signal", signal_data, channels=chans)

    async def broadcast_order_update(self, order_data: Dict[str, Any]):
        symbol = str(order_data.get("symbol") or order_data.get("pair") or "").upper()
        chans: Set[str] = {"order_update"}
        if symbol:
            chans.update({f"symbol:{symbol}", f"order:{symbol}"})
        await self.broadcast("order_update", order_data, channels=chans)

    async def broadcast_idea(self, idea_data: Dict[str, Any]):
        symbol = str(idea_data.get("symbol") or idea_data.get("pair") or "").upper()
        chans: Set[str] = {"idea"}
        if symbol:
            chans.update({f"symbol:{symbol}", f"idea:{symbol}"})
        await self.broadcast("idea", idea_data, channels=chans)

    async def broadcast_prepared_payload(self, payload_data: Dict[str, Any]):
        symbol = str(
            payload_data.get("symbol") or payload_data.get("pair") or ""
        ).upper()
        chans: Set[str] = {"prepared_payload"}
        if symbol:
            chans.update({f"symbol:{symbol}", f"prepared_payload:{symbol}"})
        await self.broadcast("prepared_payload", payload_data, channels=chans)
    
    async def flush_client_queue(self, websocket: WebSocket, max_messages: int = 100):
        """Flush pending messages from a client's queue"""
        if websocket not in self.client_queues:
            return 0
        
        sent = 0
        queue = self.client_queues[websocket]
        
        while queue and sent < max_messages:
            try:
                message = queue.popleft()
                await asyncio.wait_for(websocket.send_text(message), timeout=1.0)
                sent += 1
            except Exception as e:
                logger.debug(f"Error flushing client queue: {e}")
                break
        
        return sent
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get broadcaster metrics"""
        return {
            "clients_connected": len(self.clients),
            "authenticated_clients": len(self.authenticated_clients),
            "messages_sent": self.sent_count,
            "messages_failed": self.fail_count,
            "clients_dropped": self.drop_count,
            "auth_failures": self.auth_fail_count,
            "max_clients": self.max_clients,
            "max_queue_size": self.max_queue_size,
        }


# Global broadcaster instance
broadcaster = WebSocketBroadcaster()


# Convenience functions for easy importing
async def broadcast_tick(symbol: str, bid: float, ask: float):
    """Broadcast tick data to all connected clients"""
    await broadcaster.broadcast_tick(symbol, bid, ask)


async def broadcast_bar(bar_data: Dict[str, Any]):
    """Broadcast bar data to all connected clients"""
    await broadcaster.broadcast_bar(bar_data)


async def broadcast_signal(signal_data: Dict[str, Any]):
    """Broadcast trading signal to all connected clients"""
    await broadcaster.broadcast_signal(signal_data)


async def broadcast_order_update(order_data: Dict[str, Any]):
    """Broadcast order update to all connected clients"""
    await broadcaster.broadcast_order_update(order_data)


async def broadcast_idea(idea_data: Dict[str, Any]):
    """Broadcast trading idea to all connected clients"""
    await broadcaster.broadcast_idea(idea_data)


async def broadcast_prepared_payload(payload_data: Dict[str, Any]):
    """Broadcast prepared trading payload to all connected clients"""
    await broadcaster.broadcast_prepared_payload(payload_data)
