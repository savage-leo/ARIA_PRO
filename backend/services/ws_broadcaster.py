import json
import logging
import asyncio
import os
from typing import Dict, Set, Any, Optional
from datetime import datetime
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketBroadcaster:
    def __init__(self):
        self.clients: Set[WebSocket] = set()
        # Per-client channel subscriptions. Default subscription is {"all"}
        self.subscriptions: Dict[WebSocket, Set[str]] = {}
        # Configurable maximum number of clients (default 100)
        self.max_clients: int = int(os.environ.get("ARIA_WS_MAX_CLIENTS", "100"))
        # Simple metrics
        self.sent_count: int = 0
        self.fail_count: int = 0
        self.drop_count: int = 0
        self._lock = asyncio.Lock()

    async def add_client(self, websocket: WebSocket):
        """Add a new WebSocket client"""
        try:
            async with self._lock:
                # Limit maximum connections to prevent resource exhaustion
                if len(self.clients) >= self.max_clients:
                    logger.warning(
                        f"Maximum WebSocket connections reached ({self.max_clients}). Rejecting new connection."
                    )
                    return False
                self.clients.add(websocket)
                total = len(self.clients)
            logger.info(f"WebSocket client added successfully. Total clients: {total}")
            return True
        except Exception as e:
            logger.error(f"Error adding WebSocket client: {e}")
            return False

    async def remove_client(self, websocket: WebSocket):
        """Remove a WebSocket client"""
        try:
            async with self._lock:
                self.clients.discard(websocket)
                total = len(self.clients)
            logger.info(f"WebSocket client removed. Total clients: {total}")
        except Exception as e:
            logger.error(f"Error removing WebSocket client: {e}")

    async def get_client_count(self) -> int:
        """Return the number of connected WebSocket clients (thread-safe)."""
        async with self._lock:
            return len(self.clients)

    async def broadcast(
        self,
        message_type: str,
        data: Dict[str, Any],
        channels: Optional[Set[str]] = None,
        timeout: float = 1.0,
    ):
        """Broadcast a message to connected clients.

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
            # Subscription filter
            try:
                subs = self.subscriptions.get(client, {"all"})
                if channels and ("all" not in subs) and subs.isdisjoint(channels):
                    continue
            except Exception:
                # On any unexpected error, fall back to sending
                pass
            try:
                await asyncio.wait_for(client.send_text(message), timeout=timeout)
                self.sent_count += 1
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
                    self.drop_count += 1

    async def subscribe(
        self, websocket: WebSocket, channels: Set[str], replace: bool = False
    ) -> Set[str]:
        """Subscribe a client to specific channels.

        If replace=True, overwrite existing subscriptions. Otherwise, union.
        Returns the effective subscription set.
        """
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
