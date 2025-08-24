from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from backend.services.ws_broadcaster import broadcaster
from backend.core.config import get_settings
from backend.core.auth import verify_refresh_token
import json
import logging
import os
import asyncio
from typing import Set, Optional
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    logger.info(
        f"WebSocket connection attempt from {websocket.client.host}:{websocket.client.port}"
    )
    # Extra debug: log Origin and User-Agent to diagnose browser CORS/handshake issues
    try:
        origin = websocket.headers.get("origin")
        ua = websocket.headers.get("user-agent")
        logger.info(f"WebSocket headers: origin={origin} ua={ua}")
    except Exception:
        # Do not fail the connection on header access issues
        pass

    # REQUIRED token authentication for production security
    settings = get_settings()
    
    # Get token from query params or headers
    provided_token = (
        websocket.query_params.get("token")
        or websocket.headers.get("Authorization", "").replace("Bearer ", "")
        or websocket.headers.get("X-ARIA-TOKEN")
    )
    
    # Validate token - FAIL CLOSED for security
    if not provided_token:
        logger.warning(f"WebSocket connection rejected: no token provided from {websocket.client.host}")
        await websocket.close(code=1008, reason="Authentication required")
        return
    
    # Check against WebSocket token or admin API key
    valid_token = False
    if provided_token == settings.ARIA_WS_TOKEN:
        valid_token = True
        logger.info(f"WebSocket authenticated with WS token from {websocket.client.host}")
    elif provided_token == settings.ADMIN_API_KEY:
        valid_token = True
        logger.info(f"WebSocket authenticated with admin key from {websocket.client.host}")
    else:
        # Try JWT token validation
        try:
            jwt_payload = verify_refresh_token(provided_token)
            if jwt_payload:
                valid_token = True
                logger.info(f"WebSocket authenticated with JWT from {websocket.client.host}")
        except Exception:
            pass
    
    if not valid_token:
        logger.warning(f"WebSocket connection rejected: invalid token from {websocket.client.host}")
        await websocket.close(code=1008, reason="Invalid authentication token")
        return

    await websocket.accept()
    client_id = id(websocket)
    logger.info(f"✅ Secure WebSocket client {client_id} connected")

    hb_task: Optional[asyncio.Task] = None
    try:
        # Add to broadcaster clients
        added = await broadcaster.add_client(websocket)
        if not added:
            await websocket.close(code=1013, reason="Too many connections")
            return

        # Initial subscriptions via query param channels="chan1,chan2"
        try:
            initial_channels_param = websocket.query_params.get("channels")
            if initial_channels_param:
                chans: Set[str] = {
                    c.strip() for c in initial_channels_param.split(",") if c.strip()
                }
                effective = await broadcaster.subscribe(websocket, chans, replace=True)
            else:
                effective = await broadcaster.subscribe(websocket, set(), replace=False)
        except Exception:
            effective = {"all"}

        # Heartbeat configuration
        hb_interval = int(os.environ.get("ARIA_WS_HEARTBEAT_SEC", "30"))

        # Send welcome message
        await websocket.send_text(
            json.dumps(
                {
                    "type": "connection",
                    "message": "Connected to ARIA trading system",
                    "client_id": client_id,
                    "subscriptions": sorted(list(effective)),
                }
            )
        )

        # Background heartbeat pings
        async def _heartbeat_loop():
            while True:
                try:
                    await asyncio.sleep(max(5, hb_interval))
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "ping",
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                            }
                        )
                    )
                except Exception:
                    # Any failure will bubble to main loop via cancellation on finally
                    break

        if hb_interval > 0:
            hb_task = asyncio.create_task(_heartbeat_loop())

        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get("type") == "ping":
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "pong",
                                "timestamp": datetime.utcnow().isoformat() + "Z",
                            }
                        )
                    )
                elif message.get("type") == "subscribe":
                    chans = message.get("channels", [])
                    if not isinstance(chans, list):
                        chans = [str(chans)]
                    effective = await broadcaster.subscribe(
                        websocket, {str(c) for c in chans}, replace=False
                    )
                    # Backward-compatible event name
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "subscription_confirmed",
                                "channels": sorted(list(effective)),
                            }
                        )
                    )
                elif message.get("type") == "unsubscribe":
                    chans = set(map(str, message.get("channels", [])))
                    try:
                        subs = await broadcaster.unsubscribe(websocket, chans)
                    except Exception as e:
                        logger.warning(f"Unsubscribe error for client {client_id}: {e}")
                        subs = chans
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "subscription_updated",
                                "channels": sorted(list(subs)),
                            }
                        )
                    )
                elif message.get("type") == "set_subscriptions":
                    chans = message.get("channels", [])
                    if not isinstance(chans, list):
                        chans = [str(chans)]
                    effective = await broadcaster.subscribe(
                        websocket, {str(c) for c in chans}, replace=True
                    )
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "subscription_updated",
                                "channels": sorted(list(effective)),
                            }
                        )
                    )
                elif message.get("type") == "whoami":
                    subs = sorted(
                        list(broadcaster.subscriptions.get(websocket, {"all"}))
                    )
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "identity",
                                "client_id": client_id,
                                "subscriptions": subs,
                                "server_time": datetime.utcnow().isoformat() + "Z",
                            }
                        )
                    )
                elif message.get("type") == "echo":
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "echo",
                                "payload": message.get("payload"),
                                "server_time": datetime.utcnow().isoformat() + "Z",
                            }
                        )
                    )

            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from client {client_id}")
            except WebSocketDisconnect:
                logger.info(
                    f"WebSocket client {client_id} disconnected during message handling"
                )
                break
            except Exception as e:
                logger.error(f"Error handling client {client_id} message: {e}")
                # Don't break on general exceptions, just log them

    except WebSocketDisconnect:
        logger.info(f"WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket client {client_id} error: {e}")
    finally:
        # Stop heartbeat if running
        if hb_task:
            hb_task.cancel()
            try:
                await hb_task
            except Exception:
                pass
        # Remove from broadcaster
        try:
            await broadcaster.remove_client(websocket)
            logger.info(f"WebSocket client {client_id} removed from broadcast list")
        except Exception as e:
            logger.error(f"Error removing client {client_id} from broadcaster: {e}")
