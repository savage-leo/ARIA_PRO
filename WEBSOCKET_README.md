# WebSocket Real-Time Data Streaming

This document describes the WebSocket implementation for real-time data streaming in the ARIA PRO trading system.

## Overview

The WebSocket system provides low-latency, real-time data streaming from the backend to the frontend, enabling live updates for:

- 📊 Live tick data (bid/ask prices)
- 🎯 Trading signals from AI models
- 📋 Order status updates
- 💡 Trading ideas from SMC analysis
- 🚀 Prepared trading payloads

## Architecture

### Backend Components

1. **WebSocket Broadcaster** (`backend/services/ws_broadcaster.py`)
   - Manages WebSocket connections
   - Handles client connections/disconnections
   - Broadcasts messages to all connected clients
   - Provides convenience functions for different message types

2. **WebSocket Route** (`backend/routes/websocket.py`)
   - FastAPI WebSocket endpoint at `/ws`
   - Integrates with the broadcaster service
   - Handles client authentication and message routing

3. **Broadcasting Functions**
   - `broadcast_tick()` - Live price updates
   - `broadcast_signal()` - AI trading signals
   - `broadcast_order_update()` - Order status changes
   - `broadcast_idea()` - SMC trading ideas
   - `broadcast_prepared_payload()` - Ready-to-execute trades

### Frontend Components

1. **WebSocket Hook** (`frontend/src/hooks/useWebSocket.ts`)
   - React hook for WebSocket connection management
   - Auto-reconnection with exponential backoff
   - Message history and state management
   - Ping/pong heartbeat mechanism

2. **Real-Time Data Component** (`frontend/src/components/interfaces/RealTimeData.tsx`)
   - Displays live data streams
   - Organized into sections for different data types
   - Real-time updates with smooth animations

3. **Enhanced Trading Interface** (`frontend/src/components/interfaces/TradingInterface.tsx`)
   - Integrated with WebSocket for live price updates
   - Real-time position size calculations
   - Connection status indicators

## Message Format

All WebSocket messages follow this structure:

```json
{
  "type": "message_type",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "data": {
    // Message-specific data
  }
}
```

### Message Types

#### Tick Data
```json
{
  "type": "tick",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "data": {
    "symbol": "EURUSD",
    "bid": 1.08500,
    "ask": 1.08503,
    "spread": 0.00003
  }
}
```

#### Trading Signal
```json
{
  "type": "signal",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "data": {
    "symbol": "EURUSD",
    "side": "buy",
    "strength": 0.85,
    "confidence": 87.5,
    "model": "LSTM"
  }
}
```

#### Order Update
```json
{
  "type": "order_update",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "data": {
    "order_id": "ORD_1704110400_1234",
    "symbol": "EURUSD",
    "side": "buy",
    "status": "filled",
    "volume": 0.1,
    "price": 1.08500
  }
}
```

## Setup and Usage

### 1. Install Dependencies

```bash
# Backend
cd backend
pip install websockets

# Frontend
cd frontend
npm install
```

### 2. Start Development Environment

**Linux/Mac:**
```bash
chmod +x scripts/start_dev.sh
./scripts/start_dev.sh
```

**Windows:**
```cmd
scripts\start_dev.bat
```

### 3. Test WebSocket Data Streaming

```bash
python scripts/test_websocket.py
```

This will start a test broadcaster that simulates real-time data.

### 4. Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **WebSocket**: ws://localhost:8000/ws

## Integration with Existing Systems

### Broadcasting from Backend Services

To broadcast data from any backend service:

```python
from services.ws_broadcaster import broadcast_tick, broadcast_signal

# Broadcast tick data
await broadcast_tick("EURUSD", 1.08500, 1.08503)

# Broadcast trading signal
await broadcast_signal({
    "symbol": "EURUSD",
    "side": "buy",
    "strength": 0.85,
    "confidence": 87.5
})
```

### Using WebSocket Hook in React Components

```typescript
import { useWebSocket } from '../hooks/useWebSocket';

const MyComponent = () => {
  const { isConnected, lastMessage, messageHistory } = useWebSocket({
    onMessage: (message) => {
      console.log('Received:', message);
    }
  });

  return (
    <div>
      <div>Status: {isConnected ? 'Connected' : 'Disconnected'}</div>
      <div>Last Message: {JSON.stringify(lastMessage)}</div>
    </div>
  );
};
```

## Performance Considerations

- **Low Latency**: WebSocket connections provide minimal overhead
- **Auto-Reconnection**: Frontend automatically reconnects on connection loss
- **Message History**: Limited to prevent memory issues (configurable)
- **Heartbeat**: 30-second ping/pong to detect connection issues
- **Error Handling**: Graceful degradation on connection failures

## Security

- WebSocket connections are subject to CORS policies
- Consider implementing authentication for production use
- Rate limiting may be added for production environments

## Monitoring

The system provides logging for:
- Client connections/disconnections
- Message broadcasting
- Error conditions
- Performance metrics

## Future Enhancements

1. **Authentication**: JWT-based WebSocket authentication
2. **Channels**: Subscribe to specific data channels
3. **Compression**: Message compression for high-frequency data
4. **Clustering**: Multi-instance WebSocket support
5. **Metrics**: Prometheus metrics for monitoring

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Ensure backend is running on port 8000
   - Check firewall settings

2. **No Data Received**
   - Verify WebSocket URL in frontend
   - Check browser console for errors
   - Ensure test broadcaster is running

3. **High Latency**
   - Check network connectivity
   - Monitor server resources
   - Consider message batching for high-frequency data

### Debug Mode

Enable debug logging by setting the log level:

```python
import logging
logging.getLogger('websockets').setLevel(logging.DEBUG)
```
