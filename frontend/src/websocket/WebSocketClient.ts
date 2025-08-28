import { useState, useEffect, useRef, useCallback } from 'react';

export interface WebSocketMessage {
  type: string;
  data?: any;
  timestamp?: string;
  [key: string]: any;
}

export interface WebSocketState {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
}

export interface WebSocketClientOptions {
  url?: string;
  autoConnect?: boolean;
  reconnectInterval?: number;
  heartbeatInterval?: number;
}

class WebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectInterval: number;
  private heartbeatInterval: number;
  private reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
  private heartbeatIntervalId: ReturnType<typeof setInterval> | null = null;
  private isManuallyClosed: boolean = false;
  private listeners: Map<string, Set<(data: any) => void>> = new Map();
  private stateListeners: Set<(state: WebSocketState) => void> = new Set();

  constructor(options: WebSocketClientOptions = {}) {
    const {
      url = 'ws://localhost:8100/ws',
      autoConnect = true,
      reconnectInterval = 5000,
      heartbeatInterval = 30000
    } = options;
    
    this.url = url;
    this.reconnectInterval = reconnectInterval;
    this.heartbeatInterval = heartbeatInterval;

    if (autoConnect) {
      this.connect();
    }
  }

  public connect() {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this.isManuallyClosed = false;
    this.updateState({ isConnecting: true, isConnected: false, error: null });

    try {
      this.ws = new WebSocket(this.url);
      
      this.ws.onopen = () => {
        console.log('[WebSocket] Connected to', this.url);
        this.updateState({ isConnecting: false, isConnected: true, error: null });
        this.startHeartbeat();
      };

      this.ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          this.notifyListeners(message.type, message);
        } catch (error) {
          console.error('[WebSocket] Failed to parse message:', error);
        }
      };

      this.ws.onclose = (event) => {
        console.log('[WebSocket] Connection closed with code:', event.code);
        this.updateState({ isConnecting: false, isConnected: false, error: null });
        
        if (this.heartbeatIntervalId) {
          clearInterval(this.heartbeatIntervalId);
          this.heartbeatIntervalId = null;
        }

        // Reconnect if not manually closed
        if (!this.isManuallyClosed && event.code !== 1000) {
          this.scheduleReconnect();
        }
      };

      this.ws.onerror = (error) => {
        console.error('[WebSocket] Connection error:', error);
        this.updateState({ isConnecting: false, isConnected: false, error: 'Connection failed' });
      };

    } catch (error) {
      console.error('[WebSocket] Failed to create WebSocket connection:', error);
      this.updateState({ isConnecting: false, isConnected: false, error: 'Failed to connect' });
    }
  }

  public disconnect() {
    this.isManuallyClosed = true;
    
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    
    if (this.heartbeatIntervalId) {
      clearInterval(this.heartbeatIntervalId);
      this.heartbeatIntervalId = null;
    }
    
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    
    this.updateState({ isConnecting: false, isConnected: false, error: null });
  }

  public send(type: string, data?: any) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('[WebSocket] Attempted to send message while not connected');
      return false;
    }

    try {
      const message: WebSocketMessage = { type, data };
      this.ws.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.error('[WebSocket] Failed to send message:', error);
      return false;
    }
  }

  public subscribe(eventType: string, callback: (data: any) => void) {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, new Set());
    }
    
    this.listeners.get(eventType)!.add(callback);
    
    // Return unsubscribe function
    return () => {
      const listeners = this.listeners.get(eventType);
      if (listeners) {
        listeners.delete(callback);
        if (listeners.size === 0) {
          this.listeners.delete(eventType);
        }
      }
    };
  }

  public addStateListener(callback: (state: WebSocketState) => void) {
    this.stateListeners.add(callback);
    return () => {
      this.stateListeners.delete(callback);
    };
  }

  private notifyListeners(eventType: string, data: any) {
    const listeners = this.listeners.get(eventType);
    if (listeners) {
      listeners.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error('[WebSocket] Error in listener callback:', error);
        }
      });
    }
  }

  private updateState(state: WebSocketState) {
    this.stateListeners.forEach(callback => {
      try {
        callback(state);
      } catch (error) {
        console.error('[WebSocket] Error in state listener callback:', error);
      }
    });
  }

  private scheduleReconnect() {
    if (this.reconnectTimeout) return;
    
    console.log('[WebSocket] Scheduling reconnect in', this.reconnectInterval, 'ms');
    this.reconnectTimeout = setTimeout(() => {
      this.reconnectTimeout = null;
      this.connect();
    }, this.reconnectInterval);
  }

  private startHeartbeat() {
    if (this.heartbeatIntervalId) {
      clearInterval(this.heartbeatIntervalId);
    }
    
    this.heartbeatIntervalId = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.send('ping');
      }
    }, this.heartbeatInterval);
  }
}

// React hook wrapper for the WebSocket client
export const useWebSocketClient = (options: WebSocketClientOptions = {}) => {
  const clientRef = useRef<WebSocketClient | null>(null);
  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    isConnecting: false,
    error: null
  });

  // Initialize client only once
  if (!clientRef.current) {
    clientRef.current = new WebSocketClient({
      ...options,
      autoConnect: false // We'll handle autoConnect in useEffect
    });
    
    // Add state listener
    clientRef.current.addStateListener((newState) => {
      setState(newState);
    });
  }

  const connect = useCallback(() => {
    clientRef.current?.connect();
  }, []);

  const disconnect = useCallback(() => {
    clientRef.current?.disconnect();
  }, []);

  const send = useCallback((type: string, data?: any) => {
    return clientRef.current?.send(type, data);
  }, []);

  const subscribe = useCallback((eventType: string, callback: (data: any) => void) => {
    return clientRef.current?.subscribe(eventType, callback);
  }, []);

  // Handle autoConnect in useEffect
  useEffect(() => {
    if (options.autoConnect !== false) {
      connect();
    }
    
    return () => {
      disconnect();
    };
  }, [connect, disconnect, options.autoConnect]);

  return {
    state,
    connect,
    disconnect,
    send,
    subscribe
  };
};

export default WebSocketClient;
