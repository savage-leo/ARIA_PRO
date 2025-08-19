import { useEffect, useRef, useState, useCallback } from 'react';

export interface WebSocketMessage {
  type: string;
  timestamp: string;
  data?: any;
  message?: string;
  client_id?: string;
}

export interface WebSocketState {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  lastMessage: WebSocketMessage | null;
  messageHistory: WebSocketMessage[];
}

export interface UseWebSocketOptions {
  url?: string;
  autoConnect?: boolean;
  maxHistory?: number;
  onMessage?: (message: WebSocketMessage) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: string) => void;
}

type Env = {
  VITE_BACKEND_WS?: string;
  VITE_BACKEND_BASE?: string;
};

const getDefaultWsUrl = (): string => {
  const env = (import.meta as unknown as { env: Env }).env;
  const fromEnv = env?.VITE_BACKEND_WS;
  if (fromEnv && typeof fromEnv === 'string' && fromEnv.length > 0) {
    return fromEnv;
  }
  const base = env?.VITE_BACKEND_BASE;
  if (base && typeof base === 'string' && base.length > 0) {
    const wsBase = base.replace(/^http(s?):\/\//, (_m, s) => (s ? 'wss://' : 'ws://'));
    return `${wsBase.replace(/\/+$/, '')}/ws`;
  }
  // Fall back to same-origin WS to leverage Vite proxy in dev
  const proto = typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss' : 'ws';
  const host = typeof window !== 'undefined' ? window.location.host : 'localhost:5175';
  return `${proto}://${host}/ws`;
};

export const useWebSocket = (options: UseWebSocketOptions = {}) => {
  const {
    url = getDefaultWsUrl(),
    autoConnect = true,
    maxHistory = 100,
    onMessage,
    onConnect,
    onDisconnect,
    onError
  } = options;

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pingIntervalRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isConnectingRef = useRef<boolean>(false);
  const pendingQueueRef = useRef<any[]>([]);
  
  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    isConnecting: false,
    error: null,
    lastMessage: null,
    messageHistory: []
  });

                      // Store callbacks in refs to prevent infinite re-renders
  const onMessageRef = useRef(onMessage);
  const onConnectRef = useRef(onConnect);
  const onDisconnectRef = useRef(onDisconnect);
  const onErrorRef = useRef(onError);

  // Update refs when callbacks change
  useEffect(() => {
    onMessageRef.current = onMessage;
    onConnectRef.current = onConnect;
    onDisconnectRef.current = onDisconnect;
    onErrorRef.current = onError;
  }, [onMessage, onConnect, onDisconnect, onError]);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN || wsRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    // Prevent multiple connection attempts
    if (isConnectingRef.current || state.isConnecting) {
      return;
    }

    isConnectingRef.current = true;
    setState(prev => ({ ...prev, isConnecting: true, error: null }));

    try {
      console.log('Attempting to connect to WebSocket:', url);
      wsRef.current = new WebSocket(url);

      wsRef.current.onopen = () => {
        console.log('WebSocket connected successfully');
        isConnectingRef.current = false;
        setState(prev => ({
          ...prev,
          isConnected: true,
          isConnecting: false,
          error: null
        }));
        onConnectRef.current?.();

        // Start ping interval
        pingIntervalRef.current = setInterval(() => {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            try {
              wsRef.current.send(JSON.stringify({ type: 'ping' }));
            } catch (error) {
              console.error('Failed to send ping:', error);
            }
          }
        }, 30000); // Ping every 30 seconds

        // Flush any queued messages
        if (pendingQueueRef.current.length && wsRef.current?.readyState === WebSocket.OPEN) {
          for (const m of pendingQueueRef.current) {
            try { wsRef.current.send(JSON.stringify(m)); } catch {}
          }
          pendingQueueRef.current = [];
        }
      };

      wsRef.current.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          
          setState(prev => ({
            ...prev,
            lastMessage: message,
            messageHistory: [
              message,
              ...prev.messageHistory.slice(0, maxHistory - 1)
            ]
          }));

          onMessageRef.current?.(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      wsRef.current.onclose = (event) => {
        console.log('WebSocket closed with code:', event.code, 'reason:', event.reason);
        isConnectingRef.current = false;
        setState(prev => ({
          ...prev,
          isConnected: false,
          isConnecting: false
        }));
        
        onDisconnectRef.current?.();

        // Clear ping interval
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
          pingIntervalRef.current = null;
        }

        // Auto-reconnect if not manually closed (with exponential backoff)
        if (event.code !== 1000 && !reconnectTimeoutRef.current) {
          const delay = Math.min(3000 * Math.pow(2, 0), 30000); // Max 30 seconds
          console.log('Scheduling WebSocket reconnection in', delay, 'ms');
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectTimeoutRef.current = null;
            // Add a small delay to prevent rapid reconnections
            setTimeout(() => {
              connect();
            }, 100);
          }, delay);
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        isConnectingRef.current = false;
        const errorMessage = 'WebSocket connection error';
        setState(prev => ({
          ...prev,
          error: errorMessage,
          isConnecting: false
        }));
        onErrorRef.current?.(errorMessage);
      };

    } catch (error) {
      isConnectingRef.current = false;
      const errorMessage = 'Failed to create WebSocket connection';
      setState(prev => ({
        ...prev,
        error: errorMessage,
        isConnecting: false
      }));
      onErrorRef.current?.(errorMessage);
    }
  }, [url, maxHistory]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }

    if (wsRef.current) {
      try {
        wsRef.current.close(1000); // Normal closure
      } catch (error) {
        console.error('Error closing WebSocket:', error);
      }
      wsRef.current = null;
    }

    isConnectingRef.current = false;
    setState(prev => ({
      ...prev,
      isConnected: false,
      isConnecting: false
    }));
  }, []);

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify(message));
      } catch (error) {
        console.error('Failed to send WebSocket message:', error);
      }
    } else {
      // Queue until open to avoid warning spam and lost subscriptions
      pendingQueueRef.current.push(message);
    }
  }, []);

  const subscribe = useCallback((channels: string[]) => {
    sendMessage({
      type: 'subscribe',
      channels
    });
  }, [sendMessage]);

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect && !wsRef.current) {
      // Add a small delay to prevent rapid connections during React StrictMode
      const timeoutId = setTimeout(() => {
        connect();
      }, 100);
      
      return () => {
        clearTimeout(timeoutId);
        disconnect();
      };
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return {
    ...state,
    connect,
    disconnect,
    sendMessage,
    subscribe
  };
};
