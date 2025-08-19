import { useState, useCallback } from 'react';
import type { FC } from 'react';
import { useWebSocket, WebSocketMessage } from '../../hooks/useWebSocket';

interface TickData {
  symbol: string;
  bid: number;
  ask: number;
  spread: number;
  timestamp: string;
}

interface SignalData {
  symbol: string;
  side: 'buy' | 'sell';
  strength: number;
  confidence: number;
  timestamp: string;
}

interface OrderData {
  order_id: string;
  symbol: string;
  side: 'buy' | 'sell';
  status: string;
  volume: number;
  price: number;
  timestamp: string;
}

const RealTimeData: FC = () => {
  const [ticks, setTicks] = useState<TickData[]>([]);
  const [signals, setSignals] = useState<SignalData[]>([]);
  const [orders, setOrders] = useState<OrderData[]>([]);
  const [ideas, setIdeas] = useState<any[]>([]);

  const handleMessage = useCallback((message: WebSocketMessage) => {
    switch (message.type) {
      case 'tick':
        if (message.data) {
          setTicks(prev => [message.data, ...prev.slice(0, 49)]); // Keep last 50 ticks
        }
        break;
      case 'signal':
        if (message.data) {
          setSignals(prev => [message.data, ...prev.slice(0, 19)]); // Keep last 20 signals
        }
        break;
      case 'order_update':
        if (message.data) {
          setOrders(prev => [message.data, ...prev.slice(0, 19)]); // Keep last 20 orders
        }
        break;
      case 'idea':
        if (message.data) {
          setIdeas(prev => [message.data, ...prev.slice(0, 9)]); // Keep last 10 ideas
        }
        break;
      // omit prepared_payload in this view
    }
  }, []);

  const { isConnected, isConnecting, error, messageHistory } = useWebSocket({
    onMessage: handleMessage
  });

  const formatPrice = (price: number) => {
    return price.toFixed(5);
  };

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getSignalColor = (side: string) => {
    return side === 'buy' ? 'text-green-500' : 'text-red-500';
  };

  const getOrderStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'filled':
        return 'text-green-500';
      case 'pending':
        return 'text-yellow-500';
      case 'cancelled':
        return 'text-red-500';
      default:
        return 'text-gray-500';
    }
  };

  return (
    <div className="h-full flex flex-col space-y-4 p-4">
      {/* Connection Status */}
      <div className="flex items-center space-x-2">
        <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : isConnecting ? 'bg-yellow-500' : 'bg-red-500'}`} />
        <span className="text-sm font-medium">
          {isConnected ? 'Connected' : isConnecting ? 'Connecting...' : 'Disconnected'}
        </span>
        {error && <span className="text-red-500 text-sm">Error: {error}</span>}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-1">
        {/* Live Ticks */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-3 text-white">Live Ticks</h3>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {ticks.length === 0 ? (
              <p className="text-gray-400 text-sm">Waiting for tick data...</p>
            ) : (
              ticks.map((tick, index) => (
                <div key={index} className="flex justify-between items-center p-2 bg-slate-700 rounded">
                  <div>
                    <span className="font-medium text-white">{tick.symbol}</span>
                    <div className="text-sm text-gray-300">
                      {formatPrice(tick.bid)} / {formatPrice(tick.ask)}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-gray-400">
                      Spread: {formatPrice(tick.spread)}
                    </div>
                    <div className="text-xs text-gray-500">
                      {formatTime(tick.timestamp)}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Trading Signals */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-3 text-white">Trading Signals</h3>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {signals.length === 0 ? (
              <p className="text-gray-400 text-sm">Waiting for signals...</p>
            ) : (
              signals.map((signal, index) => (
                <div key={index} className="flex justify-between items-center p-2 bg-slate-700 rounded">
                  <div>
                    <span className={`font-medium ${getSignalColor(signal.side)}`}>
                      {signal.side.toUpperCase()} {signal.symbol}
                    </span>
                    <div className="text-sm text-gray-300">
                      Strength: {signal.strength} | Confidence: {signal.confidence}%
                    </div>
                  </div>
                  <div className="text-xs text-gray-500">
                    {formatTime(signal.timestamp)}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Order Updates */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-3 text-white">Order Updates</h3>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {orders.length === 0 ? (
              <p className="text-gray-400 text-sm">Waiting for order updates...</p>
            ) : (
              orders.map((order, index) => (
                <div key={index} className="flex justify-between items-center p-2 bg-slate-700 rounded">
                  <div>
                    <span className={`font-medium ${getOrderStatusColor(order.status)}`}>
                      {order.status.toUpperCase()}
                    </span>
                    <div className="text-sm text-gray-300">
                      {order.side.toUpperCase()} {order.symbol} @ {formatPrice(order.price)}
                    </div>
                    <div className="text-xs text-gray-400">
                      Vol: {order.volume} | ID: {order.order_id.slice(-8)}
                    </div>
                  </div>
                  <div className="text-xs text-gray-500">
                    {formatTime(order.timestamp)}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Trading Ideas */}
        <div className="bg-slate-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold mb-3 text-white">Trading Ideas</h3>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {ideas.length === 0 ? (
              <p className="text-gray-400 text-sm">Waiting for trading ideas...</p>
            ) : (
              ideas.map((idea, index) => (
                <div key={index} className="p-2 bg-slate-700 rounded">
                  <div className="text-sm text-white font-medium">
                    {idea.symbol || 'Unknown Symbol'}
                  </div>
                  <div className="text-xs text-gray-300 mt-1">
                    {JSON.stringify(idea, null, 2)}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {formatTime(idea.timestamp || new Date().toISOString())}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Message History (Debug) */}
      <div className="bg-slate-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-3 text-white">Message History (Last 10)</h3>
        <div className="space-y-1 max-h-32 overflow-y-auto">
          {messageHistory.slice(0, 10).map((msg, index) => (
            <div key={index} className="text-xs text-gray-300">
              <span className="text-blue-400">{msg.type}</span>
              <span className="text-gray-500 ml-2">
                {formatTime(msg.timestamp)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default RealTimeData;
