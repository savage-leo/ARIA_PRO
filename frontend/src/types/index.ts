// Core Trading Types
export interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnl: number;
  realizedPnl: number;
  timestamp: string;
  stopLoss?: number;
  takeProfit?: number;
}

export interface Signal {
  id: string;
  symbol: string;
  direction: 'buy' | 'sell';
  confidence: number;
  timestamp: string;
  price: number;
  stopLoss?: number;
  takeProfit?: number;
  reasoning: string;
  source: 'ai' | 'smc' | 'fusion';
}

export interface TradeIdea {
  id: number;
  symbol: string;
  bias: 'long' | 'short';
  confidence: number;
  timestamp: string;
  payload: Record<string, any>;
  meta: Record<string, any>;
  outcome?: TradeOutcome;
}

export interface TradeOutcome {
  pnl: number;
  status: 'open' | 'closed' | 'cancelled';
  exitPrice?: number;
  exitTimestamp?: string;
}

// Market Data Types
export interface OHLCV {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface MarketData {
  symbol: string;
  bid: number;
  ask: number;
  spread: number;
  timestamp: string;
  bars: OHLCV[];
}

// System Status Types
export interface ModuleStatus {
  name: string;
  status: 'healthy' | 'warning' | 'error' | 'offline';
  lastUpdate: string;
  metrics?: Record<string, number>;
  message?: string;
}

export interface SystemHealth {
  overall: 'healthy' | 'degraded' | 'critical';
  modules: ModuleStatus[];
  uptime: number;
  memoryUsage: number;
  cpuUsage: number;
}

// Performance Analytics Types
export interface EquityCurvePoint {
  timestamp: string;
  equity: number;
  drawdown: number;
  trades: number;
}

export interface PerformanceMetrics {
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  avgTrade: number;
  bestTrade: number;
  worstTrade: number;
}

// API Response Types
export interface ApiResponse<T = any> {
  ok: boolean;
  data?: T;
  error?: string;
  timestamp: string;
}

// WebSocket Message Types
export interface WSMessage {
  type: 'market_data' | 'signal' | 'position_update' | 'system_status';
  data: any;
  timestamp: string;
}
