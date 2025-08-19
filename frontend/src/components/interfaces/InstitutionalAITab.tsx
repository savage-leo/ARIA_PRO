// frontend/src/components/interfaces/InstitutionalAITab.tsx
import React, { useEffect, useState, useCallback } from "react";
import { apiGet, apiPost } from "../../services/api";
import { HUD } from "../../theme/hud";
import { useWebSocket, WebSocketMessage } from "../../hooks/useWebSocket";

interface AISignal {
  symbol: string;
  side: "buy" | "sell";
  strength: number;
  confidence: number;
  model: string;
  timestamp: number;
  features?: Record<string, any>;
  entry_price?: number;
  stop_loss?: number;
  take_profit?: number;
}

interface MarketRegime {
  symbol: string;
  regime: "trending" | "ranging" | "volatile" | "consolidating";
  strength: number;
  duration: number;
}

interface RiskMetrics {
  var_95: number;
  expected_shortfall: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  profit_factor: number;
}

interface ModelPerformance {
  model: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  trades_count: number;
  avg_return: number;
}

const InstitutionalAITab: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState("EURUSD");
  const [activeSignals, setActiveSignals] = useState<AISignal[]>([]);
  const [marketRegimes, setMarketRegimes] = useState<MarketRegime[]>([]);
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics | null>(null);
  const [modelPerformance, setModelPerformance] = useState<ModelPerformance[]>([]);
  const [signalHistory, setSignalHistory] = useState<AISignal[]>([]);
  const [isAutoTradingEnabled, setIsAutoTradingEnabled] = useState(false);
  const [loading, setLoading] = useState(false);
  
  const symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD", "BTCUSD"];

  // WebSocket for real-time updates (auto-resolves URL via env/proxy)
  const { lastMessage, isConnected } = useWebSocket();

  useEffect(() => {
    if (!lastMessage) return;
    const msg = lastMessage as WebSocketMessage;
    if (msg.type === "signal" && msg.data) {
      const sig = msg.data as AISignal;
      setActiveSignals((prev) => [sig, ...prev.slice(0, 19)]);
      setSignalHistory((prev) => [sig, ...prev.slice(0, 99)]);
    }
  }, [lastMessage]);

  const fetchMarketRegimes = useCallback(async () => {
    try {
      const response = await apiGet<{regimes: MarketRegime[]}>("/api/institutional-ai/market-regimes");
      if (response?.regimes) {
        setMarketRegimes(response.regimes);
      }
    } catch (error) {
      console.error("Failed to fetch market regimes:", error);
    }
  }, []);

  const fetchRiskMetrics = useCallback(async () => {
    try {
      const response = await apiGet<RiskMetrics>("/trading/risk-metrics");
      if (response) {
        setRiskMetrics(response);
      }
    } catch (error) {
      console.error("Failed to fetch risk metrics:", error);
    }
  }, []);

  const fetchModelPerformance = useCallback(async () => {
    try {
      const response = await apiGet<{models: ModelPerformance[]}>("/api/institutional-ai/model-performance");
      if (response?.models) {
        setModelPerformance(response.models);
      }
    } catch (error) {
      console.error("Failed to fetch model performance:", error);
    }
  }, []);

  const fetchSignalHistory = useCallback(async () => {
    try {
      const response = await apiGet<{signals: AISignal[]}>(`/signals/history?symbol=${selectedSymbol}&limit=50`);
      if (response?.signals) {
        setSignalHistory(response.signals);
      }
    } catch (error) {
      console.error("Failed to fetch signal history:", error);
    }
  }, [selectedSymbol]);

  const toggleAutoTrading = async () => {
    try {
      const response = await apiPost("/api/institutional-ai/auto-trading/toggle", {
        enabled: !isAutoTradingEnabled
      });
      if (response) {
        setIsAutoTradingEnabled(!isAutoTradingEnabled);
      }
    } catch (error) {
      console.error("Failed to toggle auto trading:", error);
    }
  };

  const executeSignal = async (signal: AISignal) => {
    try {
      await apiPost("/api/institutional-ai/execute-signal", {
        symbol: signal.symbol,
        side: signal.side,
        confidence: signal.confidence,
        model: signal.model
      });
    } catch (error) {
      console.error("Failed to execute signal:", error);
    }
  };

  const refreshData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        fetchMarketRegimes(),
        fetchRiskMetrics(),
        fetchModelPerformance(),
        fetchSignalHistory()
      ]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshData();
  }, [selectedSymbol, fetchMarketRegimes, fetchRiskMetrics, fetchModelPerformance, fetchSignalHistory]);

  const getSignalStrengthColor = (strength: number) => {
    if (strength >= 0.8) return "text-emerald-400 bg-emerald-500/20";
    if (strength >= 0.6) return "text-yellow-400 bg-yellow-500/20";
    return "text-red-400 bg-red-500/20";
  };

  const getRegimeColor = (regime: string) => {
    switch (regime) {
      case "trending": return "text-blue-400 bg-blue-500/20";
      case "ranging": return "text-purple-400 bg-purple-500/20";
      case "volatile": return "text-red-400 bg-red-500/20";
      default: return "text-gray-400 bg-gray-500/20";
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className={`text-2xl font-bold ${HUD.TEXT}`}>Institutional AI Trading Engine</h1>
          <p className="text-sm text-cyan-300/70 mt-1">Real-time AI signal generation and execution platform</p>
        </div>
        <div className="flex items-center gap-4">
          <div className={`px-3 py-2 rounded-lg ${HUD.PANEL} flex items-center gap-2`}>
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-400' : 'bg-red-400'}`}></div>
            <span className="text-sm">{isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
          <button
            onClick={refreshData}
            disabled={loading}
            className="px-4 py-2 rounded-lg bg-blue-500/20 border border-blue-400/30 text-blue-200 hover:bg-blue-500/30 disabled:opacity-50"
          >
            {loading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </div>

      {/* Symbol Selection & Auto Trading Control */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className={`p-4 rounded-xl ${HUD.CARD}`}>
          <h3 className={`text-lg font-semibold mb-3 ${HUD.TEXT}`}>Symbol Selection</h3>
          <div className="grid grid-cols-2 gap-2">
            {symbols.map(symbol => (
              <button
                key={symbol}
                onClick={() => setSelectedSymbol(symbol)}
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  selectedSymbol === symbol
                    ? 'bg-cyan-500/30 border border-cyan-400/50 text-cyan-200'
                    : 'bg-slate-700/50 border border-slate-600/50 text-slate-300 hover:bg-slate-600/50'
                }`}
              >
                {symbol}
              </button>
            ))}
          </div>
        </div>

        <div className={`p-4 rounded-xl ${HUD.CARD}`}>
          <h3 className={`text-lg font-semibold mb-3 ${HUD.TEXT}`}>Auto Trading</h3>
          <div className="space-y-3">
            <button
              onClick={toggleAutoTrading}
              className={`w-full px-4 py-3 rounded-lg font-medium transition-colors ${
                isAutoTradingEnabled
                  ? 'bg-emerald-500/20 border border-emerald-400/30 text-emerald-200'
                  : 'bg-red-500/20 border border-red-400/30 text-red-200'
              }`}
            >
              {isAutoTradingEnabled ? 'Auto Trading: ON' : 'Auto Trading: OFF'}
            </button>
            <div className="text-xs text-slate-400">
              {isAutoTradingEnabled ? 'AI signals will be executed automatically' : 'Manual execution required'}
            </div>
          </div>
        </div>

        <div className={`p-4 rounded-xl ${HUD.CARD}`}>
          <h3 className={`text-lg font-semibold mb-3 ${HUD.TEXT}`}>Risk Overview</h3>
          {riskMetrics ? (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">VaR (95%):</span>
                <span className="text-red-300">{riskMetrics.var_95.toFixed(2)}%</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Sharpe Ratio:</span>
                <span className="text-emerald-300">{riskMetrics.sharpe_ratio.toFixed(2)}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-slate-400">Win Rate:</span>
                <span className="text-blue-300">{riskMetrics.win_rate.toFixed(1)}%</span>
              </div>
            </div>
          ) : (
            <div className="text-slate-400 text-sm">Loading risk metrics...</div>
          )}
        </div>
      </div>

      {/* Market Regimes */}
      <div className={`p-4 rounded-xl ${HUD.CARD}`}>
        <h3 className={`text-lg font-semibold mb-3 ${HUD.TEXT}`}>Market Regime Analysis</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {marketRegimes.map((regime, index) => (
            <div key={index} className={`p-3 rounded-lg ${getRegimeColor(regime.regime)} border`}>
              <div className="font-medium">{regime.symbol}</div>
              <div className="text-sm opacity-80 capitalize">{regime.regime}</div>
              <div className="text-xs mt-1">Strength: {(regime.strength * 100).toFixed(0)}%</div>
            </div>
          ))}
        </div>
      </div>

      {/* Active Signals & Model Performance */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className={`p-4 rounded-xl ${HUD.CARD}`}>
          <h3 className={`text-lg font-semibold mb-3 ${HUD.TEXT}`}>Active AI Signals</h3>
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {activeSignals.length > 0 ? (
              activeSignals.map((signal, index) => (
                <div key={index} className={`p-3 rounded-lg ${HUD.PANEL} border border-slate-600/50`}>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{signal.symbol}</span>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        signal.side === 'buy' ? 'bg-emerald-500/20 text-emerald-300' : 'bg-red-500/20 text-red-300'
                      }`}>
                        {signal.side.toUpperCase()}
                      </span>
                    </div>
                    <button
                      onClick={() => executeSignal(signal)}
                      className="px-3 py-1 rounded bg-blue-500/20 text-blue-300 text-xs hover:bg-blue-500/30"
                    >
                      Execute
                    </button>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>Model: {signal.model}</div>
                    <div>Confidence: {(signal.confidence * 100).toFixed(1)}%</div>
                    <div className={`${getSignalStrengthColor(signal.strength)} px-2 py-1 rounded`}>
                      Strength: {(signal.strength * 100).toFixed(0)}%
                    </div>
                    <div className="text-slate-400">
                      {new Date(signal.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-slate-400 text-center py-8">No active signals</div>
            )}
          </div>
        </div>

        <div className={`p-4 rounded-xl ${HUD.CARD}`}>
          <h3 className={`text-lg font-semibold mb-3 ${HUD.TEXT}`}>Model Performance</h3>
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {modelPerformance.map((model, index) => (
              <div key={index} className={`p-3 rounded-lg ${HUD.PANEL} border border-slate-600/50`}>
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium">{model.model}</span>
                  <span className="text-xs text-slate-400">{model.trades_count} trades</span>
                </div>
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div>Accuracy: {(model.accuracy * 100).toFixed(1)}%</div>
                  <div>Precision: {(model.precision * 100).toFixed(1)}%</div>
                  <div>F1: {(model.f1_score * 100).toFixed(1)}%</div>
                </div>
                <div className="mt-2 text-xs">
                  Avg Return: <span className={model.avg_return >= 0 ? 'text-emerald-300' : 'text-red-300'}>
                    {(model.avg_return * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Signal History */}
      <div className={`p-4 rounded-xl ${HUD.CARD}`}>
        <h3 className={`text-lg font-semibold mb-3 ${HUD.TEXT}`}>Signal History - {selectedSymbol}</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-600/50">
                <th className="text-left py-2">Time</th>
                <th className="text-left py-2">Symbol</th>
                <th className="text-left py-2">Side</th>
                <th className="text-left py-2">Model</th>
                <th className="text-left py-2">Confidence</th>
                <th className="text-left py-2">Strength</th>
                <th className="text-left py-2">Status</th>
              </tr>
            </thead>
            <tbody>
              {signalHistory.slice(0, 20).map((signal, index) => (
                <tr key={index} className="border-b border-slate-700/30 hover:bg-slate-700/20">
                  <td className="py-2">{new Date(signal.timestamp).toLocaleString()}</td>
                  <td className="py-2">{signal.symbol}</td>
                  <td className="py-2">
                    <span className={`px-2 py-1 rounded text-xs ${
                      signal.side === 'buy' ? 'bg-emerald-500/20 text-emerald-300' : 'bg-red-500/20 text-red-300'
                    }`}>
                      {signal.side.toUpperCase()}
                    </span>
                  </td>
                  <td className="py-2">{signal.model}</td>
                  <td className="py-2">{(signal.confidence * 100).toFixed(1)}%</td>
                  <td className="py-2">
                    <span className={`px-2 py-1 rounded text-xs ${getSignalStrengthColor(signal.strength)}`}>
                      {(signal.strength * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td className="py-2">
                    <span className="px-2 py-1 rounded text-xs bg-blue-500/20 text-blue-300">
                      Active
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default InstitutionalAITab;
