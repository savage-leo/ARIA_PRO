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

interface ModelConfig {
  model: string;
  enabled: boolean;
  weight: number;
  threshold: number;
  parameters: Record<string, any>;
}

interface TuningParameters {
  learning_rate?: number;
  batch_size?: number;
  epochs?: number;
  dropout_rate?: number;
  hidden_layers?: number;
  lookback_window?: number;
  feature_selection?: string[];
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
  const [modelConfigs, setModelConfigs] = useState<ModelConfig[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [tuningParams, setTuningParams] = useState<TuningParameters>({});
  const [activeTab, setActiveTab] = useState<"signals" | "models" | "tuning" | "training">("signals");
  const [trainingStatus, setTrainingStatus] = useState<string>("idle");
  
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

  const fetchModelConfigs = useCallback(async () => {
    try {
      const response = await apiGet<{models: ModelConfig[]}>("/api/institutional-ai/model-configs");
      if (response?.models) {
        setModelConfigs(response.models);
      }
    } catch (error) {
      console.error("Failed to fetch model configs:", error);
    }
  }, []);

  const updateModelConfig = async (model: string, config: Partial<ModelConfig>) => {
    try {
      await apiPost(`/api/institutional-ai/model-configs/${model}`, config);
      await fetchModelConfigs();
    } catch (error) {
      console.error("Failed to update model config:", error);
    }
  };

  const startModelTuning = async () => {
    if (!selectedModel) return;
    setTrainingStatus("tuning");
    try {
      await apiPost("/api/institutional-ai/tune-model", {
        model: selectedModel,
        symbol: selectedSymbol,
        parameters: tuningParams
      });
      setTrainingStatus("completed");
    } catch (error) {
      console.error("Failed to start model tuning:", error);
      setTrainingStatus("error");
    }
  };

  const startModelTraining = async (command: string) => {
    setTrainingStatus("training");
    try {
      await apiPost("/api/training/parse-command", { command });
      setTrainingStatus("completed");
    } catch (error) {
      console.error("Failed to start training:", error);
      setTrainingStatus("error");
    }
  };

  const refreshData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        fetchMarketRegimes(),
        fetchRiskMetrics(),
        fetchModelPerformance(),
        fetchSignalHistory(),
        fetchModelConfigs()
      ]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshData();
  }, [selectedSymbol, fetchMarketRegimes, fetchRiskMetrics, fetchModelPerformance, fetchSignalHistory, fetchModelConfigs]);

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
          <h1 className={`text-2xl font-bold ${HUD.TEXT}`}>🤖 Institutional AI Trading Engine</h1>
          <p className="text-sm text-cyan-300/70 mt-1">Real-time AI signal generation, model tuning, and execution platform</p>
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

      {/* Tab Navigation */}
      <div className="flex space-x-1 bg-slate-800/50 p-1 rounded-lg">
        {[
          { key: "signals", label: "📡 Live Signals" },
          { key: "models", label: "🧠 Model Config" },
          { key: "tuning", label: "⚙️ Model Tuning" },
          { key: "training", label: "🎯 Training" }
        ].map(tab => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key as any)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              activeTab === tab.key
                ? 'bg-cyan-500/30 text-cyan-200 border border-cyan-400/50'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'
            }`}
          >
            {tab.label}
          </button>
        ))}
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
              {isAutoTradingEnabled ? '🟢 Auto Trading: ON' : '🔴 Auto Trading: OFF'}
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
        <h3 className={`text-lg font-semibold mb-3 ${HUD.TEXT}`}>📊 Market Regime Analysis</h3>
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

      {/* Tab Content */}
      {activeTab === "signals" && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <div className={`p-4 rounded-xl ${HUD.CARD}`}>
            <h3 className={`text-lg font-semibold mb-3 ${HUD.TEXT}`}>📡 Active AI Signals</h3>
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
            <h3 className={`text-lg font-semibold mb-3 ${HUD.TEXT}`}>🎯 Model Performance</h3>
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
      )}

      {activeTab === "models" && (
        <div className={`p-4 rounded-xl ${HUD.CARD}`}>
          <h3 className={`text-lg font-semibold mb-3 ${HUD.TEXT}`}>🧠 Model Configuration</h3>
          <div className="space-y-4">
            {modelConfigs.map((config, index) => (
              <div key={index} className={`p-4 rounded-lg ${HUD.PANEL} border border-slate-600/50`}>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <span className="font-medium text-cyan-200">{config.model}</span>
                    <label className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={config.enabled}
                        onChange={(e) => updateModelConfig(config.model, { enabled: e.target.checked })}
                        className="rounded border-cyan-400/20"
                      />
                      <span className="text-xs text-cyan-100">Enabled</span>
                    </label>
                  </div>
                  <div className={`px-2 py-1 rounded text-xs ${
                    config.enabled ? 'bg-emerald-500/20 text-emerald-300' : 'bg-slate-500/20 text-slate-300'
                  }`}>
                    {config.enabled ? 'Active' : 'Inactive'}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs text-cyan-300/70 mb-1">Weight</label>
                    <input
                      type="number"
                      value={config.weight}
                      onChange={(e) => updateModelConfig(config.model, { weight: parseFloat(e.target.value) })}
                      className="w-full bg-slate-900/60 text-cyan-100 text-xs border border-cyan-400/20 rounded px-2 py-1"
                      step="0.1"
                      min="0"
                      max="1"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-cyan-300/70 mb-1">Threshold</label>
                    <input
                      type="number"
                      value={config.threshold}
                      onChange={(e) => updateModelConfig(config.model, { threshold: parseFloat(e.target.value) })}
                      className="w-full bg-slate-900/60 text-cyan-100 text-xs border border-cyan-400/20 rounded px-2 py-1"
                      step="0.01"
                      min="0"
                      max="1"
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === "tuning" && (
        <div className={`p-4 rounded-xl ${HUD.CARD}`}>
          <h3 className={`text-lg font-semibold mb-3 ${HUD.TEXT}`}>⚙️ Model Hyperparameter Tuning</h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h4 className="text-cyan-300 font-medium mb-3">Model Selection</h4>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full bg-slate-900/60 text-cyan-100 border border-cyan-400/20 rounded px-3 py-2 mb-4"
              >
                <option value="">Select Model</option>
                <option value="LSTM">LSTM Neural Network</option>
                <option value="XGBoost">XGBoost Classifier</option>
                <option value="CNN">CNN Pattern Recognition</option>
                <option value="PPO">PPO Reinforcement Learning</option>
              </select>

              <div className="space-y-3">
                <div>
                  <label className="block text-xs text-cyan-300/70 mb-1">Learning Rate</label>
                  <input
                    type="number"
                    value={tuningParams.learning_rate || 0.001}
                    onChange={(e) => setTuningParams(prev => ({ ...prev, learning_rate: parseFloat(e.target.value) }))}
                    className="w-full bg-slate-900/60 text-cyan-100 text-xs border border-cyan-400/20 rounded px-2 py-1"
                    step="0.0001"
                  />
                </div>
                <div>
                  <label className="block text-xs text-cyan-300/70 mb-1">Batch Size</label>
                  <input
                    type="number"
                    value={tuningParams.batch_size || 32}
                    onChange={(e) => setTuningParams(prev => ({ ...prev, batch_size: parseInt(e.target.value) }))}
                    className="w-full bg-slate-900/60 text-cyan-100 text-xs border border-cyan-400/20 rounded px-2 py-1"
                  />
                </div>
                <div>
                  <label className="block text-xs text-cyan-300/70 mb-1">Epochs</label>
                  <input
                    type="number"
                    value={tuningParams.epochs || 100}
                    onChange={(e) => setTuningParams(prev => ({ ...prev, epochs: parseInt(e.target.value) }))}
                    className="w-full bg-slate-900/60 text-cyan-100 text-xs border border-cyan-400/20 rounded px-2 py-1"
                  />
                </div>
                <div>
                  <label className="block text-xs text-cyan-300/70 mb-1">Dropout Rate</label>
                  <input
                    type="number"
                    value={tuningParams.dropout_rate || 0.2}
                    onChange={(e) => setTuningParams(prev => ({ ...prev, dropout_rate: parseFloat(e.target.value) }))}
                    className="w-full bg-slate-900/60 text-cyan-100 text-xs border border-cyan-400/20 rounded px-2 py-1"
                    step="0.1"
                    min="0"
                    max="0.9"
                  />
                </div>
              </div>
            </div>

            <div>
              <h4 className="text-cyan-300 font-medium mb-3">Tuning Status</h4>
              <div className={`p-3 rounded-lg ${HUD.PANEL} mb-4`}>
                <div className="flex items-center gap-2 mb-2">
                  <div className={`w-2 h-2 rounded-full ${
                    trainingStatus === 'tuning' ? 'bg-yellow-400 animate-pulse' :
                    trainingStatus === 'completed' ? 'bg-emerald-400' :
                    trainingStatus === 'error' ? 'bg-red-400' : 'bg-slate-400'
                  }`}></div>
                  <span className="text-sm capitalize">{trainingStatus}</span>
                </div>
                <div className="text-xs text-slate-400">
                  {trainingStatus === 'idle' && 'Ready to start tuning'}
                  {trainingStatus === 'tuning' && 'Optimizing hyperparameters...'}
                  {trainingStatus === 'completed' && 'Tuning completed successfully'}
                  {trainingStatus === 'error' && 'Tuning failed - check logs'}
                </div>
              </div>

              <button
                onClick={startModelTuning}
                disabled={!selectedModel || trainingStatus === 'tuning'}
                className="w-full px-4 py-3 rounded-lg bg-cyan-500/20 border border-cyan-400/30 text-cyan-200 hover:bg-cyan-500/30 disabled:opacity-50"
              >
                {trainingStatus === 'tuning' ? 'Tuning in Progress...' : 'Start Hyperparameter Tuning'}
              </button>
            </div>
          </div>
        </div>
      )}

      {activeTab === "training" && (
        <div className={`p-4 rounded-xl ${HUD.CARD}`}>
          <h3 className={`text-lg font-semibold mb-3 ${HUD.TEXT}`}>🎯 Model Training</h3>
          <div className="space-y-4">
            <div>
              <h4 className="text-cyan-300 font-medium mb-2">Natural Language Training Commands</h4>
              <p className="text-xs text-slate-400 mb-3">Use natural language to train models with specific data</p>
              
              <div className="space-y-2">
                {[
                  "Train XAUUSD M5 for last 3 days",
                  "Train EURUSD H1 for last week",
                  "Train BTCUSD M15 for last 2 days",
                  "Retrain all models with latest data"
                ].map((command, index) => (
                  <button
                    key={index}
                    onClick={() => startModelTraining(command)}
                    disabled={trainingStatus === 'training'}
                    className="w-full text-left px-3 py-2 rounded-lg bg-slate-700/50 border border-slate-600/50 text-slate-300 hover:bg-slate-600/50 disabled:opacity-50 text-sm"
                  >
                    {command}
                  </button>
                ))}
              </div>
            </div>

            <div className={`p-3 rounded-lg ${HUD.PANEL}`}>
              <h4 className="text-cyan-300 font-medium mb-2">Training Status</h4>
              <div className="flex items-center gap-2 mb-2">
                <div className={`w-2 h-2 rounded-full ${
                  trainingStatus === 'training' ? 'bg-blue-400 animate-pulse' :
                  trainingStatus === 'completed' ? 'bg-emerald-400' :
                  trainingStatus === 'error' ? 'bg-red-400' : 'bg-slate-400'
                }`}></div>
                <span className="text-sm capitalize">{trainingStatus}</span>
              </div>
              <div className="text-xs text-slate-400">
                {trainingStatus === 'idle' && 'Ready to start training'}
                {trainingStatus === 'training' && 'Training models with new data...'}
                {trainingStatus === 'completed' && 'Training completed successfully'}
                {trainingStatus === 'error' && 'Training failed - check logs'}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Signal History - Always visible */}
      <div className={`p-4 rounded-xl ${HUD.CARD}`}>
        <h3 className={`text-lg font-semibold mb-3 ${HUD.TEXT}`}>📈 Signal History - {selectedSymbol}</h3>
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
