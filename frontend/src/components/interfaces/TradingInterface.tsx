 // frontend/src/components/interfaces/TradingInterface.tsx
import { useState } from "react";
import { apiPost } from "../../services/api";
import { useWebSocket } from "../../hooks/useWebSocket";

import { HUD } from "../../theme/hud";
interface PreparedPayload {
  symbol: string;
  bias: string;
  confidence: number;
  entry: number;
  stop: number;
  takeprofit: number;
  risk: {
    base_risk_pct: number;
    bias_factor: number;
    effective_risk_pct: number;
    score: number;
    throttle: boolean;
    reasons: Record<string, number>;
  };
  position: { size: number } | null;
  plan: any;
  dry_run: boolean;
}

const TradingInterface: React.FC = () => {
  const [symbol, setSymbol] = useState("EURUSD");
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [sl, setSl] = useState<string>("");
  const [tp, setTp] = useState<string>("");
  const [risk, setRisk] = useState<number>(0.5);
  const [volume, setVolume] = useState<number>(0.01);
  const [running, setRunning] = useState(false);
  const [currentPrice, setCurrentPrice] = useState<{bid: number, ask: number} | null>(null);
  const [preparedPayload, setPreparedPayload] = useState<PreparedPayload | null>(null);
  const [liveSignals, setLiveSignals] = useState<any[]>([]);
  const [smcIdeas, setSmcIdeas] = useState<any[]>([]);
  const [orderUpdates, setOrderUpdates] = useState<any[]>([]);

  const { isConnected } = useWebSocket({
    onMessage: (message) => {
      if (message.type === 'tick' && message.data?.symbol === symbol) {
        setCurrentPrice({
          bid: message.data.bid,
          ask: message.data.ask
        });
      } else if (message.type === 'signal') {
        setLiveSignals(prev => [message.data, ...prev.slice(0, 9)]);
      } else if (message.type === 'smc_idea') {
        setSmcIdeas(prev => [message.data, ...prev.slice(0, 9)]);
      } else if (message.type === 'order_update') {
        setOrderUpdates(prev => [message.data, ...prev.slice(0, 9)]);
      }
    }
  });

  const prepareWithBias = async () => {
    setRunning(true);
    try {
      const payload = {
        symbol,
        base_risk_pct: risk,
        market_ctx: {
          spread: currentPrice ? currentPrice.ask - currentPrice.bid : 0.00008,
          slippage: 0.00005,
          of_imbalance: 0.3
        },
        equity: 10000.0
      };
      
      const res = await apiPost("/api/smc/idea/prepare", payload);
      if (res && res.prepared_payload) {
        setPreparedPayload(res.prepared_payload as PreparedPayload);
        console.log("Bias Engine Result:", res.prepared_payload as PreparedPayload);
      } else {
        console.log("No valid idea available:", res.msg);
      }
    } catch (err: any) {
      console.error("Bias preparation failed:", err);
      alert("Bias preparation failed: " + err.message);
    } finally {
      setRunning(false);
    }
  };

  const executeIdea = async () => {
    if (!preparedPayload) {
      alert("No prepared payload available. Run Bias Engine first.");
      return;
    }

    setRunning(true);
    try {
      const payload = {
        ...preparedPayload,
        dry_run: false // LIVE TRADING - NO MOCK
      };
      
      const res = await apiPost("/api/smc/idea/execute", payload);
      console.log("Live execution result:", res);
      alert("Live order executed successfully!");
    } catch (err: any) {
      console.error("Live execution failed:", err);
      alert("Live execution failed: " + err.message);
    } finally {
      setRunning(false);
    }
  };

  const placeOrder = async (dry = false) => {
    setRunning(true);
    try {
      const payload = {
        symbol,
        order_type: side,
        volume: volume,
        sl: sl ? parseFloat(sl) : undefined,
        tp: tp ? parseFloat(tp) : undefined,
        comment: `placed-from-frontend-${dry ? 'dry-run' : 'LIVE'}`
      };
      const res = await apiPost("/trading/place-order", payload);
      console.log("Order response:", res);
      alert("Order placed successfully!");
    } catch (err: any) {
      console.error("Order failed:", err);
      const errorMessage = err.message || String(err);
      alert("Order failed: " + errorMessage);
    } finally {
      setRunning(false);
    }
  };

  // position size calculator removed (unused)

  const formatPrice = (price: number) => {
    return price.toFixed(5);
  };

  const getBiasColor = (biasFactor: number) => {
    if (biasFactor >= 1.5) return 'text-emerald-400';
    if (biasFactor >= 1.0) return 'text-amber-400';
    return 'text-rose-400';
  };

  return (
    <div className={`h-full flex flex-col lg:flex-row gap-4 p-4 ${HUD.BG}`}>
      {/* Trading Panel */}
      <div className="lg:w-1/3">
        <div className={`${HUD.CARD} rounded-xl p-6`}>
          <h2 className="text-2xl font-semibold text-cyan-200 drop-shadow mb-6">🚀 LIVE TRADING PANEL</h2>

          {/* Connection Status */}
          <div className="flex items-center space-x-2 mb-4">
            <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-emerald-400/80' : 'bg-rose-400/80'}`} />
            <span className="text-sm text-cyan-300">
              {isConnected ? '✅ LIVE MT5 Connected' : '❌ Disconnected'}
            </span>
          </div>

          {/* Current Price Display */}
          {currentPrice && (
            <div className={`${HUD.PANEL} rounded-lg p-4 mb-4`}>
              <div className="text-center">
                <div className="text-sm text-cyan-300/70">Real MT5 Price</div>
                <div className={HUD.VALUE}>
                  {formatPrice(currentPrice.bid)} / {formatPrice(currentPrice.ask)}
                </div>
                <div className="text-sm text-cyan-300/70">
                  Spread: {formatPrice(currentPrice.ask - currentPrice.bid)}
                </div>
              </div>
            </div>
          )}

          {/* Bias Engine Section */}
          <div className={`${HUD.PANEL} rounded-lg p-4 mb-4`}>
            <h3 className={`${HUD.TITLE} mb-3`}>🎯 Bias Engine</h3>
            <button
              onClick={prepareWithBias}
              disabled={running}
              className="w-full bg-cyan-600/70 hover:bg-cyan-600 disabled:opacity-60 text-cyan-50 font-semibold py-2 px-4 rounded border border-cyan-400/30 mb-3"
            >
              {running ? "Processing..." : "Run Bias Engine"}
            </button>
            
            {preparedPayload && (
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-cyan-100/90">Bias Factor:</span>
                  <span className={`font-semibold ${getBiasColor(preparedPayload.risk.bias_factor)}`}>
                    {preparedPayload.risk.bias_factor.toFixed(2)}x
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-cyan-100/90">Score:</span>
                  <span className="text-cyan-100/90">{(preparedPayload.risk.score * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-cyan-100/90">Throttle:</span>
                  <span className={preparedPayload.risk.throttle ? 'text-rose-400' : 'text-emerald-400'}>
                    {preparedPayload.risk.throttle ? '❌ BLOCKED' : '✅ ALLOWED'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-cyan-100/90">Position Size:</span>
                  <span className="text-cyan-100/90">
                    {preparedPayload.position?.size ? `${preparedPayload.position.size.toFixed(2)} lots` : 'N/A'}
                  </span>
                </div>
                
                <button
                  onClick={executeIdea}
                  disabled={running || preparedPayload.risk.throttle}
                  className="w-full bg-rose-600/80 hover:bg-rose-600 disabled:opacity-60 text-rose-50 font-semibold py-2 px-4 rounded border border-rose-400/30"
                >
                  {running ? "Executing..." : "🚀 EXECUTE LIVE ORDER"}
                </button>
              </div>
            )}
          </div>

          {/* Manual Trading */}
          <div className={`${HUD.PANEL} rounded-lg p-4`}>
            <h3 className={`${HUD.TITLE} mb-3`}>📊 Manual Trading</h3>
            
            <div className="space-y-3">
              <div>
                <label htmlFor="ti-symbol" className="block text-sm text-cyan-300/80 mb-1">Symbol</label>
                <select
                  id="ti-symbol"
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value)}
                  className="w-full bg-slate-900/60 text-cyan-100 placeholder-cyan-300/40 border border-cyan-400/20 rounded px-3 py-2 focus:outline-none focus:ring-0 focus:border-cyan-400/40"
                >
                  <option value="EURUSD">EURUSD</option>
                  <option value="GBPUSD">GBPUSD</option>
                  <option value="USDJPY">USDJPY</option>
                  <option value="AUDUSD">AUDUSD</option>
                  <option value="USDCAD">USDCAD</option>
                </select>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setSide("buy")}
                  className={`py-2 px-4 rounded font-semibold ${
                    side === "buy"
                      ? "bg-emerald-600/30 text-emerald-200 border border-emerald-400/40 shadow-[0_0_8px_rgba(16,185,129,0.35)]"
                      : "bg-slate-900/40 text-cyan-300 border border-cyan-400/20 hover:bg-slate-900/60"
                  }`}
                >
                  BUY
                </button>
                <button
                  onClick={() => setSide("sell")}
                  className={`py-2 px-4 rounded font-semibold ${
                    side === "sell"
                      ? "bg-rose-600/30 text-rose-200 border border-rose-400/40 shadow-[0_0_8px_rgba(244,63,94,0.35)]"
                      : "bg-slate-900/40 text-cyan-300 border border-cyan-400/20 hover:bg-slate-900/60"
                  }`}
                >
                  SELL
                </button>
              </div>

              <div>
                <label htmlFor="ti-risk" className="block text-sm text-cyan-300/80 mb-1">Risk %</label>
                <input
                  id="ti-risk"
                  type="number"
                  value={risk}
                  onChange={(e) => setRisk(parseFloat(e.target.value))}
                  className="w-full bg-slate-900/60 text-cyan-100 placeholder-cyan-300/40 border border-cyan-400/20 rounded px-3 py-2 focus:outline-none focus:ring-0 focus:border-cyan-400/40"
                  step="0.1"
                  min="0.1"
                  max="5"
                />
              </div>

              <div>
                <label htmlFor="ti-sl" className="block text-sm text-cyan-300/80 mb-1">Stop Loss</label>
                <input
                  id="ti-sl"
                  type="number"
                  value={sl}
                  onChange={(e) => setSl(e.target.value)}
                  className="w-full bg-slate-900/60 text-cyan-100 placeholder-cyan-300/40 border border-cyan-400/20 rounded px-3 py-2 focus:outline-none focus:ring-0 focus:border-cyan-400/40"
                  step="0.00001"
                  placeholder="0.00000"
                />
              </div>

              <div>
                <label htmlFor="ti-tp" className="block text-sm text-cyan-300/80 mb-1">Take Profit</label>
                <input
                  id="ti-tp"
                  type="number"
                  value={tp}
                  onChange={(e) => setTp(e.target.value)}
                  className="w-full bg-slate-900/60 text-cyan-100 placeholder-cyan-300/40 border border-cyan-400/20 rounded px-3 py-2 focus:outline-none focus:ring-0 focus:border-cyan-400/40"
                  step="0.00001"
                  placeholder="0.00000"
                />
              </div>

              <div>
                <label htmlFor="ti-volume" className="block text-sm text-cyan-300/80 mb-1">Position Size (lots)</label>
                <input
                  id="ti-volume"
                  type="number"
                  value={volume}
                  onChange={(e) => setVolume(parseFloat(e.target.value))}
                  className="w-full bg-slate-900/60 text-cyan-100 placeholder-cyan-300/40 border border-cyan-400/20 rounded px-3 py-2 focus:outline-none focus:ring-0 focus:border-cyan-400/40"
                  step="0.01"
                  min="0.01"
                />
              </div>

              <button
                onClick={() => placeOrder(false)}
                disabled={running}
                className="w-full bg-amber-500/80 hover:bg-amber-500 disabled:opacity-60 text-amber-50 font-semibold py-2 px-4 rounded border border-amber-400/30"
              >
                {running ? "Placing Order..." : "🚀 PLACE LIVE ORDER"}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Real-time Data Panel */}
      <div className="lg:w-2/3">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">
          {/* Live Signals */}
          <div className={`${HUD.CARD} rounded-xl p-4`}>
            <h3 className="text-cyan-200 font-semibold mb-3">📡 Live AI Signals</h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {liveSignals.map((signal, index) => (
                <div key={index} className={`${HUD.PANEL} rounded p-2`}>
                  <div className="text-sm text-cyan-100 font-semibold">
                    {signal.model} {signal.side} {signal.symbol}
                  </div>
                  <div className="text-xs text-cyan-300/70">
                    Strength: {signal.strength?.toFixed(3)} | Confidence: {signal.confidence?.toFixed(1)}%
                  </div>
                </div>
              ))}
              {liveSignals.length === 0 && (
                <div className="text-cyan-300/60 text-sm">Waiting for live signals...</div>
              )}
            </div>
          </div>

          {/* SMC Ideas */}
          <div className={`${HUD.CARD} rounded-xl p-4`}>
            <h3 className="text-cyan-200 font-semibold mb-3">🎯 SMC Trading Ideas</h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {smcIdeas.map((idea, index) => (
                <div key={index} className={`${HUD.PANEL} rounded p-2`}>
                  <div className="text-sm text-cyan-100 font-semibold">
                    {idea.type} {idea.symbol} on {idea.timeframe}
                  </div>
                  <div className="text-xs text-cyan-300/70">
                    Confidence: {idea.confidence?.toFixed(1)}%
                  </div>
                </div>
              ))}
              {smcIdeas.length === 0 && (
                <div className="text-cyan-300/60 text-sm">Waiting for SMC ideas...</div>
              )}
            </div>
          </div>

          {/* Order Updates */}
          <div className={`${HUD.CARD} rounded-xl p-4`}>
            <h3 className="text-cyan-200 font-semibold mb-3">📋 Live Order Updates</h3>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {orderUpdates.map((update, index) => (
                <div key={index} className={`${HUD.PANEL} rounded p-2`}>
                  <div className="text-sm text-cyan-100 font-semibold">
                    {update.status} {update.side} {update.symbol}
                  </div>
                  <div className="text-xs text-cyan-300/70">
                    {update.volume} lots @ {update.price?.toFixed(5)}
                  </div>
                </div>
              ))}
              {orderUpdates.length === 0 && (
                <div className="text-cyan-300/60 text-sm">Waiting for order updates...</div>
              )}
            </div>
          </div>

          {/* Bias Engine Details */}
          <div className={`${HUD.CARD} rounded-xl p-4`}>
            <h3 className="text-cyan-200 font-semibold mb-3">🧠 Bias Engine Details</h3>
            {preparedPayload ? (
              <div className="space-y-2">
                <div className="text-sm text-cyan-100/90">
                  <strong>Idea Confidence:</strong> {(preparedPayload.confidence * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-cyan-100/90">
                  <strong>Entry:</strong> {preparedPayload.entry?.toFixed(5)}
                </div>
                <div className="text-sm text-cyan-100/90">
                  <strong>Stop:</strong> {preparedPayload.stop?.toFixed(5)}
                </div>
                <div className="text-sm text-cyan-100/90">
                  <strong>Take Profit:</strong> {preparedPayload.takeprofit?.toFixed(5)}
                </div>
                <div className="text-xs text-cyan-300/60 mt-2">
                  <strong>Bias Reasons:</strong>
                  {Object.entries(preparedPayload.risk.reasons).map(([key, value]) => (
                    <div key={key} className="ml-2">
                      {key}: {(value * 100).toFixed(1)}%
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-cyan-300/60 text-sm">Run Bias Engine to see details</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TradingInterface;
