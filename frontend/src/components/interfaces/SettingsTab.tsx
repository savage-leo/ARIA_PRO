// frontend/src/components/interfaces/SettingsTab.tsx
import React, { useCallback, useEffect, useState, useRef } from "react";
import { HUD } from "../../theme/hud";
import { apiGet, apiPost } from "../../services/api";
import BackendControlPanel from "./BackendControlPanel";
import HotSwapAdminWidget from "./HotSwapAdminWidget";

interface ConfigSettings {
  MT5_LOGIN?: string;
  MT5_PASSWORD?: string;
  MT5_SERVER?: string;
  AUTO_TRADE_ENABLED?: boolean;
  AUTO_TRADE_DRY_RUN?: boolean;
  AUTO_TRADE_SYMBOLS?: string;
  ARIA_INCLUDE_XGB?: boolean;
  ARIA_ENABLE_EXEC?: boolean;
  ALLOW_LIVE?: boolean;
  RISK_BASE_PCT?: number;
  ARIA_SYMBOLS?: string;
  ARIA_USE_HMM?: boolean;
  ARIA_HMM_ALGO?: string;
  ARIA_ENABLE_CPP_SMC?: string;
}

const SettingsTab: React.FC = () => {
  const [rtt] = useState<number | null>(null);
  const [lastPong] = useState<string | null>(null);
  const [config, setConfig] = useState<ConfigSettings>({});
  const [loading, setLoading] = useState(false);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const pingRef = useRef<number | null>(null);

  const env = import.meta.env as { VITE_BACKEND_BASE?: string; VITE_BACKEND_WS?: string };
  const baseHttp = (env.VITE_BACKEND_BASE || "").replace(/\/$/, "");

  type HealthStatus = "idle" | "loading" | "ok" | "fail";
  const [health, setHealth] = useState<{
    equity: HealthStatus;
    heatmap: HealthStatus;
    perf: HealthStatus;
    models: HealthStatus;
    mt5: HealthStatus;
    lastChecked?: string;
  }>({ equity: "idle", heatmap: "idle", perf: "idle", models: "idle", mt5: "idle" });

  const fetchConfig = async () => {
    setLoading(true);
    try {
      const response = await apiGet<ConfigSettings>("/api/system/config");
      if (response) {
        setConfig(response);
      }
    } catch (error) {
      console.error("Failed to fetch config:", error);
    } finally {
      setLoading(false);
    }
  };

  const saveConfig = async () => {
    setSaveStatus('saving');
    try {
      await apiPost("/api/system/config", config);
      setSaveStatus('success');
      setTimeout(() => setSaveStatus('idle'), 2000);
    } catch (error) {
      console.error("Failed to save config:", error);
      setSaveStatus('error');
      setTimeout(() => setSaveStatus('idle'), 2000);
    }
  };

  const checkEndpoint = async (path: string): Promise<boolean> => {
    try {
      const res = await fetch(`${baseHttp}${path}`);
      const json = await res.json().catch(() => ({}));
      if (res.ok && (json?.ok === undefined || json?.ok === true)) return true;
      return false;
    } catch {
      return false;
    }
  };

  const checkAll = async () => {
    setHealth((h) => ({ ...h, equity: "loading", heatmap: "loading", perf: "loading", models: "loading", mt5: "loading" }));
    const [eok, hok, pok, mok, mt5ok] = await Promise.all([
      checkEndpoint(`/api/analytics/equity-curve`),
      checkEndpoint(`/api/analytics/trading-heatmap`),
      checkEndpoint(`/api/analytics/performance-metrics`),
      checkEndpoint(`/api/monitoring/models/status`),
      checkEndpoint(`/api/account/info`),
    ]);
    setHealth({
      equity: eok ? "ok" : "fail",
      heatmap: hok ? "ok" : "fail",
      perf: pok ? "ok" : "fail",
      models: mok ? "ok" : "fail",
      mt5: mt5ok ? "ok" : "fail",
      lastChecked: new Date().toLocaleTimeString(),
    });
  };

  const statusColor = useCallback(
    (connected: boolean) => (connected ? "text-emerald-400" : "text-rose-400"),
    []
  );

  const doPing = useCallback(() => {
    pingRef.current = Date.now();
    // sendMessage({ type: "ping" });
  }, []);

  const updateConfig = (key: keyof ConfigSettings, value: any) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  useEffect(() => {
    fetchConfig();
    checkAll();
  }, []);

  return (
    <div className={`p-4 ${HUD.BG} min-h-full`}>
      <h2 className="text-xl font-semibold text-cyan-200 drop-shadow mb-4">🔧 System Settings & Configuration</h2>

      {/* Configuration Panel */}
      <div className={`${HUD.CARD} rounded-xl p-4 mb-4`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-cyan-200 font-semibold">Backend Configuration</h3>
          <div className="flex gap-2">
            <button
              onClick={fetchConfig}
              disabled={loading}
              className={`${HUD.TAB} border border-cyan-400/20 text-xs`}
            >
              {loading ? "Loading..." : "Reload"}
            </button>
            <button
              onClick={saveConfig}
              disabled={saveStatus === 'saving'}
              className={`px-3 py-1 text-xs rounded border ${
                saveStatus === 'success' ? 'bg-emerald-600/70 border-emerald-400/30' :
                saveStatus === 'error' ? 'bg-rose-600/70 border-rose-400/30' :
                'bg-cyan-600/70 hover:bg-cyan-600 border-cyan-400/30'
              }`}
            >
              {saveStatus === 'saving' ? 'Saving...' : 
               saveStatus === 'success' ? 'Saved!' :
               saveStatus === 'error' ? 'Error!' : 'Save Config'}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* MT5 Configuration */}
          <div className={`${HUD.PANEL} p-3 rounded-lg`}>
            <h4 className="text-cyan-300 font-medium mb-2">MT5 Connection</h4>
            <div className="space-y-2">
              <div>
                <label className="block text-xs text-cyan-300/70 mb-1">Login</label>
                <input
                  type="text"
                  value={config.MT5_LOGIN || ''}
                  onChange={(e) => updateConfig('MT5_LOGIN', e.target.value)}
                  className="w-full bg-slate-900/60 text-cyan-100 text-xs border border-cyan-400/20 rounded px-2 py-1"
                  placeholder="MT5 Login"
                />
              </div>
              <div>
                <label className="block text-xs text-cyan-300/70 mb-1">Server</label>
                <input
                  type="text"
                  value={config.MT5_SERVER || ''}
                  onChange={(e) => updateConfig('MT5_SERVER', e.target.value)}
                  className="w-full bg-slate-900/60 text-cyan-100 text-xs border border-cyan-400/20 rounded px-2 py-1"
                  placeholder="FBS-Demo"
                />
              </div>
            </div>
          </div>

          {/* Trading Configuration */}
          <div className={`${HUD.PANEL} p-3 rounded-lg`}>
            <h4 className="text-cyan-300 font-medium mb-2">Trading Settings</h4>
            <div className="space-y-2">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.AUTO_TRADE_ENABLED || false}
                  onChange={(e) => updateConfig('AUTO_TRADE_ENABLED', e.target.checked)}
                  className="rounded border-cyan-400/20"
                />
                <span className="text-xs text-cyan-100">Auto Trading Enabled</span>
              </label>
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.AUTO_TRADE_DRY_RUN || false}
                  onChange={(e) => updateConfig('AUTO_TRADE_DRY_RUN', e.target.checked)}
                  className="rounded border-cyan-400/20"
                />
                <span className="text-xs text-cyan-100">Dry Run Mode</span>
              </label>
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.ALLOW_LIVE || false}
                  onChange={(e) => updateConfig('ALLOW_LIVE', e.target.checked)}
                  className="rounded border-cyan-400/20"
                />
                <span className="text-xs text-cyan-100">Allow Live Trading</span>
              </label>
              <div>
                <label className="block text-xs text-cyan-300/70 mb-1">Base Risk %</label>
                <input
                  type="number"
                  value={config.RISK_BASE_PCT || 0.5}
                  onChange={(e) => updateConfig('RISK_BASE_PCT', parseFloat(e.target.value))}
                  className="w-full bg-slate-900/60 text-cyan-100 text-xs border border-cyan-400/20 rounded px-2 py-1"
                  step="0.1"
                  min="0.1"
                  max="5"
                />
              </div>
            </div>
          </div>

          {/* AI Model Configuration */}
          <div className={`${HUD.PANEL} p-3 rounded-lg`}>
            <h4 className="text-cyan-300 font-medium mb-2">AI Models</h4>
            <div className="space-y-2">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.ARIA_INCLUDE_XGB || false}
                  onChange={(e) => updateConfig('ARIA_INCLUDE_XGB', e.target.checked)}
                  className="rounded border-cyan-400/20"
                />
                <span className="text-xs text-cyan-100">XGBoost Model</span>
              </label>
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.ARIA_USE_HMM || false}
                  onChange={(e) => updateConfig('ARIA_USE_HMM', e.target.checked)}
                  className="rounded border-cyan-400/20"
                />
                <span className="text-xs text-cyan-100">HMM Regime Detection</span>
              </label>
              <div>
                <label className="block text-xs text-cyan-300/70 mb-1">HMM Algorithm</label>
                <select
                  value={config.ARIA_HMM_ALGO || 'viterbi'}
                  onChange={(e) => updateConfig('ARIA_HMM_ALGO', e.target.value)}
                  className="w-full bg-slate-900/60 text-cyan-100 text-xs border border-cyan-400/20 rounded px-2 py-1"
                >
                  <option value="viterbi">Viterbi</option>
                  <option value="forward">Forward</option>
                  <option value="map">MAP</option>
                </select>
              </div>
            </div>
          </div>

          {/* Symbol Configuration */}
          <div className={`${HUD.PANEL} p-3 rounded-lg md:col-span-2`}>
            <h4 className="text-cyan-300 font-medium mb-2">Trading Symbols</h4>
            <div className="space-y-2">
              <div>
                <label className="block text-xs text-cyan-300/70 mb-1">Auto Trade Symbols</label>
                <input
                  type="text"
                  value={config.AUTO_TRADE_SYMBOLS || ''}
                  onChange={(e) => updateConfig('AUTO_TRADE_SYMBOLS', e.target.value)}
                  className="w-full bg-slate-900/60 text-cyan-100 text-xs border border-cyan-400/20 rounded px-2 py-1"
                  placeholder="EURUSD,GBPUSD,USDJPY,XAUUSD,BTCUSD"
                />
              </div>
              <div>
                <label className="block text-xs text-cyan-300/70 mb-1">ARIA Symbols</label>
                <input
                  type="text"
                  value={config.ARIA_SYMBOLS || ''}
                  onChange={(e) => updateConfig('ARIA_SYMBOLS', e.target.value)}
                  className="w-full bg-slate-900/60 text-cyan-100 text-xs border border-cyan-400/20 rounded px-2 py-1"
                  placeholder="EURUSD,GBPUSD,USDJPY"
                />
              </div>
            </div>
          </div>

          {/* Advanced Settings */}
          <div className={`${HUD.PANEL} p-3 rounded-lg`}>
            <h4 className="text-cyan-300 font-medium mb-2">Advanced</h4>
            <div className="space-y-2">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.ARIA_ENABLE_EXEC || false}
                  onChange={(e) => updateConfig('ARIA_ENABLE_EXEC', e.target.checked)}
                  className="rounded border-cyan-400/20"
                />
                <span className="text-xs text-cyan-100">Enable Execution</span>
              </label>
              <div>
                <label className="block text-xs text-cyan-300/70 mb-1">C++ SMC</label>
                <select
                  value={config.ARIA_ENABLE_CPP_SMC || 'auto'}
                  onChange={(e) => updateConfig('ARIA_ENABLE_CPP_SMC', e.target.value)}
                  className="w-full bg-slate-900/60 text-cyan-100 text-xs border border-cyan-400/20 rounded px-2 py-1"
                >
                  <option value="auto">Auto</option>
                  <option value="1">Enabled</option>
                  <option value="0">Disabled</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Health Check Panel */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div className={`${HUD.CARD} rounded-xl p-4`}>
          <div className="text-cyan-300 font-semibold mb-2">Environment</div>
          <div className="text-sm text-cyan-100/90">
            <div className="mb-1"><span className="text-cyan-300/70">VITE_BACKEND_BASE:</span> {env?.VITE_BACKEND_BASE || "(relative, via dev proxy)"}</div>
            <div className="mb-3"><span className="text-cyan-300/70">VITE_BACKEND_WS:</span> {env?.VITE_BACKEND_WS || "(derived from base or same-origin /ws)"}</div>
            <div className="text-xs text-cyan-300/60">Change environment variables in your Vite config or .env files and reload.</div>
          </div>
        </div>

        <div className={`${HUD.CARD} rounded-xl p-4`}>
          <div className="flex items-center justify-between mb-2">
            <div className="text-cyan-300 font-semibold">WebSocket Health</div>
            <div className={`text-xs ${statusColor(false)}`}>WS Disconnected</div>
          </div>
          <div className="text-sm text-cyan-100/90 space-y-1">
            <div>Last RTT: {rtt != null ? `${rtt} ms` : "—"}</div>
            <div>Last Pong: {lastPong || "—"}</div>
          </div>
          <button onClick={doPing} className="mt-3 px-3 py-1 rounded bg-cyan-600/70 hover:bg-cyan-600 text-cyan-50 border border-cyan-400/30">Send Ping</button>
        </div>
      </div>

      <div className={`${HUD.CARD} rounded-xl p-4 mb-4`}>
        <div className="flex items-center justify-between mb-2">
          <div className="text-cyan-300 font-semibold">System Health Check</div>
          <button onClick={checkAll} className="px-3 py-1 rounded bg-cyan-600/70 hover:bg-cyan-600 text-cyan-50 border border-cyan-400/30">Check All</button>
        </div>
        <div className="text-sm text-cyan-100/90 grid grid-cols-1 md:grid-cols-5 gap-2">
          <HealthRow label="MT5 Connection" status={health.mt5} />
          <HealthRow label="AI Models" status={health.models} />
          <HealthRow label="Equity Analytics" status={health.equity} />
          <HealthRow label="Trading Heatmap" status={health.heatmap} />
          <HealthRow label="Performance Metrics" status={health.perf} />
        </div>
        <div className="text-xs text-cyan-300/60 mt-2">Last checked: {health.lastChecked || "—"}</div>
      </div>

      {/* Hot-Swap Admin Widget */}
      <div className="mb-4">
        <HotSwapAdminWidget />
      </div>

      {/* Backend Control Panel */}
      <div>
        <BackendControlPanel />
      </div>
    </div>
  );
};

const HealthRow: React.FC<{ label: string; status: "idle" | "loading" | "ok" | "fail" }> = ({ label, status }) => {
  const pill =
    status === "ok"
      ? "bg-emerald-500/80 text-emerald-50 border border-emerald-400/30"
      : status === "fail"
      ? "bg-rose-600/80 text-rose-50 border border-rose-400/30"
      : status === "loading"
      ? "bg-cyan-700/70 text-cyan-50 border border-cyan-400/30 animate-pulse"
      : "bg-slate-700/70 text-slate-100 border border-slate-400/20";
  return (
    <div className="flex items-center justify-between rounded px-3 py-2 bg-slate-900/40 border border-cyan-400/10">
      <div className="truncate pr-2">{label}</div>
      <div className={`text-xs px-2 py-0.5 rounded ${pill}`}>{status.toUpperCase()}</div>
    </div>
  );
};

export default SettingsTab;
