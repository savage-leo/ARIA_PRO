// frontend/src/components/interfaces/SettingsTab.tsx
import React, { useCallback, useEffect, useState, useRef } from "react";
import { HUD } from "../../theme/hud";
// import { apiGet, apiPost } from "../../services/api";
import BackendControlPanel from "./BackendControlPanel";

const SettingsTab: React.FC = () => {
  const [rtt] = useState<number | null>(null);
  const [lastPong] = useState<string | null>(null);
  const pingRef = useRef<number | null>(null);

  const env = import.meta.env as { VITE_BACKEND_BASE?: string; VITE_BACKEND_WS?: string };
  const baseHttp = (env.VITE_BACKEND_BASE || "").replace(/\/$/, "");

  type HealthStatus = "idle" | "loading" | "ok" | "fail";
  const [health, setHealth] = useState<{
    equity: HealthStatus;
    heatmap: HealthStatus;
    perf: HealthStatus;
    lastChecked?: string;
  }>({ equity: "idle", heatmap: "idle", perf: "idle" });

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
    setHealth((h) => ({ ...h, equity: "loading", heatmap: "loading", perf: "loading" }));
    const [eok, hok, pok] = await Promise.all([
      checkEndpoint(`/api/analytics/equity-curve`),
      checkEndpoint(`/api/analytics/trading-heatmap`),
      checkEndpoint(`/api/analytics/performance-metrics`),
    ]);
    setHealth({
      equity: eok ? "ok" : "fail",
      heatmap: hok ? "ok" : "fail",
      perf: pok ? "ok" : "fail",
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

  useEffect(() => {
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className={`p-4 ${HUD.BG} min-h-full`}>
      <h2 className="text-xl font-semibold text-cyan-200 drop-shadow mb-4">Settings — Connectivity</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className={`${HUD.CARD} rounded-xl p-4`}>
          <div className="text-cyan-300 font-semibold mb-2">Backend Configuration</div>
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

        <div className={`${HUD.CARD} rounded-xl p-4 md:col-span-2`}>
          <div className="flex items-center justify-between mb-2">
            <div className="text-cyan-300 font-semibold">Analytics API Health</div>
            <button onClick={checkAll} className="px-3 py-1 rounded bg-cyan-600/70 hover:bg-cyan-600 text-cyan-50 border border-cyan-400/30">Check</button>
          </div>
          <div className="text-sm text-cyan-100/90 grid grid-cols-1 md:grid-cols-3 gap-2">
            <HealthRow label="/api/analytics/equity-curve" status={health.equity} />
            <HealthRow label="/api/analytics/trading-heatmap" status={health.heatmap} />
            <HealthRow label="/api/analytics/performance-metrics" status={health.perf} />
          </div>
          <div className="text-xs text-cyan-300/60 mt-2">Last checked: {health.lastChecked || "—"}</div>
        </div>
      </div>

      {/* Backend Control Panel */}
      <div className="mt-4">
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
