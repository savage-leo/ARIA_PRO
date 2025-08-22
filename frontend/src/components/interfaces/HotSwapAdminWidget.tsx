// frontend/src/components/interfaces/HotSwapAdminWidget.tsx
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { HUD } from "../../theme/hud";
import { backendBase } from "../../services/api";
import { useWebSocket, WebSocketMessage } from "../../hooks/useWebSocket";

// Types for admin hot-swap responses
interface HotSwapHistoryItem {
  id?: string;
  model_type?: string;
  model_key?: string;
  path?: string;
  status?: string; // "success" | "failed" | string
  error?: string | null;
  started_at?: string;
  completed_at?: string;
  duration_ms?: number;
  version?: string;
  [key: string]: unknown;
}

interface HotSwapManagerState {
  auto_swap_enabled?: boolean;
  active_swaps?: number;
  swaps_total?: number;
  swaps_success?: number;
  swaps_failed?: number;
  last_swap_at?: string | number | null;
  swap_history?: HotSwapHistoryItem[];
  [key: string]: unknown;
}

interface HotSwapAdminData {
  auto_swap_enabled: boolean;
  watchdog_available: boolean;
  manager: HotSwapManagerState;
}

interface HotSwapAdminResponse {
  status: string;
  message?: string;
  data: HotSwapAdminData;
}

const ADMIN_KEY_LS = "ARIA_ADMIN_KEY";

const statusPill = (on: boolean) =>
  on
    ? "bg-emerald-600/70 text-emerald-50 border border-emerald-400/30"
    : "bg-rose-600/70 text-rose-50 border border-rose-400/30";

const labelPill = "text-xs px-2 py-0.5 rounded";

const jsonPretty = (obj: unknown) => {
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
};

const useAdminFetch = (adminKey: string) => {
  // Admin endpoints live under /admin, not /api. If backendBase ends with /api, strip it to get the root.
  const rawBase = (backendBase || "").replace(/\/$/, "");
  const base = rawBase.endsWith("/api") ? rawBase.slice(0, -4) : rawBase;
  const get = useCallback(
    async (path: string): Promise<HotSwapAdminResponse> => {
      const res = await fetch(`${base}${path}`, {
        method: "GET",
        credentials: "include",
        headers: {
          Accept: "application/json",
          "X-ADMIN-KEY": adminKey,
        },
      });
      if (!res.ok) {
        let msg = res.statusText;
        try {
          const j = await res.json();
          msg = j?.detail || j?.message || JSON.stringify(j);
        } catch {}
        throw new Error(msg);
      }
      return (await res.json()) as HotSwapAdminResponse;
    },
    [adminKey, base]
  );

  const post = useCallback(
    async (path: string): Promise<HotSwapAdminResponse> => {
      const res = await fetch(`${base}${path}`, {
        method: "POST",
        credentials: "include",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
          "X-ADMIN-KEY": adminKey,
        },
        body: "{}",
      });
      if (!res.ok) {
        let msg = res.statusText;
        try {
          const j = await res.json();
          msg = j?.detail || j?.message || JSON.stringify(j);
        } catch {}
        throw new Error(msg);
      }
      return (await res.json()) as HotSwapAdminResponse;
    },
    [adminKey, base]
  );

  return { get, post };
};

const HotSwapAdminWidget: React.FC = () => {
  const [adminKey, setAdminKey] = useState<string>(() => localStorage.getItem(ADMIN_KEY_LS) || "");
  const [savingKey, setSavingKey] = useState(false);
  const [savedAt, setSavedAt] = useState<string | null>(null);

  const { get, post } = useAdminFetch(adminKey);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<HotSwapAdminData | null>(null);
  const [showRaw, setShowRaw] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const pollRef = useRef<number | null>(null);

  // WebSocket: subscribe to potential hot-swap channels if the backend starts broadcasting
  const ws = useWebSocket({
    onMessage: (msg: WebSocketMessage) => {
      if (msg?.type && ["hot_swap", "hot_swap_status", "model_hot_swap", "model_swap"].includes(msg.type)) {
        // Refresh on plausible hot-swap-related event
        void loadStatus();
      }
    },
  });
  const { subscribe } = ws;
  useEffect(() => {
    // Subscribe after connection; harmless if backend ignores these
    try {
      subscribe(["hot_swap", "admin:hot_swap"]);
    } catch {}
  }, [subscribe]);

  const loadStatus = useCallback(async () => {
    if (!adminKey) {
      setError("Admin key required");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await get("/admin/hot_swap/status");
      setData(res.data);
    } catch (e) {
      setError((e as Error).message || "Failed to load status");
    } finally {
      setLoading(false);
    }
  }, [adminKey, get]);

  const doEnable = useCallback(async () => {
    if (!adminKey) return;
    setLoading(true);
    setError(null);
    try {
      const res = await post("/admin/hot_swap/enable");
      setData(res.data);
    } catch (e) {
      setError((e as Error).message || "Failed to enable");
    } finally {
      setLoading(false);
    }
  }, [adminKey, post]);

  const doDisable = useCallback(async () => {
    if (!adminKey) return;
    setLoading(true);
    setError(null);
    try {
      const res = await post("/admin/hot_swap/disable");
      setData(res.data);
    } catch (e) {
      setError((e as Error).message || "Failed to disable");
    } finally {
      setLoading(false);
    }
  }, [adminKey, post]);

  const saveKey = useCallback(() => {
    setSavingKey(true);
    try {
      localStorage.setItem(ADMIN_KEY_LS, adminKey);
      setSavedAt(new Date().toLocaleTimeString());
    } finally {
      setSavingKey(false);
    }
  }, [adminKey]);

  useEffect(() => {
    // Initial load
    void loadStatus();
  }, [loadStatus]);

  useEffect(() => {
    if (autoRefresh) {
      // Poll every 5s
      pollRef.current = window.setInterval(() => {
        void loadStatus();
      }, 5000) as unknown as number;
    } else if (pollRef.current) {
      // If disabled, clear any existing interval immediately
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
    // Always provide a cleanup to satisfy strict TS return expectations
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
      pollRef.current = null;
    };
  }, [autoRefresh, loadStatus]);

  const manager = data?.manager;
  const history = useMemo<HotSwapHistoryItem[]>(() => {
    const h = (manager?.swap_history || []) as HotSwapHistoryItem[];
    return Array.isArray(h) ? h.slice(-10).reverse() : [];
  }, [manager]);

  return (
    <div className={`${HUD.CARD} rounded-xl p-4`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-cyan-200 font-semibold">Model Hot-Swap Control</h3>
        <div className="flex items-center gap-2">
          <label className="text-xs text-cyan-300/70 flex items-center gap-1">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh
          </label>
          <button
            onClick={() => void loadStatus()}
            disabled={loading}
            className={`${HUD.TAB} border border-cyan-400/20 text-xs`}
          >
            {loading ? "Refreshing..." : "Refresh"}
          </button>
        </div>
      </div>

      {/* Admin Key */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
        <div className="md:col-span-2">
          <label className="block text-xs text-cyan-300/70 mb-1">Admin API Key (X-ADMIN-KEY)</label>
          <input
            type="password"
            value={adminKey}
            onChange={(e) => setAdminKey(e.target.value)}
            placeholder="Enter admin key"
            className="w-full rounded bg-slate-900/60 border border-cyan-400/20 text-cyan-100 px-3 py-2 outline-none focus:border-cyan-400/50"
          />
          <div className="flex items-center gap-2 mt-2">
            <button
              onClick={saveKey}
              disabled={savingKey}
              className="px-3 py-1 text-xs bg-cyan-600/70 hover:bg-cyan-600 rounded border border-cyan-400/30"
            >
              {savingKey ? "Saving..." : "Save Key"}
            </button>
            <div className="text-xs text-cyan-300/60">{savedAt ? `Saved at ${savedAt}` : "Not saved"}</div>
          </div>
        </div>
        <div className={`${HUD.CARD} rounded p-3`}>
          <div className="text-cyan-300 font-medium mb-2">Current Status</div>
          <div className="flex items-center justify-between mb-1">
            <div className="text-sm">Auto Swap</div>
            <div className={`${labelPill} ${statusPill(!!data?.auto_swap_enabled)}`}>
              {data?.auto_swap_enabled ? "ENABLED" : "DISABLED"}
            </div>
          </div>
          <div className="flex items-center justify-between">
            <div className="text-sm">Watchdog</div>
            <div className={`${labelPill} ${statusPill(!!data?.watchdog_available)}`}>
              {data?.watchdog_available ? "AVAILABLE" : "MISSING"}
            </div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-2 mb-4">
        <button
          onClick={() => void doEnable()}
          disabled={loading || !adminKey}
          className="px-3 py-1 text-sm bg-emerald-600/70 hover:bg-emerald-600 rounded border border-emerald-400/30"
        >
          Enable Auto Swap
        </button>
        <button
          onClick={() => void doDisable()}
          disabled={loading || !adminKey}
          className="px-3 py-1 text-sm bg-rose-600/70 hover:bg-rose-600 rounded border border-rose-400/30"
        >
          Disable Auto Swap
        </button>
        {error && <div className="text-xs text-rose-300 ml-2">{error}</div>}
      </div>

      {/* Manager snapshot */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className={`${HUD.CARD} rounded p-3 md:col-span-2`}>
          <div className="flex items-center justify-between mb-2">
            <div className="text-cyan-300 font-medium">Recent Swap History</div>
            <button
              onClick={() => setShowRaw((v) => !v)}
              className={`${HUD.TAB} border border-cyan-400/20 text-xs`}
            >
              {showRaw ? "Hide Raw" : "Show Raw"}
            </button>
          </div>
          {!showRaw ? (
            <div className="space-y-2">
              {history.length === 0 && (
                <div className="text-xs text-cyan-300/60">No recent swaps</div>
              )}
              {history.map((h, idx) => (
                <div key={idx} className="flex items-start justify-between rounded px-3 py-2 bg-slate-900/40 border border-cyan-400/10">
                  <div className="text-sm">
                    <div className="text-cyan-100">
                      {h.model_type || h.model_key || h.path || "Model"}
                    </div>
                    <div className="text-xs text-cyan-300/60">
                      {(h.started_at || h.completed_at) ? `${h.started_at || "?"} → ${h.completed_at || "?"}` : ""}
                    </div>
                    {h.error && (
                      <div className="text-xs text-rose-300 mt-1">{h.error}</div>
                    )}
                  </div>
                  <div className={`${labelPill} ${statusPill((h.status || "").toLowerCase() === "success")}`}>
                    {(h.status || "").toUpperCase() || "UNKNOWN"}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <pre className="text-xs text-cyan-100/90 bg-slate-900/60 rounded p-2 overflow-auto max-h-72 border border-cyan-400/10">
              {jsonPretty(manager)}
            </pre>
          )}
        </div>

        {/* Live events preview (WS) */}
        <div className={`${HUD.CARD} rounded p-3`}>
          <div className="text-cyan-300 font-medium mb-2">Live Events</div>
          <div className="text-xs text-cyan-300/60 mb-1">
            WS: {ws.isConnected ? <span className="text-emerald-300">Connected</span> : <span className="text-rose-300">Disconnected</span>}
          </div>
          <div className="space-y-1 max-h-72 overflow-auto">
            {ws.messageHistory
              .filter((m) => ["hot_swap", "hot_swap_status", "model_hot_swap", "model_swap"].includes(m.type))
              .slice(0, 20)
              .map((m, i) => (
                <div key={i} className="rounded px-2 py-1 bg-slate-900/40 border border-cyan-400/10">
                  <div className="text-xs text-cyan-100/90">{m.type} — {m.timestamp}</div>
                  {m.data && (
                    <pre className="text-[10px] text-cyan-300/80 whitespace-pre-wrap">{jsonPretty(m.data)}</pre>
                  )}
                </div>
              ))}
            {ws.messageHistory.filter((m) => ["hot_swap", "hot_swap_status", "model_hot_swap", "model_swap"].includes(m.type)).length === 0 && (
              <div className="text-xs text-cyan-300/60">No hot-swap events received</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default HotSwapAdminWidget;
