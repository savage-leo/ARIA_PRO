// frontend/src/components/interfaces/SMCTab.tsx
import React, { useEffect, useMemo, useState } from "react";
import { useSMCStore } from "../../stores/useSMCStore";
import AISecretPanel from "./AISecretPanel";
import { apiGet } from "../../services/api";
import { HUD } from "../../theme/hud";

const SMCTab: React.FC = () => {
  const { current, history, setCurrent, addToHistory, clear } = useSMCStore();
  const [symbol, setSymbol] = useState("EURUSD");

  type SmcSignal = {
    ts: number;
    symbol?: string;
    bias?: "bullish" | "bearish" | "neutral" | string;
    confidence?: number;
    [k: string]: unknown;
  };

  type CppStatus = {
    ok: boolean;
    cpp_available?: boolean;
    market_processor?: boolean;
    smc_engine?: boolean;
    error?: string;
  };

  const [serverHistory, setServerHistory] = useState<SmcSignal[]>([]);
  const [orderBlocks, setOrderBlocks] = useState<Record<string, unknown>[]>([]);
  const [fairValueGaps, setFairValueGaps] = useState<Record<string, unknown>[]>([]);
  const [cppStatus, setCppStatus] = useState<CppStatus | null>(null);
  const [loading, setLoading] = useState(false);

  const biasColor = useMemo(() => {
    const b = (current?.bias || "").toString().toLowerCase();
    if (b.includes("bull")) return "bg-emerald-500/20 border-emerald-400/30 text-emerald-200";
    if (b.includes("bear")) return "bg-rose-500/20 border-rose-400/30 text-rose-200";
    return "bg-amber-500/20 border-amber-400/30 text-amber-200";
  }, [current?.bias]);

  const fetchCurrent = async () => {
    try {
      const res = await apiGet<{ ok: boolean; signal?: SmcSignal; msg?: string }>(
        `/api/smc/current/${symbol}`
      );
      if ((res as any)?.ok && (res as any).signal) {
        const sig = (res as any).signal as SmcSignal;
        setCurrent(sig);
        addToHistory(sig);
      }
    } catch (e) {
      // noop visual; handled by user action when needed
    }
  };

  const fetchHistory = async () => {
    try {
      const res = await apiGet<{ ok: boolean; history: SmcSignal[] }>(
        `/api/smc/history?limit=20`
      );
      if (res?.ok && Array.isArray(res.history)) setServerHistory(res.history);
    } catch {}
  };

  const fetchStructures = async () => {
    try {
      const ob = await apiGet<{ ok: boolean; order_blocks: Record<string, unknown>[] }>(
        `/api/smc/order-blocks/${symbol}`
      );
      if (ob?.ok) setOrderBlocks(ob.order_blocks || []);
    } catch {}
    try {
      const fvg = await apiGet<{ ok: boolean; fair_value_gaps: Record<string, unknown>[] }>(
        `/api/smc/fair-value-gaps/${symbol}`
      );
      if (fvg?.ok) setFairValueGaps(fvg.fair_value_gaps || []);
    } catch {}
  };

  const fetchCppStatus = async () => {
    try {
      const st = await apiGet<CppStatus>(`/api/smc/cpp/status`);
      setCppStatus(st);
    } catch {
      setCppStatus({ ok: false, error: "status unavailable" });
    }
  };

  const exportCurrent = () => {
    if (!current) return;
    const blob = new Blob([JSON.stringify(current, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `smc_${current.symbol || symbol}_${current.ts}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  useEffect(() => {
    // initial load and when symbol changes
    setLoading(true);
    Promise.all([fetchCurrent(), fetchHistory(), fetchStructures(), fetchCppStatus()])
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol]);

  return (
    <div className="p-4 space-y-4">
      <h2 className={`text-xl mb-2 ${HUD.TEXT}`}>SMC Analysis</h2>

      <div className="grid grid-cols-1 2xl:grid-cols-3 gap-4">
        <div className="2xl:col-span-2">
          <AISecretPanel />
        </div>

        <div className={`p-4 rounded-xl ${HUD.CARD} ${HUD.TEXT}`}>
          <div className="flex items-end gap-3">
            <div className="flex-1">
              <label className={`${HUD.TITLE}`}>Symbol</label>
              <input
                value={symbol}
                onChange={(e) => setSymbol(e.target.value)}
                className="w-full px-2 py-1 rounded-md bg-slate-900/60 border border-cyan-400/20 text-cyan-100 focus:outline-none focus:ring-2 focus:ring-cyan-500/40"
              />
            </div>
            <button
              onClick={() => { setLoading(true); Promise.all([fetchCurrent(), fetchStructures(), fetchHistory()]).finally(()=>setLoading(false)); }}
              className="px-3 py-2 rounded-md border border-cyan-400/20 bg-blue-500/20 text-cyan-100 hover:bg-blue-500/30"
            >{loading ? "..." : "Refresh Snapshot"}</button>
          </div>

          <div className="mt-4 grid grid-cols-2 gap-3">
            <div className={`px-3 py-2 rounded-md border ${biasColor}`}>Bias: {current?.bias ?? "n/a"}</div>
            <div className={`px-3 py-2 rounded-md border ${HUD.PANEL} border-cyan-400/10`}>CPP: {cppStatus?.cpp_available ? "Available" : "Unavailable"}</div>
            <div className={`px-3 py-2 rounded-md border ${HUD.PANEL} border-cyan-400/10`}>Market Proc: {cppStatus?.market_processor ? "OK" : "-"}</div>
            <div className={`px-3 py-2 rounded-md border ${HUD.PANEL} border-cyan-400/10`}>SMC Engine: {cppStatus?.smc_engine ? "OK" : "-"}</div>
          </div>

          <div className="mt-4">
            <div className={`${HUD.TITLE} mb-1`}>Confidence</div>
            <div className="w-full h-2 bg-slate-800 rounded">
              <div
                className="h-2 rounded bg-gradient-to-r from-cyan-500/60 to-blue-500/60 shadow-[0_0_8px_rgba(34,211,238,0.35)]"
                style={{ width: `${Math.min(100, Math.max(0, Math.round((current?.confidence ?? 0) * 100)))}%` }}
              />
            </div>
            <div className="text-xs text-cyan-300/70 mt-1">{Math.round((current?.confidence ?? 0) * 100)}%</div>
          </div>

          <div className="mt-3 flex gap-2">
            <button onClick={exportCurrent} className="px-3 py-2 rounded-md border border-cyan-400/20 bg-emerald-500/20 text-cyan-100 hover:bg-emerald-500/30">Export Current JSON</button>
            <button onClick={() => clear()} className="px-3 py-2 rounded-md border border-cyan-400/20 bg-rose-500/20 text-cyan-100 hover:bg-rose-500/30">Clear Store</button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <div className={`p-4 rounded-xl ${HUD.CARD} ${HUD.TEXT}`}>
          <h3 className="text-lg mb-2">Current Signal</h3>
          {current ? (
            <pre className="text-xs overflow-auto max-h-64">{JSON.stringify(current, null, 2)}</pre>
          ) : (
            <div className="text-cyan-300/70">No current signal</div>
          )}
        </div>

        <div className={`p-4 rounded-xl ${HUD.CARD} ${HUD.TEXT}`}>
          <h3 className="text-lg mb-2">Recent History (server)</h3>
          <div className="space-y-2 max-h-64 overflow-auto">
            {serverHistory?.length ? serverHistory.map((s, i) => (
              <div key={i} className={`px-3 py-2 rounded-md ${HUD.PANEL}`}>
                <div className="text-sm">{s.symbol} — {s.bias} ({Math.round((s.confidence ?? 0)*100)}%)</div>
                <div className="text-[10px] text-cyan-300/60">{new Date(s.ts).toLocaleString()}</div>
              </div>
            )) : <div className="text-cyan-300/70">No history</div>}
          </div>
        </div>

        <div className={`p-4 rounded-xl ${HUD.CARD} ${HUD.TEXT}`}>
          <h3 className="text-lg mb-2">Structures</h3>
          <div className="mb-2 text-sm text-cyan-300/80">Order Blocks: {orderBlocks.length} · FVG: {fairValueGaps.length}</div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-64 overflow-auto">
            <div>
              <div className={`${HUD.TITLE} mb-1`}>Order Blocks</div>
              <div className="space-y-2">
                {orderBlocks.slice(0, 8).map((ob, i) => (
                  <div key={i} className={`px-3 py-2 rounded-md ${HUD.PANEL}`}>
                    <pre className="text-[10px] whitespace-pre-wrap">{JSON.stringify(ob)}</pre>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <div className={`${HUD.TITLE} mb-1`}>Fair Value Gaps</div>
              <div className="space-y-2">
                {fairValueGaps.slice(0, 8).map((g, i) => (
                  <div key={i} className={`px-3 py-2 rounded-md ${HUD.PANEL}`}>
                    <pre className="text-[10px] whitespace-pre-wrap">{JSON.stringify(g)}</pre>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className={`p-4 rounded-xl ${HUD.CARD} ${HUD.TEXT}`}>
        <h3 className="text-lg mb-2">Local Store History</h3>
        <div className="space-y-2 max-h-64 overflow-auto">
          {history.length > 0 ? (
            history.slice(-10).map((signal, i) => (
              <div key={i} className={`px-3 py-2 rounded-md ${HUD.PANEL}`}>
                {signal.symbol} - {signal.bias} ({Math.round((signal.confidence ?? 0) * 100)}%)
              </div>
            ))
          ) : (
            <div className="text-cyan-300/70">Empty</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SMCTab;
