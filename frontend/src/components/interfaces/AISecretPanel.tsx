// frontend/src/components/interfaces/AISecretPanel.tsx
import React, { useState } from "react";
import { backendBase } from "../../services/api";
import { useSMCStore } from "../../stores/useSMCStore";
import { HUD } from "../../theme/hud";

const AISecretPanel: React.FC = () => {
  const [symbol, setSymbol] = useState("EURUSD");
  const [dryRun, setDryRun] = useState(true);
  const [loading, setLoading] = useState(false);
  type PreparedIdea = { [k: string]: unknown; dry_run?: boolean; symbol?: string };
  const [prepared, setPrepared] = useState<PreparedIdea | null>(null);
  const setCurrent = useSMCStore(s => s.setCurrent);
  // Institutional parameters
  const [baseRiskPct, setBaseRiskPct] = useState<number>(0.5);
  const [equity, setEquity] = useState<number>(10000);
  const [spread, setSpread] = useState<number>(0.00008);
  const [slippage, setSlippage] = useState<number>(0.00005);
  const [ofImbalance, setOfImbalance] = useState<number>(0.3);

  const prepare = async () => {
    setLoading(true);
    try {
      // Optional sanity check: make sure market endpoint responds
      await fetch(`${backendBase}/market/last_bar/${symbol}`).catch(() => null);

      const res = await fetch(`${backendBase}/api/smc/idea/prepare`, {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        credentials: "include",
        body: JSON.stringify({
          symbol,
          base_risk_pct: baseRiskPct,
          market_ctx: { spread, slippage, of_imbalance: ofImbalance },
          equity,
        })
      });
      const json = await res.json();
      if (!res.ok) throw new Error(JSON.stringify(json));
      const payload: PreparedIdea | null = (json.prepared_payload || json.idea) ?? null;
      if (payload) {
        // attach dry_run hint for execution step
        const withDryRun: PreparedIdea = { ...payload, dry_run: dryRun, symbol };
        setPrepared(withDryRun);
        // try to reflect any current idea in UI if present
        if (json.idea) setCurrent(json.idea);
      } else {
        setPrepared(null);
      }
    } catch (e:any) {
      alert("Prepare failed: " + String(e.message || e));
    } finally {
      setLoading(false);
    }
  };

  const execute = async () => {
    if (!prepared) return alert("No prepared payload");
    const adminKey = prompt("Admin key (required for execution):");
    if (!adminKey) return alert("admin key required");
    setLoading(true);
    try {
      const payloadToSend: PreparedIdea = { ...(prepared || {}) };
      if (typeof payloadToSend.dry_run !== "boolean") payloadToSend.dry_run = dryRun;
      const res = await fetch(`${backendBase}/api/smc/idea/execute`, {
        method: "POST",
        headers: {"Content-Type":"application/json", "X-ADMIN-KEY": adminKey},
        credentials: "include",
        body: JSON.stringify(payloadToSend)
      });
      const json = await res.json();
      if (!res.ok) throw new Error(JSON.stringify(json));
      alert("Execution result: " + JSON.stringify(json));
    } catch (e:any) {
      alert("Execution failed: " + String(e.message || e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`p-4 rounded-xl ${HUD.CARD} ${HUD.TEXT}`}>
      <h3 className="text-lg font-semibold mb-2">ARIA Secret Edge — SMC + Trap</h3>
      <div className="mt-2 grid grid-cols-1 md:grid-cols-3 gap-3">
        <div>
          <label htmlFor="ai-symbol" className={`${HUD.TITLE}`}>Symbol</label>
          <input
            id="ai-symbol"
            value={symbol}
            onChange={(e)=>setSymbol(e.target.value)}
            className="w-full px-2 py-1 rounded-md bg-slate-900/60 border border-cyan-400/20 placeholder-cyan-300/40 focus:outline-none focus:ring-2 focus:ring-cyan-500/40 text-cyan-100"
            placeholder="EURUSD"
          />
        </div>
        <div>
          <label className={`${HUD.TITLE}`}>Base Risk %</label>
          <input
            type="number"
            step="0.01"
            value={baseRiskPct}
            onChange={(e)=>setBaseRiskPct(parseFloat(e.target.value))}
            className="w-full px-2 py-1 rounded-md bg-slate-900/60 border border-cyan-400/20 text-cyan-100"
          />
        </div>
        <div>
          <label className={`${HUD.TITLE}`}>Equity</label>
          <input
            type="number"
            step="1"
            value={equity}
            onChange={(e)=>setEquity(parseFloat(e.target.value))}
            className="w-full px-2 py-1 rounded-md bg-slate-900/60 border border-cyan-400/20 text-cyan-100"
          />
        </div>
        <div>
          <label className={`${HUD.TITLE}`}>Spread</label>
          <input
            type="number"
            step="0.00001"
            value={spread}
            onChange={(e)=>setSpread(parseFloat(e.target.value))}
            className="w-full px-2 py-1 rounded-md bg-slate-900/60 border border-cyan-400/20 text-cyan-100"
          />
        </div>
        <div>
          <label className={`${HUD.TITLE}`}>Slippage</label>
          <input
            type="number"
            step="0.00001"
            value={slippage}
            onChange={(e)=>setSlippage(parseFloat(e.target.value))}
            className="w-full px-2 py-1 rounded-md bg-slate-900/60 border border-cyan-400/20 text-cyan-100"
          />
        </div>
        <div>
          <label className={`${HUD.TITLE}`}>Orderflow Imbalance</label>
          <input
            type="number"
            step="0.01"
            value={ofImbalance}
            onChange={(e)=>setOfImbalance(parseFloat(e.target.value))}
            className="w-full px-2 py-1 rounded-md bg-slate-900/60 border border-cyan-400/20 text-cyan-100"
          />
        </div>
        <div className="flex items-end gap-2">
          <label htmlFor="ai-dry" className={`${HUD.TITLE} mr-2`}>Dry-run</label>
          <input id="ai-dry" type="checkbox" checked={dryRun} onChange={()=>setDryRun(!dryRun)} />
        </div>
      </div>
      <div className="mt-3 flex gap-2">
        <button
          onClick={prepare}
          className="px-3 py-2 rounded-md border border-cyan-400/20 bg-blue-500/20 text-cyan-100 hover:bg-blue-500/30"
        >{loading ? "..." : "Prepare Idea"}</button>
        <button
          onClick={execute}
          className="px-3 py-2 rounded-md border border-cyan-400/20 bg-rose-500/20 text-cyan-100 hover:bg-rose-500/30"
        >Execute (Admin)</button>
      </div>

      {prepared !== null && (
        <div className={`mt-4 p-3 rounded-md ${HUD.PANEL}`}>
          <pre className="text-xs overflow-auto">{JSON.stringify(prepared, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default AISecretPanel;
