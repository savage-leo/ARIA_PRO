// frontend/src/components/interfaces/OrdersTab.tsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import { HUD } from "../../theme/hud";
import { useWebSocket, WebSocketMessage } from "../../hooks/useWebSocket";
import { TradingHeatmap } from "../TradingHeatmap";

interface TickData {
  symbol: string;
  bid: number;
  ask: number;
  spread?: number;
}

type TickPoint = {
  t: number; // ms timestamp (client-side arrival)
  mid: number;
  bid: number;
  ask: number;
};

const OrdersTab: React.FC = () => {
  const [symbol, setSymbol] = useState<string>("EURUSD");
  const [ticks, setTicks] = useState<TickPoint[]>([]);
  const lastSymbolRef = useRef<string>("EURUSD");

  const { isConnected, subscribe } = useWebSocket({
    onMessage: (msg: WebSocketMessage) => {
      if (msg.type === "tick") {
        const d = (msg.data || {}) as Partial<TickData>;
        const s = (d.symbol || "").toString().toUpperCase();
        const target = lastSymbolRef.current.toUpperCase();
        if (!s || s !== target) return;
        if (typeof d.bid !== "number" || typeof d.ask !== "number") return;
        const mid = (d.bid + d.ask) / 2;
        const tp: TickPoint = { t: Date.now(), mid, bid: d.bid, ask: d.ask };
        setTicks((prev) => {
          const next = [...prev, tp];
          return next.length > 600 ? next.slice(next.length - 600) : next;
        });
      }
    }
  });

  useEffect(() => {
    // Optional subscription (backend currently broadcasts to all)
    subscribe(["ticks", "signals"]);
  }, [subscribe]);

  useEffect(() => {
    lastSymbolRef.current = symbol;
    // clear buffer when symbol changes
    setTicks([]);
  }, [symbol]);

  const pipSize = useMemo(() => (sym: string) => (sym.toUpperCase().includes("JPY") ? 0.01 : 0.0001), []);

  const metrics = useMemo(() => {
    const now = Date.now();
    const windowMs = 5000; // 5s window for rate/imbalance
    const recent = ticks.filter((t) => now - t.t <= windowMs);
    const count = recent.length;
    const rate = count / (windowMs / 1000);

    const last = ticks[ticks.length - 1];
    const pips = pipSize(symbol);
    const spreadPips = last ? (last.ask - last.bid) / pips : undefined;

    // Imbalance: ratio of upticks vs downticks in window
    let up = 0, down = 0;
    for (let i = 1; i < recent.length; i++) {
      const curr = recent[i];
      const prev = recent[i - 1];
      if (!curr || !prev) continue;
      const d = curr.mid - prev.mid;
      if (d > 0) up++; else if (d < 0) down++;
    }
    const totalMoves = up + down;
    const upRatio = totalMoves > 0 ? up / totalMoves : 0.5;
    const imbalancePct = (upRatio - 0.5) * 200; // -100..100

    // Momentum proxy over window (in pips)
    const lastTick = recent[recent.length - 1];
    const firstTick = recent[0];
    const momentumPips = recent.length > 1 && lastTick && firstTick ? (lastTick.mid - firstTick.mid) / pips : 0;
    const momentumScore = Math.max(-1, Math.min(1, momentumPips / 10)); // clamp within +/-10 pips scale

    // Sentiment score blend
    const sentiment = Math.max(-100, Math.min(100, (upRatio - 0.5) * 140 + momentumScore * 60));

    return {
      lastMid: last?.mid,
      spreadPips,
      rate,
      imbalancePct,
      sentiment,
      recent,
    };
  }, [ticks, pipSize, symbol]);

  const sparkPath = useMemo(() => {
    const pts = ticks.slice(-60);
    if (pts.length < 2) return "";
    const w = 210, h = 40;
    const vals = pts.map((p) => p.mid);
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const range = Math.max(1e-9, max - min);
    const x = (i: number) => (i / (pts.length - 1)) * (w - 2) + 1;
    const y = (v: number) => h - ((v - min) / range) * (h - 4) - 2;
    const firstPt = pts[0];
    if (!firstPt) return "";
    let d = `M ${x(0).toFixed(1)} ${y(firstPt.mid).toFixed(1)}`;
    for (let i = 1; i < pts.length; i++) {
      const pt = pts[i];
      if (pt) d += ` L ${x(i).toFixed(1)} ${y(pt.mid).toFixed(1)}`;
    }
    return d;
  }, [ticks]);

  const statusColor = useMemo(() => (connected: boolean) => connected ? "text-emerald-400" : "text-rose-400", []);

  const sparkData = useMemo(() => {
    const pts = ticks.slice(-60);
    const sparkData = [];
    for (let i = 0; i < pts.length; i++) {
      const tick = pts[i];
      if (tick) {
        const prevTick = pts[i - 1];
        const velocity = prevTick ? tick.mid - prevTick.mid : 0;
        sparkData.push({ value: tick.mid, velocity });
      }
    }
    return sparkData;
  }, [ticks]);

  return (
    <div className={`p-4 ${HUD.BG} min-h-full`}>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-cyan-200 drop-shadow">Order Flow & Sentiment</h2>
        <div className="flex items-center gap-3">
          <div className={`text-xs ${statusColor(isConnected)}`}>{isConnected ? "WS Connected" : "WS Disconnected"}</div>
          <div className={`${HUD.TAB} border border-cyan-400/20 flex items-center gap-2`}>
            <span className="text-cyan-300/70 text-xs">Symbol</span>
            <input
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              className="bg-transparent outline-none w-24 uppercase text-cyan-100 placeholder-cyan-300/40"
              placeholder="EURUSD"
            />
          </div>
        </div>
      </div>

      {/* Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4">
        <div className={`${HUD.CARD} rounded-xl p-3`}>
          <div className={HUD.TITLE}>Mid</div>
          <div className={HUD.VALUE}>{
            typeof metrics.lastMid === "number" ? metrics.lastMid.toFixed(symbol.includes("JPY") ? 3 : 5) : "—"
          }</div>
        </div>
        <div className={`${HUD.CARD} rounded-xl p-3`}>
          <div className={HUD.TITLE}>Spread (pips)</div>
          <div className={HUD.VALUE}>{
            typeof metrics.spreadPips === "number" ? metrics.spreadPips.toFixed(1) : "—"
          }</div>
        </div>
        <div className={`${HUD.CARD} rounded-xl p-3`}>
          <div className={HUD.TITLE}>Tick Rate (/s)</div>
          <div className={HUD.VALUE}>{metrics.rate.toFixed(2)}</div>
        </div>
        <div className={`${HUD.CARD} rounded-xl p-3`}>
          <div className={HUD.TITLE}>Imbalance</div>
          <div className={`${HUD.VALUE} ${metrics.imbalancePct >= 0 ? "text-emerald-200" : "text-rose-200"}`}>{metrics.imbalancePct.toFixed(0)}%</div>
        </div>
      </div>

      {/* Sentiment and Sparkline */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className={`${HUD.CARD} rounded-xl p-4 col-span-1`}>
          <div className="flex items-center justify-between mb-2">
            <div className={HUD.TITLE}>Flow Sentiment</div>
            <div className={`text-sm ${metrics.sentiment >= 0 ? "text-emerald-300" : "text-rose-300"}`}>{metrics.sentiment.toFixed(0)}</div>
          </div>
          <div className="w-full h-3 rounded bg-slate-800/80 border border-cyan-400/10 overflow-hidden">
            <div
              className={`h-full ${metrics.sentiment >= 0 ? "bg-emerald-500/60" : "bg-rose-500/60"}`}
              style={{ width: `${Math.min(100, Math.abs(metrics.sentiment)).toFixed(0)}%` }}
            />
          </div>
          <div className="mt-2 text-xs text-cyan-300/70">Blend of uptick/downtick imbalance and short-window momentum</div>
        </div>
        <div className={`${HUD.CARD} rounded-xl p-4 col-span-2`}>
          <div className="flex items-center justify-between mb-2">
            <div className={HUD.TITLE}>Micro Trend (last ~60 ticks)</div>
            <div className="text-xs text-cyan-300/70">{ticks.length} pts</div>
          </div>
          <svg width="100%" viewBox="0 0 220 48" className="block">
            <path d={sparkPath} stroke="#67e8f9" strokeWidth="2" fill="none" />
          </svg>
        </div>
      </div>

      {/* Visual Flow HUD */}
      <div className={`${HUD.CARD} rounded-xl p-4 mt-4`}>
        <div className="flex items-center justify-between mb-2">
          <div className={HUD.TITLE}>Visual Flow</div>
          <div className="text-xs text-cyan-300/70">
            Rate {metrics.rate.toFixed(2)}/s • Imb {metrics.imbalancePct.toFixed(0)}%
          </div>
        </div>
        <div className="relative h-24 rounded-md bg-gradient-to-b from-slate-900/60 to-slate-800/40 border border-cyan-400/10 overflow-hidden">
          {[0,1,2].map((lane) => {
            const topPct = 20 + lane * 30;
            return (
              <div key={`lane-${lane}`}>
                {/* lane guide */}
                <div
                  className="absolute left-2 right-2 h-[2px] bg-cyan-400/20"
                  style={{ top: `${topPct}%` }}
                />
                {/* flowing shards */}
                {sparkData.map((_, i) => {
                  const base = Math.max(0.25, 1.6 - Math.min(metrics.rate, 20) * 0.05);
                  const jitter = (i % 4) * 0.07;
                  const delay = (i * 0.33) % 2;
                  const isBuy = metrics.imbalancePct >= 0;
                  const grad = isBuy
                    ? "linear-gradient(90deg, rgba(34,211,238,0.95), rgba(96,165,250,0.95))"
                    : "linear-gradient(90deg, rgba(244,63,94,0.95), rgba(251,113,133,0.95))";
                  const glow = isBuy ? "0 0 10px rgba(34,211,238,0.6)" : "0 0 10px rgba(244,63,94,0.6)";
                  return (
                    <div
                      key={`lane-${lane}-shard-${i}`}
                      className="aria-flow-shard absolute h-1 w-16 rounded-full"
                      style={{
                        top: `calc(${topPct}% - 2px)`,
                        left: "-12%",
                        background: grad,
                        boxShadow: glow,
                        animationDuration: `${(base + jitter).toFixed(2)}s`,
                        animationDelay: `${delay.toFixed(2)}s`,
                      }}
                    />
                  );
                })}
              </div>
            );
          })}
        </div>
        <div className="mt-2 text-xs text-cyan-300/60">
          Particles represent order flow; speed scales with tick rate; color indicates bias (cyan=buy, rose=sell).
        </div>
      </div>

      {/* Integrated Trading Heatmap */}
      <div className={`${HUD.CARD} rounded-xl p-4 mt-4`}>
        <div className="text-cyan-200 font-semibold mb-3">Trading Activity Heatmap</div>
        <div className="h-80">
          <TradingHeatmap />
        </div>
      </div>
    </div>
  );
};

export default OrdersTab;
