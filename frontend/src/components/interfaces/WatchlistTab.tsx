// frontend/src/components/interfaces/WatchlistTab.tsx
import React, { useEffect, useMemo, useState } from "react";
import { HUD } from "../../theme/hud";
import { useWebSocket, WebSocketMessage } from "../../hooks/useWebSocket";

interface Quote {
  symbol: string;
  bid: number;
  ask: number;
  spread: number;
  mid: number;
  ts: string;
  dir: "up" | "down" | "flat";
}

const defaultSymbols = [
  "EURUSD",
  "GBPUSD",
  "USDJPY",
  "USDCHF",
  "AUDUSD",
  "USDCAD",
  "NZDUSD",
  "EURGBP",
];

const WatchlistTab: React.FC = () => {
  const [quotes, setQuotes] = useState<Record<string, Quote>>({});

  const { isConnected, subscribe } = useWebSocket({
    onMessage: (msg: WebSocketMessage) => {
      if (msg.type === "tick") {
        const d = msg.data || {};
        const symbol = String(d.symbol || "");
        if (!symbol) return;
        const bid = Number(d.bid ?? 0);
        const ask = Number(d.ask ?? 0);
        const spread = Number(d.spread ?? Math.max(ask - bid, 0));
        const mid = (bid + ask) / 2;
        setQuotes((prev) => {
          const prevQ = prev[symbol];
          const dir: Quote["dir"] = !prevQ
            ? "flat"
            : mid > prevQ.mid
            ? "up"
            : mid < prevQ.mid
            ? "down"
            : "flat";
          return {
            ...prev,
            [symbol]: { symbol, bid, ask, spread, mid, ts: msg.timestamp, dir },
          };
        });
      }
    },
  });

  useEffect(() => {
    subscribe(["ticks"]);
  }, [subscribe]);

  const statusColor = useMemo(
    () => (connected: boolean) => (connected ? "text-emerald-400" : "text-rose-400"),
    []
  );

  const symbols = defaultSymbols;
  const visibleQuotes = symbols.map((s) => quotes[s]).filter(Boolean) as Quote[];

  return (
    <div className={`p-4 ${HUD.BG} min-h-full`}>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-cyan-200 drop-shadow">Watchlist — FX Majors</h2>
        <div className={`text-xs ${statusColor(isConnected)}`}>
          {isConnected ? "WS Connected" : "WS Disconnected"}
        </div>
      </div>

      <div className={`${HUD.CARD} rounded-xl overflow-hidden`}>
        <div className="grid grid-cols-6 gap-2 px-3 py-2 text-[11px] text-cyan-300/70 border-b border-cyan-400/10">
          <div>Symbol</div>
          <div className="text-right">Bid</div>
          <div className="text-right">Ask</div>
          <div className="text-right">Spread</div>
          <div className="text-right">Mid</div>
          <div className="text-right">Time</div>
        </div>
        <div className="divide-y divide-cyan-400/10">
          {visibleQuotes.length === 0 && (
            <div className="px-3 py-4 text-cyan-300/60 text-sm">Waiting for tick data...</div>
          )}
          {symbols.map((sym) => {
            const q = quotes[sym];
            const dirCls = q?.dir === "up" ? "text-emerald-300" : q?.dir === "down" ? "text-rose-300" : "text-cyan-100";
            return (
              <div key={sym} className="grid grid-cols-6 gap-2 px-3 py-2 text-sm text-cyan-100">
                <div className="font-semibold tracking-wide">{sym}</div>
                <div className={`text-right ${dirCls}`}>{q ? q.bid.toFixed(sym.endsWith("JPY") ? 3 : 5) : "—"}</div>
                <div className={`text-right ${dirCls}`}>{q ? q.ask.toFixed(sym.endsWith("JPY") ? 3 : 5) : "—"}</div>
                <div className="text-right text-cyan-200">{q ? q.spread.toFixed(sym.endsWith("JPY") ? 3 : 5) : "—"}</div>
                <div className={`text-right ${dirCls}`}>{q ? q.mid.toFixed(sym.endsWith("JPY") ? 3 : 5) : "—"}</div>
                <div className="text-right text-cyan-300/70 text-xs">{q ? new Date(q.ts).toLocaleTimeString() : "—"}</div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default WatchlistTab;
