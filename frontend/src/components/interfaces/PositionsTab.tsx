// frontend/src/components/interfaces/PositionsTab.tsx
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { HUD } from "../../theme/hud";
import { apiDelete, apiGet, apiPost } from "../../services/api";
import { useWebSocket, WebSocketMessage } from "../../hooks/useWebSocket";
import { EquityCurveChart } from "../EquityCurveChart";

// Normalize unknown errors to readable strings
const getErrorMessage = (e: unknown): string => {
  if (e instanceof Error) return e.message;
  if (typeof e === "string") return e;
  try {
    return JSON.stringify(e);
  } catch {
    return String(e);
  }
};

interface Mt5Position {
  ticket?: number;
  symbol?: string;
  type?: string; // buy/sell
  side?: string; // alias
  volume?: number;
  price?: number;
  profit?: number;
  sl?: number;
  tp?: number;
  time?: string | number;
  [k: string]: unknown;
}

interface PositionsResp { positions: Mt5Position[]; count?: number }

type OrderSide = "buy" | "sell";

interface OrderPayload {
  symbol: string;
  volume: number;
  order_type: OrderSide;
  sl?: number;
  tp?: number;
  comment?: string;
}

interface OrderResponse {
  ticket: number;
  status: string;
  volume: number;
  price: number;
  timestamp: string;
  slippage?: number;
}

interface AccountInfo {
  balance?: number;
  equity?: number;
  margin?: number;
  margin_free?: number;
  [k: string]: unknown;
}

type RiskMetrics = Record<string, unknown>;

const PositionsTab: React.FC = () => {
  const [positions, setPositions] = useState<Mt5Position[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [symbol, setSymbol] = useState<string>("EURUSD");
  const [side, setSide] = useState<OrderSide>("buy");
  const [volume, setVolume] = useState<number>(0.01);
  const [sl, setSl] = useState<string>("");
  const [tp, setTp] = useState<string>("");
  const [placing, setPlacing] = useState(false);
  const [account, setAccount] = useState<AccountInfo | null>(null);
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics | null>(null);

  const fetchPositions = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const json = await apiGet<PositionsResp>("/trading/positions");
      setPositions(json?.positions || []);
    } catch (e: unknown) {
      setError(getErrorMessage(e));
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchAccount = useCallback(async () => {
    try {
      type AccountInfoResponse = (
        { account?: AccountInfo; risk_metrics?: RiskMetrics } &
        Partial<AccountInfo> &
        { status?: string; error?: unknown }
      );
      const json = await apiGet<AccountInfoResponse>("/trading/account-info");
      // Accept both shapes:
      // 1) { account: {...}, risk_metrics: {...} }
      // 2) { balance, equity, margin, margin_free, risk_metrics? }
      const accountData: AccountInfo | null = json?.account
        ? json.account
        : (typeof json?.balance === "number" ||
           typeof json?.equity === "number" ||
           typeof json?.margin === "number" ||
           typeof json?.margin_free === "number")
          ? {
              balance: typeof json.balance === "number" ? json.balance : 0,
              equity: typeof json.equity === "number" ? json.equity : 0,
              margin: typeof json.margin === "number" ? json.margin : 0,
              margin_free: typeof json.margin_free === "number" ? json.margin_free : 0,
            } as AccountInfo
          : null;
      setAccount(accountData);
      setRiskMetrics(json?.risk_metrics ?? null);
    } catch (_e) {
      // non-fatal
    }
  }, []);

  const { isConnected, subscribe } = useWebSocket({
    onMessage: (msg: WebSocketMessage) => {
      if (msg.type === "order_update") {
        // Refresh on order activity
        fetchPositions();
        fetchAccount();
      }
    }
  });

  useEffect(() => {
    fetchPositions();
    fetchAccount();
    const id = setInterval(() => {
      fetchPositions();
      fetchAccount();
    }, 15000);
    return () => clearInterval(id);
  }, [fetchPositions, fetchAccount]);

  useEffect(() => {
    subscribe(["orders"]);
  }, [subscribe]);

  const closePosition = async (ticket?: number) => {
    if (!ticket) return;
    if (!confirm(`Close position ${ticket}?`)) return;
    try {
      await apiDelete(`/trading/close-position/${ticket}`);
      await fetchPositions();
    } catch (e: unknown) {
      alert("Close failed: " + getErrorMessage(e));
    }
  };

  const closeAllPositions = async () => {
    if (!positions.length) return;
    if (!confirm(`Close ALL ${positions.length} positions?`)) return;
    try {
      await Promise.allSettled(
        positions.map((p) => (p.ticket ? apiDelete(`/trading/close-position/${p.ticket}`) : Promise.resolve()))
      );
      await fetchPositions();
    } catch (e: unknown) {
      alert("Close-all failed: " + getErrorMessage(e));
    }
  };

  const placeOrder = async () => {
    setPlacing(true);
    try {
      const slNum = sl.trim() ? parseFloat(sl) : NaN;
      const tpNum = tp.trim() ? parseFloat(tp) : NaN;
      const payload: OrderPayload = {
        symbol: symbol.toUpperCase(),
        order_type: side,
        volume: Number(volume),
        comment: "positions-ticket",
        ...(Number.isFinite(slNum) ? { sl: slNum } : {}),
        ...(Number.isFinite(tpNum) ? { tp: tpNum } : {}),
      };
      const res = await apiPost("/trading/place-order", payload) as OrderResponse;
      await fetchPositions();
      await fetchAccount();
      alert(`Order placed${res?.ticket ? ` #${res.ticket}` : ""}: ${res?.status ?? "ok"}`);
    } catch (e: unknown) {
      alert("Place order failed: " + getErrorMessage(e));
    } finally {
      setPlacing(false);
    }
  };

  const statusColor = useMemo(() => (connected: boolean) => connected ? "text-emerald-400" : "text-rose-400", []);

  return (
    <div className={`p-4 ${HUD.BG} min-h-full`}>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-cyan-200 drop-shadow">Positions — Open</h2>
        <div className="flex items-center gap-3">
          <div className={`text-xs ${statusColor(isConnected)}`}>{isConnected ? "WS Connected" : "WS Disconnected"}</div>
          <button onClick={fetchPositions} className={`${HUD.TAB} border border-cyan-400/20`}>{loading ? "Refreshing..." : "Refresh"}</button>
          <button onClick={closeAllPositions} className="px-2 py-1 text-xs bg-rose-600/70 hover:bg-rose-600 rounded border border-rose-400/30">Close All</button>
        </div>
      </div>

      {error && <div className="text-rose-400 text-sm mb-3">{error}</div>}

      {/* Order Ticket + Account Metrics */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-3 mb-4">
        <div className={`${HUD.CARD} rounded-xl p-4`}>
          <div className="text-cyan-200 font-semibold mb-3">Market Order Ticket</div>
          <div className="grid grid-cols-2 gap-3">
            <div className="col-span-1">
              <div className={HUD.TITLE}>Symbol</div>
              <input
                className="mt-1 w-full bg-transparent border border-cyan-400/20 rounded px-2 py-1 uppercase"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                placeholder="EURUSD"
              />
            </div>
            <div className="col-span-1">
              <div className={HUD.TITLE}>Side</div>
              <div className="mt-1 flex gap-2">
                <button
                  onClick={() => setSide("buy")}
                  className={`${side === "buy" ? "bg-emerald-600/30 border-emerald-400/40" : "bg-slate-800/60 border-cyan-400/20"} px-3 py-1 rounded border`}
                >Buy</button>
                <button
                  onClick={() => setSide("sell")}
                  className={`${side === "sell" ? "bg-rose-600/30 border-rose-400/40" : "bg-slate-800/60 border-cyan-400/20"} px-3 py-1 rounded border`}
                >Sell</button>
              </div>
            </div>
            <div className="col-span-1">
              <div className={HUD.TITLE}>Volume (lots)</div>
              <input
                type="number"
                step="0.01"
                min="0.01"
                className="mt-1 w-full bg-transparent border border-cyan-400/20 rounded px-2 py-1"
                value={volume}
                onChange={(e) => setVolume(parseFloat(e.target.value) || 0)}
              />
            </div>
            <div className="col-span-1">
              <div className={HUD.TITLE}>SL (price)</div>
              <input
                className="mt-1 w-full bg-transparent border border-cyan-400/20 rounded px-2 py-1"
                value={sl}
                onChange={(e) => setSl(e.target.value)}
                placeholder="optional"
              />
            </div>
            <div className="col-span-1">
              <div className={HUD.TITLE}>TP (price)</div>
              <input
                className="mt-1 w-full bg-transparent border border-cyan-400/20 rounded px-2 py-1"
                value={tp}
                onChange={(e) => setTp(e.target.value)}
                placeholder="optional"
              />
            </div>
            <div className="col-span-2 flex justify-end">
              <button
                onClick={placeOrder}
                disabled={placing}
                className={`px-4 py-2 rounded border ${placing ? "opacity-60 cursor-not-allowed" : "hover:bg-cyan-500/20"} border-cyan-400/30 text-cyan-100`}
              >{placing ? "Placing..." : "Place Market Order"}</button>
            </div>
          </div>
        </div>
        <div className={`${HUD.CARD} rounded-xl p-4`}>
          <div className="text-cyan-200 font-semibold mb-3">Account</div>
          <div className="grid grid-cols-2 gap-3 text-cyan-100">
            <div>
              <div className={HUD.TITLE}>Equity</div>
              <div className={HUD.VALUE}>{typeof account?.equity === "number" ? account!.equity.toFixed(2) : "—"}</div>
            </div>
            <div>
              <div className={HUD.TITLE}>Balance</div>
              <div className={HUD.VALUE}>{typeof account?.balance === "number" ? account!.balance.toFixed(2) : "—"}</div>
            </div>
            <div>
              <div className={HUD.TITLE}>Margin</div>
              <div className={HUD.VALUE}>{typeof account?.margin === "number" ? account!.margin.toFixed(2) : "—"}</div>
            </div>
            <div>
              <div className={HUD.TITLE}>Free Margin</div>
              <div className={HUD.VALUE}>{typeof account?.margin_free === "number" ? account!.margin_free.toFixed(2) : "—"}</div>
            </div>
          </div>
        </div>
        <div className={`${HUD.CARD} rounded-xl p-4`}>
          <div className="text-cyan-200 font-semibold mb-3">Risk Snapshot</div>
          <div className="grid grid-cols-2 gap-2 text-sm text-cyan-100/90">
            {riskMetrics ? (
              Object.entries(riskMetrics)
                .filter(([, v]) => typeof v === "number")
                .slice(0, 6)
                .map(([k, v]) => (
                  <div key={k} className="flex items-center justify-between border-b border-cyan-400/10 py-1">
                    <span className="text-cyan-300/70">{k}</span>
                    <span>{(v as number).toFixed(2)}</span>
                  </div>
                ))
            ) : (
              <div className="text-cyan-300/60">No data</div>
            )}
          </div>
        </div>
      </div>

      <div className={`${HUD.CARD} rounded-xl overflow-hidden`}>
        <div className="grid grid-cols-9 gap-2 px-3 py-2 text-[11px] text-cyan-300/70 border-b border-cyan-400/10">
          <div>Ticket</div>
          <div>Symbol</div>
          <div>Side</div>
          <div>Volume</div>
          <div>Price</div>
          <div>Profit</div>
          <div>SL</div>
          <div>TP</div>
          <div className="text-right">Action</div>
        </div>
        <div className="divide-y divide-cyan-400/10">
          {positions.length === 0 && (
            <div className="px-3 py-4 text-cyan-300/60 text-sm">No open positions</div>
          )}
          {positions.map((p) => (
            <div key={p.ticket} className="grid grid-cols-9 gap-2 px-3 py-2 text-sm text-cyan-100">
              <div>{p.ticket ?? "—"}</div>
              <div>{p.symbol ?? "—"}</div>
              <div className="uppercase">{(p.side || p.type) ?? "—"}</div>
              <div>{typeof p.volume === "number" ? p.volume.toFixed(2) : "—"}</div>
              <div>{typeof p.price === "number" ? p.price.toFixed(5) : "—"}</div>
              <div className={(p.profit || 0) >= 0 ? "text-emerald-300" : "text-rose-300"}>{typeof p.profit === "number" ? p.profit.toFixed(2) : "—"}</div>
              <div>{typeof p.sl === "number" ? p.sl.toFixed(5) : "—"}</div>
              <div>{typeof p.tp === "number" ? p.tp.toFixed(5) : "—"}</div>
              <div className="text-right">
                <button onClick={() => closePosition(p.ticket)} className="px-2 py-1 text-xs bg-rose-600/70 hover:bg-rose-600 rounded border border-rose-400/30">Close</button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Integrated Equity Curve */}
      <div className={`${HUD.CARD} rounded-xl p-4 mt-4`}>
        <div className="text-cyan-200 font-semibold mb-3">Account Equity Curve</div>
        <div className="h-80">
          <EquityCurveChart />
        </div>
      </div>
    </div>
  );
};

export default PositionsTab;
