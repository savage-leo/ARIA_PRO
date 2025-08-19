import React, { useEffect, useMemo, useRef, useState } from "react";
import { HUD } from "../../theme/hud";
import { PerformanceMetrics } from "../PerformanceMetrics";
import { HedgeFundDashboard } from "../HedgeFundDashboard";

// Neon HUD palette aligned to the reference image (shared)
const BG = HUD.BG; // near-black navy
const CARD = HUD.CARD;
const TITLE = HUD.TITLE;
const VALUE = HUD.VALUE;

// Types
interface SensorData {
  flowRate: number; // units/sec
  pressure: number; // kPa
  temperature: number; // °C
  throughput: number; // msg/s
  latencyMs: number; // ms
  errorRate: number; // %
  utilization: number; // %
}

// Utilities
function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

// Simple sparkline component (no external deps)
function Sparkline({ data, width = 140, height = 40, color = "#22d3ee" }: { data: number[]; width?: number; height?: number; color?: string; }) {
  const path = useMemo(() => {
    if (!data.length) return "";
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const step = width / (data.length - 1);
    return data
      .map((v, i) => {
        const x = i * step;
        const y = height - ((v - min) / range) * (height - 4) - 2; // top/btm padding
        return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
      })
      .join(" ");
  }, [data, width, height]);

  return (
    <svg width={width} height={height} className="overflow-visible">
      <path d={path} fill="none" stroke={color} strokeWidth={2} />
    </svg>
  );
}

// Circular gauge with 270° sweep
function Gauge({ value, min = 0, max = 100, label, unit, color = "#22d3ee" }: { value: number; min?: number; max?: number; label: string; unit?: string; color?: string; }) {
  const pct = clamp((value - min) / (max - min), 0, 1);
  const size = 160;
  const r = 64;
  const cx = size / 2;
  const cy = size / 2;
  const startAngle = -225; // degrees
  const endAngle = startAngle + 270 * pct;

  function polarToCartesian(cx: number, cy: number, r: number, angleDeg: number) {
    const a = (angleDeg - 90) * (Math.PI / 180);
    return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
  }

  function arcPath(a0: number, a1: number) {
    const start = polarToCartesian(cx, cy, r, a0);
    const end = polarToCartesian(cx, cy, r, a1);
    const largeArcFlag = a1 - a0 <= 180 ? 0 : 1;
    return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 1 ${end.x} ${end.y}`;
  }

  const track = arcPath(startAngle, startAngle + 270);
  const arc = arcPath(startAngle, endAngle);

  return (
    <div className={`relative ${CARD} rounded-xl p-4`}> 
      <div className={`${TITLE} mb-2`}>{label}</div>
      <div className="flex items-center gap-4">
        <svg width={160} height={160}>
          <defs>
            <filter id="glow">
              <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor={color} floodOpacity="0.6" />
            </filter>
            <linearGradient id="grad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#22d3ee" />
              <stop offset="100%" stopColor="#60a5fa" />
            </linearGradient>
          </defs>
          <path d={track} stroke="#0ea5b7" opacity={0.2} strokeWidth={10} fill="none" />
          <path d={arc} stroke="url(#grad)" strokeWidth={10} strokeLinecap="round" filter="url(#glow)" fill="none" />
          {/* tick marks */}
          {Array.from({ length: 10 }).map((_, i) => {
            const t = i / 9; // 0..1
            const a = startAngle + 270 * t;
            const p1 = polarToCartesian(cx, cy, r + 8, a);
            const p2 = polarToCartesian(cx, cy, r + 16, a);
            return <line key={i} x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y} stroke="#1f2937" strokeWidth={2} />;
          })}
        </svg>
        <div className="flex flex-col">
          <div className={VALUE}>
            {value.toFixed(1)} {unit}
          </div>
          <div className="text-xs text-cyan-400/60">Min {min} • Max {max}</div>
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value, suffix, good = true }: { label: string; value: string | number; suffix?: string; good?: boolean; }) {
  return (
    <div className={`rounded-lg ${CARD} p-3`}> 
      <div className={`${TITLE}`}>{label}</div>
      <div className={`text-xl ${good ? "text-emerald-300" : "text-amber-300"}`}>{value}{suffix ?? ""}</div>
    </div>
  );
}

// Main Flow Monitor Tab
const FlowMonitorTab: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'system' | 'hedge-fund'>('system');
  const [data, setData] = useState<SensorData>({
    flowRate: 0,
    pressure: 0, // CPU %
    temperature: 0, // Memory %
    throughput: 0,
    latencyMs: 0,
    errorRate: 0,
    utilization: 0,
  });
  const [history, setHistory] = useState<number[]>(Array.from({ length: 60 }, () => 0));

  // Live data wiring
  const connectedRef = useRef(false);
  const wsRef = useRef<WebSocket | null>(null);
  const msgCounterRef = useRef(0);
  const totalMsgRef = useRef(0);
  const errorCountRef = useRef(0);
  const pingStartRef = useRef<number | null>(null);
  const [connected, setConnected] = useState(false);
  const [events, setEvents] = useState<{ ts: number; text: string }[]>([]);
  const signalsWindowRef = useRef<number[]>([]);
  const ordersWindowRef = useRef<number[]>([]);
  const [signalsPerMin, setSignalsPerMin] = useState(0);
  const [ordersPerMin, setOrdersPerMin] = useState(0);

  const baseHttpEnv = (import.meta.env.VITE_BACKEND_BASE as string | undefined)?.replace(/\/$/, "");
  const baseHttp = baseHttpEnv || ""; // relative, proxied in dev
  const wsUrl = (() => {
    const wsEnv = import.meta.env.VITE_BACKEND_WS as string | undefined;
    if (wsEnv && wsEnv.length > 0) return wsEnv.replace(/\/$/, "");
    const base = baseHttpEnv;
    if (base && base.length > 0) {
      const wsBase = base.replace(/^http(s?):\/\//, (_m, s) => (s ? 'wss://' : 'ws://'));
      return `${wsBase.replace(/\/+$/, '')}/ws`;
    }
    const proto = typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss' : 'ws';
    const host = typeof window !== 'undefined' ? window.location.host : 'localhost:5175';
    return `${proto}://${host}/ws`;
  })();

  // Connect WebSocket and handle messages
  useEffect(() => {
    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      ws.onopen = () => {
        connectedRef.current = true;
        setConnected(true);
        setEvents((e) => [{ ts: Date.now(), text: "WS connected" }, ...e].slice(0, 8));
        ws.send(JSON.stringify({ type: "subscribe", channels: ["ticks", "signals", "orders"] }));
      };
      ws.onclose = () => {
        connectedRef.current = false;
        setConnected(false);
        setEvents((e) => [{ ts: Date.now(), text: "WS disconnected" }, ...e].slice(0, 8));
      };
      ws.onmessage = (ev) => {
        try {
          const payload = JSON.parse(ev.data);
          totalMsgRef.current += 1;
          const t = payload.type as string | undefined;
          if (t === "pong") {
            if (pingStartRef.current != null) {
              const rtt = performance.now() - pingStartRef.current;
              setData((d) => ({ ...d, latencyMs: Math.round(rtt) }));
              pingStartRef.current = null;
            }
            return;
          }
          msgCounterRef.current += 1;
          if (t === "tick") {
            // tick received
          } else if (t === "signal") {
            signalsWindowRef.current.push(Date.now());
            const now = Date.now();
            signalsWindowRef.current = signalsWindowRef.current.filter((ts) => now - ts < 60_000);
            setSignalsPerMin(signalsWindowRef.current.length);
            setEvents((e) => [{ ts: now, text: "AI signal" }, ...e].slice(0, 8));
          } else if (t === "order_update") {
            ordersWindowRef.current.push(Date.now());
            const now = Date.now();
            ordersWindowRef.current = ordersWindowRef.current.filter((ts) => now - ts < 60_000);
            setOrdersPerMin(ordersWindowRef.current.length);
            setEvents((e) => [{ ts: now, text: "Order update" }, ...e].slice(0, 8));
          }
        } catch (err) {
          errorCountRef.current += 1;
        }
      };

      return () => {
        ws.close();
      };
    } catch (e) {
      setEvents((ev) => [{ ts: Date.now(), text: "WS error" }, ...ev].slice(0, 8));
      return () => {}; // Add explicit return for error case
    }
  }, [wsUrl]);

  // Ping RTT every 5s
  useEffect(() => {
    const id = setInterval(() => {
      if (wsRef.current && connectedRef.current) {
        pingStartRef.current = performance.now();
        try {
          wsRef.current.send(JSON.stringify({ type: "ping" }));
        } catch (e) {
          // ignore
        }
      }
    }, 5000);
    return () => clearInterval(id);
  }, []);

  // Aggregate throughput every second
  useEffect(() => {
    const id = setInterval(() => {
      const perSec = msgCounterRef.current;
      msgCounterRef.current = 0;
      const utilization = clamp((perSec / 2000) * 100, 0, 100);
      const errPct = (errorCountRef.current / Math.max(1, totalMsgRef.current)) * 100;
      setData((prev) => ({
        ...prev,
        throughput: perSec,
        flowRate: perSec,
        utilization,
        errorRate: errPct,
      }));
      setHistory((h) => [...h.slice(1), perSec]);
    }, 1000);
    return () => clearInterval(id);
  }, []);

  // Poll system metrics for CPU/Memory
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const res = await fetch(`${baseHttp}/monitoring/metrics`);
        if (!res.ok) throw new Error("metrics HTTP status " + res.status);
        const j = await res.json();
        if (!cancelled) {
          const cpu = Number(j.cpu_usage ?? 0);
          const mem = Number(j.memory_usage ?? 0);
          setData((d) => ({ ...d, pressure: cpu, temperature: mem }));
        }
      } catch (e) {
        // ignore in UI; keep last values
      }
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [baseHttp]);

  const alert = data.errorRate > 2 || data.latencyMs > 150 || data.utilization > 90;

  return (
    <div className={`h-full w-full ${BG} text-slate-200`}> 
      {/* Tab Navigation */}
      <div className="flex border-b border-gray-700 mb-4">
        <button 
          className={`px-4 py-2 font-medium transition-colors ${
            activeTab === 'system' 
              ? 'text-cyan-400 border-b-2 border-cyan-400' 
              : 'text-gray-400 hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('system')}
        >
          System Monitor
        </button>
        <button 
          className={`px-4 py-2 font-medium transition-colors ${
            activeTab === 'hedge-fund' 
              ? 'text-cyan-400 border-b-2 border-cyan-400' 
              : 'text-gray-400 hover:text-gray-300'
          }`}
          onClick={() => setActiveTab('hedge-fund')}
        >
          Hedge Fund Analytics
        </button>
      </div>

      {activeTab === 'hedge-fund' ? (
        <HedgeFundDashboard />
      ) : (
        <div className="p-3 md:p-4">
          {/* Header */}
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="text-cyan-400 uppercase tracking-widest text-xs">ARIA</div>
              <h2 className="text-2xl font-semibold text-cyan-200 drop-shadow">ARIA Flow Monitor</h2>
            </div>
            <div className="flex items-center gap-3">
              <div className={`px-3 py-1 rounded border ${connected ? "border-emerald-400 text-emerald-300" : "border-amber-400 text-amber-300"}`}>
                {connected ? "WS Connected" : "WS Disconnected"}
              </div>
              <div className={`px-3 py-1 rounded border ${alert ? "border-amber-400 text-amber-300" : "border-emerald-400 text-emerald-300"}`}>
                {alert ? "ATTN: Thresholds exceeded" : "Nominal"}
              </div>
            </div>
          </div>

      {/* ARIA Core Section */}
      <div className={`rounded-xl ${CARD} p-4 mb-4`}>
        <div className={`${TITLE} mb-2`}>ARIA Core</div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Metric label="Signals/min" value={signalsPerMin} />
          <Metric label="Orders/min" value={ordersPerMin} />
          <Metric label="RTT" value={Math.round(data.latencyMs)} suffix=" ms" good={data.latencyMs < 120} />
          <Metric label="Errors" value={data.errorRate.toFixed(2)} suffix=" %" good={data.errorRate < 1} />
        </div>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        {/* Col 1: Gauges */}
        <div className="space-y-4">
          <Gauge value={data.throughput} min={0} max={2000} label="Message Rate" unit="msg/s" />
          <div className={`grid grid-cols-2 gap-4`}>
            <Gauge value={data.pressure} min={0} max={100} label="CPU Load" unit="%" />
            <Gauge value={data.temperature} min={0} max={100} label="Memory" unit="%" />
          </div>
        </div>

        {/* Col 2: Pipeline / Schematic */}
        <div className={`rounded-xl ${CARD} p-4 relative overflow-hidden`}> 
          <div className={`${TITLE} mb-2`}>Data Flow Schematic</div>
          <svg viewBox="0 0 600 300" className="w-full h-[320px]">
            {/* nodes */}
            <defs>
              <filter id="nodeGlow">
                <feDropShadow dx="0" dy="0" stdDeviation="4" floodColor="#22d3ee" floodOpacity="0.6" />
              </filter>
              <path id="pipe1" d="M92 150 H250" />
              <path id="pipe2" d="M350 150 H508" />
            </defs>
            <circle cx="70" cy="150" r="22" fill="#0ea5b7" opacity="0.35" />
            <circle cx="70" cy="150" r="12" fill="#22d3ee" filter="url(#nodeGlow)" />
            <text x="70" y="150" textAnchor="middle" dominantBaseline="middle" fontSize="10" fill="#cffafe">SRC</text>

            <rect x="250" y="110" width="100" height="80" rx="10" fill="#1f2937" stroke="#22d3ee" opacity={0.6} />
            <text x="300" y="150" textAnchor="middle" dominantBaseline="middle" fontSize="12" fill="#93c5fd">Processor</text>

            <circle cx="530" cy="150" r="22" fill="#0ea5b7" opacity="0.35" />
            <circle cx="530" cy="150" r="12" fill="#60a5fa" filter="url(#nodeGlow)" />
            <text x="530" y="150" textAnchor="middle" dominantBaseline="middle" fontSize="10" fill="#dbeafe">DST</text>

            {/* pipes */}
            <path d="M92 150 H250" stroke="#22d3ee" strokeWidth="4" opacity="0.6" />
            <path d="M350 150 H508" stroke="#60a5fa" strokeWidth="4" opacity="0.6" />

            {/* animated particles along pipes */}
            {Array.from({ length: 6 }).map((_, i) => {
              const dur = `${clamp(2000 / Math.max(20, data.throughput), 0.2, 6).toFixed(2)}s`;
              const r = 2 + (i % 3);
              return (
                <g key={`p1-${i}`}>
                  <circle r={r} fill="#67e8f9">
                    <animateMotion dur={dur} repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1">
                      <mpath href="#pipe1" />
                    </animateMotion>
                  </circle>
                </g>
              );
            })}
            {Array.from({ length: 6 }).map((_, i) => {
              const dur = `${clamp(2200 / Math.max(20, data.throughput), 0.2, 6).toFixed(2)}s`;
              const r = 2 + (i % 3);
              return (
                <g key={`p2-${i}`}>
                  <circle r={r} fill="#93c5fd">
                    <animateMotion dur={dur} repeatCount="indefinite" keyPoints="0;1" keyTimes="0;1">
                      <mpath href="#pipe2" />
                    </animateMotion>
                  </circle>
                </g>
              );
            })}

            {/* throughput indicator */}
            <text x="300" y="210" textAnchor="middle" fontSize="11" fill="#67e8f9">Throughput: {Math.round(data.throughput)} msg/s</text>
            <text x="300" y="230" textAnchor="middle" fontSize="11" fill="#93c5fd">Latency: {Math.round(data.latencyMs)} ms</text>
          </svg>
        </div>

        {/* Col 3: Metrics & Trends */}
        <div className="space-y-4">
          <div className={`rounded-xl ${CARD} p-4`}>
            <div className="flex items-center justify-between">
              <div>
                <div className={`${TITLE}`}>Throughput</div>
                <div className="text-3xl text-cyan-200">{Math.round(data.throughput)} msg/s</div>
              </div>
              <Sparkline data={history} width={160} height={50} />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <Metric label="Latency" value={Math.round(data.latencyMs)} suffix=" ms" good={data.latencyMs < 60} />
            <Metric label="Errors" value={data.errorRate.toFixed(2)} suffix=" %" good={data.errorRate < 1} />
            <Metric label="Utilization" value={Math.round(data.utilization)} suffix=" %" good={data.utilization < 85} />
            <Metric label="Health" value={alert ? "Degraded" : "OK"} good={!alert} />
          </div>

          {/* Event ticker */}
          <div className={`rounded-xl ${CARD} p-3`}>
            <div className={`${TITLE} mb-1`}>Events</div>
            <div className="text-[11px] grid gap-1">
              {events.slice(0, 6).map((e, idx) => (
                <div key={idx} className="text-cyan-300/80">
                  {new Date(e.ts).toLocaleTimeString()} · {e.text}
                </div>
              ))}
              {events.length === 0 && <div className="text-cyan-300/50">No events yet</div>}
            </div>
          </div>
        </div>
      </div>

      {/* System Performance Matrix */}
      <div className={`${CARD} rounded-xl p-4 mt-4`}>
        <div className="text-cyan-200 font-semibold mb-3">System Performance Matrix</div>
        <div className="h-96">
          <PerformanceMetrics />
        </div>
      </div>

      {/* Footer legend */}
      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3 text-[11px] text-cyan-300/70">
        <div className={`rounded ${CARD} px-3 py-2`}>Cyan line: Source → Processor</div>
        <div className={`rounded ${CARD} px-3 py-2`}>Blue line: Processor → Destination</div>
        <div className={`rounded ${CARD} px-3 py-2`}>Glow intensity ≈ activity</div>
        <div className={`rounded ${CARD} px-3 py-2`}>Live WS + API telemetry        </div>
      </div>
        </div>
      )}
    </div>
  );
};

export default FlowMonitorTab;
