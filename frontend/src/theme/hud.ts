// frontend/src/theme/hud.ts
// Shared neon/cyber HUD theme tokens for ARIA dashboard

export const HUD = {
  BG: "bg-[#0b1020]",
  TEXT: "text-cyan-100",
  TITLE: "text-cyan-300 tracking-wider uppercase text-xs",
  VALUE: "text-2xl font-semibold text-cyan-200 drop-shadow",
  CARD: "bg-slate-900/60 border border-cyan-400/20 shadow-[0_0_20px_rgba(34,211,238,0.15)]",
  PANEL: "rounded-xl bg-slate-900/40 border border-cyan-400/10",
  TAB: "px-3 py-1 rounded-md outline-none focus:outline-none focus:ring-0 text-cyan-300/70 hover:text-cyan-200 hover:bg-slate-800/50",
  TAB_ACTIVE:
    "px-3 py-1 rounded-md outline-none focus:outline-none focus:ring-0 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 text-cyan-100 border border-cyan-400/30 shadow-[0_0_12px_rgba(34,211,238,0.35)]",
};

export type HudKeys = keyof typeof HUD;
