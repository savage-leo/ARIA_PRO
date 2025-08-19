import { create } from "zustand";

export interface SMCSignal {
  ts: number;
  symbol?: string;
  bias?: string;
  confidence?: number;
  bos?: any;
  order_block?: any;
  fvg?: any[];
}

type State = {
  current: SMCSignal | null;
  history: SMCSignal[];
  setCurrent: (s: SMCSignal | null) => void;
  addToHistory: (s: SMCSignal) => void;
  clear: () => void;
}

export const useSMCStore = create<State>((set, get) => ({
  current: null,
  history: [],
  setCurrent: (s) => set({ current: s }),
  addToHistory: (s) => set({ history: [...get().history.slice(-99), s] }),
  clear: () => set({ current: null, history: [] }),
}));
