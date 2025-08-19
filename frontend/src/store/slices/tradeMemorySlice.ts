import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import type { TradeIdea } from '@/types';

export interface TradeMemoryState {
  current: TradeIdea | null;
  history: TradeIdea[]; // newest first
  loading: boolean;
  error: string | null;
  lastUpdate: string | null;
}

const initialState: TradeMemoryState = {
  current: null,
  history: [],
  loading: false,
  error: null,
  lastUpdate: null,
};

const tradeMemorySlice = createSlice({
  name: 'tradeMemory',
  initialState,
  reducers: {
    addIdea: (state, action: PayloadAction<TradeIdea>) => {
      state.history.unshift(action.payload);
      state.current = action.payload;
      state.lastUpdate = new Date().toISOString();
    },
    setCurrentIdea: (state, action: PayloadAction<TradeIdea | null>) => {
      state.current = action.payload;
      state.lastUpdate = new Date().toISOString();
    },
    updateIdea: (state, action: PayloadAction<Partial<TradeIdea> & { id: number }>) => {
      const idx = state.history.findIndex(i => i.id === action.payload.id);
      if (idx !== -1) {
        state.history[idx] = { ...state.history[idx], ...action.payload } as TradeIdea;
        if (state.current?.id === action.payload.id) {
          state.current = state.history[idx];
        }
        state.lastUpdate = new Date().toISOString();
      }
    },
    removeIdea: (state, action: PayloadAction<number>) => {
      state.history = state.history.filter(i => i.id !== action.payload);
      if (state.current?.id === action.payload) {
        state.current = null;
      }
      state.lastUpdate = new Date().toISOString();
    },
    clearHistory: (state) => {
      state.history = [];
      state.current = null;
      state.lastUpdate = new Date().toISOString();
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
  },
});

export const { addIdea, setCurrentIdea, updateIdea, removeIdea, clearHistory, setLoading, setError } = tradeMemorySlice.actions;
export default tradeMemorySlice.reducer;
