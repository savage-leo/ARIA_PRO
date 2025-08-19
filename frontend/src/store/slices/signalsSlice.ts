import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { Signal } from '@/types';

interface SignalsState {
  signals: Signal[];
  loading: boolean;
  error: string | null;
  lastUpdate: string | null;
}

const initialState: SignalsState = {
  signals: [],
  loading: false,
  error: null,
  lastUpdate: null,
};

export const generateSignals = createAsyncThunk(
  'signals/generate',
  async (params: { symbol: string; timeframe: string; bars: number }) => {
    const response = await fetch('/api/signals/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!response.ok) throw new Error('Failed to generate signals');
    return response.json();
  }
);

export const fetchSignals = createAsyncThunk(
  'signals/fetch',
  async (params: { limit?: number; symbol?: string } = {}) => {
    const url = new URL('/api/signals/recent', window.location.origin);
    if (params.limit) url.searchParams.set('limit', params.limit.toString());
    if (params.symbol) url.searchParams.set('symbol', params.symbol);
    
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch signals');
    return response.json();
  }
);

const signalsSlice = createSlice({
  name: 'signals',
  initialState,
  reducers: {
    addSignal: (state, action: PayloadAction<Signal>) => {
      state.signals.unshift(action.payload);
      if (state.signals.length > 100) state.signals.pop();
    },
    clearSignals: (state) => {
      state.signals = [];
    },
    updateSignal: (state, action: PayloadAction<Partial<Signal> & { id: string }>) => {
      const index = state.signals.findIndex((s) => s.id === action.payload.id);
      if (index !== -1) {
        const { id, ...rest } = action.payload;
        // Apply each non-undefined property individually
        const target = state.signals[index];
        Object.entries(rest).forEach(([key, value]) => {
          if (value !== undefined) {
            (target as any)[key] = value;
          }
        });
      }
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(generateSignals.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(generateSignals.fulfilled, (state, action) => {
        state.loading = false;
        if (action.payload.signals) {
          state.signals = [...action.payload.signals, ...state.signals].slice(0, 100);
        }
        state.lastUpdate = new Date().toISOString();
      })
      .addCase(generateSignals.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to generate signals';
      })
      .addCase(fetchSignals.fulfilled, (state, action) => {
        if (action.payload.signals) {
          state.signals = action.payload.signals;
        }
        state.lastUpdate = new Date().toISOString();
      });
  },
});

export const { addSignal, clearSignals, updateSignal } = signalsSlice.actions;
export default signalsSlice.reducer;
