import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import type { EquityCurvePoint, PerformanceMetrics } from '@/types';

export interface PerformanceState {
  metrics: PerformanceMetrics | null;
  equityCurve: EquityCurvePoint[];
  loading: boolean;
  error: string | null;
  lastUpdate: string | null;
}

const initialState: PerformanceState = {
  metrics: null,
  equityCurve: [],
  loading: false,
  error: null,
  lastUpdate: null,
};

const performanceSlice = createSlice({
  name: 'performance',
  initialState,
  reducers: {
    setMetrics: (state, action: PayloadAction<PerformanceMetrics | null>) => {
      state.metrics = action.payload;
      state.lastUpdate = new Date().toISOString();
      state.error = null;
    },
    setEquityCurve: (state, action: PayloadAction<EquityCurvePoint[]>) => {
      state.equityCurve = action.payload;
      state.lastUpdate = new Date().toISOString();
    },
    clearPerformance: (state) => {
      state.metrics = null;
      state.equityCurve = [];
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

export const { setMetrics, setEquityCurve, clearPerformance, setLoading, setError } = performanceSlice.actions;
export default performanceSlice.reducer;
