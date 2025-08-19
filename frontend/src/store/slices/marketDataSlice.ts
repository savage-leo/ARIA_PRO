import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import type { MarketData, OHLCV } from '@/types';

export interface MarketDataState {
  bySymbol: Record<string, MarketData>;
  loading: boolean;
  error: string | null;
  lastUpdate: string | null;
}

const initialState: MarketDataState = {
  bySymbol: {},
  loading: false,
  error: null,
  lastUpdate: null,
};

const marketDataSlice = createSlice({
  name: 'marketData',
  initialState,
  reducers: {
    setQuote: (state, action: PayloadAction<MarketData>) => {
      state.bySymbol[action.payload.symbol] = action.payload;
      state.error = null;
      state.lastUpdate = new Date().toISOString();
    },
    updateQuote: (
      state,
      action: PayloadAction<{ symbol: string; patch: Partial<Omit<MarketData, 'symbol'>> }>
    ) => {
      const { symbol, patch } = action.payload;
      const existing = state.bySymbol[symbol];
      if (existing) {
        state.bySymbol[symbol] = { ...existing, ...patch, symbol } as MarketData;
        state.lastUpdate = new Date().toISOString();
      }
    },
    setBars: (state, action: PayloadAction<{ symbol: string; bars: OHLCV[] }>) => {
      const { symbol, bars } = action.payload;
      const existing = state.bySymbol[symbol];
      if (existing) {
        state.bySymbol[symbol] = { ...existing, bars };
      } else {
        state.bySymbol[symbol] = {
          symbol,
          bid: 0,
          ask: 0,
          spread: 0,
          timestamp: new Date().toISOString(),
          bars,
        };
      }
      state.lastUpdate = new Date().toISOString();
    },
    removeSymbol: (state, action: PayloadAction<string>) => {
      delete state.bySymbol[action.payload];
      state.lastUpdate = new Date().toISOString();
    },
    clearAll: (state) => {
      state.bySymbol = {};
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

export const { setQuote, updateQuote, setBars, removeSymbol, clearAll, setLoading, setError } = marketDataSlice.actions;
export default marketDataSlice.reducer;
