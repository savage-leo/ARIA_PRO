import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import type { Position } from '@/types';

export interface PositionsState {
  items: Position[];
  loading: boolean;
  error: string | null;
  lastUpdate: string | null;
}

const initialState: PositionsState = {
  items: [],
  loading: false,
  error: null,
  lastUpdate: null,
};

const positionsSlice = createSlice({
  name: 'positions',
  initialState,
  reducers: {
    setPositions: (state, action: PayloadAction<Position[]>) => {
      state.items = action.payload;
      state.error = null;
      state.lastUpdate = new Date().toISOString();
    },
    upsertPosition: (state, action: PayloadAction<Position>) => {
      const idx = state.items.findIndex(p => p.id === action.payload.id);
      if (idx >= 0) {
        state.items[idx] = action.payload;
      } else {
        state.items.unshift(action.payload);
      }
      state.lastUpdate = new Date().toISOString();
    },
    removePosition: (state, action: PayloadAction<string>) => {
      state.items = state.items.filter(p => p.id !== action.payload);
      state.lastUpdate = new Date().toISOString();
    },
    clearPositions: (state) => {
      state.items = [];
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

export const { setPositions, upsertPosition, removePosition, clearPositions, setLoading, setError } = positionsSlice.actions;
export default positionsSlice.reducer;
