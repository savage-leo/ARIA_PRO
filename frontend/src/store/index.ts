import { configureStore } from '@reduxjs/toolkit';
import { TypedUseSelectorHook, useDispatch, useSelector } from 'react-redux';
import signalsSlice from './slices/signalsSlice';
import positionsSlice from './slices/positionsSlice';
import tradeMemorySlice from './slices/tradeMemorySlice';
import marketDataSlice from './slices/marketDataSlice';
import systemStatusSlice from './slices/systemStatusSlice';
import performanceSlice from './slices/performanceSlice';

export const store = configureStore({
  reducer: {
    signals: signalsSlice,
    positions: positionsSlice,
    tradeMemory: tradeMemorySlice,
    marketData: marketDataSlice,
    systemStatus: systemStatusSlice,
    performance: performanceSlice,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST', 'persist/REHYDRATE'],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

export const useAppDispatch = () => useDispatch<AppDispatch>();
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;
