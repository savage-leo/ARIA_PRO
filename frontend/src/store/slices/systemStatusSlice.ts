import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { SystemHealth, ModuleStatus } from '@/types';

interface SystemStatusState {
  health: SystemHealth | null;
  modules: ModuleStatus[];
  loading: boolean;
  error: string | null;
  lastUpdate: string | null;
}

const initialState: SystemStatusState = {
  health: null,
  modules: [],
  loading: false,
  error: null,
  lastUpdate: null,
};

export const fetchSystemStatus = createAsyncThunk(
  'systemStatus/fetch',
  async () => {
    const [healthResponse, modulesResponse] = await Promise.all([
      fetch('/api/health'),
      fetch('/api/data-sources/status'),
    ]);
    
    if (!healthResponse.ok || !modulesResponse.ok) {
      throw new Error('Failed to fetch system status');
    }
    
    const health = await healthResponse.json();
    const modules = await modulesResponse.json();
    
    return { health, modules };
  }
);

export const fetchModuleStatus = createAsyncThunk(
  'systemStatus/fetchModules',
  async () => {
    const endpoints = [
      { name: 'AutoTrader', url: '/api/monitoring/auto-trader/status' },
      { name: 'Models', url: '/api/monitoring/models/status' },
      { name: 'MT5', url: '/api/market/status' },
      { name: 'SMC Engine', url: '/api/smc/cpp/status' },
      { name: 'Trade Memory', url: '/api/trade-memory/recent?limit=1' },
    ];
    
    const results = await Promise.allSettled(
      endpoints.map(async (endpoint) => {
        try {
          const response = await fetch(endpoint.url);
          const data = await response.json();
          return {
            name: endpoint.name,
            status: response.ok ? 'healthy' : 'error',
            lastUpdate: new Date().toISOString(),
            metrics: data,
            message: response.ok ? 'Operational' : 'Service unavailable',
          } as ModuleStatus;
        } catch (error) {
          return {
            name: endpoint.name,
            status: 'offline',
            lastUpdate: new Date().toISOString(),
            message: `Connection failed: ${error}`,
          } as ModuleStatus;
        }
      })
    );
    
    return results.map((result, index) => 
      result.status === 'fulfilled' ? result.value : {
        name: endpoints[index]!.name,
        status: 'error',
        lastUpdate: new Date().toISOString(),
        message: 'Failed to check status',
      } as ModuleStatus
    );
  }
);

const systemStatusSlice = createSlice({
  name: 'systemStatus',
  initialState,
  reducers: {
    updateModuleStatus: (state, action: PayloadAction<ModuleStatus>) => {
      const index = state.modules.findIndex(m => m.name === action.payload.name);
      if (index !== -1) {
        state.modules[index] = action.payload;
      } else {
        state.modules.push(action.payload);
      }
    },
    setOverallHealth: (state, action: PayloadAction<SystemHealth['overall']>) => {
      if (state.health) {
        state.health.overall = action.payload;
      }
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchSystemStatus.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchSystemStatus.fulfilled, (state, action) => {
        state.loading = false;
        state.health = action.payload.health;
        state.lastUpdate = new Date().toISOString();
      })
      .addCase(fetchSystemStatus.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch system status';
      })
      .addCase(fetchModuleStatus.fulfilled, (state, action) => {
        state.modules = action.payload;
        
        // Calculate overall health based on module statuses
        const healthyCount = action.payload.filter(m => m.status === 'healthy').length;
        const total = action.payload.length;
        
        if (healthyCount === total) {
          state.health = { 
            ...state.health, 
            overall: 'healthy',
            modules: action.payload,
            uptime: Date.now(),
            memoryUsage: 0,
            cpuUsage: 0,
          } as SystemHealth;
        } else if (healthyCount > total * 0.7) {
          state.health = { 
            ...state.health, 
            overall: 'degraded',
            modules: action.payload,
            uptime: Date.now(),
            memoryUsage: 0,
            cpuUsage: 0,
          } as SystemHealth;
        } else {
          state.health = { 
            ...state.health, 
            overall: 'critical',
            modules: action.payload,
            uptime: Date.now(),
            memoryUsage: 0,
            cpuUsage: 0,
          } as SystemHealth;
        }
        
        state.lastUpdate = new Date().toISOString();
      });
  },
});

export const { updateModuleStatus, setOverallHealth } = systemStatusSlice.actions;
export default systemStatusSlice.reducer;
