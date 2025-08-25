// API Configuration
export const API_BASE_URL = import.meta.env.VITE_BACKEND_BASE || 'http://localhost:8100';
export const WS_BASE_URL = import.meta.env.VITE_BACKEND_WS || 'ws://localhost:8100';

// Environment
export const isDevelopment = import.meta.env.DEV;
export const isProduction = import.meta.env.PROD;

// Feature flags
export const ENABLE_WEBSOCKET = true;
export const ENABLE_MONITORING = true;
export const ENABLE_AUTO_TRADING = true;
