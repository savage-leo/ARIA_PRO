// API Configuration
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';
export const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

// Environment
export const isDevelopment = import.meta.env.DEV;
export const isProduction = import.meta.env.PROD;

// Feature flags
export const ENABLE_WEBSOCKET = true;
export const ENABLE_MONITORING = true;
export const ENABLE_AUTO_TRADING = true;
