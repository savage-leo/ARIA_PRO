/**
 * API Client for ARIA PRO Backend
 * Handles HTTP requests, authentication, and error handling
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

interface ApiResponse<T = any> {
  ok: boolean;
  data?: T;
  error?: string;
  cached?: boolean;
}

class ApiClient {
  private baseUrl: string;
  private token: string | null = null;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
    this.token = localStorage.getItem('access_token');
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
      ...options.headers,
    };

    if (this.token) {
      headers.Authorization = `Bearer ${this.token}`;
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        return {
          ok: false,
          error: errorData.detail || `HTTP ${response.status}: ${response.statusText}`,
        };
      }

      const data = await response.json();
      return {
        ok: true,
        data,
      };
    } catch (error) {
      return {
        ok: false,
        error: error instanceof Error ? error.message : 'Network error',
      };
    }
  }

  // Authentication
  setToken(token: string) {
    this.token = token;
    localStorage.setItem('access_token', token);
  }

  clearToken() {
    this.token = null;
    localStorage.removeItem('access_token');
  }

  // Account endpoints
  async getAccountInfo() {
    return this.request<{
      balance: number;
      equity: number;
      margin: number;
      free_margin: number;
      cached: boolean;
    }>('/account/info');
  }

  async getAccountBalance() {
    return this.request<{
      balance: number;
      equity: number;
      cached: boolean;
    }>('/account/balance');
  }

  // Market data endpoints
  async getLastBar(symbol: string) {
    return this.request<{
      bar: {
        time: string;
        open: number;
        high: number;
        low: number;
        close: number;
        volume: number;
      };
      cached: boolean;
    }>(`/market/last_bar/${symbol}`);
  }

  // Trading endpoints
  async placeOrder(order: {
    symbol: string;
    action: 'buy' | 'sell';
    volume: number;
    price?: number;
    sl?: number;
    tp?: number;
  }) {
    return this.request('/trading/order', {
      method: 'POST',
      body: JSON.stringify(order),
    });
  }

  // Positions endpoints
  async getOpenPositions() {
    return this.request('/positions/open');
  }

  // Signals endpoints
  async getLatestSignals() {
    return this.request('/signals/latest');
  }

  // Health check
  async getHealth() {
    return this.request('/health');
  }
}

export const apiClient = new ApiClient();
export default apiClient;
