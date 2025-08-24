/**
 * Main Dashboard Component for ARIA PRO
 * Displays account info, market data, and trading interface
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, Grid, Alert, CircularProgress } from '@mui/material';
import { apiClient } from '../api/client';

interface AccountInfo {
  balance: number;
  equity: number;
  margin?: number;
  free_margin?: number;
  cached: boolean;
}

interface MarketBar {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export const Dashboard: React.FC = () => {
  const [accountInfo, setAccountInfo] = useState<AccountInfo | null>(null);
  const [marketData, setMarketData] = useState<Record<string, MarketBar>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'];

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    setError(null);

    try {
      // Load account info
      const accountResponse = await apiClient.getAccountInfo();
      if (accountResponse.ok && accountResponse.data) {
        setAccountInfo(accountResponse.data);
      } else {
        setError(accountResponse.error || 'Failed to load account info');
      }

      // Load market data for each symbol
      const marketPromises = symbols.map(async (symbol) => {
        const response = await apiClient.getLastBar(symbol);
        if (response.ok && response.data) {
          return { symbol, bar: response.data.bar };
        }
        return null;
      });

      const marketResults = await Promise.all(marketPromises);
      const newMarketData: Record<string, MarketBar> = {};
      
      marketResults.forEach((result) => {
        if (result) {
          newMarketData[result.symbol] = result.bar;
        }
      });

      setMarketData(newMarketData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', padding: '2rem' }}>
        <CircularProgress />
      </div>
    );
  }

  return (
    <div style={{ padding: '1rem' }}>
      <Typography variant="h4" gutterBottom>
        ARIA PRO Dashboard
      </Typography>

      {error && (
        <Alert severity="error" style={{ marginBottom: '1rem' }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Account Information */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Account Information
              </Typography>
              {accountInfo ? (
                <>
                  <Typography>
                    Balance: ${accountInfo.balance?.toFixed(2) || 'N/A'}
                  </Typography>
                  <Typography>
                    Equity: ${accountInfo.equity?.toFixed(2) || 'N/A'}
                  </Typography>
                  {accountInfo.margin && (
                    <Typography>
                      Margin: ${accountInfo.margin.toFixed(2)}
                    </Typography>
                  )}
                  {accountInfo.free_margin && (
                    <Typography>
                      Free Margin: ${accountInfo.free_margin.toFixed(2)}
                    </Typography>
                  )}
                  <Typography variant="caption" color="textSecondary">
                    {accountInfo.cached ? 'Cached data' : 'Live data'}
                  </Typography>
                </>
              ) : (
                <Typography color="textSecondary">
                  Account information unavailable
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Market Data */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Market Data
              </Typography>
              {Object.entries(marketData).map(([symbol, bar]) => (
                <div key={symbol} style={{ marginBottom: '0.5rem' }}>
                  <Typography variant="subtitle2">
                    {symbol}: {bar.close.toFixed(5)}
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    H: {bar.high.toFixed(5)} L: {bar.low.toFixed(5)} V: {bar.volume}
                  </Typography>
                </div>
              ))}
              {Object.keys(marketData).length === 0 && (
                <Typography color="textSecondary">
                  No market data available
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Connection Status */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Status
              </Typography>
              <Typography color="primary">
                ✅ Frontend connected to backend
              </Typography>
              <Typography variant="caption" color="textSecondary">
                API Base: {apiClient['baseUrl']}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </div>
  );
};
