import React, { useEffect, useState } from 'react';
import { Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { Box, Card, CardContent, Typography, FormControl, InputLabel, Select, MenuItem, CircularProgress, TextField } from '@mui/material';
import type { SelectChangeEvent } from '@mui/material/Select';

interface EquityPoint {
  timestamp: string;
  equity: number;
  drawdown: number;
  trades: number;
}

interface EquityCurveChartProps {
  symbol?: string;
  days?: number;
}

export const EquityCurveChart: React.FC<EquityCurveChartProps> = ({ symbol, days = 30 }) => {
  const [data, setData] = useState<EquityPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedDays, setSelectedDays] = useState(days);
  const [symbolInput, setSymbolInput] = useState(symbol ?? '');
  const [error, setError] = useState<string | null>(null);

  // Environment-aware base HTTP (dev uses Vite proxy via relative paths)
  const baseHttpEnv = (import.meta.env.VITE_BACKEND_BASE as string | undefined)?.replace(/\/$/, "");
  const baseHttp = baseHttpEnv || "";

  useEffect(() => {
    const fetchEquityData = async () => {
      setLoading(true);
      setError(null);
      try {
        const url = new URL(`${baseHttp}/api/analytics/equity-curve`, window.location.origin);
        url.searchParams.set('days', selectedDays.toString());
        if (symbolInput) url.searchParams.set('symbol', symbolInput);

        const response = await fetch(url);
        const result = await response.json();
        
        if (result.ok && result.data) {
          type RawPoint = { timestamp: string | number | Date; equity: number; drawdown: number; trades: number };
          setData((result.data as RawPoint[]).map((point) => ({
            equity: point.equity,
            drawdown: point.drawdown,
            trades: point.trades,
            timestamp: new Date(point.timestamp).toLocaleDateString(),
          })));
        } else if (!result.ok) {
          setError(result.message || 'Failed to load equity curve');
        }
      } catch (error) {
        console.error('Failed to fetch equity curve:', error);
        setError('Failed to fetch equity curve');
      } finally {
        setLoading(false);
      }
    };

    fetchEquityData();
  }, [symbolInput, selectedDays, baseHttp]);

  return (
    <Card sx={{ backgroundColor: '#0b1020', border: '1px solid rgba(34,211,238,0.3)' }}>
      <CardContent>
        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'space-between', alignItems: 'center', mb: 2, flexWrap: 'wrap' }}>
          <Typography variant="h6" sx={{ color: '#22d3ee' }}>
            EQUITY CURVE {symbolInput ? `- ${symbolInput}` : ''}
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <TextField
              label="Symbol"
              value={symbolInput}
              onChange={(e) => setSymbolInput(e.target.value.toUpperCase())}
              size="small"
              sx={{
                minWidth: 120,
                '& .MuiInputBase-input': { color: '#e0fbff' },
                '& .MuiInputLabel-root': { color: '#22d3ee' },
                '& .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(34,211,238,0.3)' },
              }}
            />
            <FormControl size="small" sx={{ minWidth: 120 }}>
              <InputLabel sx={{ color: '#22d3ee' }}>Period</InputLabel>
              <Select
                value={selectedDays.toString()}
                onChange={(e: SelectChangeEvent) => setSelectedDays(Number(e.target.value))}
                sx={{ color: '#e0fbff', '& .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(34,211,238,0.3)' } }}
              >
                <MenuItem value={"7"}>7 Days</MenuItem>
                <MenuItem value={"30"}>30 Days</MenuItem>
                <MenuItem value={"90"}>90 Days</MenuItem>
                <MenuItem value={"365"}>1 Year</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </Box>

        {error && (
          <Box sx={{ color: '#ff6b6b', border: '1px solid rgba(255,107,107,0.5)', p: 1, borderRadius: 1, mb: 2 }}>
            {error}
          </Box>
        )}

        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <CircularProgress sx={{ color: '#22d3ee' }} />
          </Box>
        ) : (
          <Box sx={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data}>
                <defs>
                  <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.1}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis 
                  dataKey="timestamp" 
                  stroke="#22d3ee"
                  fontSize={12}
                />
                <YAxis 
                  stroke="#22d3ee"
                  fontSize={12}
                  tickFormatter={(value) => `$${value.toFixed(0)}`}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#0b1020', 
                    border: '1px solid rgba(34,211,238,0.3)',
                    color: '#e0fbff'
                  }}
                  formatter={(value: any, name: string) => [
                    name === 'equity' ? `$${value.toFixed(2)}` : `${value.toFixed(2)}%`,
                    name === 'equity' ? 'Equity' : 'Drawdown'
                  ]}
                />
                <Area
                  type="monotone"
                  dataKey="equity"
                  stroke="#22d3ee"
                  fillOpacity={1}
                  fill="url(#equityGradient)"
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="drawdown"
                  stroke="#ff0040"
                  strokeWidth={1}
                  dot={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};
