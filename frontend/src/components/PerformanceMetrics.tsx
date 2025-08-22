import React, { useEffect, useState } from 'react';
import { Box, Card, CardContent, Typography, Grid, CircularProgress, Chip, FormControl, Select, MenuItem, InputLabel } from '@mui/material';
import type { SelectChangeEvent } from '@mui/material/Select';
import { RiseOutlined, FallOutlined, LineChartOutlined, BarChartOutlined } from '@ant-design/icons';

interface PerformanceData {
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  profit_factor: number;
  total_trades: number;
  avg_trade: number;
  best_trade: number;
  worst_trade: number;
}

export const PerformanceMetrics: React.FC = () => {
  const [data, setData] = useState<PerformanceData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [days, setDays] = useState<number>(30);

  // Environment-aware base HTTP (dev uses Vite proxy via relative paths)
  const baseHttpEnv = (import.meta.env.VITE_BACKEND_BASE as string | undefined)?.replace(/\/$/, "");
  const baseHttp = baseHttpEnv || "";

  useEffect(() => {
    const fetchPerformanceData = async () => {
      setError(null);
      setLoading(true);
      try {
        const response = await fetch(`${baseHttp}/api/analytics/performance-metrics?days=${days}`);
        const result = await response.json();
        
        if (result.ok && result.data) {
          setData(result.data as PerformanceData);
        } else if (!result.ok) {
          setError(result.message || 'Failed to load performance metrics');
          setData(null);
        }
      } catch (error) {
        console.error('Failed to fetch performance metrics:', error);
        setError('Failed to fetch performance metrics');
        setData(null);
      } finally {
        setLoading(false);
      }
    };

    fetchPerformanceData();
  }, [days, baseHttp]);

  if (loading) {
    return (
      <Card sx={{ backgroundColor: '#0b1020', border: '1px solid rgba(34,211,238,0.3)' }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <CircularProgress sx={{ color: '#22d3ee' }} />
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card sx={{ backgroundColor: '#0b1020', border: '1px solid rgba(34,211,238,0.3)' }}>
        <CardContent>
          <Typography sx={{ color: '#22d3ee', textAlign: 'center' }}>
            No performance data available
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const MetricCard = ({ 
    title, 
    value, 
    icon, 
    color = '#22d3ee',
    suffix = '',
    prefix = '' 
  }: {
    title: string;
    value: number;
    icon: React.ReactNode;
    color?: string;
    suffix?: string;
    prefix?: string;
  }) => (
    <Card sx={{ backgroundColor: '#0b1020', border: `1px solid ${color}55`, height: '100%' }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
          {icon}
          <Typography variant="subtitle2" sx={{ color, fontWeight: 'bold' }}>
            {title}
          </Typography>
        </Box>
        <Typography variant="h4" sx={{ color, fontWeight: 'bold' }}>
          {prefix}{typeof value === 'number' ? value.toFixed(2) : value}{suffix}
        </Typography>
      </CardContent>
    </Card>
  );

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2, gap: 2, flexWrap: 'wrap' }}>
        <Typography variant="h6" sx={{ color: '#22d3ee' }}>
          PERFORMANCE METRICS ({days} DAYS)
        </Typography>
        <FormControl size="small" sx={{ minWidth: 140 }}>
          <InputLabel sx={{ color: '#22d3ee' }}>Period</InputLabel>
          <Select
            value={days.toString()}
            onChange={(e: SelectChangeEvent) => setDays(Number(e.target.value))}
            sx={{ color: '#e0fbff', '& .MuiOutlinedInput-notchedOutline': { borderColor: 'rgba(34,211,238,0.3)' } }}
          >
            <MenuItem value={"7"}>7 Days</MenuItem>
            <MenuItem value={"30"}>30 Days</MenuItem>
            <MenuItem value={"90"}>90 Days</MenuItem>
            <MenuItem value={"365"}>1 Year</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {error && (
        <Box sx={{ color: '#ff6b6b', border: '1px solid rgba(255,107,107,0.5)', p: 1, borderRadius: 1, mb: 2 }}>
          {error}
        </Box>
      )}
      
      <Grid container spacing={2}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="TOTAL RETURN"
            value={data.total_return}
            icon={<RiseOutlined style={{ color: data.total_return >= 0 ? '#22d3ee' : '#ff4040' }} />}
            color={data.total_return >= 0 ? '#22d3ee' : '#ff4040'}
            prefix="$"
          />
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="WIN RATE"
            value={data.win_rate}
            icon={<BarChartOutlined style={{ color: '#22d3ee' }} />}
            suffix="%"
          />
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="SHARPE RATIO"
            value={data.sharpe_ratio}
            icon={<LineChartOutlined style={{ color: data.sharpe_ratio >= 1 ? '#22d3ee' : '#ffaa00' }} />}
            color={data.sharpe_ratio >= 1 ? '#22d3ee' : '#ffaa00'}
          />
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="MAX DRAWDOWN"
            value={data.max_drawdown}
            icon={<FallOutlined style={{ color: '#ff4040' }} />}
            color="#ff4040"
            suffix="%"
          />
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="PROFIT FACTOR"
            value={data.profit_factor === Infinity ? 999 : data.profit_factor}
            icon={<BarChartOutlined style={{ color: data.profit_factor >= 1.5 ? '#22d3ee' : '#ffaa00' }} />}
            color={data.profit_factor >= 1.5 ? '#22d3ee' : '#ffaa00'}
          />
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="TOTAL TRADES"
            value={data.total_trades}
            icon={<BarChartOutlined style={{ color: '#22d3ee' }} />}
          />
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="AVG TRADE"
            value={data.avg_trade}
            icon={<LineChartOutlined style={{ color: data.avg_trade >= 0 ? '#22d3ee' : '#ff4040' }} />}
            color={data.avg_trade >= 0 ? '#22d3ee' : '#ff4040'}
            prefix="$"
          />
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card sx={{ backgroundColor: '#0b1020', border: '1px solid rgba(34,211,238,0.3)', height: '100%' }}>
            <CardContent>
              <Typography variant="subtitle2" sx={{ color: '#22d3ee', fontWeight: 'bold', mb: 1 }}>
                BEST / WORST
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Chip
                  label={`Best: $${data.best_trade.toFixed(2)}`}
                  sx={{ backgroundColor: '#22d3ee', color: '#000', fontWeight: 'bold' }}
                  size="small"
                />
                <Chip
                  label={`Worst: $${data.worst_trade.toFixed(2)}`}
                  sx={{ backgroundColor: '#ff4040', color: '#fff', fontWeight: 'bold' }}
                  size="small"
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};
