import React, { useEffect, useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  LinearProgress,
  Alert,
  IconButton,
  Tooltip,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  AccountBalance,
  Timeline,
  Memory,
  Warning,
  Refresh,
  ShowChart,
  Psychology,
} from '@mui/icons-material';

interface PortfolioMetrics {
  total_return: number;
  total_return_pct: number;
  sharpe_ratio: number;
  max_drawdown: number;
  calmar_ratio: number;
  win_rate: number;
  total_trades: number;
  avg_trade: number;
  volatility_annualized: number;
  current_drawdown: number;
}

interface StrategyAttribution {
  [strategy: string]: {
    total_pnl: number;
    pnl_contribution_pct: number;
    win_rate: number;
    avg_win: number;
    avg_loss: number;
    trade_count: number;
    max_drawdown: number;
    profit_factor: number;
  };
}

interface RiskMetrics {
  var_1d_99: number;
  var_1d_95: number;
  expected_shortfall: number;
  volatility: number;
  skewness: number;
  kurtosis: number;
  max_loss: number;
  downside_deviation: number;
}

interface HedgeFundData {
  timestamp: number;
  portfolio: PortfolioMetrics;
  current_pnl: number;
  daily_pnl: number;
  inception_pnl: number;
  active_positions: number;
  top_strategies: [string, number][];
  attribution: StrategyAttribution;
  risk_metrics: RiskMetrics;
  total_strategies: number;
  data_points: number;
}

const MetricCard: React.FC<{
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ReactNode;
  color?: 'success' | 'warning' | 'error' | 'info';
  trend?: 'up' | 'down' | 'flat';
}> = ({ title, value, subtitle, icon, color = 'info', trend }) => {
  const getColorStyles = () => {
    switch (color) {
      case 'success': return { bg: '#0f1a0f', border: '#10b981', text: '#10b981' };
      case 'error': return { bg: '#1a0f0f', border: '#ef4444', text: '#ef4444' };
      case 'warning': return { bg: '#1a1a0f', border: '#f59e0b', text: '#f59e0b' };
      default: return { bg: '#0f1419', border: '#3b82f6', text: '#3b82f6' };
    }
  };

  const styles = getColorStyles();

  return (
    <Card
      sx={{
        backgroundColor: styles.bg,
        border: `1px solid ${styles.border}`,
        borderRadius: 2,
        height: '100%',
      }}
    >
      <CardContent sx={{ p: 2 }}>
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
          <Typography variant="body2" color="#94a3b8" fontWeight={500}>
            {title}
          </Typography>
          {icon && <Box sx={{ color: styles.text }}>{icon}</Box>}
        </Box>
        
        <Box display="flex" alignItems="center" gap={1}>
          <Typography variant="h5" fontWeight={600} color="#f1f5f9">
            {value}
          </Typography>
          {trend && (
            <Box sx={{ color: trend === 'up' ? '#10b981' : trend === 'down' ? '#ef4444' : '#94a3b8' }}>
              {trend === 'up' ? <TrendingUp fontSize="small" /> : 
               trend === 'down' ? <TrendingDown fontSize="small" /> : null}
            </Box>
          )}
        </Box>
        
        {subtitle && (
          <Typography variant="caption" color="#64748b">
            {subtitle}
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

const StrategyTable: React.FC<{ strategies: [string, number][] }> = ({ strategies }) => (
  <Paper sx={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}>
    <Table size="small">
      <TableHead>
        <TableRow>
          <TableCell sx={{ color: '#94a3b8', fontWeight: 600 }}>Strategy</TableCell>
          <TableCell align="right" sx={{ color: '#94a3b8', fontWeight: 600 }}>P&L</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {strategies.slice(0, 6).map(([name, pnl]) => (
          <TableRow key={name}>
            <TableCell sx={{ color: '#f1f5f9', borderColor: '#374151' }}>
              <Box display="flex" alignItems="center" gap={1}>
                <Psychology fontSize="small" sx={{ color: '#64748b' }} />
                {name}
              </Box>
            </TableCell>
            <TableCell 
              align="right" 
              sx={{ 
                color: pnl >= 0 ? '#10b981' : '#ef4444',
                fontWeight: 600,
                borderColor: '#374151'
              }}
            >
              ${pnl.toFixed(2)}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  </Paper>
);

export const HedgeFundDashboard: React.FC = () => {
  const [data, setData] = useState<HedgeFundData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchDashboardData = async () => {
    try {
      const response = await fetch('/hedge-fund/performance');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const dashboardData = await response.json();
      setData(dashboardData);
      setError(null);
      setLastUpdate(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
      console.error('Hedge fund dashboard error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const handleRefresh = () => {
    setLoading(true);
    fetchDashboardData();
  };

  if (loading && !data) {
    return (
      <Box 
        sx={{ 
          height: '100%',
          backgroundColor: '#0a0e1a',
          color: '#e1e5e9',
          p: 3,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}
      >
        <Box textAlign="center">
          <LinearProgress sx={{ mb: 2, width: 200 }} />
          <Typography>Loading hedge fund data...</Typography>
        </Box>
      </Box>
    );
  }

  if (error && !data) {
    return (
      <Box sx={{ height: '100%', backgroundColor: '#0a0e1a', color: '#e1e5e9', p: 3 }}>
        <Alert 
          severity="error" 
          action={
            <IconButton color="inherit" onClick={handleRefresh}>
              <Refresh />
            </IconButton>
          }
        >
          {error}
        </Alert>
      </Box>
    );
  }

  if (!data) return null;

  const { portfolio, risk_metrics, top_strategies } = data;

  return (
    <Box sx={{ 
      height: '100%',
      backgroundColor: '#0a0e1a',
      color: '#e1e5e9',
      p: 3,
      overflow: 'auto'
    }}>
      {/* Header */}
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={3}>
        <Box>
          <Typography variant="h4" fontWeight={600} color="#3b82f6">
            🏛️ Multi-Strategy Hedge Fund
          </Typography>
          <Typography variant="body2" color="#64748b">
            ARIA Institutional Trading System • Real-time Analytics
          </Typography>
        </Box>
        <Box display="flex" alignItems="center" gap={2}>
          <Chip
            label={`${data.active_positions} Active Positions`}
            color="primary"
            variant="outlined"
          />
          <Chip
            label={`${data.total_strategies} Strategies`}
            color="secondary"
            variant="outlined"
          />
          <Tooltip title="Refresh Data">
            <IconButton onClick={handleRefresh} sx={{ color: '#3b82f6' }}>
              <Refresh />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {error && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          {error} (showing cached data)
        </Alert>
      )}

      {/* Portfolio Performance Grid */}
      <Grid container spacing={2} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total P&L"
            value={`$${data.inception_pnl.toFixed(2)}`}
            subtitle="Inception to Date"
            icon={<AccountBalance />}
            color={data.inception_pnl >= 0 ? 'success' : 'error'}
            trend={data.inception_pnl >= 0 ? 'up' : 'down'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Daily P&L"
            value={`$${data.daily_pnl.toFixed(2)}`}
            subtitle="Today's Performance"
            icon={<Timeline />}
            color={data.daily_pnl >= 0 ? 'success' : 'error'}
            trend={data.daily_pnl >= 0 ? 'up' : 'down'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Sharpe Ratio"
            value={portfolio.sharpe_ratio.toFixed(3)}
            subtitle="Risk-Adjusted Return"
            icon={<ShowChart />}
            color={portfolio.sharpe_ratio > 1.5 ? 'success' : portfolio.sharpe_ratio > 0.8 ? 'warning' : 'error'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Max Drawdown"
            value={`${portfolio.max_drawdown.toFixed(1)}%`}
            subtitle="Peak to Trough"
            icon={<TrendingDown />}
            color={portfolio.max_drawdown < 5 ? 'success' : portfolio.max_drawdown < 10 ? 'warning' : 'error'}
          />
        </Grid>
      </Grid>

      {/* Secondary Metrics */}
      <Grid container spacing={2} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Win Rate"
            value={`${portfolio.win_rate.toFixed(1)}%`}
            subtitle={`${portfolio.total_trades} trades`}
            color={portfolio.win_rate > 60 ? 'success' : portfolio.win_rate > 50 ? 'warning' : 'error'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Volatility"
            value={`${portfolio.volatility_annualized.toFixed(1)}%`}
            subtitle="Annualized"
            color={portfolio.volatility_annualized < 15 ? 'success' : portfolio.volatility_annualized < 25 ? 'warning' : 'error'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="VaR (99%)"
            value={`$${risk_metrics.var_1d_99.toFixed(4)}`}
            subtitle="1-Day Value at Risk"
            icon={<Warning />}
            color="error"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Data Points"
            value={data.data_points.toLocaleString()}
            subtitle="Live Analytics"
            icon={<Memory />}
            color="info"
          />
        </Grid>
      </Grid>

      {/* Strategy Performance and Risk */}
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Card sx={{ backgroundColor: '#1e293b', border: '1px solid #334155', height: '100%' }}>
            <CardContent>
              <Typography variant="h6" color="#f1f5f9" mb={2}>
                🎯 Top Performing Strategies
              </Typography>
              <StrategyTable strategies={top_strategies} />
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card sx={{ backgroundColor: '#1e293b', border: '1px solid #334155', height: '100%' }}>
            <CardContent>
              <Typography variant="h6" color="#f1f5f9" mb={2}>
                ⚠️ Risk Analytics
              </Typography>
              <Grid container spacing={1}>
                <Grid item xs={6}>
                  <Box mb={1}>
                    <Typography variant="caption" color="#94a3b8">Expected Shortfall</Typography>
                    <Typography variant="body2" color="#ef4444" fontWeight={600}>
                      ${risk_metrics.expected_shortfall.toFixed(4)}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6}>
                  <Box mb={1}>
                    <Typography variant="caption" color="#94a3b8">Skewness</Typography>
                    <Typography variant="body2" color="#f1f5f9" fontWeight={600}>
                      {risk_metrics.skewness.toFixed(3)}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6}>
                  <Box mb={1}>
                    <Typography variant="caption" color="#94a3b8">Kurtosis</Typography>
                    <Typography variant="body2" color="#f1f5f9" fontWeight={600}>
                      {risk_metrics.kurtosis.toFixed(3)}
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6}>
                  <Box mb={1}>
                    <Typography variant="caption" color="#94a3b8">Max Loss</Typography>
                    <Typography variant="body2" color="#ef4444" fontWeight={600}>
                      ${risk_metrics.max_loss.toFixed(4)}
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Footer */}
      <Box mt={3} textAlign="center">
        <Typography variant="caption" color="#64748b">
          Last Updated: {lastUpdate.toLocaleTimeString()} • 
          T470 Optimized • CPU-Only Multi-Strategy System
        </Typography>
      </Box>
    </Box>
  );
};
