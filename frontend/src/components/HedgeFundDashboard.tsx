import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { 
  Box, 
  Card,
  CardContent,
  Grid,
  IconButton,
  Paper,
  Tooltip,
  Typography,
  Chip,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  Button,
  Alert,
  LinearProgress,
  Skeleton,
  useTheme,
} from '@mui/material';
import {
  InfoCircleOutlined,
  WarningOutlined,
  SyncOutlined,
  BankOutlined,
  StockOutlined,
  LineChartOutlined,
  HddOutlined,
  BulbOutlined,
  FallOutlined,
  ReloadOutlined,
  CloseOutlined,
  RiseOutlined,
} from '@ant-design/icons';
// Using Ant Design icons instead of MUI icons

// Theme and types
import { hedgeFundTheme, commonStyles } from '../theme/hedgeFundTheme';
// Type definitions are already defined in this file, no need to import

// Constants
const DEFAULT_VISIBLE_STRATEGIES = 6;
const REFRESH_INTERVAL_MS = 30000; // 30 seconds

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

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: 'up' | 'down' | 'neutral';
  icon?: React.ReactNode;
  color?: 'success' | 'error' | 'warning' | 'info';
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  icon,
  color = 'info',
  trend,
  className = '',
  style = {},
}) => {
  const getColorStyles = useCallback(() => {
    switch (color) {
      case 'success':
        return { 
          bg: hedgeFundTheme.colors.background.success, 
          border: hedgeFundTheme.colors.border.success, 
          text: hedgeFundTheme.colors.text.success 
        };
      case 'error':
        return { 
          bg: hedgeFundTheme.colors.background.error, 
          border: hedgeFundTheme.colors.border.error, 
          text: hedgeFundTheme.colors.text.error 
        };
      case 'warning':
        return { 
          bg: hedgeFundTheme.colors.background.warning, 
          border: hedgeFundTheme.colors.border.warning, 
          text: hedgeFundTheme.colors.text.warning 
        };
      default:
        return { 
          bg: hedgeFundTheme.colors.background.card, 
          border: hedgeFundTheme.colors.border.info, 
          text: hedgeFundTheme.colors.text.info 
        };
    }
  }, [color]);

  const styles = useMemo(() => getColorStyles(), [getColorStyles]);

  return (
    <Card
      sx={[
        {
          ...commonStyles.card,
          backgroundColor: styles.bg || hedgeFundTheme.colors.surface.paper,
          border: `1px solid ${styles.border || hedgeFundTheme.colors.surface.border}`,
          height: '100%',
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
            transform: 'translateY(-2px)',
          },
        },
        ...(Array.isArray(style) ? style : [style])
      ]}
      className={className}
      aria-label={`${title} metric card`}
    >
      <CardContent sx={{ p: 2 }}>
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
          <Typography variant="body2" color={hedgeFundTheme.colors.text.secondary} fontWeight={500}>
            {title}
          </Typography>
          {icon && <Box sx={{ color: styles.text }}>{icon}</Box>}
        </Box>
        
        <Box display="flex" alignItems="center" gap={1} minHeight={32}>
          <Typography variant="h5" fontWeight={600} color={hedgeFundTheme.colors.text.primary}>
            {value}
          </Typography>
          {trend && (
            <Box sx={{ 
              color: trend === 'up' 
                ? hedgeFundTheme.colors.text.success 
                : trend === 'down' 
                  ? hedgeFundTheme.colors.text.error 
                  : hedgeFundTheme.colors.text.disabled 
            }}>
              {trend === 'up' ? (
                <RiseOutlined style={{ fontSize: '14px' }} />
              ) : trend === 'down' ? (
                <FallOutlined style={{ fontSize: '14px' }} />
              ) : null}
            </Box>
          )}
        </Box>
        
        {subtitle && (
          <Typography variant="caption" color={hedgeFundTheme.colors.text.disabled}>
            {subtitle}
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

interface StrategyTableProps {
  strategies: Array<{ name: string; pnl: number }> | [string, number][];
  maxVisible?: number;
  isLoading?: boolean;
  error?: string | null;
  className?: string;
  style?: React.CSSProperties;
  onStrategySelect?: (strategy: string) => void;
}

const StrategyTable: React.FC<StrategyTableProps> = ({
  strategies = [],
  maxVisible = DEFAULT_VISIBLE_STRATEGIES,
  isLoading = false,
  error = null,
  className = '',
  style = {},
  onStrategySelect,
}) => {
  const visibleStrategies = useMemo(() => {
    try {
      if (!strategies || !Array.isArray(strategies) || strategies.length === 0) {
        return [];
      }
      
      // Handle both array formats: Array<{name, pnl}> and [string, number][]
      const normalized = 'name' in strategies[0] || (strategies[0] && typeof strategies[0] === 'object' && 'name' in strategies[0])
        ? (strategies as Array<{name: string; pnl: number}>)
        : (strategies as [string, number][]).map(([name, pnl]) => ({ 
            name: String(name || 'Unnamed Strategy'), 
            pnl: typeof pnl === 'number' ? pnl : 0 
          }));
      
      return normalized
        .filter(strat => strat && typeof strat === 'object' && 'name' in strat && 'pnl' in strat)
        .slice(0, maxVisible);
    } catch (error) {
      console.error('Error normalizing strategies:', error);
      return [];
    }
  }, [strategies, maxVisible]);

  const theme = useTheme();

  if (error) {
    return (
      <Alert 
        severity="error"
        sx={{ 
          mt: 2,
          '& .MuiAlert-message': {
            width: '100%',
          },
        }}
      >
        <Box>
          <Typography variant="subtitle2" fontWeight={600} gutterBottom>
            Failed to load strategies
          </Typography>
          <Typography variant="body2">{error}</Typography>
        </Box>
      </Alert>
    );
  }

  if (isLoading) {
    return (
      <Box sx={{ p: 2 }}>
        {Array.from({ length: Math.min(4, maxVisible) }).map((_, i) => (
          <Box 
            key={i}
            sx={{
              display: 'flex',
              alignItems: 'center',
              p: 2,
              mb: 1,
              borderRadius: 1,
              backgroundColor: theme.palette.background.paper,
              border: `1px solid ${hedgeFundTheme.colors.surface.border}`,
            }}
          >
            <Skeleton 
              variant="circular" 
              width={32} 
              height={32} 
              sx={{ mr: 2 }}
            />
            <Box sx={{ flexGrow: 1 }}>
              <Skeleton width="60%" height={20} />
              <Skeleton width="40%" height={16} sx={{ mt: 0.5 }} />
            </Box>
            <Skeleton width={60} height={24} />
          </Box>
        ))}
      </Box>
    );
  }

  if (visibleStrategies.length === 0) {
    return (
      <Card sx={{ 
        background: hedgeFundTheme.colors.surface.paper,
        borderRadius: 2,
        border: `1px solid ${hedgeFundTheme.colors.surface.border}`,
        transition: 'all 0.2s ease-in-out',
        '&:hover': {
          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
        },
      }}
        className={className}
      >
        <Box 
          sx={[
            {
              p: 3, 
              textAlign: 'center',
              backgroundColor: hedgeFundTheme.colors.surface.paper,
              borderRadius: 2,
              border: `1px solid ${hedgeFundTheme.colors.surface.border}`,
              transition: 'all 0.2s ease-in-out',
              '&:hover': {
                boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
              },
            },
            ...(Array.isArray(style) ? style : [style])
          ]}
        >
          <InfoCircleOutlined 
            style={{ 
              fontSize: 32, 
              opacity: 0.5, 
              marginBottom: 8,
              color: hedgeFundTheme.colors.text.disabled
            }} 
          />
          <Typography variant="body2" color="text.secondary">
            0 data available
          </Typography>
        </Box>
      </Card>
    );
  }

  return (
    <Paper 
      sx={[
        {
          backgroundColor: hedgeFundTheme.colors.surface.paper,
          border: `1px solid ${hedgeFundTheme.colors.surface.border}`,
          borderRadius: 2,
          overflow: 'hidden',
          boxShadow: 'none',
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
          },
        },
        ...(Array.isArray(style) ? style : [style])
      ]}
      className={className}
    >
      <Table size="small" aria-label="Strategy performance table">
        <TableHead>
          <TableRow>
            <TableCell 
              sx={{
                ...commonStyles.tableHeader,
                backgroundColor: theme.palette.background.paper,
                borderBottom: `1px solid ${hedgeFundTheme.colors.surface.border}`,
                py: 1.5,
                pl: 2,
              }}
            >
              Strategy
            </TableCell>
            <TableCell 
              align="right" 
              sx={{
                ...commonStyles.tableHeader,
                backgroundColor: theme.palette.background.paper,
                borderBottom: `1px solid ${hedgeFundTheme.colors.surface.border}`,
                py: 1.5,
                pr: 2,
              }}
            >
              P&L
            </TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {visibleStrategies.map((strategy) => {
            const name = strategy?.name || 'Unnamed Strategy';
            const percentage = Math.abs(strategy?.change || 0);
            const pnl = typeof strategy?.pnl === 'number' ? strategy.pnl : 0;
            
            return (
            <TableRow 
              key={name} 
              hover
              onClick={() => onStrategySelect?.(name)}
              sx={[{
                cursor: onStrategySelect ? 'pointer' : 'default',
                transition: 'background-color 0.2s ease-in-out',
                '&:hover': {
                  backgroundColor: 'rgba(0, 0, 0, 0.04)',
                  transform: 'scale(1.02)',
                  transition: 'transform 100ms',
                },
                '&:last-child td': { 
                  borderBottomLeftRadius: 2,
                  borderBottomRightRadius: 2,
                },
                '&:last-child': {
                  borderBottomRightRadius: 2,
                },
              }]}
            >
              <TableCell 
                sx={{
                  ...commonStyles.tableCell,
                  py: 1.5,
                  pl: 2,
                  borderBottom: `1px solid ${hedgeFundTheme.colors.surface.border}`,
                }}
              >
                <Box display="flex" alignItems="center" gap={1.5}>
                  <BulbOutlined 
                    style={{ 
                      fontSize: '14px', 
                      color: hedgeFundTheme.colors.text.disabled,
                      flexShrink: 0,
                      transform: 'scale(1.1)',
                      transition: 'transform 150ms',
                    }} 
                  />
                  <Typography 
                    variant="body2" 
                    noWrap
                    sx={{
                      fontWeight: 500,
                      color: 'rgba(0, 0, 0, 0.6)',
                    }}
                  >
                    {name}
                  </Typography>
                </Box>
              </TableCell>
              <TableCell 
                align="right" 
                sx={{
                  ...commonStyles.tableCell,
                  py: 1.5,
                  pr: 2,
                  borderBottom: `1px solid ${hedgeFundTheme.colors.surface.border}`,
                  color: pnl >= 0 
                    ? hedgeFundTheme.colors.text.success 
                    : hedgeFundTheme.colors.text.error,
                  fontWeight: 600,
                  fontFamily: 'monospace',
                  fontSize: '0.9rem',
                }}
              >
                {pnl >= 0 ? (
                  <RiseOutlined style={{ marginRight: 4, fontSize: '0.8em' }} />
                ) : (
                  <FallOutlined style={{ marginRight: 4, fontSize: '0.8em' }} />
                )}
                ${Math.abs(pnl).toLocaleString('en-US', {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2
                })}
              </TableCell>
            </TableRow>
          );
          })}
        </TableBody>
      </Table>
    </Paper>
  );
};

export const HedgeFundDashboard: React.FC = () => {
  const theme = useTheme();
  const [data, setData] = useState<HedgeFundData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchDashboardData = useCallback(async () => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10s timeout
      
      const response = await fetch('/hedge-fund/performance', {
        signal: controller.signal,
        headers: {
          'Cache-Control': 'no-cache',
          'Pragma': 'no-cache',
        },
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const dashboardData = await response.json();
      
      // Validate response shape
      if (!dashboardData || typeof dashboardData !== 'object') {
        throw new Error('Invalid response format');
      }
      
      setData(dashboardData);
      setError(null);
      setLastUpdate(new Date());
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch data';
      setError(errorMessage);
      console.error('Hedge fund dashboard error:', err);
      
      // Only clear data if it's the first load
      if (!data) {
        setData(null);
      }
    } finally {
      setLoading(false);
    }
  }, [data]);

  useEffect(() => {
    let isMounted = true;
    
    const loadData = async () => {
      if (isMounted) {
        await fetchDashboardData();
      }
    };
    
    loadData();
    
    // Set up refresh interval
    const intervalId = setInterval(() => {
      if (document.visibilityState === 'visible') {
        loadData();
      }
    }, REFRESH_INTERVAL_MS);
    
    // Handle tab visibility changes
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        loadData();
      }
    };
    
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    return () => {
      isMounted = false;
      clearInterval(intervalId);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [fetchDashboardData]);

  const handleRefresh = useCallback(() => {
    if (!loading) {
      setLoading(true);
      fetchDashboardData();
    }
  }, [fetchDashboardData, loading]);

  // Loading state (initial load)
  if (loading && !data) {
    return (
      <Box 
        sx={{ 
          height: '100vh',
          backgroundColor: theme.palette.background.default,
          color: theme.palette.text.primary,
          p: 3,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 2,
        }}
      >
        <Box textAlign="center">
          <LinearProgress 
            sx={{ 
              width: 300, 
              height: 6, 
              borderRadius: 3,
              mb: 3,
              '& .MuiLinearProgress-bar': {
                backgroundColor: theme.palette.primary.main,
              }
            }} 
          />
          <Typography variant="h6" color="text.secondary" gutterBottom>
            Loading Hedge Fund Dashboard
          </Typography>
          <Typography variant="body2" color="text.disabled">
            Fetching the latest performance data...
          </Typography>
        </Box>
      </Box>
    );
  }

  // Error state (only show if no cached data)
  if (error && !data) {
    return (
      <Box 
        sx={{ 
          height: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          p: 3,
          backgroundColor: theme.palette.background.default,
        }}
      >
        <Alert 
          severity="error"
          sx={{
            maxWidth: 600,
            width: '100%',
            '& .MuiAlert-message': {
              width: '100%',
            },
          }}
          action={
            <IconButton 
              aria-label="retry"
              color="inherit" 
              size="small"
              onClick={handleRefresh}
              disabled={loading}
              sx={{ alignSelf: 'flex-start' }}
            >
              <ReloadOutlined />
            </IconButton>
          }
        >
          <Box>
            <Typography variant="subtitle1" fontWeight={600} gutterBottom>
              Failed to load dashboard data
            </Typography>
            <Typography variant="body2" component="div">
              {error}
            </Typography>
            <Typography variant="caption" display="block" sx={{ mt: 1, opacity: 0.8 }}>
              Last attempt: {lastUpdate.toLocaleTimeString()}
            </Typography>
          </Box>
        </Alert>
      </Box>
    );
  }

  // If we have no data and no error (shouldn't happen, but just in case)
  if (!data) {
    return (
      <Box 
        sx={{ 
          height: '100vh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          p: 3,
          textAlign: 'center',
          backgroundColor: theme.palette.background.default,
        }}
      >
        <InfoCircleOutlined style={{ fontSize: 48, marginBottom: 16, opacity: 0.5 }} />
        <Typography variant="h6" gutterBottom>
          No Data Available
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          We couldn't load any hedge fund data at this time.
        </Typography>
        <Button
          variant="contained"
          color="primary"
          startIcon={<ReloadOutlined />}
          onClick={handleRefresh}
          disabled={loading}
          sx={{ mt: 2 }}
        >
          {loading ? 'Loading...' : 'Try Again'}
        </Button>
      </Box>
    );
  }

  const { portfolio, risk_metrics, top_strategies = [] } = data;
  
  // Transform data for the StrategyTable component
  const strategyData = useMemo(() => {
    return top_strategies.map(([name, pnl]) => ({
      name: name || 'Unnamed Strategy',
      pnl: typeof pnl === 'number' ? pnl : 0,
    }));
  }, [top_strategies]);
  
  // Format last update time
  const lastUpdateFormatted = useMemo(() => {
    return new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: true,
    }).format(lastUpdate);
  }, [lastUpdate]);

  return (
    <Box 
      component="main"
      sx={{ 
        minHeight: '100vh',
        backgroundColor: theme.palette.background.default,
        color: theme.palette.text.primary,
        p: { xs: 2, md: 3 },
        overflow: 'auto',
        '&::-webkit-scrollbar': {
          width: '8px',
          height: '8px',
        },
        '&::-webkit-scrollbar-track': {
          backgroundColor: theme.palette.background.paper,
        },
        '&::-webkit-scrollbar-thumb': {
          backgroundColor: theme.palette.action.hover,
          borderRadius: '4px',
          '&:hover': {
            backgroundColor: theme.palette.action.selected,
          },
        },
      }}
    >
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
        <Box display="flex" alignItems="center" gap={2} flexWrap="wrap">
          <Chip
            label={
              <Box display="flex" alignItems="center" gap={0.5}>
                <FallOutlined style={{ fontSize: '20px' }} />
                {`${data.active_positions} Active Position${data.active_positions !== 1 ? 's' : ''}`}
              </Box>
            }
            color="default"
            variant="outlined"
            size="small"
            sx={{
              borderColor: hedgeFundTheme.colors.surface.border,
              '& .MuiChip-label': {
                display: 'flex',
                alignItems: 'center',
                gap: 0.5,
              },
            }}
          />
          <Chip
            label={`${data.total_strategies} Strateg${data.total_strategies !== 1 ? 'ies' : 'y'}`}
            color="default"
            variant="outlined"
            size="small"
            sx={{ borderColor: hedgeFundTheme.colors.surface.border }}
          />
          <Box flexGrow={1} />
          <Tooltip 
            title={
              <Box>
                <div>Last updated: {lastUpdateFormatted}</div>
                <div>Click to refresh data</div>
              </Box>
            }
            arrow
            placement="bottom-end"
          >
            <IconButton 
              onClick={handleRefresh} 
              color="primary"
              disabled={loading}
              aria-label="Refresh data"
              sx={{
                transition: 'transform 0.3s ease-in-out',
                transform: loading ? 'rotate(360deg)' : 'rotate(0deg)',
                animation: loading ? 'spin 2s linear infinite' : 'none',
                '@keyframes spin': {
                  '0%': { transform: 'rotate(0deg)' },
                  '100%': { transform: 'rotate(360deg)' },
                },
              }}
            >
              <ReloadOutlined />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Error banner (shows when there's an error but we have cached data) */}
      {error && data && (
        <Alert 
          severity="warning" 
          sx={{ 
            mb: 3,
            '& .MuiAlert-message': {
              width: '100%',
            },
          }}
          action={
            <IconButton 
              aria-label="close" 
              color="inherit" 
              size="small" 
              onClick={() => setError(null)}
            >
              <CloseOutlined fontSize="inherit" />
            </IconButton>
          }
        >
          <Box>
            <Typography variant="subtitle2" fontWeight={600} gutterBottom>
              Data might be outdated
            </Typography>
            <Typography variant="body2">
              {error} Showing last known data from {lastUpdateFormatted}.
            </Typography>
            <Button 
              size="small" 
              onClick={handleRefresh} 
              disabled={loading}
              startIcon={<SyncOutlined spin={loading} />}
              sx={{ mt: 1 }}
            >
              {loading ? 'Refreshing...' : 'Try Again'}
            </Button>
          </Box>
        </Alert>
      )}

      {/* Portfolio Performance Grid */}
      <Grid container spacing={2} mb={3}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total P&L"
            value={`$${data.inception_pnl.toFixed(2)}`}
            subtitle="Inception to Date"
            icon={<BankOutlined />}
            color={data.inception_pnl >= 0 ? 'success' : 'error'}
            trend={data.inception_pnl >= 0 ? 'up' : 'down'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Daily P&L"
            value={`$${data.daily_pnl.toFixed(2)}`}
            subtitle="Today's Performance"
            icon={<StockOutlined />}
            color={data.daily_pnl >= 0 ? 'success' : 'error'}
            trend={data.daily_pnl >= 0 ? 'up' : 'down'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Sharpe Ratio"
            value={portfolio.sharpe_ratio.toFixed(3)}
            subtitle="Risk-Adjusted Return"
            icon={<LineChartOutlined />}
            color={portfolio.sharpe_ratio > 1.5 ? 'success' : portfolio.sharpe_ratio > 0.8 ? 'warning' : 'error'}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Max Drawdown"
            value={`${portfolio.max_drawdown.toFixed(1)}%`}
            subtitle="Peak to Trough"
            icon={<FallOutlined />}
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
            icon={<WarningOutlined />}
            color="error"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Data Points"
            value={data.data_points.toLocaleString()}
            subtitle="Live Analytics"
            icon={<HddOutlined />}
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
              <StrategyTable 
                strategies={strategyData}
                isLoading={loading}
                error={error}
              />
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
