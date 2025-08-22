// Type definitions for Hedge Fund Dashboard

export interface PortfolioMetrics {
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

export interface StrategyAttribution {
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

export interface RiskMetrics {
  var_1d_99: number;
  var_1d_95: number;
  expected_shortfall: number;
  volatility: number;
  skewness: number;
  kurtosis: number;
  max_loss: number;
  downside_deviation: number;
}

export interface HedgeFundData {
  timestamp: number;
  portfolio: PortfolioMetrics;
  current_pnl: number;
  daily_pnl: number;
  inception_pnl: number;
  active_positions: number;
  top_strategies: StrategyPerformance[];
  attribution: StrategyAttribution;
  risk_metrics: RiskMetrics;
  total_strategies: number;
  data_points: number;
}

export interface StrategyPerformance {
  name: string;
  pnl: number;
}

export type TrendDirection = 'up' | 'down' | 'flat';

export interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ReactNode;
  color?: 'success' | 'warning' | 'error' | 'info';
  trend?: TrendDirection;
  isLoading?: boolean;
}

export interface StrategyTableProps {
  strategies: StrategyPerformance[];
  maxVisible?: number;
  isLoading?: boolean;
  error?: string | null;
}

export interface DashboardState {
  data: HedgeFundData | null;
  loading: boolean;
  error: string | null;
  lastUpdate: Date;
}
