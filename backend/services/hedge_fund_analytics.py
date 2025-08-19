# -*- coding: utf-8 -*-
"""
Hedge Fund Analytics: Institutional-grade performance monitoring and analytics
Multi-strategy fund metrics, model attribution, risk analytics for T470
"""
from __future__ import annotations
import os, time, math, pathlib, threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np
import json

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.getenv("ARIA_DATA_ROOT", PROJECT_ROOT / "data"))


@dataclass
class StrategyMetrics:
    """Performance metrics for individual strategies/models"""

    name: str
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.5
    avg_win: float = 0.0
    avg_loss: float = 0.0
    trade_count: int = 0
    current_drawdown: float = 0.0
    volatility: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0


@dataclass
class RiskMetrics:
    """Risk analytics for hedge fund monitoring"""

    var_1d_99: float = 0.0  # 1-day 99% VaR
    var_1d_95: float = 0.0  # 1-day 95% VaR
    expected_shortfall: float = 0.0
    beta_to_market: float = 0.0
    correlation_to_market: float = 0.0
    max_leverage: float = 0.0
    current_leverage: float = 0.0
    concentration_risk: float = 0.0


class HedgeFundAnalytics:
    """
    Institutional-grade analytics for multi-strategy hedge fund
    Optimized for T470 constraints with real-time performance monitoring
    """

    def __init__(self):
        # Performance tracking
        self.strategy_metrics = {}  # Per-model/strategy metrics
        self.portfolio_returns = deque(maxlen=5000)  # ~3 months of M15 data
        self.daily_returns = deque(maxlen=252)  # 1 year of daily returns
        self.risk_metrics = RiskMetrics()

        # Attribution tracking
        self.model_attributions = deque(maxlen=1000)  # Model contribution tracking
        self.regime_performance = {"T": [], "R": [], "B": []}

        # Real-time monitoring
        self.current_positions = {}
        self.current_pnl = 0.0
        self.daily_pnl = 0.0
        self.inception_pnl = 0.0
        self.high_water_mark = 0.0

        # Risk monitoring
        self.position_limits = {}
        self.risk_budget_used = 0.0
        self.correlation_matrix = {}

        self.lock = threading.Lock()

        # Initialize strategy metrics for all models
        self._initialize_strategies()

    def _initialize_strategies(self):
        """Initialize metrics for all available strategies"""
        strategy_names = [
            "LSTM",
            "CNN",
            "PPO",
            "XGB",
            "XGBoost_Enhanced",
            "LightGBM",
            "MiniTransformer",
            "TinyAutoencoder",
            "BayesianLite",
            "MicroRL",
            "Ensemble",  # Overall ensemble performance
        ]

        for name in strategy_names:
            self.strategy_metrics[name] = StrategyMetrics(name=name)

    def update_trade(
        self,
        symbol: str,
        model_name: str,
        entry_price: float,
        exit_price: Optional[float],
        position_size: float,
        regime: str,
        timestamp: float,
        meta: Optional[Dict] = None,
    ):
        """Update analytics with new trade data"""
        with self.lock:
            if exit_price is not None:
                # Complete trade
                pnl = (exit_price - entry_price) * position_size
                self._update_strategy_pnl(model_name, pnl, timestamp)
                self._update_regime_performance(regime, pnl)
                self._update_portfolio_pnl(pnl, timestamp)
            else:
                # New position
                self.current_positions[f"{symbol}_{model_name}"] = {
                    "symbol": symbol,
                    "model": model_name,
                    "entry_price": entry_price,
                    "position_size": position_size,
                    "regime": regime,
                    "timestamp": timestamp,
                }

    def update_model_attribution(
        self,
        model_contributions: Dict[str, float],
        ensemble_decision: float,
        actual_pnl: Optional[float] = None,
    ):
        """Track model attribution for ensemble decisions"""
        attribution_record = {
            "timestamp": time.time(),
            "contributions": model_contributions.copy(),
            "ensemble_decision": ensemble_decision,
            "actual_pnl": actual_pnl,
        }

        self.model_attributions.append(attribution_record)

        # Update model attribution metrics
        if actual_pnl is not None:
            for model_name, contribution in model_contributions.items():
                attributed_pnl = actual_pnl * (
                    contribution / sum(model_contributions.values())
                )
                self._update_strategy_pnl(model_name, attributed_pnl, time.time())

    def _update_strategy_pnl(self, strategy_name: str, pnl: float, timestamp: float):
        """Update strategy-specific performance metrics"""
        if strategy_name not in self.strategy_metrics:
            self.strategy_metrics[strategy_name] = StrategyMetrics(name=strategy_name)

        strategy = self.strategy_metrics[strategy_name]

        # Update PnL
        strategy.total_pnl += pnl
        strategy.trade_count += 1

        # Update win/loss statistics
        if pnl > 0:
            win_count = strategy.trade_count * strategy.win_rate
            new_win_count = win_count + 1
            strategy.avg_win = (strategy.avg_win * win_count + pnl) / new_win_count
            strategy.win_rate = new_win_count / strategy.trade_count
        elif pnl < 0:
            loss_count = strategy.trade_count * (1 - strategy.win_rate)
            new_loss_count = loss_count + 1
            strategy.avg_loss = (strategy.avg_loss * loss_count + pnl) / new_loss_count

        # Update drawdown
        if strategy.total_pnl > strategy.high_water_mark:
            strategy.high_water_mark = strategy.total_pnl
            strategy.current_drawdown = 0.0
        else:
            strategy.current_drawdown = (
                strategy.high_water_mark - strategy.total_pnl
            ) / abs(strategy.high_water_mark)
            strategy.max_drawdown = max(
                strategy.max_drawdown, strategy.current_drawdown
            )

    def _update_regime_performance(self, regime: str, pnl: float):
        """Track performance by market regime"""
        if regime in self.regime_performance:
            self.regime_performance[regime].append(
                {"pnl": pnl, "timestamp": time.time()}
            )

            # Keep only recent performance (last 1000 trades per regime)
            if len(self.regime_performance[regime]) > 1000:
                self.regime_performance[regime].pop(0)

    def _update_portfolio_pnl(self, pnl: float, timestamp: float):
        """Update overall portfolio performance"""
        self.current_pnl += pnl
        self.daily_pnl += pnl
        self.inception_pnl += pnl

        # Update high water mark and drawdown
        if self.inception_pnl > self.high_water_mark:
            self.high_water_mark = self.inception_pnl

        # Add to returns series
        self.portfolio_returns.append(
            {"pnl": pnl, "cumulative_pnl": self.inception_pnl, "timestamp": timestamp}
        )

    def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics"""
        if len(self.portfolio_returns) < 10:
            return {"error": "Insufficient data for metrics calculation"}

        returns = [r["pnl"] for r in self.portfolio_returns]
        cumulative_pnls = [r["cumulative_pnl"] for r in self.portfolio_returns]

        # Basic metrics
        total_return = (
            cumulative_pnls[-1] - cumulative_pnls[0] if cumulative_pnls else 0.0
        )
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Sharpe ratio (annualized)
        periods_per_year = 35040  # M15 periods
        sharpe_ratio = (mean_return * math.sqrt(periods_per_year)) / (std_return + 1e-8)

        # Maximum drawdown
        max_dd = 0.0
        peak = cumulative_pnls[0]
        for pnl in cumulative_pnls:
            if pnl > peak:
                peak = pnl
            dd = (peak - pnl) / abs(peak) if peak != 0 else 0.0
            max_dd = max(max_dd, dd)

        # Win rate
        wins = sum(1 for r in returns if r > 0)
        win_rate = wins / len(returns)

        # Calmar ratio
        calmar_ratio = (mean_return * periods_per_year) / (max_dd + 1e-8)

        return {
            "total_return": round(total_return, 2),
            "total_return_pct": (
                round(total_return / abs(cumulative_pnls[0]) * 100, 2)
                if cumulative_pnls[0] != 0
                else 0.0
            ),
            "sharpe_ratio": round(sharpe_ratio, 3),
            "max_drawdown": round(max_dd * 100, 2),
            "calmar_ratio": round(calmar_ratio, 3),
            "win_rate": round(win_rate * 100, 1),
            "total_trades": len(returns),
            "avg_trade": round(mean_return, 4),
            "volatility_annualized": round(
                std_return * math.sqrt(periods_per_year) * 100, 2
            ),
            "current_drawdown": (
                round(
                    (self.high_water_mark - self.inception_pnl)
                    / abs(self.high_water_mark)
                    * 100,
                    2,
                )
                if self.high_water_mark != 0
                else 0.0
            ),
        }

    def calculate_strategy_attribution(self) -> Dict[str, Any]:
        """Calculate strategy attribution metrics"""
        attribution = {}

        for strategy_name, metrics in self.strategy_metrics.items():
            if metrics.trade_count > 0:
                attribution[strategy_name] = {
                    "total_pnl": round(metrics.total_pnl, 2),
                    "pnl_contribution_pct": (
                        round(metrics.total_pnl / abs(self.inception_pnl) * 100, 1)
                        if self.inception_pnl != 0
                        else 0.0
                    ),
                    "win_rate": round(metrics.win_rate * 100, 1),
                    "avg_win": round(metrics.avg_win, 4),
                    "avg_loss": round(metrics.avg_loss, 4),
                    "trade_count": metrics.trade_count,
                    "max_drawdown": round(metrics.max_drawdown * 100, 2),
                    "profit_factor": (
                        round(abs(metrics.avg_win / metrics.avg_loss), 2)
                        if metrics.avg_loss != 0
                        else 0.0
                    ),
                }

        return attribution

    def calculate_regime_analysis(self) -> Dict[str, Any]:
        """Analyze performance by market regime"""
        regime_analysis = {}

        for regime, trades in self.regime_performance.items():
            if len(trades) > 0:
                pnls = [trade["pnl"] for trade in trades]
                wins = sum(1 for pnl in pnls if pnl > 0)

                regime_analysis[regime] = {
                    "total_pnl": round(sum(pnls), 2),
                    "avg_pnl": round(np.mean(pnls), 4),
                    "win_rate": round(wins / len(pnls) * 100, 1),
                    "trade_count": len(pnls),
                    "volatility": round(np.std(pnls), 4),
                    "best_trade": round(max(pnls), 4),
                    "worst_trade": round(min(pnls), 4),
                }

        return regime_analysis

    def calculate_risk_metrics(self) -> Dict[str, Any]:
        """Calculate institutional risk metrics"""
        if len(self.portfolio_returns) < 50:
            return {"error": "Insufficient data for risk calculation"}

        returns = [r["pnl"] for r in self.portfolio_returns]

        # VaR calculation (99% and 95%)
        sorted_returns = sorted(returns)
        var_99_idx = int(len(sorted_returns) * 0.01)
        var_95_idx = int(len(sorted_returns) * 0.05)

        var_1d_99 = (
            sorted_returns[var_99_idx]
            if var_99_idx < len(sorted_returns)
            else sorted_returns[0]
        )
        var_1d_95 = (
            sorted_returns[var_95_idx]
            if var_95_idx < len(sorted_returns)
            else sorted_returns[0]
        )

        # Expected Shortfall (CVaR)
        tail_returns = (
            sorted_returns[: var_99_idx + 1] if var_99_idx >= 0 else [sorted_returns[0]]
        )
        expected_shortfall = np.mean(tail_returns) if tail_returns else 0.0

        # Update risk metrics
        self.risk_metrics.var_1d_99 = var_1d_99
        self.risk_metrics.var_1d_95 = var_1d_95
        self.risk_metrics.expected_shortfall = expected_shortfall

        return {
            "var_1d_99": round(var_1d_99, 4),
            "var_1d_95": round(var_1d_95, 4),
            "expected_shortfall": round(expected_shortfall, 4),
            "volatility": round(np.std(returns), 4),
            "skewness": round(self._calculate_skewness(returns), 3),
            "kurtosis": round(self._calculate_kurtosis(returns), 3),
            "max_loss": round(min(returns), 4),
            "downside_deviation": round(self._calculate_downside_deviation(returns), 4),
        }

    def _calculate_skewness(self, returns: List[float]) -> float:
        """Calculate skewness of returns"""
        if len(returns) < 3:
            return 0.0

        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        if std_ret == 0:
            return 0.0

        skew = np.mean([((r - mean_ret) / std_ret) ** 3 for r in returns])
        return float(skew)

    def _calculate_kurtosis(self, returns: List[float]) -> float:
        """Calculate kurtosis of returns"""
        if len(returns) < 4:
            return 0.0

        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        if std_ret == 0:
            return 0.0

        kurt = np.mean([((r - mean_ret) / std_ret) ** 4 for r in returns]) - 3
        return float(kurt)

    def _calculate_downside_deviation(self, returns: List[float]) -> float:
        """Calculate downside deviation (negative returns only)"""
        negative_returns = [r for r in returns if r < 0]
        if len(negative_returns) == 0:
            return 0.0
        return float(np.std(negative_returns))

    def get_live_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data for T470 monitoring"""
        portfolio_metrics = self.calculate_portfolio_metrics()
        strategy_attribution = self.calculate_strategy_attribution()
        regime_analysis = self.calculate_regime_analysis()
        risk_metrics = self.calculate_risk_metrics()

        # Top performing strategies
        top_strategies = sorted(
            [
                (name, metrics.get("total_pnl", 0))
                for name, metrics in strategy_attribution.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        return {
            "timestamp": time.time(),
            "portfolio": portfolio_metrics,
            "current_pnl": round(self.current_pnl, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "inception_pnl": round(self.inception_pnl, 2),
            "active_positions": len(self.current_positions),
            "top_strategies": top_strategies,
            "regime_performance": regime_analysis,
            "risk_metrics": risk_metrics,
            "attribution": strategy_attribution,
            "total_strategies": len(self.strategy_metrics),
            "data_points": len(self.portfolio_returns),
        }

    def export_performance_report(self) -> Dict[str, Any]:
        """Export comprehensive performance report"""
        dashboard_data = self.get_live_dashboard_data()

        # Add detailed analytics
        report = {
            "report_timestamp": time.time(),
            "summary": dashboard_data,
            "detailed_attribution": self.calculate_strategy_attribution(),
            "regime_breakdown": self.calculate_regime_analysis(),
            "risk_analytics": self.calculate_risk_metrics(),
            "position_details": self.current_positions,
            "recent_attributions": (
                list(self.model_attributions)[-100:] if self.model_attributions else []
            ),
        }

        return report

    def reset_daily_metrics(self):
        """Reset daily metrics (call at start of trading day)"""
        with self.lock:
            self.daily_pnl = 0.0

            # Archive daily performance if significant
            if abs(self.daily_pnl) > 0.01:  # Only if meaningful P&L
                self.daily_returns.append(self.daily_pnl)


# Global analytics instance
hedge_fund_analytics = HedgeFundAnalytics()
