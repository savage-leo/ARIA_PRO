"""
Meta-Reinforcement Learning Strategy Selector
Dynamically selects and weights trading strategies based on performance and market conditions
"""

import logging
import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """Performance metrics for a strategy"""
    name: str
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_return: float = 0.0
    max_drawdown: float = 0.0
    recent_performance: float = 0.0  # Last N trades
    regime_performance: Dict[str, float] = field(default_factory=dict)
    execution_quality: float = 0.0
    confidence: float = 0.5


@dataclass
class MetaAction:
    """Meta-RL action: strategy weights and parameters"""
    strategy_weights: Dict[str, float]
    risk_multiplier: float
    time_horizon: str  # 'scalp', 'intraday', 'swing'
    focus_symbols: List[str]
    confidence: float
    reasoning: str


class MetaRLBrain:
    """
    Meta-RL brain using multi-armed bandit with Thompson Sampling
    and contextual awareness for strategy selection
    """
    
    def __init__(self, strategies: List[str]):
        self.strategies = strategies
        
        # Thompson Sampling parameters (Beta distribution)
        self.alpha = {s: 1.0 for s in strategies}  # Successes
        self.beta = {s: 1.0 for s in strategies}   # Failures
        
        # Performance history
        self.performance_window = deque(maxlen=100)
        self.strategy_history = deque(maxlen=500)
        
        # Context embeddings for state representation
        self.context_dim = 10
        self.strategy_embeddings = {}
        self._init_embeddings()
        
        # Q-table for meta-learning
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.1
        
    def _init_embeddings(self):
        """Initialize strategy embeddings"""
        for i, strategy in enumerate(self.strategies):
            # Create orthogonal embeddings
            embedding = np.zeros(self.context_dim)
            embedding[i % self.context_dim] = 1.0
            self.strategy_embeddings[strategy] = embedding
            
    def _get_state_key(self, market_context: Dict) -> str:
        """Convert market context to discrete state key"""
        regime = market_context.get('regime', 'unknown')
        volatility = 'high' if market_context.get('volatility', 0) > 0.02 else 'low'
        trend = 'up' if market_context.get('trend', 0) > 0 else 'down'
        session = market_context.get('session', 'unknown')
        
        return f"{regime}_{volatility}_{trend}_{session}"
        
    def thompson_sample(self) -> Dict[str, float]:
        """Sample strategy weights using Thompson Sampling"""
        samples = {}
        for strategy in self.strategies:
            # Sample from Beta distribution
            samples[strategy] = np.random.beta(
                self.alpha[strategy],
                self.beta[strategy]
            )
        
        # Normalize to sum to 1
        total = sum(samples.values())
        if total > 0:
            weights = {s: v/total for s, v in samples.items()}
        else:
            weights = {s: 1.0/len(self.strategies) for s in self.strategies}
            
        return weights
        
    def update_thompson(self, strategy: str, reward: float):
        """Update Thompson Sampling parameters"""
        # Convert reward to success/failure
        if reward > 0:
            self.alpha[strategy] += reward
        else:
            self.beta[strategy] += abs(reward)
            
    def get_q_value(self, state: str, action: str) -> float:
        """Get Q-value for state-action pair"""
        return self.q_table.get(f"{state}_{action}", 0.0)
        
    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-value using Q-learning"""
        key = f"{state}_{action}"
        
        # Get current Q-value
        current_q = self.q_table.get(key, 0.0)
        
        # Get max Q-value for next state
        next_actions = [s for s in self.strategies]
        max_next_q = max([self.get_q_value(next_state, a) for a in next_actions])
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[key] = new_q
        
    def select_action(self, market_context: Dict) -> Tuple[Dict[str, float], str]:
        """
        Select strategy weights using epsilon-greedy with Q-learning
        Returns: (weights, selected_primary_strategy)
        """
        state = self._get_state_key(market_context)
        
        # Epsilon-greedy exploration
        if np.random.random() < self.exploration_rate:
            # Explore: use Thompson Sampling
            weights = self.thompson_sample()
            primary = max(weights, key=weights.get)
        else:
            # Exploit: use Q-values
            q_values = {s: self.get_q_value(state, s) for s in self.strategies}
            
            # Softmax over Q-values for weights
            exp_q = {s: np.exp(v) for s, v in q_values.items()}
            total = sum(exp_q.values())
            
            if total > 0:
                weights = {s: v/total for s, v in exp_q.items()}
            else:
                weights = {s: 1.0/len(self.strategies) for s in self.strategies}
                
            primary = max(q_values, key=q_values.get)
            
        return weights, primary


class StrategyPerformanceTracker:
    """Track and analyze strategy performance"""
    
    def __init__(self):
        self.trades = deque(maxlen=1000)
        self.daily_returns = {}
        self.strategy_stats = {}
        
    def record_trade(self, strategy: str, symbol: str, return_pct: float, 
                    regime: str, execution_quality: float):
        """Record a trade outcome"""
        self.trades.append({
            'strategy': strategy,
            'symbol': symbol,
            'return': return_pct,
            'regime': regime,
            'execution': execution_quality,
            'timestamp': time.time()
        })
        
    def calculate_metrics(self, strategy: str, window: int = 100) -> StrategyMetrics:
        """Calculate performance metrics for a strategy"""
        # Filter trades for this strategy
        strategy_trades = [t for t in self.trades if t['strategy'] == strategy]
        
        if not strategy_trades:
            return StrategyMetrics(name=strategy)
            
        recent_trades = strategy_trades[-window:]
        returns = [t['return'] for t in recent_trades]
        
        # Calculate metrics
        metrics = StrategyMetrics(name=strategy)
        
        if returns:
            metrics.avg_return = np.mean(returns)
            metrics.win_rate = len([r for r in returns if r > 0]) / len(returns)
            
            # Sharpe ratio (simplified)
            if len(returns) > 1:
                return_std = np.std(returns)
                if return_std > 0:
                    metrics.sharpe_ratio = metrics.avg_return / return_std
                    
            # Max drawdown
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            metrics.max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
            
            # Recent performance (last 20 trades)
            if len(returns) >= 20:
                metrics.recent_performance = np.mean(returns[-20:])
            else:
                metrics.recent_performance = metrics.avg_return
                
            # Regime-specific performance
            for regime in ['trend', 'range', 'breakout']:
                regime_trades = [t for t in recent_trades if t['regime'] == regime]
                if regime_trades:
                    regime_returns = [t['return'] for t in regime_trades]
                    metrics.regime_performance[regime] = np.mean(regime_returns)
                    
            # Execution quality
            exec_scores = [t['execution'] for t in recent_trades]
            metrics.execution_quality = np.mean(exec_scores) if exec_scores else 0.5
            
            # Overall confidence (0-1)
            metrics.confidence = min(1.0, max(0.0,
                0.3 * (metrics.win_rate) +
                0.3 * (metrics.sharpe_ratio / 2.0 + 0.5) +
                0.2 * (1.0 - abs(metrics.max_drawdown) / 0.1) +
                0.2 * metrics.execution_quality
            ))
            
        return metrics


class MetaRLSelector:
    """Main Meta-RL Strategy Selector"""
    
    def __init__(self):
        # Available strategies
        self.strategies = [
            'momentum',
            'mean_reversion', 
            'breakout',
            'arbitrage',
            'market_making',
            'trend_following',
            'volatility',
            'pairs_trading'
        ]
        
        # Components
        self.brain = MetaRLBrain(self.strategies)
        self.tracker = StrategyPerformanceTracker()
        
        # Current state
        self.current_weights = {s: 1.0/len(self.strategies) for s in self.strategies}
        self.current_primary = 'momentum'
        self.last_market_state = None
        
        # Meta parameters
        self.risk_appetite = 1.0
        self.adaptation_speed = 0.1
        
    async def select_strategies(self, market_context: Dict) -> MetaAction:
        """
        Select optimal strategy mix based on market conditions
        Returns MetaAction with strategy weights and parameters
        """
        
        # Calculate performance metrics for all strategies
        strategy_metrics = {}
        for strategy in self.strategies:
            metrics = self.tracker.calculate_metrics(strategy)
            strategy_metrics[strategy] = metrics
            
        # Get brain's recommendation
        weights, primary = self.brain.select_action(market_context)
        
        # Adjust weights based on recent performance
        adjusted_weights = self._adjust_weights_by_performance(
            weights, strategy_metrics, market_context
        )
        
        # Determine risk multiplier based on overall performance
        risk_mult = self._calculate_risk_multiplier(strategy_metrics, market_context)
        
        # Select time horizon based on market conditions
        time_horizon = self._select_time_horizon(market_context)
        
        # Focus on best performing symbols
        focus_symbols = self._select_focus_symbols(market_context)
        
        # Calculate meta confidence
        meta_confidence = self._calculate_meta_confidence(
            strategy_metrics, market_context
        )
        
        # Update current state
        self.current_weights = adjusted_weights
        self.current_primary = primary
        self.last_market_state = self.brain._get_state_key(market_context)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            adjusted_weights, primary, strategy_metrics, market_context
        )
        
        return MetaAction(
            strategy_weights=adjusted_weights,
            risk_multiplier=risk_mult,
            time_horizon=time_horizon,
            focus_symbols=focus_symbols,
            confidence=meta_confidence,
            reasoning=reasoning
        )
        
    def _adjust_weights_by_performance(self, base_weights: Dict[str, float],
                                      metrics: Dict[str, StrategyMetrics],
                                      context: Dict) -> Dict[str, float]:
        """Adjust strategy weights based on recent performance"""
        
        adjusted = base_weights.copy()
        current_regime = context.get('regime', 'unknown')
        
        for strategy, metric in metrics.items():
            if strategy not in adjusted:
                continue
                
            # Boost strategies performing well in current regime
            if current_regime in metric.regime_performance:
                regime_perf = metric.regime_performance[current_regime]
                if regime_perf > 0:
                    adjusted[strategy] *= (1.0 + regime_perf)
                    
            # Reduce weight for underperforming strategies
            if metric.recent_performance < -0.02:  # -2% recent performance
                adjusted[strategy] *= 0.7
                
            # Boost high Sharpe strategies
            if metric.sharpe_ratio > 1.5:
                adjusted[strategy] *= 1.2
                
        # Renormalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {s: v/total for s, v in adjusted.items()}
            
        return adjusted
        
    def _calculate_risk_multiplier(self, metrics: Dict[str, StrategyMetrics],
                                  context: Dict) -> float:
        """Calculate dynamic risk multiplier"""
        
        # Base risk from market volatility
        volatility = context.get('volatility', 0.01)
        base_risk = 1.0
        
        if volatility > 0.03:  # High volatility
            base_risk = 0.7
        elif volatility < 0.01:  # Low volatility
            base_risk = 1.2
            
        # Adjust by overall strategy performance
        avg_sharpe = np.mean([m.sharpe_ratio for m in metrics.values()])
        avg_win_rate = np.mean([m.win_rate for m in metrics.values()])
        
        performance_mult = 1.0
        if avg_sharpe > 1.0 and avg_win_rate > 0.55:
            performance_mult = 1.3  # Increase risk when performing well
        elif avg_sharpe < 0.5 or avg_win_rate < 0.45:
            performance_mult = 0.7  # Reduce risk when underperforming
            
        return base_risk * performance_mult * self.risk_appetite
        
    def _select_time_horizon(self, context: Dict) -> str:
        """Select optimal time horizon based on market conditions"""
        
        volatility = context.get('volatility', 0.01)
        session = context.get('session', 'unknown')
        regime = context.get('regime', 'range')
        
        # High volatility -> shorter timeframe
        if volatility > 0.025:
            return 'scalp'
            
        # Trending market -> longer timeframe
        if regime == 'trend':
            return 'swing'
            
        # Active session -> intraday
        if session in ['london', 'newyork']:
            return 'intraday'
            
        return 'intraday'  # Default
        
    def _select_focus_symbols(self, context: Dict) -> List[str]:
        """Select symbols to focus on"""
        
        # Get symbol performance from context
        symbol_performance = context.get('symbol_performance', {})
        
        if not symbol_performance:
            # Default focus
            return ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
            
        # Sort by recent performance
        sorted_symbols = sorted(
            symbol_performance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Take top performers
        return [s[0] for s in sorted_symbols[:4]]
        
    def _calculate_meta_confidence(self, metrics: Dict[str, StrategyMetrics],
                                  context: Dict) -> float:
        """Calculate overall meta-strategy confidence"""
        
        # Average strategy confidence
        avg_confidence = np.mean([m.confidence for m in metrics.values()])
        
        # Market condition clarity
        regime_clarity = 1.0 if context.get('regime', 'unknown') != 'unknown' else 0.5
        
        # Recent win rate
        recent_trades = list(self.tracker.trades)[-50:]
        if recent_trades:
            recent_win_rate = len([t for t in recent_trades if t['return'] > 0]) / len(recent_trades)
        else:
            recent_win_rate = 0.5
            
        # Combined confidence
        meta_confidence = (
            avg_confidence * 0.4 +
            regime_clarity * 0.3 +
            recent_win_rate * 0.3
        )
        
        return min(1.0, max(0.0, meta_confidence))
        
    def _generate_reasoning(self, weights: Dict[str, float], primary: str,
                          metrics: Dict[str, StrategyMetrics], context: Dict) -> str:
        """Generate human-readable reasoning for strategy selection"""
        
        regime = context.get('regime', 'unknown')
        volatility = context.get('volatility', 0.01)
        
        # Find top weighted strategies
        top_strategies = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
        
        reasoning = f"Primary: {primary} ({weights[primary]:.1%}). "
        reasoning += f"Market: {regime} regime, "
        reasoning += f"{'high' if volatility > 0.02 else 'normal'} volatility. "
        
        # Add performance reason
        if primary in metrics:
            m = metrics[primary]
            reasoning += f"{primary} Sharpe: {m.sharpe_ratio:.2f}, "
            reasoning += f"Win rate: {m.win_rate:.1%}. "
            
        reasoning += f"Top mix: {', '.join([f'{s[0]}({s[1]:.0%})' for s in top_strategies])}"
        
        return reasoning
        
    def record_outcome(self, strategy: str, symbol: str, return_pct: float,
                      regime: str, execution_quality: float):
        """Record trade outcome for learning"""
        
        # Record in tracker
        self.tracker.record_trade(strategy, symbol, return_pct, regime, execution_quality)
        
        # Update Thompson Sampling
        self.brain.update_thompson(strategy, return_pct)
        
        # Update Q-learning if we have state transition
        if self.last_market_state:
            current_state = self.brain._get_state_key({'regime': regime})
            self.brain.update_q_value(
                self.last_market_state,
                strategy,
                return_pct,
                current_state
            )
            
    def get_strategy_report(self) -> Dict[str, Any]:
        """Get comprehensive strategy performance report"""
        
        report = {
            'current_weights': self.current_weights,
            'current_primary': self.current_primary,
            'strategy_metrics': {},
            'recent_performance': {},
            'q_table_size': len(self.brain.q_table),
            'total_trades': len(self.tracker.trades)
        }
        
        for strategy in self.strategies:
            metrics = self.tracker.calculate_metrics(strategy)
            report['strategy_metrics'][strategy] = {
                'sharpe_ratio': metrics.sharpe_ratio,
                'win_rate': metrics.win_rate,
                'avg_return': metrics.avg_return,
                'max_drawdown': metrics.max_drawdown,
                'confidence': metrics.confidence
            }
            
        # Recent 24h performance
        cutoff = time.time() - 86400
        recent = [t for t in self.tracker.trades if t['timestamp'] > cutoff]
        if recent:
            report['recent_performance'] = {
                'total_trades': len(recent),
                'win_rate': len([t for t in recent if t['return'] > 0]) / len(recent),
                'avg_return': np.mean([t['return'] for t in recent])
            }
            
        return report


# Global instance
_meta_selector = None


def get_meta_selector() -> MetaRLSelector:
    """Get or create the global Meta-RL selector"""
    global _meta_selector
    if _meta_selector is None:
        _meta_selector = MetaRLSelector()
    return _meta_selector
