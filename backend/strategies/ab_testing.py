"""
A/B Testing Framework for Trading Strategies
Statistical evaluation and performance comparison of trading strategies
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import defaultdict, deque
import json
from scipy import stats
import hashlib

logger = logging.getLogger(__name__)


class ABTestingFramework:
    """
    A/B Testing framework for trading strategy evaluation
    Implements statistical tests and performance metrics
    """
    
    def __init__(self):
        self.active_tests = {}
        self.test_results = defaultdict(list)
        self.strategy_metrics = defaultdict(lambda: {
            'trades': [],
            'pnl': [],
            'win_rate': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'avg_trade_duration': 0,
            'total_volume': 0
        })
        
        # Test configuration
        self.min_samples = 30  # Minimum trades for statistical significance
        self.confidence_level = 0.95
        self.test_duration = timedelta(days=7)  # Default test duration
        
    def create_test(self, test_name: str, strategies: List[str], 
                   allocation: Dict[str, float] = None) -> str:
        """
        Create a new A/B test
        
        Args:
            test_name: Name of the test
            strategies: List of strategy names to test
            allocation: Traffic allocation percentages (default: equal split)
        
        Returns:
            Test ID
        """
        test_id = hashlib.md5(f"{test_name}_{datetime.now()}".encode()).hexdigest()[:8]
        
        if allocation is None:
            # Equal allocation by default
            allocation = {s: 1.0 / len(strategies) for s in strategies}
        
        self.active_tests[test_id] = {
            'name': test_name,
            'strategies': strategies,
            'allocation': allocation,
            'start_time': datetime.now(),
            'end_time': datetime.now() + self.test_duration,
            'status': 'active',
            'results': {}
        }
        
        logger.info(f"Created A/B test {test_id}: {test_name} with strategies {strategies}")
        
        return test_id
    
    def select_strategy(self, test_id: str, context: Dict = None) -> str:
        """
        Select strategy based on test allocation
        
        Args:
            test_id: Test identifier
            context: Optional context for contextual bandits
        
        Returns:
            Selected strategy name
        """
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test = self.active_tests[test_id]
        
        if test['status'] != 'active':
            # Return winner if test completed
            if 'winner' in test:
                return test['winner']
            # Otherwise return first strategy
            return test['strategies'][0]
        
        # Multi-armed bandit selection (epsilon-greedy)
        epsilon = 0.1  # Exploration rate
        
        if np.random.random() < epsilon:
            # Explore: random selection based on allocation
            return self._weighted_random_selection(test['allocation'])
        else:
            # Exploit: select best performing strategy
            return self._get_best_strategy(test_id)
    
    def _weighted_random_selection(self, allocation: Dict[str, float]) -> str:
        """Select strategy based on weighted allocation"""
        strategies = list(allocation.keys())
        weights = list(allocation.values())
        return np.random.choice(strategies, p=weights)
    
    def _get_best_strategy(self, test_id: str) -> str:
        """Get best performing strategy based on current metrics"""
        test = self.active_tests[test_id]
        best_strategy = test['strategies'][0]
        best_score = -float('inf')
        
        for strategy in test['strategies']:
            metrics = self.strategy_metrics[strategy]
            if len(metrics['trades']) > 0:
                # Score based on Sharpe ratio and win rate
                score = metrics['sharpe_ratio'] * 0.6 + metrics['win_rate'] * 0.4
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
        
        return best_strategy
    
    def record_trade_result(self, test_id: str, strategy: str, result: Dict):
        """
        Record trade result for a strategy
        
        Args:
            test_id: Test identifier
            strategy: Strategy name
            result: Trade result with pnl, duration, etc.
        """
        if test_id not in self.active_tests:
            return
        
        # Update strategy metrics
        metrics = self.strategy_metrics[strategy]
        metrics['trades'].append(result)
        metrics['pnl'].append(result.get('pnl', 0))
        
        # Calculate updated metrics
        self._update_metrics(strategy)
        
        # Check if test should be concluded
        self._check_test_completion(test_id)
    
    def _update_metrics(self, strategy: str):
        """Update performance metrics for a strategy"""
        metrics = self.strategy_metrics[strategy]
        trades = metrics['trades']
        
        if len(trades) == 0:
            return
        
        # Win rate
        wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
        metrics['win_rate'] = wins / len(trades)
        
        # Sharpe ratio (simplified)
        if len(metrics['pnl']) > 1:
            returns = np.array(metrics['pnl'])
            if np.std(returns) > 0:
                metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                metrics['sharpe_ratio'] = 0
        
        # Max drawdown
        cumulative_pnl = np.cumsum(metrics['pnl'])
        if len(cumulative_pnl) > 0:
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = (cumulative_pnl - running_max) / (running_max + 1e-10)
            metrics['max_drawdown'] = np.min(drawdown)
        
        # Average trade duration
        durations = [t.get('duration', 0) for t in trades]
        metrics['avg_trade_duration'] = np.mean(durations) if durations else 0
        
        # Total volume
        metrics['total_volume'] = sum(t.get('volume', 0) for t in trades)
    
    def _check_test_completion(self, test_id: str):
        """Check if test should be concluded"""
        test = self.active_tests[test_id]
        
        # Check time limit
        if datetime.now() > test['end_time']:
            self._conclude_test(test_id)
            return
        
        # Check sample size for all strategies
        all_have_min_samples = True
        for strategy in test['strategies']:
            if len(self.strategy_metrics[strategy]['trades']) < self.min_samples:
                all_have_min_samples = False
                break
        
        if all_have_min_samples:
            # Perform statistical significance test
            if self._is_statistically_significant(test_id):
                self._conclude_test(test_id)
    
    def _is_statistically_significant(self, test_id: str) -> bool:
        """Check if results are statistically significant"""
        test = self.active_tests[test_id]
        
        if len(test['strategies']) != 2:
            # For multiple strategies, use ANOVA
            return self._perform_anova(test_id)
        
        # For two strategies, use t-test
        strategy_a, strategy_b = test['strategies']
        pnl_a = self.strategy_metrics[strategy_a]['pnl']
        pnl_b = self.strategy_metrics[strategy_b]['pnl']
        
        if len(pnl_a) < self.min_samples or len(pnl_b) < self.min_samples:
            return False
        
        # Perform Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(pnl_a, pnl_b, equal_var=False)
        
        # Check if p-value is below significance level
        alpha = 1 - self.confidence_level
        return p_value < alpha
    
    def _perform_anova(self, test_id: str) -> bool:
        """Perform ANOVA test for multiple strategies"""
        test = self.active_tests[test_id]
        
        groups = []
        for strategy in test['strategies']:
            pnl = self.strategy_metrics[strategy]['pnl']
            if len(pnl) >= self.min_samples:
                groups.append(pnl)
        
        if len(groups) < 2:
            return False
        
        # Perform one-way ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        alpha = 1 - self.confidence_level
        return p_value < alpha
    
    def _conclude_test(self, test_id: str):
        """Conclude test and determine winner"""
        test = self.active_tests[test_id]
        test['status'] = 'completed'
        test['end_time'] = datetime.now()
        
        # Determine winner based on Sharpe ratio
        best_strategy = None
        best_sharpe = -float('inf')
        
        for strategy in test['strategies']:
            metrics = self.strategy_metrics[strategy]
            if metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = metrics['sharpe_ratio']
                best_strategy = strategy
        
        test['winner'] = best_strategy
        
        # Store results
        test['results'] = {
            strategy: {
                'win_rate': self.strategy_metrics[strategy]['win_rate'],
                'sharpe_ratio': self.strategy_metrics[strategy]['sharpe_ratio'],
                'total_pnl': sum(self.strategy_metrics[strategy]['pnl']),
                'num_trades': len(self.strategy_metrics[strategy]['trades']),
                'max_drawdown': self.strategy_metrics[strategy]['max_drawdown']
            }
            for strategy in test['strategies']
        }
        
        logger.info(f"Test {test_id} concluded. Winner: {best_strategy} with Sharpe {best_sharpe:.2f}")
    
    def get_test_results(self, test_id: str) -> Dict:
        """Get results for a specific test"""
        if test_id not in self.active_tests:
            return {}
        
        test = self.active_tests[test_id]
        
        return {
            'test_id': test_id,
            'name': test['name'],
            'status': test['status'],
            'strategies': test['strategies'],
            'start_time': test['start_time'].isoformat(),
            'end_time': test['end_time'].isoformat(),
            'winner': test.get('winner'),
            'results': test.get('results', {}),
            'current_metrics': {
                strategy: {
                    'win_rate': self.strategy_metrics[strategy]['win_rate'],
                    'sharpe_ratio': self.strategy_metrics[strategy]['sharpe_ratio'],
                    'num_trades': len(self.strategy_metrics[strategy]['trades']),
                    'total_pnl': sum(self.strategy_metrics[strategy]['pnl'])
                }
                for strategy in test['strategies']
            }
        }
    
    def get_all_active_tests(self) -> List[Dict]:
        """Get all active tests"""
        active_tests = []
        
        for test_id, test in self.active_tests.items():
            if test['status'] == 'active':
                active_tests.append(self.get_test_results(test_id))
        
        return active_tests
    
    def calculate_statistical_power(self, effect_size: float, 
                                  sample_size: int) -> float:
        """
        Calculate statistical power of a test
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            sample_size: Sample size per group
        
        Returns:
            Statistical power (0-1)
        """
        # Simplified power calculation
        # In production, use statsmodels.stats.power
        
        alpha = 1 - self.confidence_level
        critical_z = stats.norm.ppf(1 - alpha/2)
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(sample_size / 2)
        
        # Power = P(|Z| > critical_z | H1 is true)
        power = 1 - stats.norm.cdf(critical_z - ncp) + stats.norm.cdf(-critical_z - ncp)
        
        return power
    
    def recommend_sample_size(self, effect_size: float = 0.5, 
                             power: float = 0.8) -> int:
        """
        Recommend sample size for desired power
        
        Args:
            effect_size: Expected effect size
            power: Desired statistical power
        
        Returns:
            Recommended sample size per group
        """
        # Binary search for sample size
        low, high = 10, 10000
        
        while low < high:
            mid = (low + high) // 2
            calculated_power = self.calculate_statistical_power(effect_size, mid)
            
            if calculated_power < power:
                low = mid + 1
            else:
                high = mid
        
        return low


# Singleton instance
ab_testing = ABTestingFramework()


# Example usage functions
async def run_strategy_test():
    """Example of running an A/B test"""
    
    # Create test comparing strategies
    test_id = ab_testing.create_test(
        "Quick Profit vs Traditional",
        strategies=["quick_profit", "traditional_fusion", "arbitrage"],
        allocation={"quick_profit": 0.4, "traditional_fusion": 0.3, "arbitrage": 0.3}
    )
    
    # Simulate trades
    for i in range(100):
        # Select strategy for this trade
        strategy = ab_testing.select_strategy(test_id)
        
        # Simulate trade result
        result = {
            'pnl': np.random.randn() * 10,  # Random P&L
            'duration': np.random.randint(60, 3600),  # Duration in seconds
            'volume': np.random.uniform(0.1, 2.0)  # Volume in lots
        }
        
        # Record result
        ab_testing.record_trade_result(test_id, strategy, result)
        
        await asyncio.sleep(0.1)
    
    # Get results
    results = ab_testing.get_test_results(test_id)
    print(f"Test Results: {json.dumps(results, indent=2)}")
    
    # Get recommended sample size
    sample_size = ab_testing.recommend_sample_size(effect_size=0.3, power=0.9)
    print(f"Recommended sample size: {sample_size} trades per strategy")
