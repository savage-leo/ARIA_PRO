"""
Institutional-Grade Volatility Stress Tests
Tests AI models under extreme market conditions and cross-pair validation
"""

import numpy as np
import pandas as pd
import logging
import time
import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import asyncio

from backend.core.model_loader import cached_models
from backend.services.real_ai_signal_generator import RealAISignalGenerator
from backend.services.mt5_market_data import mt5_market_feed

logger = logging.getLogger(__name__)

@dataclass
class StressTestResult:
    """Result of a stress test scenario"""
    scenario: str
    symbol: str
    duration: float
    signals_generated: int
    error_rate: float
    avg_signal_strength: float
    volatility_handled: float
    performance_degradation: float
    passed: bool
    details: Dict[str, Any]


class VolatilityStressTester:
    """Comprehensive volatility and cross-pair stress testing"""
    
    def __init__(self):
        self.signal_generator = RealAISignalGenerator()
        self.results: List[StressTestResult] = []
        
        # Major currency pairs for cross-validation
        self.major_pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
        self.exotic_pairs = ["EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURCHF", "GBPCHF"]
        self.commodity_pairs = ["XAUUSD", "XAGUSD", "USOIL", "BTCUSD"]
        
    def generate_synthetic_volatility_data(self, base_price: float = 1.1000, 
                                         volatility_factor: float = 1.0,
                                         bars_count: int = 1000,
                                         regime: str = "normal") -> List[Dict[str, Any]]:
        """Generate synthetic market data with controlled volatility"""
        
        np.random.seed(42)  # Reproducible tests
        
        bars = []
        current_price = base_price
        
        for i in range(bars_count):
            # Different volatility regimes
            if regime == "high_volatility":
                price_change = np.random.normal(0, 0.01 * volatility_factor)
            elif regime == "flash_crash":
                if i == bars_count // 2:  # Flash crash in middle
                    price_change = -0.05 * volatility_factor
                elif i > bars_count // 2 and i < bars_count // 2 + 10:
                    price_change = np.random.normal(0.002, 0.02 * volatility_factor)  # Recovery
                else:
                    price_change = np.random.normal(0, 0.002 * volatility_factor)
            elif regime == "trending":
                trend = 0.0001 * volatility_factor if i < bars_count // 2 else -0.0001 * volatility_factor
                price_change = trend + np.random.normal(0, 0.003 * volatility_factor)
            elif regime == "sideways":
                # Mean reversion
                deviation = current_price - base_price
                price_change = -deviation * 0.1 + np.random.normal(0, 0.002 * volatility_factor)
            else:  # normal
                price_change = np.random.normal(0, 0.005 * volatility_factor)
            
            current_price += price_change
            
            # Generate OHLC from current price
            spread = 0.0002 * volatility_factor
            high = current_price + abs(np.random.normal(0, spread))
            low = current_price - abs(np.random.normal(0, spread))
            open_price = current_price - price_change
            
            bar = {
                "ts": time.time() - (bars_count - i) * 60,
                "o": max(low, min(high, open_price)),
                "h": high,
                "l": low,
                "c": current_price,
                "v": max(1, int(np.random.normal(1000, 200 * volatility_factor))),
                "symbol": "SYNTHETIC"
            }
            bars.append(bar)
        
        return bars
    
    def test_high_volatility_scenario(self, symbol: str = "EURUSD") -> StressTestResult:
        """Test model performance under high volatility conditions"""
        logger.info(f"Testing high volatility scenario for {symbol}")
        
        start_time = time.time()
        
        # Generate high volatility data
        volatile_bars = self.generate_synthetic_volatility_data(
            volatility_factor=5.0,
            regime="high_volatility",
            bars_count=500
        )
        
        # Update symbol in bars
        for bar in volatile_bars:
            bar["symbol"] = symbol
        
        signals_generated = 0
        errors = 0
        signal_strengths = []
        
        try:
            # Test signal generation under stress
            for i in range(0, len(volatile_bars), 50):
                test_bars = volatile_bars[max(0, i-100):i+1]
                if len(test_bars) < 50:
                    continue
                
                try:
                    signals = self.signal_generator._generate_ai_signals(symbol, test_bars)
                    signals_generated += 1
                    
                    # Calculate average signal strength
                    if signals:
                        avg_strength = np.mean([abs(v) for v in signals.values() if v is not None])
                        signal_strengths.append(avg_strength)
                
                except Exception as e:
                    errors += 1
                    logger.warning(f"Signal generation error in high volatility: {e}")
        
        except Exception as e:
            logger.error(f"High volatility test failed: {e}")
            errors += 1
        
        duration = time.time() - start_time
        error_rate = errors / max(signals_generated + errors, 1)
        avg_signal_strength = np.mean(signal_strengths) if signal_strengths else 0.0
        
        # Calculate volatility handled
        price_changes = [abs(volatile_bars[i]["c"] - volatile_bars[i-1]["c"]) / volatile_bars[i-1]["c"] 
                        for i in range(1, len(volatile_bars))]
        volatility_handled = np.mean(price_changes) * 100  # As percentage
        
        # Performance criteria
        passed = (error_rate < 0.1 and  # Less than 10% error rate
                 avg_signal_strength > 0.1 and  # Meaningful signals
                 signals_generated > 5)  # Generated reasonable number of signals
        
        result = StressTestResult(
            scenario="high_volatility",
            symbol=symbol,
            duration=duration,
            signals_generated=signals_generated,
            error_rate=error_rate,
            avg_signal_strength=avg_signal_strength,
            volatility_handled=volatility_handled,
            performance_degradation=error_rate * 100,
            passed=passed,
            details={
                "total_bars": len(volatile_bars),
                "errors": errors,
                "max_price_change": max(price_changes) * 100,
                "signal_distribution": dict(zip(["lstm", "cnn", "ppo", "xgb"], 
                                              [len([s for s in signal_strengths if s > 0.1]) for _ in range(4)]))
            }
        )
        
        self.results.append(result)
        return result
    
    def test_flash_crash_scenario(self, symbol: str = "EURUSD") -> StressTestResult:
        """Test model resilience during flash crash events"""
        logger.info(f"Testing flash crash scenario for {symbol}")
        
        start_time = time.time()
        
        # Generate flash crash data
        crash_bars = self.generate_synthetic_volatility_data(
            volatility_factor=3.0,
            regime="flash_crash",
            bars_count=200
        )
        
        for bar in crash_bars:
            bar["symbol"] = symbol
        
        signals_generated = 0
        errors = 0
        signal_strengths = []
        crash_signals = []
        
        try:
            # Focus on crash period and recovery
            crash_start = len(crash_bars) // 2 - 10
            crash_end = len(crash_bars) // 2 + 20
            
            for i in range(crash_start, crash_end, 5):
                test_bars = crash_bars[max(0, i-50):i+1]
                if len(test_bars) < 30:
                    continue
                
                try:
                    signals = self.signal_generator._generate_ai_signals(symbol, test_bars)
                    signals_generated += 1
                    
                    if signals:
                        avg_strength = np.mean([abs(v) for v in signals.values() if v is not None])
                        signal_strengths.append(avg_strength)
                        
                        # Track signals during crash
                        if crash_start <= i <= crash_start + 10:
                            crash_signals.append(signals)
                
                except Exception as e:
                    errors += 1
                    logger.warning(f"Signal generation error during flash crash: {e}")
        
        except Exception as e:
            logger.error(f"Flash crash test failed: {e}")
            errors += 1
        
        duration = time.time() - start_time
        error_rate = errors / max(signals_generated + errors, 1)
        avg_signal_strength = np.mean(signal_strengths) if signal_strengths else 0.0
        
        # Calculate crash magnitude
        pre_crash_price = crash_bars[len(crash_bars) // 2 - 1]["c"]
        crash_price = crash_bars[len(crash_bars) // 2]["c"]
        crash_magnitude = abs(crash_price - pre_crash_price) / pre_crash_price * 100
        
        # Performance criteria - more lenient for extreme events
        passed = (error_rate < 0.2 and  # Less than 20% error rate during crash
                 signals_generated > 3 and  # Some signals generated
                 len(crash_signals) > 0)  # Signals during crash period
        
        result = StressTestResult(
            scenario="flash_crash",
            symbol=symbol,
            duration=duration,
            signals_generated=signals_generated,
            error_rate=error_rate,
            avg_signal_strength=avg_signal_strength,
            volatility_handled=crash_magnitude,
            performance_degradation=error_rate * 100,
            passed=passed,
            details={
                "crash_magnitude_percent": crash_magnitude,
                "crash_signals": len(crash_signals),
                "errors": errors,
                "recovery_signals": signals_generated - len(crash_signals)
            }
        )
        
        self.results.append(result)
        return result
    
    def test_cross_pair_validation(self, test_pairs: List[str] = None) -> List[StressTestResult]:
        """Test model consistency across different currency pairs"""
        logger.info("Testing cross-pair validation")
        
        if test_pairs is None:
            test_pairs = self.major_pairs[:5]  # Test top 5 major pairs
        
        results = []
        baseline_signals = {}
        
        # Generate consistent test data for all pairs
        base_bars = self.generate_synthetic_volatility_data(
            volatility_factor=1.0,
            regime="normal",
            bars_count=200
        )
        
        for symbol in test_pairs:
            start_time = time.time()
            
            # Adjust base data for different pairs
            test_bars = []
            for bar in base_bars:
                adjusted_bar = bar.copy()
                adjusted_bar["symbol"] = symbol
                
                # Adjust price levels for different pairs
                if "JPY" in symbol:
                    price_multiplier = 110.0
                elif "XAU" in symbol:
                    price_multiplier = 1800.0
                elif "BTC" in symbol:
                    price_multiplier = 45000.0
                else:
                    price_multiplier = 1.0
                
                for price_key in ["o", "h", "l", "c"]:
                    adjusted_bar[price_key] *= price_multiplier
                
                test_bars.append(adjusted_bar)
            
            signals_generated = 0
            errors = 0
            signal_strengths = []
            pair_signals = []
            
            try:
                # Test signal generation for this pair
                for i in range(50, len(test_bars), 25):
                    test_window = test_bars[i-50:i+1]
                    
                    try:
                        signals = self.signal_generator._generate_ai_signals(symbol, test_window)
                        signals_generated += 1
                        
                        if signals:
                            avg_strength = np.mean([abs(v) for v in signals.values() if v is not None])
                            signal_strengths.append(avg_strength)
                            pair_signals.append(signals)
                    
                    except Exception as e:
                        errors += 1
                        logger.warning(f"Cross-pair validation error for {symbol}: {e}")
            
            except Exception as e:
                logger.error(f"Cross-pair test failed for {symbol}: {e}")
                errors += 1
            
            duration = time.time() - start_time
            error_rate = errors / max(signals_generated + errors, 1)
            avg_signal_strength = np.mean(signal_strengths) if signal_strengths else 0.0
            
            # Store baseline for comparison
            if symbol == test_pairs[0]:
                baseline_signals = {
                    "avg_strength": avg_signal_strength,
                    "error_rate": error_rate,
                    "signals_count": signals_generated
                }
            
            # Calculate consistency with baseline
            consistency_score = 1.0
            if baseline_signals:
                strength_diff = abs(avg_signal_strength - baseline_signals["avg_strength"])
                error_diff = abs(error_rate - baseline_signals["error_rate"])
                consistency_score = max(0.0, 1.0 - (strength_diff + error_diff))
            
            # Performance criteria
            passed = (error_rate < 0.15 and  # Less than 15% error rate
                     avg_signal_strength > 0.05 and  # Some signal strength
                     consistency_score > 0.7)  # Reasonable consistency
            
            result = StressTestResult(
                scenario="cross_pair_validation",
                symbol=symbol,
                duration=duration,
                signals_generated=signals_generated,
                error_rate=error_rate,
                avg_signal_strength=avg_signal_strength,
                volatility_handled=consistency_score * 100,
                performance_degradation=(1.0 - consistency_score) * 100,
                passed=passed,
                details={
                    "consistency_score": consistency_score,
                    "baseline_comparison": baseline_signals.copy() if baseline_signals else {},
                    "pair_type": self._classify_pair(symbol),
                    "signal_samples": len(pair_signals)
                }
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def test_regime_switching(self, symbol: str = "EURUSD") -> StressTestResult:
        """Test model adaptation to different market regimes"""
        logger.info(f"Testing regime switching for {symbol}")
        
        start_time = time.time()
        
        # Test different regimes
        regimes = ["trending", "sideways", "high_volatility"]
        regime_results = {}
        
        total_signals = 0
        total_errors = 0
        all_signal_strengths = []
        
        for regime in regimes:
            regime_bars = self.generate_synthetic_volatility_data(
                volatility_factor=2.0 if regime == "high_volatility" else 1.0,
                regime=regime,
                bars_count=150
            )
            
            for bar in regime_bars:
                bar["symbol"] = symbol
            
            regime_signals = 0
            regime_errors = 0
            regime_strengths = []
            
            try:
                for i in range(50, len(regime_bars), 20):
                    test_window = regime_bars[i-50:i+1]
                    
                    try:
                        signals = self.signal_generator._generate_ai_signals(symbol, test_window)
                        regime_signals += 1
                        total_signals += 1
                        
                        if signals:
                            avg_strength = np.mean([abs(v) for v in signals.values() if v is not None])
                            regime_strengths.append(avg_strength)
                            all_signal_strengths.append(avg_strength)
                    
                    except Exception as e:
                        regime_errors += 1
                        total_errors += 1
                        logger.warning(f"Regime switching error ({regime}): {e}")
            
            except Exception as e:
                logger.error(f"Regime test failed for {regime}: {e}")
                regime_errors += 1
                total_errors += 1
            
            regime_results[regime] = {
                "signals": regime_signals,
                "errors": regime_errors,
                "avg_strength": np.mean(regime_strengths) if regime_strengths else 0.0,
                "error_rate": regime_errors / max(regime_signals + regime_errors, 1)
            }
        
        duration = time.time() - start_time
        error_rate = total_errors / max(total_signals + total_errors, 1)
        avg_signal_strength = np.mean(all_signal_strengths) if all_signal_strengths else 0.0
        
        # Calculate regime adaptation score
        regime_consistency = np.std([r["avg_strength"] for r in regime_results.values()])
        adaptation_score = max(0.0, 1.0 - regime_consistency)
        
        # Performance criteria
        passed = (error_rate < 0.2 and  # Less than 20% error rate across regimes
                 avg_signal_strength > 0.1 and  # Meaningful signals
                 adaptation_score > 0.5)  # Reasonable adaptation
        
        result = StressTestResult(
            scenario="regime_switching",
            symbol=symbol,
            duration=duration,
            signals_generated=total_signals,
            error_rate=error_rate,
            avg_signal_strength=avg_signal_strength,
            volatility_handled=adaptation_score * 100,
            performance_degradation=(1.0 - adaptation_score) * 100,
            passed=passed,
            details={
                "regime_results": regime_results,
                "adaptation_score": adaptation_score,
                "regime_consistency": regime_consistency
            }
        )
        
        self.results.append(result)
        return result
    
    def _classify_pair(self, symbol: str) -> str:
        """Classify currency pair type"""
        if symbol in self.major_pairs:
            return "major"
        elif symbol in self.exotic_pairs:
            return "exotic"
        elif symbol in self.commodity_pairs:
            return "commodity"
        else:
            return "unknown"
    
    def run_comprehensive_stress_tests(self) -> Dict[str, Any]:
        """Run all stress tests and generate comprehensive report"""
        logger.info("Starting comprehensive volatility stress tests...")
        
        # Clear previous results
        self.results.clear()
        
        test_symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        
        try:
            # Run different stress test scenarios
            for symbol in test_symbols[:2]:  # Test on 2 major pairs
                self.test_high_volatility_scenario(symbol)
                self.test_flash_crash_scenario(symbol)
                self.test_regime_switching(symbol)
            
            # Cross-pair validation
            self.test_cross_pair_validation(test_symbols)
            
        except Exception as e:
            logger.error(f"Stress testing error: {e}")
        
        # Analyze results
        analysis = self._analyze_stress_results()
        
        return {
            "test_results": [
                {
                    "scenario": r.scenario,
                    "symbol": r.symbol,
                    "duration": r.duration,
                    "signals_generated": r.signals_generated,
                    "error_rate": r.error_rate,
                    "avg_signal_strength": r.avg_signal_strength,
                    "volatility_handled": r.volatility_handled,
                    "performance_degradation": r.performance_degradation,
                    "passed": r.passed,
                    "details": r.details
                }
                for r in self.results
            ],
            "analysis": analysis,
            "recommendations": self._generate_stress_recommendations(analysis)
        }
    
    def _analyze_stress_results(self) -> Dict[str, Any]:
        """Analyze stress test results"""
        if not self.results:
            return {}
        
        # Overall statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        pass_rate = passed_tests / total_tests
        
        # Performance metrics
        avg_error_rate = np.mean([r.error_rate for r in self.results])
        max_error_rate = max([r.error_rate for r in self.results])
        avg_signal_strength = np.mean([r.avg_signal_strength for r in self.results])
        
        # Scenario analysis
        scenario_stats = {}
        for result in self.results:
            if result.scenario not in scenario_stats:
                scenario_stats[result.scenario] = {
                    "tests": 0,
                    "passed": 0,
                    "avg_error_rate": 0,
                    "avg_signal_strength": 0
                }
            
            stats = scenario_stats[result.scenario]
            stats["tests"] += 1
            if result.passed:
                stats["passed"] += 1
            stats["avg_error_rate"] += result.error_rate
            stats["avg_signal_strength"] += result.avg_signal_strength
        
        # Calculate averages
        for scenario, stats in scenario_stats.items():
            stats["pass_rate"] = stats["passed"] / stats["tests"]
            stats["avg_error_rate"] /= stats["tests"]
            stats["avg_signal_strength"] /= stats["tests"]
        
        # Identify problematic scenarios
        problematic_scenarios = [
            scenario for scenario, stats in scenario_stats.items()
            if stats["pass_rate"] < 0.7 or stats["avg_error_rate"] > 0.15
        ]
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": pass_rate,
            "avg_error_rate": avg_error_rate,
            "max_error_rate": max_error_rate,
            "avg_signal_strength": avg_signal_strength,
            "scenario_stats": scenario_stats,
            "problematic_scenarios": problematic_scenarios,
            "robustness_score": pass_rate * (1.0 - avg_error_rate)
        }
    
    def _generate_stress_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on stress test results"""
        recommendations = []
        
        pass_rate = analysis.get("pass_rate", 0.0)
        avg_error_rate = analysis.get("avg_error_rate", 0.0)
        problematic_scenarios = analysis.get("problematic_scenarios", [])
        
        if pass_rate < 0.8:
            recommendations.append(f"Overall pass rate is {pass_rate:.1%} - improve model robustness")
        
        if avg_error_rate > 0.1:
            recommendations.append(f"Average error rate is {avg_error_rate:.1%} - add error handling")
        
        if "high_volatility" in problematic_scenarios:
            recommendations.append("Improve high volatility handling - consider volatility-aware models")
        
        if "flash_crash" in problematic_scenarios:
            recommendations.append("Enhance flash crash resilience - add circuit breakers")
        
        if "cross_pair_validation" in problematic_scenarios:
            recommendations.append("Improve cross-pair consistency - normalize features by pair type")
        
        if "regime_switching" in problematic_scenarios:
            recommendations.append("Add regime detection - implement adaptive model weighting")
        
        # General recommendations
        recommendations.extend([
            "Implement model ensemble voting for extreme conditions",
            "Add volatility-based confidence scoring",
            "Create fallback models for high-stress scenarios",
            "Monitor model performance in production",
            "Set up automated stress testing in CI/CD pipeline"
        ])
        
        return recommendations
    
    def save_results(self, output_path: str = None) -> str:
        """Save stress test results to file"""
        if not output_path:
            output_path = f"stress_test_results_{int(time.time())}.json"
        
        results_data = self.run_comprehensive_stress_tests()
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Stress test results saved to {output_path}")
        return output_path


def main():
    """Run volatility stress tests"""
    logging.basicConfig(level=logging.INFO)
    
    tester = VolatilityStressTester()
    results = tester.run_comprehensive_stress_tests()
    
    print("\n" + "="*70)
    print("ARIA PRO VOLATILITY STRESS TEST RESULTS")
    print("="*70)
    
    # Display test results
    for result in results["test_results"]:
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"\n{result['scenario'].upper()} - {result['symbol']} {status}")
        print(f"  Duration: {result['duration']:.2f}s")
        print(f"  Signals Generated: {result['signals_generated']}")
        print(f"  Error Rate: {result['error_rate']:.1%}")
        print(f"  Avg Signal Strength: {result['avg_signal_strength']:.3f}")
        print(f"  Volatility Handled: {result['volatility_handled']:.1f}%")
        
        if result['details']:
            print("  Details:")
            for key, value in result['details'].items():
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value}")
                elif isinstance(value, dict) and len(value) <= 3:
                    print(f"    {key}: {value}")
    
    # Display analysis
    print(f"\n{'='*70}")
    print("STRESS TEST ANALYSIS")
    print("="*70)
    
    analysis = results["analysis"]
    print(f"Total Tests: {analysis.get('total_tests', 0)}")
    print(f"Passed Tests: {analysis.get('passed_tests', 0)}")
    print(f"Pass Rate: {analysis.get('pass_rate', 0):.1%}")
    print(f"Average Error Rate: {analysis.get('avg_error_rate', 0):.1%}")
    print(f"Robustness Score: {analysis.get('robustness_score', 0):.3f}")
    
    # Display scenario statistics
    scenario_stats = analysis.get('scenario_stats', {})
    if scenario_stats:
        print(f"\nScenario Performance:")
        for scenario, stats in scenario_stats.items():
            print(f"  {scenario}: {stats['pass_rate']:.1%} pass rate, "
                  f"{stats['avg_error_rate']:.1%} error rate")
    
    # Display recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print("="*70)
    
    for i, rec in enumerate(results["recommendations"], 1):
        print(f"{i}. {rec}")
    
    # Save detailed results
    output_file = tester.save_results()
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
