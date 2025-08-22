"""
Institutional-Grade Inference Profiler
Identifies CPU bottlenecks and performance optimization opportunities
"""

import time
import cProfile
import pstats
import io
import logging
import numpy as np
import psutil
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json

# Import models for profiling
from backend.core.model_loader import cached_models, aria_models, ModelLoader
from backend.services.real_ai_signal_generator import RealAISignalGenerator
from backend.services.mt5_market_data import mt5_market_feed

logger = logging.getLogger(__name__)

@dataclass
class ProfileResult:
    """Profiling result data"""
    operation: str
    duration: float
    cpu_percent: float
    memory_mb: float
    calls: int
    bottlenecks: List[str]


class InferenceProfiler:
    """Profiles AI model inference performance and identifies bottlenecks"""
    
    def __init__(self):
        self.results: List[ProfileResult] = []
        self.model_loader = ModelLoader(use_cache=True)
        self.signal_generator = RealAISignalGenerator()
        
        # Test data
        self.test_data = self._generate_test_data()
        
    def _generate_test_data(self) -> Dict[str, Any]:
        """Generate realistic test data for profiling"""
        np.random.seed(42)  # Reproducible results
        
        return {
            "sequence": np.random.randn(50).astype(np.float32),
            "image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation": np.random.randn(10).astype(np.float32),
            "tabular_features": {"tabular": np.random.randn(6).astype(np.float32)},
            "bars": [
                {
                    "ts": time.time() - i * 60,
                    "o": 1.1000 + np.random.randn() * 0.001,
                    "h": 1.1005 + np.random.randn() * 0.001,
                    "l": 1.0995 + np.random.randn() * 0.001,
                    "c": 1.1002 + np.random.randn() * 0.001,
                    "v": 1000 + np.random.randint(-100, 100),
                    "symbol": "EURUSD"
                }
                for i in range(100)
            ]
        }
    
    def profile_model_loading(self) -> ProfileResult:
        """Profile model loading performance"""
        logger.info("Profiling model loading...")
        
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Profile model loading
        pr = cProfile.Profile()
        pr.enable()
        
        # Test both cached and uncached loading
        cached_loader = ModelLoader(use_cache=True)
        uncached_loader = ModelLoader(use_cache=False)
        
        pr.disable()
        
        end_time = time.time()
        end_cpu = psutil.cpu_percent()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Analyze profile
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)
        
        bottlenecks = self._extract_bottlenecks(s.getvalue())
        
        result = ProfileResult(
            operation="model_loading",
            duration=end_time - start_time,
            cpu_percent=end_cpu - start_cpu,
            memory_mb=end_memory - start_memory,
            calls=ps.total_calls,
            bottlenecks=bottlenecks
        )
        
        self.results.append(result)
        return result
    
    def profile_lstm_inference(self, iterations: int = 100) -> ProfileResult:
        """Profile LSTM inference performance"""
        logger.info(f"Profiling LSTM inference ({iterations} iterations)...")
        
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        pr = cProfile.Profile()
        pr.enable()
        
        # Run LSTM inference multiple times
        for _ in range(iterations):
            cached_models.predict_lstm(self.test_data["sequence"])
        
        pr.disable()
        
        end_time = time.time()
        end_cpu = psutil.cpu_percent()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Analyze profile
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)
        
        bottlenecks = self._extract_bottlenecks(s.getvalue())
        
        result = ProfileResult(
            operation="lstm_inference",
            duration=end_time - start_time,
            cpu_percent=end_cpu - start_cpu,
            memory_mb=end_memory - start_memory,
            calls=ps.total_calls,
            bottlenecks=bottlenecks
        )
        
        self.results.append(result)
        return result
    
    def profile_cnn_inference(self, iterations: int = 50) -> ProfileResult:
        """Profile CNN inference performance"""
        logger.info(f"Profiling CNN inference ({iterations} iterations)...")
        
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        pr = cProfile.Profile()
        pr.enable()
        
        # Run CNN inference multiple times
        for _ in range(iterations):
            cached_models.predict_cnn(self.test_data["image"])
        
        pr.disable()
        
        end_time = time.time()
        end_cpu = psutil.cpu_percent()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Analyze profile
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)
        
        bottlenecks = self._extract_bottlenecks(s.getvalue())
        
        result = ProfileResult(
            operation="cnn_inference",
            duration=end_time - start_time,
            cpu_percent=end_cpu - start_cpu,
            memory_mb=end_memory - start_memory,
            calls=ps.total_calls,
            bottlenecks=bottlenecks
        )
        
        self.results.append(result)
        return result
    
    def profile_ppo_inference(self, iterations: int = 100) -> ProfileResult:
        """Profile PPO inference performance"""
        logger.info(f"Profiling PPO inference ({iterations} iterations)...")
        
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        pr = cProfile.Profile()
        pr.enable()
        
        # Run PPO inference multiple times
        for _ in range(iterations):
            cached_models.trade_with_ppo(self.test_data["observation"])
        
        pr.disable()
        
        end_time = time.time()
        end_cpu = psutil.cpu_percent()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Analyze profile
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)
        
        bottlenecks = self._extract_bottlenecks(s.getvalue())
        
        result = ProfileResult(
            operation="ppo_inference",
            duration=end_time - start_time,
            cpu_percent=end_cpu - start_cpu,
            memory_mb=end_memory - start_memory,
            calls=ps.total_calls,
            bottlenecks=bottlenecks
        )
        
        self.results.append(result)
        return result
    
    def profile_xgb_inference(self, iterations: int = 200) -> ProfileResult:
        """Profile XGBoost inference performance"""
        logger.info(f"Profiling XGBoost inference ({iterations} iterations)...")
        
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        pr = cProfile.Profile()
        pr.enable()
        
        # Run XGBoost inference multiple times
        for _ in range(iterations):
            cached_models.predict_xgb(self.test_data["tabular_features"])
        
        pr.disable()
        
        end_time = time.time()
        end_cpu = psutil.cpu_percent()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Analyze profile
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)
        
        bottlenecks = self._extract_bottlenecks(s.getvalue())
        
        result = ProfileResult(
            operation="xgb_inference",
            duration=end_time - start_time,
            cpu_percent=end_cpu - start_cpu,
            memory_mb=end_memory - start_memory,
            calls=ps.total_calls,
            bottlenecks=bottlenecks
        )
        
        self.results.append(result)
        return result
    
    def profile_signal_generation(self, iterations: int = 20) -> ProfileResult:
        """Profile end-to-end signal generation"""
        logger.info(f"Profiling signal generation ({iterations} iterations)...")
        
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        pr = cProfile.Profile()
        pr.enable()
        
        # Run full signal generation multiple times
        for _ in range(iterations):
            self.signal_generator._generate_ai_signals("EURUSD", self.test_data["bars"])
        
        pr.disable()
        
        end_time = time.time()
        end_cpu = psutil.cpu_percent()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Analyze profile
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(30)
        
        bottlenecks = self._extract_bottlenecks(s.getvalue())
        
        result = ProfileResult(
            operation="signal_generation",
            duration=end_time - start_time,
            cpu_percent=end_cpu - start_cpu,
            memory_mb=end_memory - start_memory,
            calls=ps.total_calls,
            bottlenecks=bottlenecks
        )
        
        self.results.append(result)
        return result
    
    def profile_cache_performance(self) -> ProfileResult:
        """Profile cache hit/miss performance"""
        logger.info("Profiling cache performance...")
        
        # Clear cache first
        cached_models.invalidate_cache()
        
        start_time = time.time()
        
        # Cold cache - first access
        cold_start = time.time()
        cached_models.predict_lstm(self.test_data["sequence"])
        cached_models.predict_cnn(self.test_data["image"])
        cached_models.trade_with_ppo(self.test_data["observation"])
        cached_models.predict_xgb(self.test_data["tabular_features"])
        cold_time = time.time() - cold_start
        
        # Warm cache - subsequent accesses
        warm_times = []
        for _ in range(10):
            warm_start = time.time()
            cached_models.predict_lstm(self.test_data["sequence"])
            cached_models.predict_cnn(self.test_data["image"])
            cached_models.trade_with_ppo(self.test_data["observation"])
            cached_models.predict_xgb(self.test_data["tabular_features"])
            warm_times.append(time.time() - warm_start)
        
        avg_warm_time = np.mean(warm_times)
        speedup = cold_time / avg_warm_time if avg_warm_time > 0 else 0
        
        total_time = time.time() - start_time
        
        result = ProfileResult(
            operation="cache_performance",
            duration=total_time,
            cpu_percent=0.0,
            memory_mb=0.0,
            calls=44,  # 4 models * 11 iterations
            bottlenecks=[
                f"Cold cache time: {cold_time:.3f}s",
                f"Warm cache time: {avg_warm_time:.3f}s",
                f"Cache speedup: {speedup:.1f}x"
            ]
        )
        
        self.results.append(result)
        return result
    
    def profile_concurrent_inference(self, threads: int = 4) -> ProfileResult:
        """Profile concurrent inference performance"""
        logger.info(f"Profiling concurrent inference ({threads} threads)...")
        
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        def worker():
            for _ in range(25):
                cached_models.predict_lstm(self.test_data["sequence"])
                cached_models.predict_cnn(self.test_data["image"])
                cached_models.trade_with_ppo(self.test_data["observation"])
                cached_models.predict_xgb(self.test_data["tabular_features"])
        
        # Run concurrent workers
        thread_list = []
        for _ in range(threads):
            t = threading.Thread(target=worker)
            thread_list.append(t)
            t.start()
        
        for t in thread_list:
            t.join()
        
        end_time = time.time()
        end_cpu = psutil.cpu_percent()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        result = ProfileResult(
            operation="concurrent_inference",
            duration=end_time - start_time,
            cpu_percent=end_cpu - start_cpu,
            memory_mb=end_memory - start_memory,
            calls=threads * 25 * 4,
            bottlenecks=[f"Threads: {threads}", f"Total calls: {threads * 25 * 4}"]
        )
        
        self.results.append(result)
        return result
    
    def _extract_bottlenecks(self, profile_output: str) -> List[str]:
        """Extract performance bottlenecks from profile output"""
        bottlenecks = []
        lines = profile_output.split('\n')
        
        # Look for high-time functions
        for line in lines:
            if 'cumulative' in line or 'ncalls' in line:
                continue
            if any(keyword in line.lower() for keyword in ['onnx', 'predict', 'inference', 'session']):
                parts = line.strip().split()
                if len(parts) >= 6:
                    try:
                        cumtime = float(parts[3])
                        if cumtime > 0.01:  # Functions taking >10ms
                            func_name = parts[-1] if parts else "unknown"
                            bottlenecks.append(f"{func_name}: {cumtime:.3f}s")
                    except (ValueError, IndexError):
                        continue
        
        return bottlenecks[:5]  # Top 5 bottlenecks
    
    def run_comprehensive_profile(self) -> Dict[str, Any]:
        """Run comprehensive performance profiling"""
        logger.info("Starting comprehensive inference profiling...")
        
        # Clear previous results
        self.results.clear()
        
        # Profile different aspects
        try:
            self.profile_model_loading()
            self.profile_lstm_inference()
            self.profile_cnn_inference()
            self.profile_ppo_inference()
            self.profile_xgb_inference()
            self.profile_signal_generation()
            self.profile_cache_performance()
            self.profile_concurrent_inference()
        except Exception as e:
            logger.error(f"Profiling error: {e}")
        
        # Analyze results
        analysis = self._analyze_results()
        
        return {
            "results": [
                {
                    "operation": r.operation,
                    "duration": r.duration,
                    "cpu_percent": r.cpu_percent,
                    "memory_mb": r.memory_mb,
                    "calls": r.calls,
                    "bottlenecks": r.bottlenecks
                }
                for r in self.results
            ],
            "analysis": analysis,
            "recommendations": self._generate_recommendations(analysis)
        }
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze profiling results"""
        if not self.results:
            return {}
        
        # Find slowest operations
        slowest = max(self.results, key=lambda r: r.duration)
        
        # Calculate average inference times
        inference_results = [r for r in self.results if 'inference' in r.operation]
        avg_inference_time = np.mean([r.duration for r in inference_results]) if inference_results else 0
        
        # Memory usage analysis
        max_memory = max(self.results, key=lambda r: r.memory_mb)
        
        # CPU usage analysis
        max_cpu = max(self.results, key=lambda r: r.cpu_percent)
        
        return {
            "slowest_operation": {
                "name": slowest.operation,
                "duration": slowest.duration,
                "bottlenecks": slowest.bottlenecks
            },
            "average_inference_time": avg_inference_time,
            "max_memory_usage": {
                "operation": max_memory.operation,
                "memory_mb": max_memory.memory_mb
            },
            "max_cpu_usage": {
                "operation": max_cpu.operation,
                "cpu_percent": max_cpu.cpu_percent
            },
            "total_operations": len(self.results)
        }
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if analysis.get("slowest_operation", {}).get("duration", 0) > 1.0:
            recommendations.append(
                f"Optimize {analysis['slowest_operation']['name']} - taking {analysis['slowest_operation']['duration']:.2f}s"
            )
        
        if analysis.get("max_memory_usage", {}).get("memory_mb", 0) > 500:
            recommendations.append(
                f"Reduce memory usage in {analysis['max_memory_usage']['operation']} - using {analysis['max_memory_usage']['memory_mb']:.1f} MB"
            )
        
        if analysis.get("max_cpu_usage", {}).get("cpu_percent", 0) > 80:
            recommendations.append(
                f"Optimize CPU usage in {analysis['max_cpu_usage']['operation']} - using {analysis['max_cpu_usage']['cpu_percent']:.1f}%"
            )
        
        # General recommendations
        recommendations.extend([
            "Consider model quantization for faster inference",
            "Implement batch processing for multiple symbols",
            "Use thread pool for concurrent inference",
            "Monitor cache hit rates and adjust TTL",
            "Profile with production data for realistic results"
        ])
        
        return recommendations
    
    def save_results(self, output_path: str = None):
        """Save profiling results to file"""
        if not output_path:
            output_path = f"inference_profile_{int(time.time())}.json"
        
        results_data = self.run_comprehensive_profile()
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Profiling results saved to {output_path}")
        return output_path


def main():
    """Run inference profiling"""
    logging.basicConfig(level=logging.INFO)
    
    profiler = InferenceProfiler()
    results = profiler.run_comprehensive_profile()
    
    print("\n" + "="*60)
    print("ARIA PRO INFERENCE PROFILING RESULTS")
    print("="*60)
    
    for result in results["results"]:
        print(f"\n{result['operation'].upper()}:")
        print(f"  Duration: {result['duration']:.3f}s")
        print(f"  CPU: {result['cpu_percent']:.1f}%")
        print(f"  Memory: {result['memory_mb']:.1f} MB")
        print(f"  Calls: {result['calls']}")
        if result['bottlenecks']:
            print("  Bottlenecks:")
            for bottleneck in result['bottlenecks']:
                print(f"    - {bottleneck}")
    
    print(f"\n{'='*60}")
    print("ANALYSIS & RECOMMENDATIONS")
    print("="*60)
    
    analysis = results["analysis"]
    print(f"Slowest Operation: {analysis.get('slowest_operation', {}).get('name', 'N/A')}")
    print(f"Average Inference Time: {analysis.get('average_inference_time', 0):.3f}s")
    print(f"Max Memory Usage: {analysis.get('max_memory_usage', {}).get('memory_mb', 0):.1f} MB")
    print(f"Max CPU Usage: {analysis.get('max_cpu_usage', {}).get('cpu_percent', 0):.1f}%")
    
    print("\nRecommendations:")
    for i, rec in enumerate(results["recommendations"], 1):
        print(f"{i}. {rec}")
    
    # Save results
    output_file = profiler.save_results()
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
