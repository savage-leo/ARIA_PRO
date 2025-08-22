"""
Stress testing framework for ARIA performance monitoring.
Tests model inference under various load conditions and measures system performance.
"""

import asyncio
import random
import time
import json
import logging
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from backend.core.performance_monitor import get_performance_monitor, track_performance
from backend.services.real_ai_signal_generator import real_ai_signal_generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StressTester:
    """Stress testing framework for ARIA performance monitoring."""
    
    def __init__(self):
        self.monitor = get_performance_monitor()
        self.models = ["LSTM", "CNN", "XGBoost", "PPO", "VisualAI", "LLM_Macro"]
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
        
    @track_performance("StressTest.simulate_model_inference")
    async def simulate_model_inference(self, model_name: str, latency_range: tuple = (0.01, 0.2)):
        """Simulate model inference with configurable latency."""
        # Simulate variable processing time
        processing_time = random.uniform(*latency_range)
        await asyncio.sleep(processing_time)
        
        # Simulate memory allocation
        dummy_data = np.random.random((100, 50))
        result = np.mean(dummy_data)
        
        return {"model": model_name, "score": result, "processing_time": processing_time}
    
    async def burst_test(self, burst_size: int = 100, concurrent_limit: int = 10):
        """Test system under burst load conditions."""
        logger.info(f"Starting burst test: {burst_size} requests, {concurrent_limit} concurrent")
        
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        async def limited_inference():
            async with semaphore:
                model = random.choice(self.models)
                return await self.simulate_model_inference(model)
        
        start_time = time.time()
        tasks = [limited_inference() for _ in range(burst_size)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Calculate metrics
        successful = [r for r in results if not isinstance(r, Exception)]
        failed = [r for r in results if isinstance(r, Exception)]
        
        total_time = end_time - start_time
        throughput = len(successful) / total_time
        
        logger.info(f"Burst test completed:")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Successful: {len(successful)}")
        logger.info(f"  Failed: {len(failed)}")
        logger.info(f"  Throughput: {throughput:.2f} req/s")
        
        return {
            "total_time": total_time,
            "successful": len(successful),
            "failed": len(failed),
            "throughput": throughput,
            "errors": [str(e) for e in failed]
        }
    
    async def sustained_load_test(self, duration_seconds: int = 60, requests_per_second: int = 5):
        """Test system under sustained load."""
        logger.info(f"Starting sustained load test: {duration_seconds}s at {requests_per_second} req/s")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        request_count = 0
        
        while time.time() < end_time:
            # Launch batch of requests
            batch_tasks = []
            for _ in range(requests_per_second):
                model = random.choice(self.models)
                task = asyncio.create_task(self.simulate_model_inference(model))
                batch_tasks.append(task)
                request_count += 1
            
            # Wait for batch completion or timeout
            try:
                await asyncio.wait_for(asyncio.gather(*batch_tasks), timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning("Batch timeout - system overloaded")
            
            # Wait for next second
            await asyncio.sleep(max(0, 1.0 - (time.time() % 1.0)))
        
        actual_duration = time.time() - start_time
        actual_rps = request_count / actual_duration
        
        logger.info(f"Sustained load test completed:")
        logger.info(f"  Duration: {actual_duration:.2f}s")
        logger.info(f"  Requests: {request_count}")
        logger.info(f"  Actual RPS: {actual_rps:.2f}")
        
        return {
            "duration": actual_duration,
            "requests": request_count,
            "target_rps": requests_per_second,
            "actual_rps": actual_rps
        }
    
    async def memory_stress_test(self, iterations: int = 50):
        """Test memory usage under load."""
        logger.info(f"Starting memory stress test: {iterations} iterations")
        
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_samples = []
        
        for i in range(iterations):
            # Create large data structures
            model = random.choice(self.models)
            
            # Simulate heavy memory usage
            with self.monitor.track_model(f"MemoryStress_{model}"):
                large_array = np.random.random((1000, 1000))
                result = await self.simulate_model_inference(model, (0.05, 0.15))
                
                # Sample memory usage
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
                
                # Clean up
                del large_array
            
            if i % 10 == 0:
                logger.info(f"Memory stress progress: {i}/{iterations}")
        
        final_memory = process.memory_info().rss / 1024 / 1024
        peak_memory = max(memory_samples)
        avg_memory = np.mean(memory_samples)
        
        logger.info(f"Memory stress test completed:")
        logger.info(f"  Initial memory: {initial_memory:.1f} MB")
        logger.info(f"  Peak memory: {peak_memory:.1f} MB")
        logger.info(f"  Average memory: {avg_memory:.1f} MB")
        logger.info(f"  Final memory: {final_memory:.1f} MB")
        logger.info(f"  Memory growth: {final_memory - initial_memory:.1f} MB")
        
        return {
            "initial_memory_mb": initial_memory,
            "peak_memory_mb": peak_memory,
            "average_memory_mb": avg_memory,
            "final_memory_mb": final_memory,
            "memory_growth_mb": final_memory - initial_memory,
            "samples": memory_samples
        }
    
    async def latency_distribution_test(self, samples: int = 1000):
        """Test latency distribution across models."""
        logger.info(f"Starting latency distribution test: {samples} samples")
        
        results = {}
        
        for model in self.models:
            latencies = []
            
            for _ in range(samples // len(self.models)):
                start_time = time.time()
                await self.simulate_model_inference(model, (0.01, 0.3))
                latency = (time.time() - start_time) * 1000  # ms
                latencies.append(latency)
            
            results[model] = {
                "samples": len(latencies),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "mean_ms": np.mean(latencies),
                "median_ms": np.median(latencies),
                "p95_ms": np.percentile(latencies, 95),
                "p99_ms": np.percentile(latencies, 99),
                "std_ms": np.std(latencies)
            }
            
            logger.info(f"{model}: mean={results[model]['mean_ms']:.1f}ms, p95={results[model]['p95_ms']:.1f}ms")
        
        return results
    
    async def run_comprehensive_test(self):
        """Run all stress tests and generate comprehensive report."""
        logger.info("Starting comprehensive stress test suite")
        
        # Start performance monitoring
        await self.monitor.start_monitoring()
        
        test_results = {}
        
        try:
            # 1. Burst test
            logger.info("\n=== BURST TEST ===")
            test_results["burst_test"] = await self.burst_test(burst_size=200, concurrent_limit=20)
            await asyncio.sleep(5)  # Cool down
            
            # 2. Sustained load test
            logger.info("\n=== SUSTAINED LOAD TEST ===")
            test_results["sustained_load"] = await self.sustained_load_test(duration_seconds=30, requests_per_second=10)
            await asyncio.sleep(5)  # Cool down
            
            # 3. Memory stress test
            logger.info("\n=== MEMORY STRESS TEST ===")
            test_results["memory_stress"] = await self.memory_stress_test(iterations=30)
            await asyncio.sleep(5)  # Cool down
            
            # 4. Latency distribution test
            logger.info("\n=== LATENCY DISTRIBUTION TEST ===")
            test_results["latency_distribution"] = await self.latency_distribution_test(samples=500)
            
            # 5. Get final performance metrics
            test_results["final_metrics"] = {
                "system": self.monitor.get_system_metrics(),
                "models": self.monitor.get_metrics()
            }
            
        except Exception as e:
            logger.error(f"Test suite error: {e}")
            test_results["error"] = str(e)
        
        # Generate report
        report_file = f"stress_test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        logger.info(f"\nComprehensive test completed. Report saved to: {report_file}")
        
        # Print summary
        self._print_summary(test_results)
        
        return test_results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print test summary."""
        print("\n" + "="*60)
        print("ARIA PERFORMANCE STRESS TEST SUMMARY")
        print("="*60)
        
        if "burst_test" in results:
            burst = results["burst_test"]
            print(f"Burst Test: {burst['throughput']:.1f} req/s, {burst['failed']} failures")
        
        if "sustained_load" in results:
            sustained = results["sustained_load"]
            print(f"Sustained Load: {sustained['actual_rps']:.1f} req/s for {sustained['duration']:.1f}s")
        
        if "memory_stress" in results:
            memory = results["memory_stress"]
            print(f"Memory Usage: {memory['peak_memory_mb']:.1f} MB peak, {memory['memory_growth_mb']:.1f} MB growth")
        
        if "latency_distribution" in results:
            latency = results["latency_distribution"]
            avg_p95 = np.mean([model_data["p95_ms"] for model_data in latency.values()])
            print(f"Latency P95: {avg_p95:.1f} ms average across models")
        
        print("="*60)

async def main():
    """Main stress test execution."""
    tester = StressTester()
    
    # Run individual tests or comprehensive suite
    import sys
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        
        if test_type == "burst":
            await tester.burst_test()
        elif test_type == "sustained":
            await tester.sustained_load_test()
        elif test_type == "memory":
            await tester.memory_stress_test()
        elif test_type == "latency":
            await tester.latency_distribution_test()
        else:
            print("Unknown test type. Use: burst, sustained, memory, latency, or no argument for comprehensive")
    else:
        # Run comprehensive test suite
        await tester.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main())
