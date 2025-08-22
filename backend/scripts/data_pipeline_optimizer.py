"""
Data Pipeline Memory Efficiency Optimizer
Audits and optimizes MT5 data fetching for streaming and memory efficiency
"""

import time
import logging
import psutil
import numpy as np
import threading
from typing import Dict, List, Any, Optional, Generator
from dataclasses import dataclass
from collections import deque
import gc
import sys

from backend.services.mt5_market_data import mt5_market_feed
from backend.services.data_source_manager import data_source_manager

logger = logging.getLogger(__name__)

@dataclass
class MemoryProfile:
    """Memory usage profile for data operations"""
    operation: str
    before_mb: float
    after_mb: float
    peak_mb: float
    duration: float
    data_size: int
    efficiency_score: float


class StreamingDataBuffer:
    """Memory-efficient streaming data buffer with automatic cleanup"""
    
    def __init__(self, max_size: int = 1000, cleanup_threshold: float = 0.8):
        self.max_size = max_size
        self.cleanup_threshold = cleanup_threshold
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()
        self.total_bytes = 0
        self.access_count = 0
        
    def append(self, data: Dict[str, Any]):
        """Add data with automatic memory management"""
        with self.lock:
            # Estimate data size
            data_size = sys.getsizeof(data) + sum(sys.getsizeof(v) for v in data.values())
            
            # Trigger cleanup if needed
            if len(self.buffer) / self.max_size > self.cleanup_threshold:
                self._cleanup_old_data()
            
            self.buffer.append(data)
            self.total_bytes += data_size
            
    def get_recent(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent data efficiently"""
        with self.lock:
            self.access_count += 1
            return list(self.buffer)[-count:]
    
    def _cleanup_old_data(self):
        """Clean up old data to free memory"""
        cleanup_count = int(self.max_size * 0.2)  # Remove 20% oldest
        for _ in range(min(cleanup_count, len(self.buffer))):
            if self.buffer:
                old_data = self.buffer.popleft()
                self.total_bytes -= sys.getsizeof(old_data)
        
        # Force garbage collection
        gc.collect()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self.lock:
            return {
                "size": len(self.buffer),
                "max_size": self.max_size,
                "total_bytes": self.total_bytes,
                "access_count": self.access_count,
                "memory_mb": self.total_bytes / 1024 / 1024
            }


class DataPipelineOptimizer:
    """Optimizes data pipeline for memory efficiency and streaming"""
    
    def __init__(self):
        self.profiles: List[MemoryProfile] = []
        self.streaming_buffers: Dict[str, StreamingDataBuffer] = {}
        
    def profile_mt5_data_fetch(self, symbol: str = "EURUSD", bars_count: int = 1000) -> MemoryProfile:
        """Profile MT5 data fetching memory usage"""
        logger.info(f"Profiling MT5 data fetch for {symbol} ({bars_count} bars)")
        
        # Measure initial memory
        process = psutil.Process()
        before_memory = process.memory_info().rss / 1024 / 1024
        peak_memory = before_memory
        
        start_time = time.time()
        
        try:
            # Monitor memory during fetch
            def memory_monitor():
                nonlocal peak_memory
                while True:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    peak_memory = max(peak_memory, current_memory)
                    time.sleep(0.1)
            
            monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
            monitor_thread.start()
            
            # Fetch data
            bars = mt5_market_feed.get_historical_bars(symbol, "M1", bars_count)
            data_size = len(bars) if bars else 0
            
            # Stop monitoring
            monitor_thread = None
            
        except Exception as e:
            logger.error(f"MT5 data fetch failed: {e}")
            bars = []
            data_size = 0
        
        end_time = time.time()
        after_memory = process.memory_info().rss / 1024 / 1024
        
        # Calculate efficiency score
        memory_used = peak_memory - before_memory
        efficiency_score = data_size / max(memory_used, 0.1)  # bars per MB
        
        profile = MemoryProfile(
            operation=f"mt5_fetch_{symbol}_{bars_count}",
            before_mb=before_memory,
            after_mb=after_memory,
            peak_mb=peak_memory,
            duration=end_time - start_time,
            data_size=data_size,
            efficiency_score=efficiency_score
        )
        
        self.profiles.append(profile)
        return profile
    
    def profile_streaming_buffer(self, symbol: str = "EURUSD", iterations: int = 1000) -> MemoryProfile:
        """Profile streaming buffer memory efficiency"""
        logger.info(f"Profiling streaming buffer for {symbol} ({iterations} iterations)")
        
        process = psutil.Process()
        before_memory = process.memory_info().rss / 1024 / 1024
        
        start_time = time.time()
        
        # Create streaming buffer
        buffer = StreamingDataBuffer(max_size=500)
        
        # Simulate streaming data
        for i in range(iterations):
            tick_data = {
                "symbol": symbol,
                "bid": 1.1000 + np.random.randn() * 0.001,
                "ask": 1.1002 + np.random.randn() * 0.001,
                "time": time.time(),
                "volume": np.random.randint(1, 100)
            }
            buffer.append(tick_data)
            
            # Simulate data access
            if i % 100 == 0:
                recent_data = buffer.get_recent(50)
        
        end_time = time.time()
        after_memory = process.memory_info().rss / 1024 / 1024
        
        buffer_stats = buffer.get_stats()
        efficiency_score = buffer_stats["size"] / max(buffer_stats["memory_mb"], 0.1)
        
        profile = MemoryProfile(
            operation=f"streaming_buffer_{symbol}_{iterations}",
            before_mb=before_memory,
            after_mb=after_memory,
            peak_mb=after_memory,
            duration=end_time - start_time,
            data_size=buffer_stats["size"],
            efficiency_score=efficiency_score
        )
        
        self.profiles.append(profile)
        return profile
    
    def profile_batch_processing(self, batch_sizes: List[int] = [10, 50, 100, 500]) -> List[MemoryProfile]:
        """Profile different batch processing sizes"""
        logger.info(f"Profiling batch processing with sizes: {batch_sizes}")
        
        profiles = []
        
        for batch_size in batch_sizes:
            process = psutil.Process()
            before_memory = process.memory_info().rss / 1024 / 1024
            
            start_time = time.time()
            
            # Simulate batch processing
            batches_processed = 0
            total_items = 0
            
            for batch_num in range(20):  # Process 20 batches
                batch_data = []
                
                # Create batch
                for i in range(batch_size):
                    item = {
                        "id": batch_num * batch_size + i,
                        "data": np.random.randn(100),  # Simulate feature data
                        "metadata": {"batch": batch_num, "item": i}
                    }
                    batch_data.append(item)
                
                # Process batch (simulate computation)
                processed_batch = []
                for item in batch_data:
                    processed_item = {
                        "id": item["id"],
                        "result": np.mean(item["data"]),
                        "processed_at": time.time()
                    }
                    processed_batch.append(processed_item)
                
                batches_processed += 1
                total_items += len(processed_batch)
                
                # Clear batch data to simulate memory cleanup
                batch_data.clear()
                processed_batch.clear()
                
                # Periodic garbage collection
                if batch_num % 5 == 0:
                    gc.collect()
            
            end_time = time.time()
            after_memory = process.memory_info().rss / 1024 / 1024
            
            memory_used = after_memory - before_memory
            efficiency_score = total_items / max(memory_used, 0.1)
            
            profile = MemoryProfile(
                operation=f"batch_processing_{batch_size}",
                before_mb=before_memory,
                after_mb=after_memory,
                peak_mb=after_memory,
                duration=end_time - start_time,
                data_size=total_items,
                efficiency_score=efficiency_score
            )
            
            profiles.append(profile)
            self.profiles.append(profile)
        
        return profiles
    
    def create_optimized_data_fetcher(self) -> 'OptimizedDataFetcher':
        """Create optimized data fetcher based on profiling results"""
        return OptimizedDataFetcher(self.get_optimization_recommendations())
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Generate optimization recommendations based on profiling"""
        if not self.profiles:
            return {}
        
        # Analyze profiles
        avg_efficiency = np.mean([p.efficiency_score for p in self.profiles])
        memory_intensive = [p for p in self.profiles if p.peak_mb - p.before_mb > 50]
        slow_operations = [p for p in self.profiles if p.duration > 1.0]
        
        # Find optimal batch size
        batch_profiles = [p for p in self.profiles if "batch_processing" in p.operation]
        optimal_batch_size = 100  # default
        if batch_profiles:
            best_batch = max(batch_profiles, key=lambda p: p.efficiency_score)
            optimal_batch_size = int(best_batch.operation.split('_')[-1])
        
        recommendations = {
            "optimal_batch_size": optimal_batch_size,
            "use_streaming_buffers": True,
            "enable_memory_monitoring": len(memory_intensive) > 0,
            "enable_gc_optimization": True,
            "max_buffer_size": 1000 if avg_efficiency > 10 else 500,
            "cleanup_threshold": 0.7 if len(memory_intensive) > 0 else 0.8,
            "memory_intensive_operations": [p.operation for p in memory_intensive],
            "slow_operations": [p.operation for p in slow_operations]
        }
        
        return recommendations
    
    def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Run comprehensive data pipeline audit"""
        logger.info("Starting comprehensive data pipeline audit...")
        
        # Clear previous profiles
        self.profiles.clear()
        
        try:
            # Profile different aspects
            self.profile_mt5_data_fetch("EURUSD", 100)
            self.profile_mt5_data_fetch("EURUSD", 500)
            self.profile_mt5_data_fetch("EURUSD", 1000)
            
            self.profile_streaming_buffer("EURUSD", 500)
            self.profile_streaming_buffer("GBPUSD", 1000)
            
            self.profile_batch_processing([10, 50, 100, 200, 500])
            
        except Exception as e:
            logger.error(f"Audit error: {e}")
        
        # Generate analysis
        analysis = self._analyze_profiles()
        recommendations = self.get_optimization_recommendations()
        
        return {
            "profiles": [
                {
                    "operation": p.operation,
                    "duration": p.duration,
                    "memory_used_mb": p.peak_mb - p.before_mb,
                    "data_size": p.data_size,
                    "efficiency_score": p.efficiency_score
                }
                for p in self.profiles
            ],
            "analysis": analysis,
            "recommendations": recommendations,
            "optimized_fetcher_config": recommendations
        }
    
    def _analyze_profiles(self) -> Dict[str, Any]:
        """Analyze profiling results"""
        if not self.profiles:
            return {}
        
        # Memory analysis
        memory_usage = [p.peak_mb - p.before_mb for p in self.profiles]
        avg_memory = np.mean(memory_usage)
        max_memory = max(memory_usage)
        
        # Performance analysis
        durations = [p.duration for p in self.profiles]
        avg_duration = np.mean(durations)
        max_duration = max(durations)
        
        # Efficiency analysis
        efficiencies = [p.efficiency_score for p in self.profiles]
        avg_efficiency = np.mean(efficiencies)
        best_efficiency = max(efficiencies)
        
        # Find bottlenecks
        bottlenecks = []
        for profile in self.profiles:
            memory_used = profile.peak_mb - profile.before_mb
            if memory_used > avg_memory * 2:
                bottlenecks.append(f"High memory: {profile.operation} ({memory_used:.1f} MB)")
            if profile.duration > avg_duration * 2:
                bottlenecks.append(f"Slow operation: {profile.operation} ({profile.duration:.2f}s)")
        
        return {
            "total_profiles": len(self.profiles),
            "memory_stats": {
                "average_mb": avg_memory,
                "max_mb": max_memory,
                "total_mb": sum(memory_usage)
            },
            "performance_stats": {
                "average_duration": avg_duration,
                "max_duration": max_duration,
                "total_duration": sum(durations)
            },
            "efficiency_stats": {
                "average_efficiency": avg_efficiency,
                "best_efficiency": best_efficiency,
                "efficiency_range": [min(efficiencies), max(efficiencies)]
            },
            "bottlenecks": bottlenecks
        }


class OptimizedDataFetcher:
    """Memory-optimized data fetcher with streaming capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.buffers: Dict[str, StreamingDataBuffer] = {}
        self.batch_size = config.get("optimal_batch_size", 100)
        self.max_buffer_size = config.get("max_buffer_size", 1000)
        self.cleanup_threshold = config.get("cleanup_threshold", 0.8)
        
    def get_streaming_buffer(self, symbol: str) -> StreamingDataBuffer:
        """Get or create streaming buffer for symbol"""
        if symbol not in self.buffers:
            self.buffers[symbol] = StreamingDataBuffer(
                max_size=self.max_buffer_size,
                cleanup_threshold=self.cleanup_threshold
            )
        return self.buffers[symbol]
    
    def fetch_data_stream(self, symbol: str, timeframe: str = "M1") -> Generator[Dict[str, Any], None, None]:
        """Memory-efficient streaming data fetch"""
        buffer = self.get_streaming_buffer(symbol)
        
        try:
            # Fetch data in batches
            offset = 0
            while True:
                batch = mt5_market_feed.get_historical_bars(
                    symbol, timeframe, self.batch_size, offset
                )
                
                if not batch:
                    break
                
                for bar in batch:
                    buffer.append(bar)
                    yield bar
                
                offset += self.batch_size
                
                # Memory management
                if self.config.get("enable_gc_optimization", True):
                    if offset % (self.batch_size * 5) == 0:
                        gc.collect()
                
        except Exception as e:
            logger.error(f"Streaming fetch error for {symbol}: {e}")
    
    def get_recent_data(self, symbol: str, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent data efficiently from buffer"""
        buffer = self.get_streaming_buffer(symbol)
        return buffer.get_recent(count)
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get statistics for all buffers"""
        return {
            symbol: buffer.get_stats()
            for symbol, buffer in self.buffers.items()
        }


def main():
    """Run data pipeline optimization audit"""
    logging.basicConfig(level=logging.INFO)
    
    optimizer = DataPipelineOptimizer()
    results = optimizer.run_comprehensive_audit()
    
    print("\n" + "="*60)
    print("DATA PIPELINE OPTIMIZATION AUDIT")
    print("="*60)
    
    # Display profiles
    for profile in results["profiles"]:
        print(f"\n{profile['operation'].upper()}:")
        print(f"  Duration: {profile['duration']:.3f}s")
        print(f"  Memory Used: {profile['memory_used_mb']:.1f} MB")
        print(f"  Data Size: {profile['data_size']} items")
        print(f"  Efficiency: {profile['efficiency_score']:.1f} items/MB")
    
    # Display analysis
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print("="*60)
    
    analysis = results["analysis"]
    print(f"Total Profiles: {analysis.get('total_profiles', 0)}")
    print(f"Average Memory Usage: {analysis.get('memory_stats', {}).get('average_mb', 0):.1f} MB")
    print(f"Average Duration: {analysis.get('performance_stats', {}).get('average_duration', 0):.3f}s")
    print(f"Average Efficiency: {analysis.get('efficiency_stats', {}).get('average_efficiency', 0):.1f} items/MB")
    
    # Display bottlenecks
    bottlenecks = analysis.get('bottlenecks', [])
    if bottlenecks:
        print("\nBottlenecks:")
        for bottleneck in bottlenecks:
            print(f"  - {bottleneck}")
    
    # Display recommendations
    print(f"\n{'='*60}")
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*60)
    
    recommendations = results["recommendations"]
    print(f"Optimal Batch Size: {recommendations.get('optimal_batch_size', 100)}")
    print(f"Max Buffer Size: {recommendations.get('max_buffer_size', 1000)}")
    print(f"Cleanup Threshold: {recommendations.get('cleanup_threshold', 0.8)}")
    print(f"Use Streaming Buffers: {recommendations.get('use_streaming_buffers', True)}")
    print(f"Enable Memory Monitoring: {recommendations.get('enable_memory_monitoring', False)}")
    print(f"Enable GC Optimization: {recommendations.get('enable_gc_optimization', True)}")
    
    # Test optimized fetcher
    print(f"\n{'='*60}")
    print("TESTING OPTIMIZED FETCHER")
    print("="*60)
    
    optimized_fetcher = optimizer.create_optimized_data_fetcher()
    
    # Test streaming
    stream_count = 0
    for bar in optimized_fetcher.fetch_data_stream("EURUSD"):
        stream_count += 1
        if stream_count >= 50:  # Test first 50 bars
            break
    
    print(f"Streamed {stream_count} bars successfully")
    
    # Test buffer stats
    buffer_stats = optimized_fetcher.get_buffer_stats()
    for symbol, stats in buffer_stats.items():
        print(f"{symbol} Buffer: {stats['size']} items, {stats['memory_mb']:.1f} MB")
    
    print(f"\nData pipeline optimization audit completed!")


if __name__ == "__main__":
    main()
