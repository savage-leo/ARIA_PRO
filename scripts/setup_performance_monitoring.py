"""
Setup script for ARIA performance monitoring integration.
Initializes monitoring, validates components, and provides usage examples.
"""

import asyncio
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.performance_monitor import get_performance_monitor
from backend.services.auto_trader import auto_trader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_monitoring():
    """Initialize and validate performance monitoring system."""
    logger.info("Setting up ARIA performance monitoring...")
    
    # Initialize performance monitor
    monitor = get_performance_monitor()
    await monitor.start_monitoring()
    
    # Test basic functionality
    with monitor.track_model("Setup_Test"):
        await asyncio.sleep(0.1)  # Simulate work
    
    # Get initial metrics
    system_metrics = monitor.get_system_metrics()
    logger.info(
        "System metrics initialized: %s models tracked, %s symbols tracked",
        system_metrics.get('models_tracked', 0),
        system_metrics.get('symbols_tracked', 0),
    )
    logger.info("Thresholds: %s", monitor.thresholds)
    
    # Validate AutoTrader integration
    if hasattr(auto_trader, 'performance_monitor'):
        logger.info("✅ AutoTrader performance monitoring integrated")
    else:
        logger.warning("⚠️ AutoTrader performance monitoring not integrated")
    
    logger.info("Performance monitoring setup complete!")
    
    return monitor

def print_usage_guide():
    """Print usage guide for performance monitoring."""
    monitor = get_performance_monitor()
    thr = monitor.thresholds
    print("\n" + "="*60)
    print("ARIA PERFORMANCE MONITORING - USAGE GUIDE")
    print("="*60)
    
    print("\n📊 DASHBOARD ACCESS:")
    print("  • WebSocket: ws://localhost:8001/monitoring/ws/performance")
    print("  • REST API: GET /monitoring/performance/metrics")
    print("  • REST API: GET /monitoring/performance/symbols")
    print("  • REST API: GET /monitoring/performance/thresholds")
    print("  • Model-specific: GET /monitoring/performance/models/{model_name}")
    
    print("\n🔧 INTEGRATION EXAMPLES:")
    print("  # Decorator usage:")
    print("  @track_performance('MyModel')")
    print("  async def my_function():")
    print("      # Your code here")
    print("      pass")
    
    print("\n  # Context manager usage:")
    print("  monitor = get_performance_monitor()")
    print("  with monitor.track_model('MyModel'):")
    print("      # Your code here")
    print("      pass")
    
    print("\n🧪 STRESS TESTING:")
    print("  python scripts/stress_test_performance.py          # Full suite")
    print("  python scripts/stress_test_performance.py burst    # Burst test only")
    print("  python scripts/stress_test_performance.py memory   # Memory test only")
    
    print("\n📈 MONITORING FEATURES:")
    print("  • Real-time latency tracking (EMA smoothing)")
    print("  • Per-thread CPU and memory monitoring")
    print("  • Automatic threshold-based alerting (env-configurable)")
    print("  • WebSocket streaming for live dashboards")
    print("  • Historical data with configurable retention")
    
    print("\n⚠️ ALERT THRESHOLDS:")
    print(f"  • CPU Usage > {thr.get('cpu_warn_percent', 90)}%: Warning alert")
    print(f"  • Memory Available < {thr.get('mem_available_percent_crit', 10)}%: Critical alert")
    print(f"  • Model/Per-Symbol Latency > {thr.get('latency_warn_ms', 1000)}ms: Warning alert")
    
    print("="*60)

async def main():
    """Main setup and validation."""
    try:
        # Setup monitoring
        monitor = await setup_monitoring()
        
        # Print usage guide
        print_usage_guide()
        
        # Run quick validation
        logger.info("\nRunning validation tests...")
        
        # Test multiple model tracking
        models = ["LSTM", "CNN", "XGBoost"]
        tasks = []
        
        for model in models:
            async def test_model(name):
                with monitor.track_model(name):
                    await asyncio.sleep(0.05)  # Simulate inference
                    return f"{name}_result"
            
            tasks.append(test_model(model))
        
        results = await asyncio.gather(*tasks)
        logger.info(f"Validation complete: {len(results)} models tested")
        
        # Validate per-symbol metrics
        syms = ["EURUSD", "GBPUSD", "USDJPY"]
        sym_tasks = []
        for s in syms:
            async def test_symbol(sym):
                with monitor.track_model("SymbolPipeline", symbol=sym):
                    await asyncio.sleep(0.03)
                    return sym
            sym_tasks.append(test_symbol(s))

        _ = await asyncio.gather(*sym_tasks)
        sym_metrics = monitor.get_symbol_metrics()
        logger.info("Per-symbol metrics collected for %d symbols", len(sym_metrics))

        # Show final metrics
        final_metrics = monitor.get_metrics()
        logger.info(f"Final state: {len(final_metrics)} models tracked")
        
        for model_name, metrics in final_metrics.items():
            logger.info(f"  {model_name}: {metrics['calls']} calls, {metrics['latency_ms']['avg']:.1f}ms avg")
        
        logger.info("\n🎉 ARIA Performance Monitoring is ready!")
        logger.info("Start your FastAPI server and access the monitoring endpoints.")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
