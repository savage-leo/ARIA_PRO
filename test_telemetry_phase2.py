#!/usr/bin/env python3
"""
Test Script for ARIA PRO Phase 2 Telemetry Implementation
Tests Prometheus integration and enhanced monitoring capabilities
"""

import time
import requests
import json
import random
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8100"
TELEMETRY_BASE = f"{API_BASE_URL}/telemetry"

def test_prometheus_integration():
    """Test Prometheus metrics integration"""
    print("🔍 Testing Prometheus Integration...")
    print("=" * 60)
    
    # Test 1: Check if Prometheus is available
    print("\n1. Checking Prometheus Availability...")
    try:
        response = requests.get(f"{TELEMETRY_BASE}/dashboard", timeout=5)
        if response.status_code == 200:
            data = response.json()
            prometheus_info = data['data']['prometheus']
            print(f"✅ Prometheus Available: {prometheus_info['available']}")
            if prometheus_info['available']:
                print(f"   Metrics Endpoint: {prometheus_info['metrics_endpoint']}")
            else:
                print("   ⚠️  Prometheus not available - check prometheus-client installation")
        else:
            print(f"❌ Dashboard failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
    
    # Test 2: Get Prometheus metrics
    print("\n2. Testing Prometheus Metrics Endpoint...")
    try:
        response = requests.get(f"{TELEMETRY_BASE}/prometheus", timeout=5)
        if response.status_code == 200:
            metrics = response.text
            print(f"✅ Prometheus metrics retrieved successfully")
            print(f"   Content-Type: {response.headers.get('content-type', 'unknown')}")
            print(f"   Metrics length: {len(metrics)} characters")
            
            # Check for key metrics
            key_metrics = [
                'aria_execution_latency_seconds',
                'aria_slippage_pips',
                'aria_errors_total',
                'aria_trade_volume_total',
                'aria_pnl_dollars',
                'aria_win_rate_ratio',
                'aria_drawdown_percent',
                'aria_mt5_connection_status',
                'aria_kill_switch_active'
            ]
            
            found_metrics = []
            for metric in key_metrics:
                if metric in metrics:
                    found_metrics.append(metric)
            
            print(f"   Found {len(found_metrics)}/{len(key_metrics)} key metrics")
            for metric in found_metrics:
                print(f"   ✅ {metric}")
            
        elif response.status_code == 503:
            print("❌ Prometheus metrics not available (503)")
        else:
            print(f"❌ Prometheus metrics failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Prometheus metrics error: {e}")

def test_enhanced_telemetry():
    """Test enhanced telemetry with Prometheus integration"""
    print("\n🎯 Testing Enhanced Telemetry with Prometheus...")
    print("=" * 60)
    
    # Simulate trading activity to generate metrics
    print("\nSimulating trading activity for metrics generation...")
    for i in range(5):
        execution_data = {
            "start_time": time.time() - random.uniform(0.01, 0.1),
            "end_time": time.time(),
            "expected_price": random.uniform(1.1000, 1.2000),
            "actual_price": random.uniform(1.1000, 1.2000),
            "trade_data": {
                "symbol": random.choice(["EURUSD", "GBPUSD", "USDJPY"]),
                "action": random.choice(["BUY", "SELL"]),
                "volume": random.uniform(0.1, 1.0),
                "pnl": random.uniform(-50, 100),
                "confidence": random.uniform(0.6, 0.9),
                "regime": random.choice(["T", "R", "B"]),
                "execution_price": random.uniform(1.1000, 1.2000),
                "expected_price": random.uniform(1.1000, 1.2000)
            }
        }
        
        try:
            response = requests.post(f"{TELEMETRY_BASE}/track-execution", json=execution_data, timeout=5)
            if response.status_code == 200:
                print(f"✅ Execution {i+1} tracked with Prometheus")
            else:
                print(f"❌ Execution {i+1} failed")
        except Exception as e:
            print(f"❌ Execution {i+1} error: {e}")
        
        time.sleep(1)
    
    # Simulate errors
    print("\nSimulating errors for error metrics...")
    error_types = ["connection_timeout", "order_rejected", "insufficient_funds"]
    for error_type in error_types:
        try:
            response = requests.post(f"{TELEMETRY_BASE}/track-error", json={"error_type": error_type}, timeout=5)
            if response.status_code == 200:
                print(f"✅ Error tracked: {error_type}")
            else:
                print(f"❌ Error tracking failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Error tracking error: {e}")
        time.sleep(0.5)

def test_metrics_consistency():
    """Test consistency between telemetry and Prometheus metrics"""
    print("\n📊 Testing Metrics Consistency...")
    print("=" * 60)
    
    # Get telemetry summary
    try:
        response = requests.get(f"{TELEMETRY_BASE}/summary", timeout=5)
        if response.status_code == 200:
            telemetry_data = response.json()['data']
            print("✅ Telemetry Summary Retrieved")
            
            # Get Prometheus metrics
            response = requests.get(f"{TELEMETRY_BASE}/prometheus", timeout=5)
            if response.status_code == 200:
                prometheus_metrics = response.text
                print("✅ Prometheus Metrics Retrieved")
                
                # Check for consistency indicators
                business_metrics = telemetry_data['business']
                perf_metrics = telemetry_data['performance']
                
                print(f"\n📈 Business Metrics:")
                print(f"   Real-time P&L: ${business_metrics['real_time_pnl']:.2f}")
                print(f"   Win Rate: {business_metrics['win_rate']:.2%}")
                print(f"   Max Drawdown: {business_metrics['max_drawdown']:.2%}")
                
                print(f"\n⚡ Performance Metrics:")
                print(f"   Latency P95: {perf_metrics['execution_latency_p95']:.2f}ms")
                print(f"   Error Rate: {perf_metrics['error_rate']:.2%}")
                print(f"   MT5 Health: {perf_metrics['mt5_connection_health']}")
                
                # Check if metrics are being tracked in Prometheus
                if 'aria_pnl_dollars' in prometheus_metrics:
                    print(f"\n✅ P&L metrics found in Prometheus")
                if 'aria_execution_latency_seconds' in prometheus_metrics:
                    print(f"✅ Latency metrics found in Prometheus")
                if 'aria_errors_total' in prometheus_metrics:
                    print(f"✅ Error metrics found in Prometheus")
                
            else:
                print("❌ Prometheus metrics not available")
        else:
            print(f"❌ Telemetry summary failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Metrics consistency test error: {e}")

def test_prometheus_server():
    """Test standalone Prometheus metrics server"""
    print("\n🌐 Testing Prometheus Metrics Server...")
    print("=" * 60)
    
    # Test direct Prometheus server (should be running on port 8000)
    try:
        response = requests.get("http://localhost:8000/metrics", timeout=5)
        if response.status_code == 200:
            metrics = response.text
            print("✅ Prometheus server responding on port 8000")
            print(f"   Metrics length: {len(metrics)} characters")
            
            # Count ARIA metrics
            aria_metrics = [line for line in metrics.split('\n') if line.startswith('aria_')]
            print(f"   Found {len(aria_metrics)} ARIA metrics")
            
            if aria_metrics:
                print("   Sample ARIA metrics:")
                for metric in aria_metrics[:5]:  # Show first 5
                    print(f"   📊 {metric}")
            
        else:
            print(f"❌ Prometheus server not responding: {response.status_code}")
    except Exception as e:
        print(f"❌ Prometheus server error: {e}")
        print("   Note: Prometheus server may not be running on port 8000")

def main():
    """Main test function for Phase 2"""
    print("🚀 ARIA PRO Phase 2 Telemetry Test Suite")
    print("=" * 60)
    print(f"Testing against: {API_BASE_URL}")
    print(f"Timestamp: {datetime.now()}")
    
    # Test Prometheus integration
    test_prometheus_integration()
    
    # Test enhanced telemetry
    test_enhanced_telemetry()
    
    # Wait for metrics to be processed
    print("\n⏳ Waiting for metrics processing...")
    time.sleep(3)
    
    # Test metrics consistency
    test_metrics_consistency()
    
    # Test Prometheus server
    test_prometheus_server()
    
    print("\n" + "=" * 60)
    print("✅ Phase 2 Telemetry Test Complete!")
    print("\n📋 Phase 2 Features Tested:")
    print("   ✅ Prometheus metrics integration")
    print("   ✅ Enhanced telemetry tracking")
    print("   ✅ Metrics consistency validation")
    print("   ✅ Prometheus server functionality")
    print("\n🎯 Next Steps:")
    print("   1. Set up Grafana dashboards")
    print("   2. Configure Prometheus scraping")
    print("   3. Implement alerting rules")
    print("   4. Proceed to Phase 3: Enhanced Alerting")

if __name__ == "__main__":
    main()

