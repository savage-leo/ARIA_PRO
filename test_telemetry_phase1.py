#!/usr/bin/env python3
"""
Test Script for ARIA PRO Phase 1 Telemetry Implementation
Tests real-time performance monitoring and business metrics tracking
"""

import time
import requests
import json
import random
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8100"
TELEMETRY_BASE = f"{API_BASE_URL}/telemetry"

def test_telemetry_endpoints():
    """Test all telemetry endpoints"""
    print("🧪 Testing ARIA PRO Phase 1 Telemetry Implementation")
    print("=" * 60)
    
    # Test 1: Health endpoint
    print("\n1. Testing Telemetry Health...")
    try:
        response = requests.get(f"{TELEMETRY_BASE}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Status: {data['data']['status']}")
            print(f"   MT5 Connected: {data['data']['mt5_connected']}")
            print(f"   Critical Alerts: {data['data']['critical_alerts']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test 2: Performance metrics
    print("\n2. Testing Performance Metrics...")
    try:
        response = requests.get(f"{TELEMETRY_BASE}/performance", timeout=5)
        if response.status_code == 200:
            data = response.json()
            perf = data['data']
            print(f"✅ Execution Latency P95: {perf['execution_latency']['p95_ms']:.2f}ms")
            print(f"   Slippage P95: {perf['slippage']['p95']:.4f}")
            print(f"   Error Rate: {perf['error_rate']:.2%}")
            print(f"   MT5 Connection: {perf['mt5_connection']['healthy']}")
        else:
            print(f"❌ Performance metrics failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Performance metrics error: {e}")
    
    # Test 3: Business metrics
    print("\n3. Testing Business Metrics...")
    try:
        response = requests.get(f"{TELEMETRY_BASE}/business", timeout=5)
        if response.status_code == 200:
            data = response.json()
            business = data['data']
            print(f"✅ Real-time P&L: ${business['pnl']['real_time']:.2f}")
            print(f"   Daily P&L: ${business['pnl']['daily']:.2f}")
            print(f"   Win Rate: {business['performance']['win_rate']:.2%}")
            print(f"   Max Drawdown: {business['performance']['max_drawdown']:.2%}")
            print(f"   Total Trades: {business['trades']['total']}")
        else:
            print(f"❌ Business metrics failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Business metrics error: {e}")
    
    # Test 4: Alerts
    print("\n4. Testing Alerts...")
    try:
        response = requests.get(f"{TELEMETRY_BASE}/alerts", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Total Alerts: {data['data']['total_alerts']}")
            print(f"   Critical Alerts: {data['data']['critical_alerts']}")
        else:
            print(f"❌ Alerts failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Alerts error: {e}")
    
    # Test 5: Dashboard
    print("\n5. Testing Dashboard...")
    try:
        response = requests.get(f"{TELEMETRY_BASE}/dashboard", timeout=5)
        if response.status_code == 200:
            data = response.json()
            dashboard = data['data']
            print(f"✅ Overall Status: {dashboard['overall_status']}")
            print(f"   Performance Score: {dashboard['health_scores']['performance']:.0f}/100")
            print(f"   Business Score: {dashboard['health_scores']['business']:.0f}/100")
            print(f"   Overall Score: {dashboard['health_scores']['overall']:.0f}/100")
        else:
            print(f"❌ Dashboard failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Dashboard error: {e}")

def simulate_trading_activity():
    """Simulate trading activity to generate telemetry data"""
    print("\n🎯 Simulating Trading Activity...")
    print("=" * 60)
    
    # Simulate multiple executions
    for i in range(5):
        print(f"\nSimulating execution {i+1}/5...")
        
        # Simulate execution data
        start_time = time.time() - random.uniform(0.01, 0.1)  # Random latency
        end_time = time.time()
        expected_price = random.uniform(1.1000, 1.2000)
        actual_price = expected_price + random.uniform(-0.0005, 0.0005)  # Some slippage
        
        trade_data = {
            "symbol": "EURUSD",
            "action": random.choice(["BUY", "SELL"]),
            "volume": random.uniform(0.1, 1.0),
            "pnl": random.uniform(-50, 100),  # Random P&L
            "confidence": random.uniform(0.6, 0.9),
            "regime": random.choice(["T", "R", "B"]),
            "execution_price": actual_price,
            "expected_price": expected_price
        }
        
        execution_data = {
            "start_time": start_time,
            "end_time": end_time,
            "expected_price": expected_price,
            "actual_price": actual_price,
            "trade_data": trade_data
        }
        
        try:
            response = requests.post(
                f"{TELEMETRY_BASE}/track-execution",
                json=execution_data,
                timeout=5
            )
            if response.status_code == 200:
                result = response.json()
                latency = result['data']['latency_ms']
                slippage = result['data']['slippage']
                print(f"   ✅ Execution tracked - Latency: {latency:.2f}ms, Slippage: {slippage:.4f}")
            else:
                print(f"   ❌ Execution tracking failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Execution tracking error: {e}")
        
        time.sleep(1)  # Wait between simulations
    
    # Simulate some errors
    print("\nSimulating errors...")
    error_types = ["connection_timeout", "order_rejected", "insufficient_funds", "market_closed"]
    for error_type in error_types:
        try:
            response = requests.post(
                f"{TELEMETRY_BASE}/track-error",
                json={"error_type": error_type},
                timeout=5
            )
            if response.status_code == 200:
                print(f"   ✅ Error tracked: {error_type}")
            else:
                print(f"   ❌ Error tracking failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error tracking error: {e}")
        time.sleep(0.5)

def test_summary_endpoint():
    """Test the comprehensive summary endpoint"""
    print("\n📊 Testing Summary Endpoint...")
    print("=" * 60)
    
    try:
        response = requests.get(f"{TELEMETRY_BASE}/summary", timeout=5)
        if response.status_code == 200:
            data = response.json()
            summary = data['data']
            
            print(f"✅ Overall Status: {summary['status']}")
            print(f"   Timestamp: {datetime.fromtimestamp(summary['timestamp'])}")
            
            # Performance summary
            perf = summary['performance']
            print(f"\n📈 Performance Metrics:")
            print(f"   Latency P95: {perf['execution_latency_p95']:.2f}ms")
            print(f"   Slippage P95: {perf['slippage_p95']:.4f}")
            print(f"   Error Rate: {perf['error_rate']:.2%}")
            print(f"   MT5 Health: {perf['mt5_connection_health']}")
            
            # Business summary
            business = summary['business']
            print(f"\n💰 Business Metrics:")
            print(f"   Real-time P&L: ${business['real_time_pnl']:.2f}")
            print(f"   Daily P&L: ${business['daily_pnl']:.2f}")
            print(f"   Win Rate: {business['win_rate']:.2%}")
            print(f"   Max Drawdown: {business['max_drawdown']:.2%}")
            print(f"   Total Trades: {business['total_trades']}")
            
            # Alerts summary
            alerts = summary['alerts']
            print(f"\n🚨 Recent Alerts: {len(alerts)}")
            for alert in alerts[-3:]:  # Show last 3 alerts
                print(f"   [{alert['severity'].upper()}] {alert['message']}")
                
        else:
            print(f"❌ Summary failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Summary error: {e}")

def main():
    """Main test function"""
    print("🚀 ARIA PRO Phase 1 Telemetry Test Suite")
    print("=" * 60)
    print(f"Testing against: {API_BASE_URL}")
    print(f"Timestamp: {datetime.now()}")
    
    # Test basic endpoints
    test_telemetry_endpoints()
    
    # Simulate trading activity
    simulate_trading_activity()
    
    # Wait a moment for data to be processed
    print("\n⏳ Waiting for data processing...")
    time.sleep(2)
    
    # Test summary endpoint with new data
    test_summary_endpoint()
    
    print("\n" + "=" * 60)
    print("✅ Phase 1 Telemetry Test Complete!")
    print("\n📋 Next Steps:")
    print("   1. Check telemetry data files in data/telemetry/")
    print("   2. Monitor alerts in data/alerts/")
    print("   3. Verify metrics are being tracked in real-time")
    print("   4. Proceed to Phase 2: Prometheus integration")

if __name__ == "__main__":
    main()

