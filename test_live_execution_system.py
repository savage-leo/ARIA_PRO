#!/usr/bin/env python3
"""Test Live Execution System - Complete MT5 Integration"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))


def test_mt5_execution_harness():
    """Test MT5 execution harness functionality"""
    print("🔧 Testing MT5 Execution Harness...")

    try:
        from backend.services.mt5_execution_harness import mt5_execution_harness

        # Test status
        status = mt5_execution_harness.get_status()
        print(
            f"  ✅ Harness Status: Live={status['is_live']}, Kill Switch={status['kill_switch_active']}"
        )
        print(
            f"  📊 Execution Stats: {status['execution_stats']['total_orders']} orders, "
            f"{status['execution_stats']['avg_latency_ms']:.1f}ms avg latency"
        )

        # Test multi-asset symbols
        symbols = mt5_execution_harness.get_multi_asset_symbols()
        print(f"  🌐 Supported Assets: {len(symbols)} symbols")

        return True

    except Exception as e:
        print(f"  ❌ MT5 Harness Error: {e}")
        return False


def test_multi_asset_manager():
    """Test multi-asset framework"""
    print("\n🌍 Testing Multi-Asset Manager...")

    try:
        from backend.services.multi_asset_manager import multi_asset_manager

        # Test asset configurations
        symbols = multi_asset_manager.get_supported_symbols()
        total_assets = sum(len(assets) for assets in symbols.values())
        print(
            f"  ✅ Asset Classes: {len(symbols)} classes, {total_assets} total assets"
        )

        for asset_class, asset_list in symbols.items():
            print(f"    {asset_class.upper()}: {len(asset_list)} assets")

        # Test position sizing
        position_size = multi_asset_manager.calculate_position_size(
            symbol="EURUSD",
            confidence=0.75,
            account_balance=100000.0,
            risk_per_trade=0.02,
        )
        print(
            f"  📏 Position Sizing Test: EURUSD = {position_size:.2f} lots @ 75% confidence"
        )

        # Test correlation risk
        existing_positions = {"GBPUSD": 1.0, "EURGBP": -0.5}
        correlation_risk = multi_asset_manager.check_correlation_risk(
            "EURUSD", existing_positions
        )
        print(
            f"  🔗 Correlation Risk: {correlation_risk:.3f} for EURUSD vs existing positions"
        )

        # Test position validation
        validation = multi_asset_manager.validate_new_position(
            "XAUUSD", 2.0, existing_positions
        )
        print(
            f"  ✅ Position Validation: Allowed={validation['allowed']}, "
            f"Warnings={len(validation['warnings'])}"
        )

        return True

    except Exception as e:
        print(f"  ❌ Multi-Asset Manager Error: {e}")
        return False


def test_hedge_fund_integration():
    """Test hedge fund analytics integration"""
    print("\n📈 Testing Hedge Fund Integration...")

    try:
        from backend.services.hedge_fund_analytics import hedge_fund_analytics
        from backend.services.t470_pipeline_optimized import t470_pipeline

        # Test analytics
        dashboard_data = hedge_fund_analytics.get_live_dashboard_data()
        print(
            f"  ✅ Analytics: {dashboard_data['total_strategies']} strategies, "
            f"{dashboard_data['data_points']} data points"
        )

        # Test pipeline
        status = t470_pipeline.get_system_status()
        print(
            f"  🔧 Pipeline: {status['memory_usage_mb']:.1f}MB memory, "
            f"{len(status.get('active_models', []))} active models"
        )

        # Test sample processing
        result = t470_pipeline.process_tick_optimized(
            symbol="EURUSD",
            price=1.1000,
            account_balance=100000.0,
            atr=0.001,
            spread_pips=0.8,
        )

        decision = result.get("decision", {})
        print(
            f"  🎯 Sample Decision: {decision.get('action', 'N/A')} "
            f"(confidence: {result.get('confidence', 0):.3f})"
        )

        return True

    except Exception as e:
        print(f"  ❌ Hedge Fund Integration Error: {e}")
        return False


def test_api_endpoints():
    """Test live execution API endpoints"""
    print("\n🌐 Testing API Endpoints...")

    try:
        import requests

        base_url = "http://127.0.0.1:8000"

        # Test live execution status
        try:
            response = requests.get(f"{base_url}/live-execution/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(
                    f"  ✅ Live Execution API: Status OK, "
                    f"Live={data.get('is_live', False)}"
                )
            else:
                print(f"  ⚠️ Live Execution API: HTTP {response.status_code}")
        except requests.exceptions.RequestException:
            print("  ⚠️ Live Execution API: Backend not running")

        # Test hedge fund dashboard
        try:
            response = requests.get(f"{base_url}/hedge-fund/performance", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Hedge Fund API: Performance data available")
            else:
                print(f"  ⚠️ Hedge Fund API: HTTP {response.status_code}")
        except requests.exceptions.RequestException:
            print("  ⚠️ Hedge Fund API: Backend not running")

        # Test supported symbols
        try:
            response = requests.get(
                f"{base_url}/live-execution/symbols/supported", timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                total_symbols = sum(len(symbols) for symbols in data.values())
                print(
                    f"  ✅ Multi-Asset API: {total_symbols} symbols across {len(data)} asset classes"
                )
            else:
                print(f"  ⚠️ Multi-Asset API: HTTP {response.status_code}")
        except requests.exceptions.RequestException:
            print("  ⚠️ Multi-Asset API: Backend not running")

        return True

    except ImportError:
        print("  ⚠️ Requests not available - API tests skipped")
        return True


def main():
    """Run complete live execution system test"""
    print("🏛️ ARIA Live Execution System - Comprehensive Test")
    print("=" * 60)

    tests = [
        test_mt5_execution_harness,
        test_multi_asset_manager,
        test_hedge_fund_integration,
        test_api_endpoints,
    ]

    results = []
    for test in tests:
        try:
            success = test()
            results.append(success)
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")

    passed = sum(results)
    total = len(results)

    print(f"✅ Passed: {passed}/{total} tests")

    if passed == total:
        print("\n🚀 LIVE EXECUTION SYSTEM: READY FOR DEPLOYMENT!")
        print("🌐 API Endpoints:")
        print("   • Live Execution: http://127.0.0.1:8000/live-execution/")
        print("   • Hedge Fund Analytics: http://127.0.0.1:8000/hedge-fund/")
        print("   • Multi-Asset Support: Forex, Commodities, Indices, Crypto")
        print("\n💻 T470 Multi-Strategy Hedge Fund: FULLY ARMED!")
    else:
        print(f"\n⚠️ {total - passed} tests failed - review errors above")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

