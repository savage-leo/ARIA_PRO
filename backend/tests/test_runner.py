"""
Test runner script for ARIA PRO comprehensive test suite
Provides organized test execution with reporting and coverage
"""

import pytest
import sys
import os
import time
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def run_unit_tests():
    """Run unit tests for individual components"""
    print("=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)
    
    test_files = [
        "test_enhanced_auto_trader.py::TestEnhancedAutoTrader",
        "test_enhanced_auto_trader.py::TestRiskEngineEnhancements", 
        "test_enhanced_auto_trader.py::TestAISignalGeneratorEnhancements",
        "test_enhanced_auto_trader.py::TestSecurityEnhancements",
        "test_enhanced_auto_trader.py::TestDistributedLocking",
        "test_enhanced_auto_trader.py::TestComprehensiveMonitoring"
    ]
    
    for test_file in test_files:
        print(f"\n--- Running {test_file} ---")
        result = pytest.main([
            test_file,
            "-v",
            "--tb=short",
            "-x"  # Stop on first failure
        ])
        if result != 0:
            print(f"❌ Unit test failed: {test_file}")
            return False
    
    print("\n✅ All unit tests passed!")
    return True


def run_integration_tests():
    """Run integration tests for component interactions"""
    print("\n" + "=" * 60)
    print("RUNNING INTEGRATION TESTS")
    print("=" * 60)
    
    result = pytest.main([
        "test_integration_suite.py",
        "-v",
        "--tb=short",
        "-x"
    ])
    
    if result == 0:
        print("\n✅ All integration tests passed!")
        return True
    else:
        print("\n❌ Integration tests failed!")
        return False


def run_load_tests():
    """Run load and performance tests"""
    print("\n" + "=" * 60)
    print("RUNNING LOAD & PERFORMANCE TESTS")
    print("=" * 60)
    
    result = pytest.main([
        "test_load_performance.py",
        "-v",
        "--tb=short",
        "-s",  # Show print statements
        "-x"
    ])
    
    if result == 0:
        print("\n✅ All load tests passed!")
        return True
    else:
        print("\n❌ Load tests failed!")
        return False


def run_specific_test_suite(suite_name):
    """Run a specific test suite"""
    suites = {
        "unit": run_unit_tests,
        "integration": run_integration_tests,
        "load": run_load_tests
    }
    
    if suite_name not in suites:
        print(f"❌ Unknown test suite: {suite_name}")
        print(f"Available suites: {', '.join(suites.keys())}")
        return False
    
    return suites[suite_name]()


def run_all_tests():
    """Run complete test suite"""
    print("🚀 Starting ARIA PRO Comprehensive Test Suite")
    print(f"Test directory: {Path(__file__).parent}")
    
    start_time = time.time()
    
    # Run test suites in order
    results = {
        "Unit Tests": run_unit_tests(),
        "Integration Tests": run_integration_tests(),
        "Load Tests": run_load_tests()
    }
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Summary report
    print("\n" + "=" * 60)
    print("TEST SUITE SUMMARY")
    print("=" * 60)
    
    total_passed = sum(results.values())
    total_suites = len(results)
    
    for suite_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{suite_name:<20} {status}")
    
    print(f"\nTotal Duration: {duration:.2f} seconds")
    print(f"Suites Passed: {total_passed}/{total_suites}")
    
    if total_passed == total_suites:
        print("\n🎉 ALL TESTS PASSED! System is ready for production.")
        return True
    else:
        print(f"\n⚠️  {total_suites - total_passed} test suite(s) failed. Review and fix issues.")
        return False


def main():
    """Main test runner entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ARIA PRO Test Runner")
    parser.add_argument(
        "suite",
        nargs="?",
        default="all",
        choices=["all", "unit", "integration", "load"],
        help="Test suite to run (default: all)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage",
        "-c",
        action="store_true",
        help="Run with coverage reporting"
    )
    
    args = parser.parse_args()
    
    # Change to test directory
    os.chdir(Path(__file__).parent)
    
    # Run tests
    if args.suite == "all":
        success = run_all_tests()
    else:
        success = run_specific_test_suite(args.suite)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
