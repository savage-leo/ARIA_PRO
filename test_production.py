#!/usr/bin/env python3
"""
ARIA PRO Production Test Script
Tests all major components of the production system
"""

import os
import sys
import time
import requests
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ARIA.TEST")


def test_environment():
    """Test environment configuration"""
    logger.info("=== Testing Environment Configuration ===")

    # Check production.env exists
    env_file = Path("production.env")
    if env_file.exists():
        logger.info("✅ production.env found")
    else:
        logger.warning("⚠️ production.env not found")

    # Check key environment variables
    key_vars = [
        "ARIA_ENABLE_MT5",
        "ARIA_SYMBOLS",
        "ARIA_ENABLE_EXEC",
        "ARIA_RISK_PER_TRADE",
    ]

    for var in key_vars:
        value = os.environ.get(var, "NOT_SET")
        logger.info(f"{var}: {value}")

    return True


def test_dependencies():
    """Test required dependencies"""
    logger.info("=== Testing Dependencies ===")

    required_modules = ["fastapi", "uvicorn", "numpy", "requests"]

    missing = []
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✅ {module}")
        except ImportError:
            logger.error(f"❌ {module} - MISSING")
            missing.append(module)

    if missing:
        logger.error(f"Missing modules: {missing}")
        return False

    return True


def test_backend_api():
    """Test backend API endpoints"""
    logger.info("=== Testing Backend API ===")

    base_url = "http://localhost:8000"

    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Backend health check passed")
            logger.info(f"Response: {response.json()}")
        else:
            logger.error(f"❌ Backend health check failed: {response.status_code}")
            return False

        # Test root endpoint
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Backend root endpoint working")
            logger.info(f"Response: {response.json()}")
        else:
            logger.error(f"❌ Backend root endpoint failed: {response.status_code}")

        # Test debug endpoints
        response = requests.get(f"{base_url}/debug/health/detailed", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Debug API working")
            data = response.json()
            logger.info(f"Components: {data.get('components', {})}")
        else:
            logger.warning(f"⚠️ Debug API not responding: {response.status_code}")

        return True

    except requests.exceptions.ConnectionError:
        logger.error("❌ Backend API not accessible - is the server running?")
        return False
    except Exception as e:
        logger.error(f"❌ API test failed: {e}")
        return False


def test_orchestrator():
    """Test Phase 3 Orchestrator"""
    logger.info("=== Testing Phase 3 Orchestrator ===")

    try:
        # Test if orchestrator module can be imported
        from backend.core.phase3_orchestrator import Phase3Orchestrator, BarBuilder

        logger.info("✅ Phase 3 Orchestrator module import successful")

        # Test if Enhanced SMC Fusion Core can be imported
        from backend.smc.smc_fusion_core import (
            EnhancedSMCFusionCore,
            EnhancedFusionConfig,
        )

        logger.info("✅ Enhanced SMC Fusion Core module import successful")

        # Test basic orchestrator creation (without starting)
        symbols = ["EURUSD"]
        orch = Phase3Orchestrator(symbols, timeframe=60)
        logger.info("✅ Phase 3 Orchestrator creation successful")

        return True

    except ImportError as e:
        logger.error(f"❌ Module import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Orchestrator test failed: {e}")
        return False


def test_mt5_integration():
    """Test MT5 integration"""
    logger.info("=== Testing MT5 Integration ===")

    try:
        from backend.services.mt5_client import MT5Client

        logger.info("✅ MT5 Client module import successful")

        # Test MT5 client creation
        mt5 = MT5Client()
        logger.info("✅ MT5 Client creation successful")

        # Check if MT5 is enabled
        mt5_enabled = os.environ.get("ARIA_ENABLE_MT5", "0") == "1"
        logger.info(f"MT5 Enabled: {mt5_enabled}")

        return True

    except ImportError as e:
        logger.error(f"❌ MT5 module import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ MT5 test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("🚀 ARIA PRO Production System Test")
    logger.info("=" * 50)

    tests = [
        test_environment,
        test_dependencies,
        test_backend_api,
        test_orchestrator,
        test_mt5_integration,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with exception: {e}")
            results.append(False)

    # Summary
    logger.info("=" * 50)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 50)

    passed = sum(results)
    total = len(results)

    logger.info(f"Tests passed: {passed}/{total}")

    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - System ready for production!")
    else:
        logger.warning("⚠️ Some tests failed - check configuration")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
