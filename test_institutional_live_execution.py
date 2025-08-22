#!/usr/bin/env python3
"""
Institutional Live Execution Test Suite
Comprehensive validation of ARIA's live trading capabilities
"""

import sys
import os
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InstitutionalLiveExecutionTester:
    """Comprehensive institutional live execution test suite"""
    
    def __init__(self):
        self.test_results = {}
        self.critical_issues = []
        self.warnings = []
        self.passed_tests = 0
        self.total_tests = 0
        
    def run_comprehensive_test(self):
        """Run all institutional tests"""
        print("🏛️ ARIA Institutional Live Execution - Comprehensive Test Suite")
        print("=" * 80)
        
        # Test categories
        test_categories = [
            ("Environment & Configuration", self.test_environment_configuration),
            ("MT5 Connection & Validation", self.test_mt5_connection),
            ("Live Data Validation", self.test_live_data_validation),
            ("Execution Latency", self.test_execution_latency),
            ("Risk Management", self.test_risk_management),
            ("Security & Compliance", self.test_security_compliance),
            ("Production Readiness", self.test_production_readiness),
        ]
        
        for category_name, test_func in test_categories:
            print(f"\n🔍 {category_name}")
            print("-" * 50)
            try:
                test_func()
            except Exception as e:
                logger.error(f"Test category {category_name} failed: {e}")
                self.critical_issues.append(f"{category_name}: {str(e)}")
        
        self.print_summary()
    
    def test_environment_configuration(self):
        """Test environment configuration for institutional deployment"""
        print("Testing environment configuration...")
        
        # Test 1: Check for demo account configuration
        self.total_tests += 1
        try:
            mt5_server = os.getenv("MT5_SERVER", "")
            mt5_login = os.getenv("MT5_LOGIN", "")
            
            if "demo" in mt5_server.lower():
                self.critical_issues.append("CRITICAL: Demo account configured - not suitable for live trading")
                print("  ❌ Demo account detected in MT5_SERVER")
            else:
                print("  ✅ Non-demo account configured")
                self.passed_tests += 1
                
        except Exception as e:
            self.critical_issues.append(f"Environment check failed: {e}")
            print(f"  ❌ Environment check failed: {e}")
        
        # Test 2: Check required environment variables
        self.total_tests += 1
        required_vars = ["MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER"]
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.critical_issues.append(f"Missing required environment variables: {missing_vars}")
            print(f"  ❌ Missing environment variables: {missing_vars}")
        else:
            print("  ✅ All required environment variables present")
            self.passed_tests += 1
        
        # Test 3: Check live trading flags
        self.total_tests += 1
        auto_exec_enabled = os.getenv("AUTO_EXEC_ENABLED", "false").lower() in ("1", "true", "yes")
        allow_live = os.getenv("ALLOW_LIVE", "0") == "1"
        
        if not (auto_exec_enabled and allow_live):
            self.warnings.append("Live trading not enabled - check AUTO_EXEC_ENABLED and ALLOW_LIVE")
            print("  ⚠️ Live trading not enabled")
        else:
            print("  ✅ Live trading enabled")
            self.passed_tests += 1
    
    def test_mt5_connection(self):
        """Test MT5 connection and validation"""
        print("Testing MT5 connection...")
        
        # Test 1: Import MT5 modules
        self.total_tests += 1
        try:
            from backend.services.mt5_executor import MT5Executor
            from backend.services.mt5_client import MT5Client
            print("  ✅ MT5 modules imported successfully")
            self.passed_tests += 1
        except Exception as e:
            self.critical_issues.append(f"MT5 module import failed: {e}")
            print(f"  ❌ MT5 module import failed: {e}")
            return
        
        # Test 2: Test MT5 connection
        self.total_tests += 1
        try:
            mt5_executor = MT5Executor()
            connected = mt5_executor.connect()
            
            if connected:
                print("  ✅ MT5 connection successful")
                self.passed_tests += 1
                
                # Test 3: Get account information
                self.total_tests += 1
                try:
                    account_info = mt5_executor.get_account_info()
                    print(f"  ✅ Account info retrieved: {account_info.get('login')} @ {account_info.get('server')}")
                    self.passed_tests += 1
                    
                    # Check if it's a demo account
                    if "demo" in account_info.get("server", "").lower():
                        self.critical_issues.append("CRITICAL: Connected to demo account - not suitable for live trading")
                        print("  ❌ Connected to demo account")
                    else:
                        print("  ✅ Connected to live account")
                        
                except Exception as e:
                    self.critical_issues.append(f"Account info retrieval failed: {e}")
                    print(f"  ❌ Account info retrieval failed: {e}")
                    
            else:
                self.critical_issues.append("MT5 connection failed")
                print("  ❌ MT5 connection failed")
                
        except Exception as e:
            self.critical_issues.append(f"MT5 connection test failed: {e}")
            print(f"  ❌ MT5 connection test failed: {e}")
    
    def test_live_data_validation(self):
        """Test live market data validation"""
        print("Testing live data validation...")
        
        # Test 1: Check if live data validation exists
        self.total_tests += 1
        try:
            from backend.services.mt5_execution_harness import MT5ExecutionHarness
            harness = MT5ExecutionHarness()
            
            # Check if validate_live_market_data method exists
            if hasattr(harness, 'validate_live_market_data'):
                print("  ✅ Live data validation method exists")
                self.passed_tests += 1
            else:
                self.critical_issues.append("Missing live data validation method")
                print("  ❌ Missing live data validation method")
                
        except Exception as e:
            self.critical_issues.append(f"Live data validation test failed: {e}")
            print(f"  ❌ Live data validation test failed: {e}")
        
        # Test 2: Check data source validation
        self.total_tests += 1
        try:
            from backend.services.mt5_market_data import mt5_market_feed
            
            # Check if we can get market data
            symbols = ["EURUSD", "GBPUSD"]
            for symbol in symbols:
                try:
                    # This would need to be implemented to check data source
                    print(f"  ⚠️ Data source validation not implemented for {symbol}")
                except Exception as e:
                    print(f"  ❌ Data source validation failed for {symbol}: {e}")
                    
            self.warnings.append("Data source validation not fully implemented")
            
        except Exception as e:
            self.critical_issues.append(f"Data source validation test failed: {e}")
            print(f"  ❌ Data source validation test failed: {e}")
    
    def test_execution_latency(self):
        """Test execution latency monitoring"""
        print("Testing execution latency...")
        
        # Test 1: Check if latency monitoring exists
        self.total_tests += 1
        try:
            from backend.services.mt5_execution_harness import MT5ExecutionHarness
            harness = MT5ExecutionHarness()
            
            # Check execution stats
            stats = harness.get_status()
            if "execution_stats" in stats:
                print("  ✅ Execution statistics tracking exists")
                self.passed_tests += 1
            else:
                self.critical_issues.append("Missing execution statistics tracking")
                print("  ❌ Missing execution statistics tracking")
                
        except Exception as e:
            self.critical_issues.append(f"Execution latency test failed: {e}")
            print(f"  ❌ Execution latency test failed: {e}")
        
        # Test 2: Check latency thresholds
        self.total_tests += 1
        try:
            # Check if latency monitoring is implemented
            print("  ⚠️ Real-time latency monitoring not fully implemented")
            self.warnings.append("Real-time latency monitoring needs enhancement")
            
        except Exception as e:
            self.critical_issues.append(f"Latency threshold test failed: {e}")
            print(f"  ❌ Latency threshold test failed: {e}")
    
    def test_risk_management(self):
        """Test risk management capabilities"""
        print("Testing risk management...")
        
        # Test 1: Check risk engine
        self.total_tests += 1
        try:
            from backend.services.risk_engine import risk_engine
            
            # Check risk configuration
            config = risk_engine.get_risk_config()
            if config:
                print("  ✅ Risk engine configuration available")
                self.passed_tests += 1
            else:
                self.critical_issues.append("Risk engine configuration missing")
                print("  ❌ Risk engine configuration missing")
                
        except Exception as e:
            self.critical_issues.append(f"Risk engine test failed: {e}")
            print(f"  ❌ Risk engine test failed: {e}")
        
        # Test 2: Check kill switch functionality
        self.total_tests += 1
        try:
            from backend.services.mt5_execution_harness import MT5ExecutionHarness
            harness = MT5ExecutionHarness()
            
            if hasattr(harness, 'activate_kill_switch'):
                print("  ✅ Kill switch functionality exists")
                self.passed_tests += 1
            else:
                self.critical_issues.append("Missing kill switch functionality")
                print("  ❌ Missing kill switch functionality")
                
        except Exception as e:
            self.critical_issues.append(f"Kill switch test failed: {e}")
            print(f"  ❌ Kill switch test failed: {e}")
        
        # Test 3: Check correlation risk management
        self.total_tests += 1
        try:
            # Check if correlation risk management exists
            print("  ⚠️ Correlation risk management not fully implemented")
            self.warnings.append("Correlation risk management needs enhancement")
            
        except Exception as e:
            self.critical_issues.append(f"Correlation risk test failed: {e}")
            print(f"  ❌ Correlation risk test failed: {e}")
    
    def test_security_compliance(self):
        """Test security and compliance features"""
        print("Testing security and compliance...")
        
        # Test 1: Check API rate limiting
        self.total_tests += 1
        try:
            from backend.routes.trading import router
            
            # Check if rate limiting is implemented
            if hasattr(router, 'limiter'):
                print("  ✅ API rate limiting implemented")
                self.passed_tests += 1
            else:
                self.warnings.append("API rate limiting not fully implemented")
                print("  ⚠️ API rate limiting not fully implemented")
                
        except Exception as e:
            self.critical_issues.append(f"API rate limiting test failed: {e}")
            print(f"  ❌ API rate limiting test failed: {e}")
        
        # Test 2: Check authentication
        self.total_tests += 1
        try:
            admin_key = os.getenv("ADMIN_API_KEY", "")
            if admin_key and admin_key != "changeme":
                print("  ✅ Admin API key configured")
                self.passed_tests += 1
            else:
                self.warnings.append("Admin API key not properly configured")
                print("  ⚠️ Admin API key not properly configured")
                
        except Exception as e:
            self.critical_issues.append(f"Authentication test failed: {e}")
            print(f"  ❌ Authentication test failed: {e}")
        
        # Test 3: Check audit logging
        self.total_tests += 1
        try:
            from backend.services.mt5_execution_harness import MT5ExecutionHarness
            harness = MT5ExecutionHarness()
            
            if hasattr(harness, 'setup_audit_logging'):
                print("  ✅ Audit logging implemented")
                self.passed_tests += 1
            else:
                self.critical_issues.append("Missing audit logging")
                print("  ❌ Missing audit logging")
                
        except Exception as e:
            self.critical_issues.append(f"Audit logging test failed: {e}")
            print(f"  ❌ Audit logging test failed: {e}")
    
    def test_production_readiness(self):
        """Test production readiness"""
        print("Testing production readiness...")
        
        # Test 1: Check backend startup
        self.total_tests += 1
        try:
            from backend.main import app
            print("  ✅ Backend application loads successfully")
            self.passed_tests += 1
        except Exception as e:
            self.critical_issues.append(f"Backend startup failed: {e}")
            print(f"  ❌ Backend startup failed: {e}")
        
        # Test 2: Check required services
        self.total_tests += 1
        try:
            required_services = [
                "backend.services.mt5_executor",
                "backend.services.risk_engine",
                "backend.services.auto_trader",
                "backend.services.mt5_market_data"
            ]
            
            missing_services = []
            for service in required_services:
                try:
                    __import__(service)
                except ImportError:
                    missing_services.append(service)
            
            if missing_services:
                self.critical_issues.append(f"Missing required services: {missing_services}")
                print(f"  ❌ Missing required services: {missing_services}")
            else:
                print("  ✅ All required services available")
                self.passed_tests += 1
                
        except Exception as e:
            self.critical_issues.append(f"Service availability test failed: {e}")
            print(f"  ❌ Service availability test failed: {e}")
        
        # Test 3: Check configuration files
        self.total_tests += 1
        try:
            config_files = ["production.env", "backend/requirements.txt"]
            missing_files = []
            
            for file in config_files:
                if not os.path.exists(file):
                    missing_files.append(file)
            
            if missing_files:
                self.warnings.append(f"Missing configuration files: {missing_files}")
                print(f"  ⚠️ Missing configuration files: {missing_files}")
            else:
                print("  ✅ Configuration files present")
                self.passed_tests += 1
                
        except Exception as e:
            self.critical_issues.append(f"Configuration test failed: {e}")
            print(f"  ❌ Configuration test failed: {e}")
    
    def print_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 80)
        print("📊 INSTITUTIONAL LIVE EXECUTION TEST SUMMARY")
        print("=" * 80)
        
        # Test results
        print(f"\n✅ Tests Passed: {self.passed_tests}/{self.total_tests}")
        print(f"❌ Critical Issues: {len(self.critical_issues)}")
        print(f"⚠️ Warnings: {len(self.warnings)}")
        
        # Critical issues
        if self.critical_issues:
            print(f"\n🔴 CRITICAL ISSUES (Must Fix Before Live Trading):")
            for i, issue in enumerate(self.critical_issues, 1):
                print(f"  {i}. {issue}")
        
        # Warnings
        if self.warnings:
            print(f"\n🟡 WARNINGS (Should Address):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        # Overall assessment
        print(f"\n🎯 OVERALL ASSESSMENT:")
        if len(self.critical_issues) == 0:
            print("  ✅ READY FOR INSTITUTIONAL DEPLOYMENT")
        elif len(self.critical_issues) <= 3:
            print("  ⚠️ NEARLY READY - Critical fixes required")
        else:
            print("  ❌ NOT READY - Multiple critical issues")
        
        # Recommendations
        print(f"\n📋 RECOMMENDATIONS:")
        if len(self.critical_issues) > 0:
            print("  1. Fix all critical issues before live trading")
            print("  2. Implement live account validation")
            print("  3. Add real-time market data validation")
            print("  4. Enhance security measures")
        else:
            print("  1. Proceed with caution in live environment")
            print("  2. Monitor system performance closely")
            print("  3. Implement additional safety measures")
        
        print(f"\n🏛️ ARIA Institutional Platform Status: {'READY' if len(self.critical_issues) == 0 else 'NOT READY'}")


def main():
    """Main test execution"""
    tester = InstitutionalLiveExecutionTester()
    tester.run_comprehensive_test()


if __name__ == "__main__":
    main()

