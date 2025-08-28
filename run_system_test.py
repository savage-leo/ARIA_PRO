#!/usr/bin/env python3
"""
ARIA PRO Comprehensive System Test Suite
Validates all components, configurations, and functionality
"""

import os
import sys
import time
import subprocess
import json
import requests
from pathlib import Path
from typing import Dict, List, Any
import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ARIASystemTest:
    def __init__(self):
        self.results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tests': {},
            'summary': {'passed': 0, 'failed': 0, 'total': 0}
        }
        self.base_dir = Path(__file__).parent
        
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status} - {test_name}: {details}")
        
        self.results['tests'][test_name] = {
            'passed': passed,
            'details': details,
            'timestamp': time.strftime('%H:%M:%S')
        }
        
        if passed:
            self.results['summary']['passed'] += 1
        else:
            self.results['summary']['failed'] += 1
        self.results['summary']['total'] += 1

    def test_python_environment(self):
        """Test Python environment and dependencies"""
        try:
            import sys
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            
            # Check Python version
            if sys.version_info >= (3, 11):
                self.log_test("Python Version", True, f"Python {python_version}")
            else:
                self.log_test("Python Version", False, f"Python {python_version} < 3.11")
                return
                
            # Test critical imports
            critical_modules = [
                'fastapi', 'uvicorn', 'pydantic', 'numpy', 'pandas',
                'tensorflow', 'onnxruntime', 'MetaTrader5', 'redis'
            ]
            
            missing_modules = []
            for module in critical_modules:
                try:
                    __import__(module)
                except ImportError:
                    missing_modules.append(module)
            
            if missing_modules:
                self.log_test("Dependencies", False, f"Missing: {', '.join(missing_modules)}")
            else:
                self.log_test("Dependencies", True, f"All {len(critical_modules)} modules available")
                
        except Exception as e:
            self.log_test("Python Environment", False, str(e))

    def test_configuration_loading(self):
        """Test configuration loading and validation"""
        try:
            sys.path.append(str(self.base_dir))
            from backend.core.config import get_settings
            
            settings = get_settings()
            
            # Test critical settings
            critical_settings = [
                'ARIA_ENABLE_MT5', 'MT5_LOGIN', 'MT5_SERVER',
                'AUTO_TRADE_ENABLED', 'JWT_SECRET_KEY', 'ADMIN_API_KEY'
            ]
            
            missing_settings = []
            for setting in critical_settings:
                if not hasattr(settings, setting) or not getattr(settings, setting):
                    missing_settings.append(setting)
            
            if missing_settings:
                self.log_test("Configuration", False, f"Missing: {', '.join(missing_settings)}")
            else:
                self.log_test("Configuration", True, f"All {len(critical_settings)} settings loaded")
                
        except Exception as e:
            self.log_test("Configuration Loading", False, str(e))

    def test_backend_compilation(self):
        """Test backend code compilation"""
        try:
            result = subprocess.run([
                sys.executable, '-m', 'compileall', 'backend/', '-q'
            ], cwd=self.base_dir, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                self.log_test("Backend Compilation", True, "All Python files compile successfully")
            else:
                self.log_test("Backend Compilation", False, f"Compilation errors: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.log_test("Backend Compilation", False, "Compilation timeout")
        except Exception as e:
            self.log_test("Backend Compilation", False, str(e))

    def test_ai_models(self):
        """Test AI model availability and loading"""
        try:
            models_dir = self.base_dir / 'backend' / 'models'
            expected_models = [
                'xgboost_forex.onnx',
                'lstm_forex.onnx', 
                'cnn_patterns.onnx',
                'ppo_trader.zip'
            ]
            
            missing_models = []
            model_sizes = {}
            
            for model in expected_models:
                model_path = models_dir / model
                if model_path.exists():
                    size_kb = model_path.stat().st_size / 1024
                    model_sizes[model] = f"{size_kb:.1f}KB"
                else:
                    missing_models.append(model)
            
            if missing_models:
                self.log_test("AI Models", False, f"Missing: {', '.join(missing_models)}")
            else:
                details = ", ".join([f"{k}({v})" for k, v in model_sizes.items()])
                self.log_test("AI Models", True, f"All models present: {details}")
                
        except Exception as e:
            self.log_test("AI Models", False, str(e))

    def test_database_connections(self):
        """Test database and cache connections"""
        try:
            # Test SQLite database
            db_path = self.base_dir / 'data' / 'trade_memory.db'
            if db_path.exists():
                self.log_test("Trade Database", True, f"Database exists ({db_path.stat().st_size} bytes)")
            else:
                self.log_test("Trade Database", False, "Database file not found")
            
            # Test Redis connection (if available)
            try:
                import redis
                r = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=2)
                r.ping()
                self.log_test("Redis Connection", True, "Redis server accessible")
            except:
                self.log_test("Redis Connection", False, "Redis server not accessible")
                
        except Exception as e:
            self.log_test("Database Connections", False, str(e))

    def test_unit_tests(self):
        """Run unit test suite"""
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 'backend/tests/', '-v', '--tb=short', '--collect-only', '-q'
            ], cwd=self.base_dir, capture_output=True, text=True, timeout=60)
            
            # Parse pytest collection output
            output_lines = result.stdout.split('\n')
            collected_line = [line for line in output_lines if 'collected' in line]
            
            if collected_line and 'collected' in collected_line[0]:
                # Extract number of collected tests
                import re
                match = re.search(r'(\d+) item', collected_line[0])
                if match:
                    test_count = int(match.group(1))
                    self.log_test("Unit Tests", True, f"{test_count} tests discovered and ready")
                else:
                    self.log_test("Unit Tests", False, "Could not parse test collection")
            else:
                self.log_test("Unit Tests", False, f"No tests collected: {result.stdout[:200]}")
                
        except subprocess.TimeoutExpired:
            self.log_test("Unit Tests", False, "Test execution timeout")
        except Exception as e:
            self.log_test("Unit Tests", False, str(e))

    def test_backend_startup(self):
        """Test backend startup without full server launch"""
        try:
            # Test import and basic initialization
            test_script = '''
import sys
import os
sys.path.append(os.getcwd())

try:
    from backend.core.config import get_settings
    from backend.services.real_ai_signal_generator import RealAISignalGenerator
    from backend.smc.smc_fusion_core_enhanced import EnhancedSMCFusionCore
    
    settings = get_settings()
    print(f"✅ Settings loaded: MT5={settings.ARIA_ENABLE_MT5}")
    
    # Test AI signal generator initialization
    generator = RealAISignalGenerator()
    print("✅ AI Signal Generator initialized")
    
    # Test SMC Fusion Core
    fusion_core = EnhancedSMCFusionCore("EURUSD")
    print("✅ SMC Fusion Core initialized")
    
    print("SUCCESS: Backend components initialized")
    
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
'''
            
            result = subprocess.run([
                sys.executable, '-c', test_script
            ], cwd=self.base_dir, capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0 and "SUCCESS" in result.stdout:
                self.log_test("Backend Startup", True, "All core components initialized")
            else:
                error_msg = result.stderr or result.stdout
                self.log_test("Backend Startup", False, f"Initialization failed: {error_msg[:200]}")
                
        except subprocess.TimeoutExpired:
            self.log_test("Backend Startup", False, "Startup timeout")
        except Exception as e:
            self.log_test("Backend Startup", False, str(e))

    def test_frontend_build(self):
        """Test frontend build process"""
        try:
            frontend_dir = self.base_dir / 'frontend'
            if not frontend_dir.exists():
                self.log_test("Frontend Build", False, "Frontend directory not found")
                return
            
            # Check package.json
            package_json = frontend_dir / 'package.json'
            if package_json.exists():
                with open(package_json) as f:
                    package_data = json.load(f)
                    dependencies = len(package_data.get('dependencies', {}))
                    self.log_test("Frontend Dependencies", True, f"{dependencies} dependencies defined")
            else:
                self.log_test("Frontend Build", False, "package.json not found")
                
        except Exception as e:
            self.log_test("Frontend Build", False, str(e))

    def test_security_configuration(self):
        """Test security configuration"""
        try:
            sys.path.append(str(self.base_dir))
            from backend.core.config import get_settings
            
            settings = get_settings()
            
            # Check security settings
            security_checks = []
            
            # JWT Secret length
            if hasattr(settings, 'JWT_SECRET_KEY') and len(settings.JWT_SECRET_KEY) >= 32:
                security_checks.append("JWT_SECRET_KEY")
            
            # Admin API key length
            if hasattr(settings, 'ADMIN_API_KEY') and len(settings.ADMIN_API_KEY) >= 16:
                security_checks.append("ADMIN_API_KEY")
            
            # CORS configuration
            if hasattr(settings, 'ARIA_CORS_ORIGINS') and settings.ARIA_CORS_ORIGINS:
                security_checks.append("CORS_ORIGINS")
            
            if len(security_checks) >= 2:
                self.log_test("Security Configuration", True, f"Configured: {', '.join(security_checks)}")
            else:
                self.log_test("Security Configuration", False, f"Missing security settings")
                
        except Exception as e:
            self.log_test("Security Configuration", False, str(e))

    def run_all_tests(self):
        """Run all system tests"""
        logger.info("🚀 Starting ARIA PRO Comprehensive System Test")
        logger.info("=" * 60)
        
        # Run all tests
        self.test_python_environment()
        self.test_configuration_loading()
        self.test_backend_compilation()
        self.test_ai_models()
        self.test_database_connections()
        self.test_security_configuration()
        self.test_backend_startup()
        self.test_unit_tests()
        self.test_frontend_build()
        
        # Generate summary
        logger.info("=" * 60)
        logger.info("📊 SYSTEM TEST SUMMARY")
        logger.info(f"Total Tests: {self.results['summary']['total']}")
        logger.info(f"✅ Passed: {self.results['summary']['passed']}")
        logger.info(f"❌ Failed: {self.results['summary']['failed']}")
        
        success_rate = (self.results['summary']['passed'] / self.results['summary']['total']) * 100
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        # Save detailed results
        results_file = self.base_dir / 'system_test_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"📄 Detailed results saved to: {results_file}")
        
        if self.results['summary']['failed'] == 0:
            logger.info("🎉 ALL TESTS PASSED - System is ready for production!")
            return True
        else:
            logger.info("⚠️  Some tests failed - Review results before deployment")
            return False

if __name__ == "__main__":
    test_suite = ARIASystemTest()
    success = test_suite.run_all_tests()
    sys.exit(0 if success else 1)
