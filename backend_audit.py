#!/usr/bin/env python3
"""
ARIA PRO Backend Comprehensive Audit Script
Identifies all missing dependencies and import issues
"""

import sys
import os
import importlib.util
import subprocess
from pathlib import Path

# Add backend to path
sys.path.insert(0, r'c:\savage\ARIA_PRO')

# Track missing modules
missing_modules = set()
import_errors = []
successful_imports = []

# Core dependencies to check
REQUIRED_PACKAGES = [
    # FastAPI core
    'fastapi', 'uvicorn', 'pydantic', 'python-dotenv',
    # FastAPI extensions
    'python-multipart', 'slowapi', 'aiofiles',
    # Database
    'sqlalchemy', 'alembic',
    # Trading
    'MetaTrader5', 'pandas', 'numpy', 'ta',
    # ML/AI
    'scikit-learn', 'xgboost', 'onnxruntime', 'stable_baselines3',
    'torch', 'transformers',
    # Async/Network
    'aiohttp', 'websockets', 'httpx',
    # Monitoring
    'psutil', 'prometheus_client',
    # Utils
    'coloredlogs', 'python-json-logger', 'cryptography',
    'llama_cpp', 'joblib', 'matplotlib', 'seaborn'
]

def check_module(module_name):
    """Check if a module can be imported."""
    try:
        spec = importlib.util.find_spec(module_name.replace('-', '_'))
        if spec is None:
            return False
        return True
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False

def get_installed_version(package_name):
    """Get installed version of a package."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', package_name],
            capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':')[1].strip()
    except:
        pass
    return None

print("=" * 60)
print("ARIA PRO BACKEND AUDIT")
print("=" * 60)

# 1. Check core dependencies
print("\n1. CHECKING CORE DEPENDENCIES")
print("-" * 30)
for package in REQUIRED_PACKAGES:
    module_name = package.replace('-', '_')
    if check_module(module_name):
        version = get_installed_version(package)
        successful_imports.append(f"{package}=={version}" if version else package)
        print(f"✓ {package}: {version or 'installed'}")
    else:
        missing_modules.add(package)
        print(f"✗ {package}: MISSING")

# 2. Try to import backend main
print("\n2. TESTING BACKEND MAIN IMPORT")
print("-" * 30)
try:
    from backend.main import app
    print("✓ backend.main imported successfully")
except ImportError as e:
    import_errors.append(f"backend.main: {str(e)}")
    print(f"✗ backend.main import failed: {e}")
except Exception as e:
    import_errors.append(f"backend.main: {str(e)}")
    print(f"✗ backend.main error: {e}")

# 3. Test specific backend modules
print("\n3. TESTING BACKEND MODULES")
print("-" * 30)
backend_modules = [
    'backend.core.config',
    'backend.core.live_guard',
    'backend.core.performance_monitor',
    'backend.services.mt5_market_data',
    'backend.services.auto_trader',
    'backend.services.real_ai_signal_generator',
    'backend.services.data_source_manager',
    'backend.services.cpp_integration',
    'backend.routes.account',
    'backend.routes.market',
    'backend.routes.signals',
    'backend.routes.monitoring',
    'backend.routes.websocket',
    'backend.routes.trade_memory_api'
]

for module in backend_modules:
    try:
        importlib.import_module(module)
        print(f"✓ {module}")
    except ImportError as e:
        error_msg = str(e)
        print(f"✗ {module}: {error_msg}")
        import_errors.append(f"{module}: {error_msg}")
        # Extract missing module from error
        if "No module named" in error_msg:
            missing = error_msg.split("'")[1].split('.')[0]
            if missing not in ['backend']:
                missing_modules.add(missing)
    except Exception as e:
        print(f"✗ {module}: {e}")
        import_errors.append(f"{module}: {str(e)}")

# 4. Generate installation commands
print("\n4. MISSING PACKAGES TO INSTALL")
print("-" * 30)
if missing_modules:
    print("Missing packages detected:")
    for module in sorted(missing_modules):
        print(f"  - {module}")
    
    # Map common import names to package names
    package_map = {
        'sklearn': 'scikit-learn',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'yaml': 'pyyaml',
        'dotenv': 'python-dotenv',
        'multipart': 'python-multipart',
        'limits': 'slowapi',
        'starlette': 'starlette',
        'llama_cpp': 'llama-cpp-python',
        'stable_baselines3': 'stable-baselines3[extra]',
        'MetaTrader5': 'MetaTrader5',
        'websockets': 'websockets',
        'aiohttp': 'aiohttp',
        'httpx': 'httpx',
        'prometheus_client': 'prometheus-client',
        'sqlalchemy': 'sqlalchemy',
        'alembic': 'alembic',
        'joblib': 'joblib',
        'cryptography': 'cryptography',
        'transformers': 'transformers',
        'torch': 'torch',
        'onnxruntime': 'onnxruntime'
    }
    
    install_packages = []
    for module in missing_modules:
        package = package_map.get(module, module)
        install_packages.append(package)
    
    print(f"\nInstall command:")
    print(f"  .\.venv\Scripts\pip install {' '.join(sorted(install_packages))}")
else:
    print("✓ All required packages are installed")

# 5. Check environment variables
print("\n5. CHECKING ENVIRONMENT VARIABLES")
print("-" * 30)
env_vars = [
    'PYTHONPATH',
    'ARIA_ENABLE_MT5',
    'MT5_LOGIN',
    'MT5_PASSWORD',
    'MT5_SERVER',
    'AUTO_TRADE_ENABLED',
    'ADMIN_API_KEY'
]
for var in env_vars:
    value = os.environ.get(var)
    if value:
        # Mask sensitive values
        if 'PASSWORD' in var or 'KEY' in var:
            display = '*' * 8
        else:
            display = value[:20] + '...' if len(value) > 20 else value
        print(f"✓ {var}: {display}")
    else:
        print(f"✗ {var}: Not set")

# 6. Summary
print("\n" + "=" * 60)
print("AUDIT SUMMARY")
print("=" * 60)
print(f"Missing packages: {len(missing_modules)}")
print(f"Import errors: {len(import_errors)}")
print(f"Successful imports: {len(successful_imports)}")

if missing_modules or import_errors:
    print("\nACTION REQUIRED:")
    print("1. Install missing packages (see command above)")
    print("2. Set PYTHONPATH environment variable:")
    print("   $env:PYTHONPATH = 'c:\\savage\\ARIA_PRO'")
    print("3. Fix any remaining import errors")
    sys.exit(1)
else:
    print("\n✓ Backend is ready to run!")
    sys.exit(0)
