#!/usr/bin/env python3
"""
ARIA_PRO Backend Diagnostic Script
Identifies and reports backend issues
"""

import sys
import os
import importlib
import traceback


def check_imports():
    """Check if all required modules can be imported"""
    print("Checking imports...")

    modules = [
        "backend.main",
        "backend.routes.trading",
        "backend.routes.account",
        "backend.routes.market",
        "backend.routes.positions",
        "backend.routes.signals",
        "backend.routes.smc_routes",
        "backend.routes.websocket",
        "backend.services.mt5_executor",
        "backend.services.ws_broadcaster",
        "backend.services.cpp_integration",
        "backend.core.trade_memory",
        "backend.core.risk_engine",
        "backend.smc.smc_edge_core",
        "backend.smc.smc_fusion_core",
        "backend.smc.trap_detector",
    ]

    failed_imports = []

    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {e}")
            failed_imports.append((module, e))

    return failed_imports


def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")

    packages = ["fastapi", "uvicorn", "pydantic", "websockets", "requests", "pybind11"]

    missing_packages = []

    for package in packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - not installed")
            missing_packages.append(package)

    return missing_packages


def check_files():
    """Check if required files exist"""
    print("\nChecking files...")

    files = [
        "backend/main.py",
        "backend/routes/trading.py",
        "backend/routes/account.py",
        "backend/routes/market.py",
        "backend/routes/positions.py",
        "backend/routes/signals.py",
        "backend/routes/smc_routes.py",
        "backend/routes/websocket.py",
        "backend/services/mt5_executor.py",
        "backend/services/ws_broadcaster.py",
        "backend/services/cpp_integration.py",
        "backend/core/trade_memory.py",
        "backend/core/risk_engine.py",
        "backend/smc/smc_edge_core.py",
        "backend/smc/smc_fusion_core.py",
        "backend/smc/trap_detector.py",
    ]

    missing_files = []

    for file_path in files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - missing")
            missing_files.append(file_path)

    return missing_files


def test_app_creation():
    """Test if the FastAPI app can be created"""
    print("\nTesting app creation...")

    try:
        from backend.main import app

        print("✅ FastAPI app created successfully")
        print(f"   Title: {app.title}")
        print(f"   Routes: {len(app.routes)}")
        return True
    except Exception as e:
        print(f"❌ Failed to create app: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all diagnostics"""
    print("ARIA_PRO Backend Diagnostics")
    print("=" * 40)

    # Check imports
    failed_imports = check_imports()

    # Check dependencies
    missing_packages = check_dependencies()

    # Check files
    missing_files = check_files()

    # Test app creation
    app_ok = test_app_creation()

    # Summary
    print("\n" + "=" * 40)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 40)

    if failed_imports:
        print(f"❌ {len(failed_imports)} import failures")
        for module, error in failed_imports:
            print(f"   - {module}: {error}")
    else:
        print("✅ All imports successful")

    if missing_packages:
        print(f"❌ {len(missing_packages)} missing packages")
        for package in missing_packages:
            print(f"   - {package}")
    else:
        print("✅ All dependencies installed")

    if missing_files:
        print(f"❌ {len(missing_files)} missing files")
        for file_path in missing_files:
            print(f"   - {file_path}")
    else:
        print("✅ All files present")

    if app_ok:
        print("✅ App creation successful")
    else:
        print("❌ App creation failed")

    if not any([failed_imports, missing_packages, missing_files]) and app_ok:
        print("\n🎉 Backend is ready to start!")
    else:
        print("\n⚠️  Backend has issues that need to be resolved")


if __name__ == "__main__":
    main()
