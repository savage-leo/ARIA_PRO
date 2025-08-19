#!/usr/bin/env python3
"""
Test script for frontend accessibility
"""

import requests
import time


def test_frontend():
    """Test if frontend is accessible"""
    print("Testing Frontend Accessibility...")

    try:
        # Test frontend on default Vite port
        response = requests.get("http://localhost:5173", timeout=5)
        print(f"Frontend Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Frontend is accessible")
        else:
            print("⚠️  Frontend returned non-200 status")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Frontend not accessible - is it running on http://localhost:5173?")
        return False
    except Exception as e:
        print(f"❌ Frontend test failed: {e}")
        return False


def test_backend():
    """Test if backend is accessible"""
    print("\nTesting Backend Accessibility...")

    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"Backend Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Backend is accessible")
            return True
        else:
            print("⚠️  Backend returned non-200 status")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Backend not accessible - is it running on http://localhost:8000?")
        return False
    except Exception as e:
        print(f"❌ Backend test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("ARIA_PRO Frontend/Backend Connectivity Test")
    print("=" * 50)

    backend_ok = test_backend()
    frontend_ok = test_frontend()

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    if backend_ok and frontend_ok:
        print("🎉 Both frontend and backend are accessible!")
        print("   Frontend: http://localhost:5173")
        print("   Backend:  http://localhost:8000")
    elif backend_ok:
        print("⚠️  Backend is running but frontend is not accessible")
        print("   Try running: cd frontend && npm run dev")
    elif frontend_ok:
        print("⚠️  Frontend is running but backend is not accessible")
        print("   Try running: python start_backend.py")
    else:
        print("❌ Neither frontend nor backend are accessible")
        print("   Start backend: python start_backend.py")
        print("   Start frontend: cd frontend && npm run dev")


if __name__ == "__main__":
    main()
