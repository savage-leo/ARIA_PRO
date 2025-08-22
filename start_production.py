#!/usr/bin/env python3
"""
ARIA PRO Production Startup Script
Launches both FastAPI backend and Phase 3 Orchestrator
"""

import os
import sys
import subprocess
import time
import signal
import logging
from pathlib import Path
from dotenv import load_dotenv

# Resolve project root relative to this script
ROOT = Path(__file__).parent.resolve()
LOGS_DIR = ROOT / "logs"

# Ensure logs directory exists before configuring logging
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
    handlers=[
        logging.FileHandler(str(LOGS_DIR / "production.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ARIA.PRODUCTION")


def load_environment():
    """Load environment variables from production.env"""
    env_file = ROOT / "production.env"
    if env_file.exists():
        load_dotenv(env_file)
        logger.info("Loaded production environment from production.env")
    else:
        logger.warning("production.env not found, using system environment")


def check_dependencies():
    """Check if all required dependencies are available"""
    required_modules = [
        "fastapi",
        "uvicorn",
        "numpy",
        "MetaTrader5",
        "sklearn",  # scikit-learn runtime import path
        "joblib",
    ]

    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        logger.error(f"Missing required modules: {missing}")
        logger.error("Please install: pip install -r backend/requirements.txt")
        return False

    logger.info("All dependencies verified")
    return True


def start_backend():
    """Start the FastAPI backend server"""
    logger.info("Starting ARIA PRO FastAPI backend...")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--workers",
        "1",
    ]

    try:
        backend_out = open(LOGS_DIR / "backend.out", "a", buffering=1, encoding="utf-8")
        process = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=backend_out, stderr=backend_out, text=True
        )
        logger.info(f"Backend started with PID: {process.pid}")
        return process
    except Exception as e:
        logger.error(f"Failed to start backend: {e}")
        return None


def start_orchestrator():
    """Start the Phase 3 Orchestrator"""
    logger.info("Starting Phase 3 Orchestrator...")

    cmd = [sys.executable, "-m", "backend.core.phase3_orchestrator"]

    try:
        orch_out = open(
            LOGS_DIR / "orchestrator.out", "a", buffering=1, encoding="utf-8"
        )
        process = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=orch_out, stderr=orch_out, text=True
        )
        logger.info(f"Orchestrator started with PID: {process.pid}")
        return process
    except Exception as e:
        logger.error(f"Failed to start orchestrator: {e}")
        return None


def monitor_processes(backend_proc, orchestrator_proc):
    """Monitor running processes and handle shutdown"""
    try:
        while True:
            # Check if processes are still running
            if backend_proc and backend_proc.poll() is not None:
                logger.error("Backend process terminated unexpectedly")
                break

            if orchestrator_proc and orchestrator_proc.poll() is not None:
                logger.error("Orchestrator process terminated unexpectedly")
                break

            time.sleep(5)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        # Clean shutdown
        if backend_proc:
            logger.info("Stopping backend...")
            backend_proc.terminate()
            backend_proc.wait(timeout=10)

        if orchestrator_proc:
            logger.info("Stopping orchestrator...")
            orchestrator_proc.terminate()
            orchestrator_proc.wait(timeout=10)


def main():
    """Main production startup function"""
    logger.info("=== ARIA PRO Production Startup ===")

    # Load environment
    load_environment()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Create logs directory (already ensured above; keep for idempotency)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Start backend
    backend_proc = start_backend()
    if not backend_proc:
        sys.exit(1)

    # Wait a moment for backend to initialize
    time.sleep(3)

    # Start orchestrator
    orchestrator_proc = start_orchestrator()
    if not orchestrator_proc:
        logger.error("Failed to start orchestrator, stopping backend")
        backend_proc.terminate()
        sys.exit(1)

    logger.info("=== ARIA PRO Production System Started ===")
    logger.info("Backend: http://localhost:8000")
    logger.info("Health Check: http://localhost:8000/health")
    logger.info("Press Ctrl+C to shutdown")

    # Monitor processes
    monitor_processes(backend_proc, orchestrator_proc)

    logger.info("=== ARIA PRO Production System Shutdown Complete ===")


if __name__ == "__main__":
    main()
