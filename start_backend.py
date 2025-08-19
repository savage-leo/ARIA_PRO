#!/usr/bin/env python3
"""
ARIA_PRO Backend Startup Script
Starts the backend without auto-reload to avoid issues
"""

import uvicorn
import sys
import os


def main():
    """Start the backend server"""
    print("Starting ARIA_PRO Backend...")
    print("=" * 40)

    # Set environment variables
    os.environ.setdefault("ADMIN_API_KEY", "changeme")
    os.environ.setdefault("AUTO_EXEC_ENABLED", "true")
    os.environ.setdefault("ALLOW_LIVE", "1")

    # Start server without reload to avoid issues
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload to avoid issues
        log_level="info",
    )


if __name__ == "__main__":
    main()
