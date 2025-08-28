#!/usr/bin/env python
"""
Direct startup script for CPU-friendly backend
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    print("=" * 50)
    print("ARIA CPU BACKEND - DIRECT START")
    print("=" * 50)
    
    import uvicorn
    uvicorn.run(
        "backend.main_cpu:app",
        host="0.0.0.0",
        port=8100,
        reload=False,
        log_level="info",
        workers=1,  # Single worker for institutional trading consistency
        access_log=True,
        use_colors=True,
        server_header=False,  # Security: hide server info
        date_header=False,    # Security: hide date header
        timeout_keep_alive=30,
        timeout_graceful_shutdown=10,
        limit_concurrency=1000,
        limit_max_requests=10000,
        backlog=2048
    )
