"""
Error Boundary Middleware for ARIA PRO
Global error handling and logging for production
"""

import sys
import traceback
import logging
from typing import Any
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime

logger = logging.getLogger(__name__)


class ErrorBoundaryMiddleware(BaseHTTPMiddleware):
    """
    Global error handler that catches all exceptions and returns proper error responses
    """
    
    def __init__(self, app, debug: bool = False):
        super().__init__(app)
        self.debug = debug
    
    async def dispatch(self, request: Request, call_next):
        """Process request with error boundary"""
        try:
            response = await call_next(request)
            return response
        except HTTPException as e:
            # Let HTTPExceptions pass through (they have proper status codes)
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": e.detail,
                    "status_code": e.status_code,
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            # Log the full exception
            logger.error(f"Unhandled exception in {request.url.path}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Prepare error response
            error_id = f"ERR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            content = {
                "error": "Internal Server Error",
                "message": "An unexpected error occurred processing your request",
                "error_id": error_id,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add debug info if in debug mode
            if self.debug:
                content["debug"] = {
                    "exception": str(e),
                    "type": type(e).__name__,
                    "traceback": traceback.format_exc().split('\n')
                }
            
            return JSONResponse(
                status_code=500,
                content=content
            )
