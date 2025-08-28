# Production Security Middleware for ARIA Pro
# Implements CORS, Security Headers, and Error Boundary

import logging
import traceback
from typing import Dict, Any, Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    def __init__(self, app: ASGIApp, csp_connect_src: str = ""):
        super().__init__(app)
        self.csp_connect_src = csp_connect_src
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # HSTS for HTTPS
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        
        # Content Security Policy
        csp_parts = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "font-src 'self' data:",
            f"connect-src 'self' ws: wss: {self.csp_connect_src}".strip(),
            "object-src 'none'",
            "base-uri 'self'",
            "form-action 'self'"
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_parts)
        
        # Cache control for sensitive endpoints
        if any(path in request.url.path for path in ["/api/account", "/api/positions", "/api/trading"]):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        return response

class ProductionErrorBoundary(BaseHTTPMiddleware):
    """Global error handler that sanitizes error messages in production"""
    
    def __init__(self, app: ASGIApp, debug: bool = False):
        super().__init__(app)
        self.debug = debug
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except HTTPException:
            # Let FastAPI handle HTTP exceptions normally
            raise
        except Exception as e:
            # Log the full error for debugging
            logger.error(
                f"Unhandled exception in {request.method} {request.url.path}: {e}",
                exc_info=True
            )
            
            # Return sanitized error response
            if self.debug:
                # Development mode - show detailed error
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={
                        "detail": "Internal server error",
                        "error": str(e),
                        "type": type(e).__name__,
                        "traceback": traceback.format_exc().split('\n')
                    }
                )
            else:
                # Production mode - generic error message
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={
                        "detail": "Internal server error",
                        "error_id": f"ERR_{hash(str(e)) % 10000:04d}"
                    }
                )

def setup_cors_middleware(app, settings):
    """Configure CORS middleware with production settings"""
    
    # Development: allow everything for ease of local testing
    if getattr(settings, "is_development", False):
        logger.warning("Development mode detected: enabling permissive CORS (allow_origins=['*'])")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining", 
                "X-RateLimit-Reset",
                "Retry-After"
            ],
            max_age=600,
        )
        return
    
    # Production/Test: use configured origins only
    origins = settings.cors_origins_list
    if not origins:
        logger.warning("No CORS origins configured - API will reject browser requests in production/test")
        origins = []  # Empty list = no origins allowed
    
    logger.info(f"CORS configured for origins: {origins}")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Requested-With",
            "Accept",
            "Origin",
            "User-Agent",
            "DNT",
            "Cache-Control",
            "X-Mx-ReqToken",
            "Keep-Alive",
            "If-Modified-Since",
            "X-CSRFToken"
        ],
        expose_headers=[
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining", 
            "X-RateLimit-Reset",
            "Retry-After"
        ]
    )

def setup_trusted_host_middleware(app, settings):
    """Configure trusted host middleware"""
    
    allowed_hosts = settings.allowed_hosts_list
    if not allowed_hosts:
        logger.warning("No allowed hosts configured - will accept requests from any host")
        return
    
    logger.info(f"Trusted hosts configured: {allowed_hosts}")
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=allowed_hosts
    )

def setup_security_middleware(app, settings, debug: bool = False):
    """Setup all security middleware in correct order"""
    
    # 1. Trusted Host (outermost)
    setup_trusted_host_middleware(app, settings)
    
    # 2. CORS
    setup_cors_middleware(app, settings)
    
    # 3. Security Headers
    app.add_middleware(
        SecurityHeadersMiddleware,
        csp_connect_src=" ".join(settings.csp_connect_src_extra)
    )
    
    # 4. Error Boundary (innermost - catches all other errors)
    app.add_middleware(
        ProductionErrorBoundary,
        debug=debug
    )
    
    logger.info("Security middleware stack configured")
