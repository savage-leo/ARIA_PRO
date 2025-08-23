"""
Comprehensive Error Handling and User-Friendly Error Boundaries
Production-grade error handling with proper logging and user messaging
"""

import logging
import traceback
from typing import Dict, Any, Optional
from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class ErrorType(str, Enum):
    """Categorized error types for better user experience"""
    MT5_CONNECTION = "mt5_connection"
    MODEL_INFERENCE = "model_inference"
    TRADING_EXECUTION = "trading_execution"
    DATA_VALIDATION = "data_validation"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    SYSTEM_RESOURCE = "system_resource"
    EXTERNAL_API = "external_api"
    INTERNAL_SERVER = "internal_server"

class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class UserFriendlyError:
    """User-friendly error with proper categorization"""
    
    def __init__(
        self,
        error_type: ErrorType,
        severity: ErrorSeverity,
        user_message: str,
        technical_details: str = None,
        suggested_action: str = None,
        retry_after: Optional[int] = None
    ):
        self.error_type = error_type
        self.severity = severity
        self.user_message = user_message
        self.technical_details = technical_details
        self.suggested_action = suggested_action
        self.retry_after = retry_after
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format"""
        response = {
            "error": {
                "type": self.error_type,
                "severity": self.severity,
                "message": self.user_message,
                "timestamp": "2025-08-23T13:56:32+03:00"
            }
        }
        
        if self.suggested_action:
            response["error"]["suggested_action"] = self.suggested_action
        
        if self.retry_after:
            response["error"]["retry_after"] = self.retry_after
        
        if self.technical_details:
            response["error"]["technical_details"] = self.technical_details
        
        return response

class ErrorBoundaryMiddleware(BaseHTTPMiddleware):
    """Global error boundary middleware for FastAPI"""
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            return await self.handle_exception(request, exc)
    
    async def handle_exception(self, request: Request, exc: Exception) -> Response:
        """Handle exceptions with user-friendly responses"""
        
        # Log the full exception for debugging
        logger.error(
            f"Unhandled exception in {request.method} {request.url.path}: {exc}",
            exc_info=True
        )
        
        # Categorize and create user-friendly error
        error = self.categorize_exception(exc)
        
        # Determine HTTP status code
        status_code = self.get_status_code(error.error_type, error.severity)
        
        return JSONResponse(
            status_code=status_code,
            content=error.to_dict()
        )
    
    def categorize_exception(self, exc: Exception) -> UserFriendlyError:
        """Categorize exception into user-friendly error"""
        
        exc_str = str(exc).lower()
        exc_type = type(exc).__name__
        
        # MT5 Connection Errors
        if "mt5" in exc_str or "metatrader" in exc_str:
            return UserFriendlyError(
                error_type=ErrorType.MT5_CONNECTION,
                severity=ErrorSeverity.HIGH,
                user_message="Trading platform connection is temporarily unavailable",
                technical_details=str(exc),
                suggested_action="The system will automatically retry. Please wait a moment and try again.",
                retry_after=30
            )
        
        # Model Inference Errors
        if "model" in exc_str or "onnx" in exc_str or "inference" in exc_str:
            return UserFriendlyError(
                error_type=ErrorType.MODEL_INFERENCE,
                severity=ErrorSeverity.MEDIUM,
                user_message="AI analysis is temporarily unavailable",
                technical_details=str(exc),
                suggested_action="Please try again in a few moments. If the issue persists, contact support."
            )
        
        # Trading Execution Errors
        if "trade" in exc_str or "order" in exc_str or "position" in exc_str:
            return UserFriendlyError(
                error_type=ErrorType.TRADING_EXECUTION,
                severity=ErrorSeverity.CRITICAL,
                user_message="Trading operation could not be completed",
                technical_details=str(exc),
                suggested_action="Please verify your account status and try again. Contact support if the issue persists."
            )
        
        # Authentication Errors
        if exc_type in ["HTTPException"] and hasattr(exc, 'status_code'):
            if exc.status_code in [401, 403]:
                return UserFriendlyError(
                    error_type=ErrorType.AUTHENTICATION,
                    severity=ErrorSeverity.MEDIUM,
                    user_message="Authentication required or access denied",
                    suggested_action="Please check your credentials and try again."
                )
        
        # Rate Limiting Errors
        if "rate limit" in exc_str or "too many requests" in exc_str:
            return UserFriendlyError(
                error_type=ErrorType.RATE_LIMIT,
                severity=ErrorSeverity.LOW,
                user_message="Too many requests. Please slow down.",
                suggested_action="Wait a moment before making another request.",
                retry_after=60
            )
        
        # System Resource Errors
        if "memory" in exc_str or "disk" in exc_str or "cpu" in exc_str:
            return UserFriendlyError(
                error_type=ErrorType.SYSTEM_RESOURCE,
                severity=ErrorSeverity.HIGH,
                user_message="System resources are temporarily constrained",
                technical_details=str(exc),
                suggested_action="Please try again in a few minutes. The system is working to resolve this issue."
            )
        
        # External API Errors
        if "api" in exc_str or "http" in exc_str or "connection" in exc_str:
            return UserFriendlyError(
                error_type=ErrorType.EXTERNAL_API,
                severity=ErrorSeverity.MEDIUM,
                user_message="External service is temporarily unavailable",
                suggested_action="Please try again later. The issue is likely temporary.",
                retry_after=120
            )
        
        # Default Internal Server Error
        return UserFriendlyError(
            error_type=ErrorType.INTERNAL_SERVER,
            severity=ErrorSeverity.HIGH,
            user_message="An unexpected error occurred",
            technical_details=f"{exc_type}: {str(exc)}",
            suggested_action="Please try again. If the problem persists, contact support with the error details."
        )
    
    def get_status_code(self, error_type: ErrorType, severity: ErrorSeverity) -> int:
        """Determine appropriate HTTP status code"""
        
        status_map = {
            ErrorType.MT5_CONNECTION: 503,
            ErrorType.MODEL_INFERENCE: 503,
            ErrorType.TRADING_EXECUTION: 503,
            ErrorType.DATA_VALIDATION: 400,
            ErrorType.AUTHENTICATION: 401,
            ErrorType.RATE_LIMIT: 429,
            ErrorType.SYSTEM_RESOURCE: 503,
            ErrorType.EXTERNAL_API: 502,
            ErrorType.INTERNAL_SERVER: 500
        }
        
        return status_map.get(error_type, 500)

def create_user_friendly_exception(
    error_type: ErrorType,
    severity: ErrorSeverity,
    user_message: str,
    technical_details: str = None,
    suggested_action: str = None
) -> HTTPException:
    """Create HTTPException with user-friendly error format"""
    
    error = UserFriendlyError(
        error_type=error_type,
        severity=severity,
        user_message=user_message,
        technical_details=technical_details,
        suggested_action=suggested_action
    )
    
    status_code = ErrorBoundaryMiddleware().get_status_code(error_type, severity)
    
    return HTTPException(
        status_code=status_code,
        detail=error.to_dict()["error"]
    )

# Convenience functions for common error scenarios
def mt5_unavailable_error(details: str = None) -> HTTPException:
    """Create MT5 unavailable error"""
    return create_user_friendly_exception(
        error_type=ErrorType.MT5_CONNECTION,
        severity=ErrorSeverity.HIGH,
        user_message="Trading platform is temporarily unavailable",
        technical_details=details,
        suggested_action="The system will automatically retry. Please wait and try again."
    )

def model_inference_error(details: str = None) -> HTTPException:
    """Create model inference error"""
    return create_user_friendly_exception(
        error_type=ErrorType.MODEL_INFERENCE,
        severity=ErrorSeverity.MEDIUM,
        user_message="AI analysis is temporarily unavailable",
        technical_details=details,
        suggested_action="Please try again in a few moments."
    )

def trading_execution_error(details: str = None) -> HTTPException:
    """Create trading execution error"""
    return create_user_friendly_exception(
        error_type=ErrorType.TRADING_EXECUTION,
        severity=ErrorSeverity.CRITICAL,
        user_message="Trading operation could not be completed",
        technical_details=details,
        suggested_action="Please verify your account status and try again."
    )
