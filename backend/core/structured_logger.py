"""
Structured Logging for ARIA Pro
JSON-formatted logs with context for production observability
"""

import logging
import json
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, Optional
from logging.handlers import RotatingFileHandler
import os

class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        # Base log structure
        log_obj = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_obj['user_id'] = record.user_id
        if hasattr(record, 'session_id'):
            log_obj['session_id'] = record.session_id
        if hasattr(record, 'request_id'):
            log_obj['request_id'] = record.request_id
        if hasattr(record, 'symbol'):
            log_obj['symbol'] = record.symbol
        if hasattr(record, 'order_id'):
            log_obj['order_id'] = record.order_id
        if hasattr(record, 'trade_id'):
            log_obj['trade_id'] = record.trade_id
        if hasattr(record, 'ip_address'):
            log_obj['ip_address'] = record.ip_address
        if hasattr(record, 'endpoint'):
            log_obj['endpoint'] = record.endpoint
        if hasattr(record, 'method'):
            log_obj['method'] = record.method
        if hasattr(record, 'status_code'):
            log_obj['status_code'] = record.status_code
        if hasattr(record, 'response_time'):
            log_obj['response_time'] = record.response_time
        if hasattr(record, 'error_type'):
            log_obj['error_type'] = record.error_type
        if hasattr(record, 'metrics'):
            log_obj['metrics'] = record.metrics
        
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add stack info if present
        if record.stack_info:
            log_obj['stack_trace'] = record.stack_info
        
        return json.dumps(log_obj, default=str)


class StructuredLogger:
    """Wrapper for structured logging with context"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context = {}
    
    def set_context(self, **kwargs):
        """Set persistent context for all logs"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear persistent context"""
        self.context = {}
    
    def _log(self, level: int, msg: str, **kwargs):
        """Internal log method with context injection"""
        extra = {**self.context, **kwargs}
        self.logger.log(level, msg, extra=extra)
    
    def debug(self, msg: str, **kwargs):
        """Log debug message"""
        self._log(logging.DEBUG, msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        """Log info message"""
        self._log(logging.INFO, msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Log warning message"""
        self._log(logging.WARNING, msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        """Log error message"""
        self._log(logging.ERROR, msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        """Log critical message"""
        self._log(logging.CRITICAL, msg, **kwargs)
    
    def exception(self, msg: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(msg, extra={**self.context, **kwargs})
    
    def order_event(
        self,
        event: str,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        order_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs
    ):
        """Log order-related event with sanitized data"""
        # Sanitize sensitive data for production logs
        sanitized_kwargs = {}
        for key, value in kwargs.items():
            if key in ['password', 'secret', 'token', 'key', 'credentials']:
                sanitized_kwargs[key] = '[REDACTED]'
            elif key == 'raw_market_data' or key == 'tick_data':
                # Don't log raw market data at INFO level
                continue
            else:
                sanitized_kwargs[key] = value
        
        self.info(
            f"Order event: {event}",
            symbol=symbol,
            order_type=order_type,
            volume=round(volume, 4) if volume else None,  # Limit precision
            price=round(price, 5) if price else None,     # Limit precision
            order_id=order_id,
            user_id=user_id,
            **sanitized_kwargs
        )
    
    def trade_event(
        self,
        event: str,
        symbol: str,
        action: str,
        confidence: float,
        trade_id: Optional[str] = None,
        profit: Optional[float] = None,
        **kwargs
    ):
        """Log trade-related event"""
        self.info(
            f"Trade event: {event}",
            symbol=symbol,
            action=action,
            confidence=confidence,
            trade_id=trade_id,
            profit=profit,
            **kwargs
        )
    
    def model_event(
        self,
        model_name: str,
        event: str,
        inference_time: Optional[float] = None,
        accuracy: Optional[float] = None,
        **kwargs
    ):
        """Log model-related event"""
        self.info(
            f"Model event: {event}",
            model_name=model_name,
            inference_time=inference_time,
            accuracy=accuracy,
            **kwargs
        )
    
    def api_event(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        response_time: float,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        **kwargs
    ):
        """Log API request event"""
        level = logging.INFO if status_code < 400 else logging.WARNING
        self._log(
            level,
            f"API {method} {endpoint} - {status_code}",
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            response_time=response_time,
            user_id=user_id,
            ip_address=ip_address,
            **kwargs
        )
    
    def security_event(
        self,
        event: str,
        severity: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        **kwargs
    ):
        """Log security-related event"""
        level = logging.WARNING if severity == "medium" else logging.ERROR
        self._log(
            level,
            f"Security event: {event}",
            security_event=event,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            **kwargs
        )
    
    def performance_event(
        self,
        component: str,
        metrics: Dict[str, Any],
        **kwargs
    ):
        """Log performance metrics"""
        self.info(
            f"Performance metrics for {component}",
            component=component,
            metrics=metrics,
            **kwargs
        )


def setup_structured_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
):
    """
    Setup structured logging for the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path for file output
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create structured formatter
    formatter = StructuredFormatter()
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set format for external libraries to reduce noise
    for lib in ['urllib3', 'asyncio', 'aiohttp']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    return root_logger


def get_structured_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance"""
    return StructuredLogger(name)
