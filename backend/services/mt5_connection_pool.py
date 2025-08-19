"""
MT5 Connection Pool with Circuit Breaker
Provides production-ready MT5 connection management.
"""

import logging
import threading
import time
import os
from typing import Optional, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open" 
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: type = Exception

class CircuitBreaker:
    """Circuit breaker for MT5 operations"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self._lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise RuntimeError("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.config.expected_exception as e:
                self._on_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.config.recovery_timeout
        )
    
    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN

class MT5Connection:
    """Individual MT5 connection wrapper"""
    
    def __init__(self, login: int, password: str, server: str):
        self.login = login
        self.password = password
        self.server = server
        self.connected = False
        self.last_used = time.time()
    
    def connect(self) -> bool:
        """Establish MT5 connection"""
        if mt5 is None:
            logger.warning("MetaTrader5 not available")
            return False
        
        try:
            # Initialize MT5 terminal
            if not mt5.initialize():
                error = mt5.last_error()
                logger.warning(f"MT5 initialize failed: {error}")
                return False
            
            # Attempt login with credentials
            if not mt5.login(self.login, password=self.password, server=self.server):
                error = mt5.last_error()
                logger.warning(f"MT5 login failed: {error}")
                mt5.shutdown()
                return False
            
            # Verify connection with account info
            account = mt5.account_info()
            if account is None:
                logger.warning("MT5 account info unavailable after login")
                mt5.shutdown()
                return False
            
            self.connected = True
            self.last_used = time.time()
            logger.info(f"MT5 connected successfully: Account {account.login}, Balance {account.balance}")
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection failed: {e}")
            try:
                mt5.shutdown()
            except:
                pass
            return False
    
    def disconnect(self):
        """Close MT5 connection"""
        if self.connected and mt5:
            try:
                mt5.shutdown()
            except Exception:
                pass
        self.connected = False
    
    def is_healthy(self) -> bool:
        """Check if connection is still healthy"""
        if not self.connected or not mt5:
            return False
        
        try:
            account = mt5.account_info()
            return account is not None
        except Exception:
            return False

class MT5ConnectionPool:
    """Thread-safe MT5 connection pool with async retry logic"""
    
    def __init__(self, max_connections: int = 5):
        self.max_connections = max_connections
        self.pool = []
        self.active_connections = {}
        self._lock = threading.Lock()
        self.circuit_breaker = CircuitBreaker(CircuitBreakerConfig())
        
        # Connection settings from environment (no hardcoded defaults)
        login_str = os.getenv("MT5_LOGIN", "").strip()
        self.login = int(login_str) if login_str.isdigit() else 0
        self.password = os.getenv("MT5_PASSWORD", "")
        self.server = os.getenv("MT5_SERVER", "")
        if not self.login or not self.password or not self.server:
            logger.warning(
                "MT5 credentials incomplete. Ensure MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER are set in environment."
            )
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 2.0
        self.connection_timeout = 30.0
        
        logger.info(f"MT5 Pool initialized: login={self.login}, server={self.server}")
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with automatic cleanup"""
        conn = None
        try:
            conn = self._acquire_connection()
            if conn and conn.is_healthy():
                yield conn
            else:
                raise RuntimeError("No healthy MT5 connection available")
        finally:
            if conn:
                self._release_connection(conn)
    
    def _acquire_connection(self) -> Optional[MT5Connection]:
        """Acquire connection from pool with retry logic and enhanced error handling"""
        with self._lock:
            # Try to get from pool
            while self.pool:
                conn = self.pool.pop()
                if conn.is_healthy():
                    thread_id = threading.get_ident()
                    self.active_connections[thread_id] = conn
                    logger.debug(f"Reused MT5 connection from pool")
                    return conn
                else:
                    logger.debug(f"Discarding unhealthy connection from pool")
                    conn.disconnect()
            
            # Create new connection if under limit
            if len(self.active_connections) < self.max_connections:
                # Implement retry logic at pool level
                for attempt in range(self.max_retries):
                    conn = MT5Connection(self.login, self.password, self.server)
                    try:
                        if self.circuit_breaker.call(conn.connect):
                            thread_id = threading.get_ident()
                            self.active_connections[thread_id] = conn
                            logger.info(f"Created new MT5 connection (active: {len(self.active_connections)})")
                            return conn
                        else:
                            logger.warning(f"MT5 connection attempt {attempt + 1}/{self.max_retries} failed")
                            if attempt < self.max_retries - 1:
                                time.sleep(self.retry_delay * (attempt + 1))
                    except Exception as e:
                        logger.error(f"Circuit breaker rejected MT5 connection attempt {attempt + 1}: {e}")
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay * (attempt + 1))
                
                logger.error("All MT5 connection attempts exhausted")
            else:
                logger.warning(f"MT5 connection pool exhausted (max: {self.max_connections})")
            
            return None
    
    def _release_connection(self, conn: MT5Connection):
        """Return connection to pool"""
        with self._lock:
            thread_id = threading.get_ident()
            if thread_id in self.active_connections:
                del self.active_connections[thread_id]
            
            if conn.is_healthy() and len(self.pool) < self.max_connections:
                self.pool.append(conn)
            else:
                conn.disconnect()
    
    def health_check(self) -> Dict[str, Any]:
        """Get pool health status"""
        with self._lock:
            return {
                "pool_size": len(self.pool),
                "active_connections": len(self.active_connections),
                "max_connections": self.max_connections,
                "circuit_breaker_state": self.circuit_breaker.state.value,
                "failure_count": self.circuit_breaker.failure_count
            }
    
    def shutdown(self):
        """Shutdown all connections"""
        with self._lock:
            for conn in self.pool:
                conn.disconnect()
            for conn in self.active_connections.values():
                conn.disconnect()
            self.pool.clear()
            self.active_connections.clear()

# Global connection pool instance
_connection_pool = None

def get_connection_pool() -> MT5ConnectionPool:
    """Get global MT5 connection pool"""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = MT5ConnectionPool()
    return _connection_pool
