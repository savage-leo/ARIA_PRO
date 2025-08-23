"""
Audit Trail System for ARIA Pro
Immutable logging of all trading operations for compliance
"""

import json
import hashlib
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

class AuditEventType(str, Enum):
    ORDER_SUBMIT = "order_submit"
    ORDER_MODIFY = "order_modify"
    ORDER_CANCEL = "order_cancel"
    ORDER_EXECUTE = "order_execute"
    ORDER_REJECT = "order_reject"
    POSITION_OPEN = "position_open"
    POSITION_CLOSE = "position_close"
    RISK_OVERRIDE = "risk_override"
    KILL_SWITCH = "kill_switch"
    MODEL_SIGNAL = "model_signal"
    AUTH_LOGIN = "auth_login"
    AUTH_LOGOUT = "auth_logout"
    CONFIG_CHANGE = "config_change"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"

class AuditLogger:
    """Thread-safe audit logger with SQLite backend"""
    
    def __init__(self, db_path: str = "./data/audit_trail.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        """Initialize audit database with immutable table"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    username TEXT,
                    ip_address TEXT,
                    session_id TEXT,
                    symbol TEXT,
                    order_id TEXT,
                    position_id TEXT,
                    action TEXT NOT NULL,
                    details TEXT,
                    risk_metrics TEXT,
                    hash TEXT NOT NULL,
                    prev_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_log(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_log(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_username ON audit_log(username)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_order_id ON audit_log(order_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON audit_log(symbol)")
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper cleanup"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _compute_hash(self, data: Dict[str, Any], prev_hash: Optional[str] = None) -> str:
        """Compute SHA-256 hash of audit entry for immutability"""
        # Create deterministic string representation
        hash_input = json.dumps(data, sort_keys=True) + (prev_hash or "")
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        position_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        risk_metrics: Optional[Dict[str, Any]] = None
    ) -> int:
        """Log an audit event with chain-of-custody hash"""
        
        with self._lock:
            timestamp = time.time()
            
            # Get previous hash for chain
            prev_hash = None
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1"
                )
                row = cursor.fetchone()
                if row:
                    prev_hash = row["hash"]
            
            # Prepare audit data
            audit_data = {
                "timestamp": timestamp,
                "event_type": event_type,
                "username": username,
                "ip_address": ip_address,
                "session_id": session_id,
                "symbol": symbol,
                "order_id": order_id,
                "position_id": position_id,
                "action": action,
                "details": details,
                "risk_metrics": risk_metrics
            }
            
            # Compute hash
            entry_hash = self._compute_hash(audit_data, prev_hash)
            
            # Insert into database
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO audit_log (
                        timestamp, event_type, username, ip_address, session_id,
                        symbol, order_id, position_id, action, details,
                        risk_metrics, hash, prev_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp,
                    event_type,
                    username,
                    ip_address,
                    session_id,
                    symbol,
                    order_id,
                    position_id,
                    action,
                    json.dumps(details) if details else None,
                    json.dumps(risk_metrics) if risk_metrics else None,
                    entry_hash,
                    prev_hash
                ))
                conn.commit()
                audit_id = cursor.lastrowid
            
            # Also log to standard logger for real-time monitoring
            logger.info(
                f"AUDIT: {event_type} - {action} | "
                f"User: {username} | Symbol: {symbol} | "
                f"Order: {order_id} | Hash: {entry_hash[:8]}..."
            )
            
            return audit_id
    
    def verify_integrity(self, start_id: int = 1, end_id: Optional[int] = None) -> bool:
        """Verify audit log integrity by checking hash chain"""
        with self._get_connection() as conn:
            query = "SELECT * FROM audit_log WHERE id >= ?"
            params = [start_id]
            
            if end_id:
                query += " AND id <= ?"
                params.append(end_id)
            
            query += " ORDER BY id ASC"
            
            cursor = conn.execute(query, params)
            prev_hash = None
            
            for row in cursor:
                # Reconstruct audit data
                audit_data = {
                    "timestamp": row["timestamp"],
                    "event_type": row["event_type"],
                    "username": row["username"],
                    "ip_address": row["ip_address"],
                    "session_id": row["session_id"],
                    "symbol": row["symbol"],
                    "order_id": row["order_id"],
                    "position_id": row["position_id"],
                    "action": row["action"],
                    "details": json.loads(row["details"]) if row["details"] else None,
                    "risk_metrics": json.loads(row["risk_metrics"]) if row["risk_metrics"] else None
                }
                
                # Verify hash
                expected_hash = self._compute_hash(audit_data, prev_hash)
                if expected_hash != row["hash"]:
                    logger.error(f"Audit integrity violation at ID {row['id']}")
                    return False
                
                if row["prev_hash"] != prev_hash:
                    logger.error(f"Chain integrity violation at ID {row['id']}")
                    return False
                
                prev_hash = row["hash"]
        
        return True
    
    def get_recent_events(
        self,
        event_type: Optional[AuditEventType] = None,
        username: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query recent audit events with filters"""
        with self._get_connection() as conn:
            query = "SELECT * FROM audit_log WHERE 1=1"
            params = []
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)
            
            if username:
                query += " AND username = ?"
                params.append(username)
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            
            events = []
            for row in cursor:
                events.append({
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "datetime": datetime.fromtimestamp(row["timestamp"]).isoformat(),
                    "event_type": row["event_type"],
                    "username": row["username"],
                    "ip_address": row["ip_address"],
                    "session_id": row["session_id"],
                    "symbol": row["symbol"],
                    "order_id": row["order_id"],
                    "position_id": row["position_id"],
                    "action": row["action"],
                    "details": json.loads(row["details"]) if row["details"] else None,
                    "risk_metrics": json.loads(row["risk_metrics"]) if row["risk_metrics"] else None,
                    "hash": row["hash"]
                })
            
            return events

# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None

def get_audit_logger() -> AuditLogger:
    """Get or create singleton audit logger"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger

# Convenience functions for common audit events
def audit_order_submit(
    username: str,
    symbol: str,
    order_id: str,
    order_type: str,
    volume: float,
    price: Optional[float] = None,
    sl: Optional[float] = None,
    tp: Optional[float] = None,
    ip_address: Optional[str] = None,
    risk_metrics: Optional[Dict[str, Any]] = None
):
    """Audit an order submission"""
    logger = get_audit_logger()
    return logger.log_event(
        event_type=AuditEventType.ORDER_SUBMIT,
        action=f"Submit {order_type} order",
        username=username,
        ip_address=ip_address,
        symbol=symbol,
        order_id=order_id,
        details={
            "order_type": order_type,
            "volume": volume,
            "price": price,
            "stop_loss": sl,
            "take_profit": tp
        },
        risk_metrics=risk_metrics
    )

def audit_position_close(
    username: str,
    symbol: str,
    position_id: str,
    profit: float,
    close_reason: str,
    ip_address: Optional[str] = None
):
    """Audit a position closure"""
    logger = get_audit_logger()
    return logger.log_event(
        event_type=AuditEventType.POSITION_CLOSE,
        action=f"Close position: {close_reason}",
        username=username,
        ip_address=ip_address,
        symbol=symbol,
        position_id=position_id,
        details={
            "profit": profit,
            "close_reason": close_reason
        }
    )

def audit_kill_switch(
    username: str,
    reason: str,
    affected_symbols: List[str],
    ip_address: Optional[str] = None
):
    """Audit kill switch activation"""
    logger = get_audit_logger()
    return logger.log_event(
        event_type=AuditEventType.KILL_SWITCH,
        action="Activate kill switch",
        username=username,
        ip_address=ip_address,
        details={
            "reason": reason,
            "affected_symbols": affected_symbols
        }
    )
