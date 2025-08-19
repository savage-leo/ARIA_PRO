# backend/core/trade_memory.py
"""
TradeMemory: enriched trade memory CRUD for TMI
Stores ideas and outcomes in sqlite (trade_memory table)
"""

import sqlite3
import os
import json
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aria.core.trade_memory")
DB = os.getenv("TRADE_DB", "logs/trade_memory.sqlite")
_db_dir = os.path.dirname(DB)
if _db_dir:
    os.makedirs(_db_dir, exist_ok=True)


def _init_db():
    # Initialize SQLite database with WAL for better concurrent reads
    with sqlite3.connect(DB) as conn:
        cur = conn.cursor()
        try:
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            # Pragmas are best-effort; continue even if not supported
            logger.debug("SQLite PRAGMA setup skipped or failed", exc_info=True)

        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS trade_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            symbol TEXT,
            payload TEXT,
            metadata TEXT
        )
        """
        )

        # Helpful indexes for common queries
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_trade_memory_symbol ON trade_memory(symbol)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_trade_memory_ts ON trade_memory(ts)"
        )
        conn.commit()


_init_db()


def _init_db_for_path(db_path: str) -> None:
    """Initialize schema for a specific database path (used for custom db_path)."""
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        try:
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            logger.debug(
                "SQLite PRAGMA setup skipped or failed for %s", db_path, exc_info=True
            )
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS trade_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            symbol TEXT,
            payload TEXT,
            metadata TEXT
        )
        """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_trade_memory_symbol ON trade_memory(symbol)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_trade_memory_ts ON trade_memory(ts)"
        )
        conn.commit()


class TradeMemory:
    def __init__(self, db_path: str = DB):
        self.db_path = db_path
        # Ensure schema exists for this specific path as well
        if self.db_path != DB:
            try:
                _init_db_for_path(self.db_path)
            except Exception:
                logger.exception("Failed to initialize DB at %s", self.db_path)

    def insert_trade_idea(
        self, idea: Dict[str, Any], meta: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        """Insert a trade idea and return its row id if successful."""
        try:
            ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            symbol = str(idea.get("symbol", "UNK"))
            payload_json = json.dumps(idea, separators=(",", ":"), ensure_ascii=False)
            meta_json = json.dumps(
                meta or {}, separators=(",", ":"), ensure_ascii=False
            )
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO trade_memory (ts, symbol, payload, metadata) VALUES (?, ?, ?, ?)",
                    (ts, symbol, payload_json, meta_json),
                )
                inserted_id = cur.lastrowid
                conn.commit()
                return int(inserted_id) if inserted_id is not None else None
        except Exception:
            logger.exception("Failed to insert trade idea")
            return None

    def list_recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, ts, symbol, payload, metadata FROM trade_memory ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()

        results: List[Dict[str, Any]] = []
        for r in rows:
            try:
                payload = json.loads(r[3]) if r[3] else {}
            except Exception:
                payload = {}
            try:
                meta = json.loads(r[4]) if r[4] else {}
            except Exception:
                meta = {}
            results.append(
                {
                    "id": r[0],
                    "ts": r[1],
                    "symbol": r[2],
                    "payload": payload,
                    "meta": meta,
                }
            )
        return results

    def mark_outcome(self, id: int, outcome: Dict[str, Any]) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.cursor()
                # append/update outcome inside metadata JSON
                cur.execute("SELECT metadata FROM trade_memory WHERE id=?", (id,))
                r = cur.fetchone()
                if not r:
                    return False
                try:
                    meta = json.loads(r[0]) if r[0] else {}
                except Exception:
                    meta = {}
                meta["outcome"] = outcome
                cur.execute(
                    "UPDATE trade_memory SET metadata=? WHERE id=?",
                    (json.dumps(meta, separators=(",", ":"), ensure_ascii=False), id),
                )
                conn.commit()
                return True
        except Exception:
            logger.exception("Failed to mark outcome")
            return False

    def get_by_id(self, id: int) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, ts, symbol, payload, metadata FROM trade_memory WHERE id=?",
                (id,),
            )
            r = cur.fetchone()
        if not r:
            return None
        try:
            payload = json.loads(r[3]) if r[3] else {}
        except Exception:
            payload = {}
        try:
            meta = json.loads(r[4]) if r[4] else {}
        except Exception:
            meta = {}
        return {
            "id": r[0],
            "ts": r[1],
            "symbol": r[2],
            "payload": payload,
            "meta": meta,
        }

    def list_by_symbol(
        self, symbol: str, limit: int = 100, since: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        symbol = symbol.upper()
        query = "SELECT id, ts, symbol, payload, metadata FROM trade_memory WHERE symbol = ?"
        params: List[Any] = [symbol]
        if since:
            query += " AND ts >= ?"
            params.append(since)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(query, tuple(params))
            rows = cur.fetchall()

        results: List[Dict[str, Any]] = []
        for r in rows:
            try:
                payload = json.loads(r[3]) if r[3] else {}
            except Exception:
                payload = {}
            try:
                meta = json.loads(r[4]) if r[4] else {}
            except Exception:
                meta = {}
            results.append(
                {
                    "id": r[0],
                    "ts": r[1],
                    "symbol": r[2],
                    "payload": payload,
                    "meta": meta,
                }
            )
        return results

    def delete(self, id: int) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM trade_memory WHERE id=?", (id,))
            conn.commit()
            return cur.rowcount > 0

    def purge_older_than(self, days: int) -> int:
        """Delete records older than N days. Returns number of rows deleted."""
        cutoff = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        # SQLite lacks INTERVAL; compare text timestamps if consistent ISO8601
        # Compute cutoff in Python
        try:
            from datetime import timedelta

            cutoff_dt = datetime.utcnow() - timedelta(days=days)
            cutoff = cutoff_dt.isoformat(timespec="seconds") + "Z"
        except Exception:
            pass
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM trade_memory WHERE ts < ?", (cutoff,))
            conn.commit()
            return cur.rowcount
