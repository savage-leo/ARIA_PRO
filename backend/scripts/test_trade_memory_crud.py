"""
Quick test script for TradeMemory CRUD operations.
Usage:
  python backend/scripts/test_trade_memory_crud.py
"""

import sys
import os

# Add the project root to the path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.core.trade_memory import TradeMemory
from datetime import datetime
import os


def main() -> None:
    test_db = os.environ.get("TEST_TRADE_DB", "logs/test_trade_memory.sqlite")
    tm = TradeMemory(db_path=test_db)

    print(f"Using test DB: {test_db}")

    # 1) Insert
    idea = {
        "symbol": "EURUSD",
        "bias": "long",
        "confidence": 0.82,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    meta = {"source": "unit-test", "note": "initial idea"}
    inserted_id = tm.insert_trade_idea(idea, meta)
    print(f"Inserted id: {inserted_id}")

    # 2) List recent
    recent = tm.list_recent(limit=5)
    print("Recent items (up to 5):", recent)

    # 3) Get by id
    if inserted_id is not None:
        item = tm.get_by_id(inserted_id)
        print("Fetched by id:", item)

        # 4) Mark outcome
        ok = tm.mark_outcome(inserted_id, {"pnl": 12.5, "status": "closed"})
        print("Marked outcome ok:", ok)
        print("After outcome:", tm.get_by_id(inserted_id))

        # 5) Delete
        deleted = tm.delete(inserted_id)
        print("Deleted:", deleted)

    # 6) Purge older than N days (no-op for fresh records)
    purged = tm.purge_older_than(3650)
    print("Purged rows older than 10 years:", purged)


if __name__ == "__main__":
    main()
