"""
Trade Memory API: CRUD endpoints for trade ideas and outcomes.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from backend.core.trade_memory import TradeMemory

router = APIRouter(prefix="/trade-memory", tags=["Trade Memory"])
_tm = TradeMemory()


class TradeIdeaIn(BaseModel):
    idea: Dict[str, Any]
    meta: Optional[Dict[str, Any]] = None


class OutcomeIn(BaseModel):
    outcome: Dict[str, Any]


@router.post("/ideas")
def create_trade_idea(body: TradeIdeaIn):
    inserted_id = _tm.insert_trade_idea(body.idea, body.meta)
    if inserted_id is None:
        raise HTTPException(status_code=500, detail="Failed to insert trade idea")
    return {"ok": True, "id": inserted_id}


@router.get("/recent")
def list_recent(limit: int = Query(100, ge=1, le=1000)) -> Dict[str, Any]:
    items = _tm.list_recent(limit=limit)
    return {"ok": True, "items": items, "count": len(items)}


@router.get("/{id}")
def get_by_id(id: int) -> Dict[str, Any]:
    item = _tm.get_by_id(id)
    if not item:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True, "item": item}


@router.get("/symbol/{symbol}")
def list_by_symbol(
    symbol: str, limit: int = Query(100, ge=1, le=1000), since: Optional[str] = None
) -> Dict[str, Any]:
    items = _tm.list_by_symbol(symbol, limit=limit, since=since)
    return {"ok": True, "items": items, "count": len(items)}


@router.patch("/{id}/outcome")
def mark_outcome(id: int, body: OutcomeIn) -> Dict[str, Any]:
    ok = _tm.mark_outcome(id, body.outcome)
    if not ok:
        raise HTTPException(status_code=404, detail="Not found or failed to update")
    return {"ok": True}


@router.delete("/{id}")
def delete_item(id: int) -> Dict[str, Any]:
    deleted = _tm.delete(id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}


@router.delete("/purge")
def purge(days: int = Query(30, ge=1, le=3650)) -> Dict[str, Any]:
    deleted = _tm.purge_older_than(days)
    return {"ok": True, "deleted": int(deleted)}
