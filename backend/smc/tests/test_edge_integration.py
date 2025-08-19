# backend/smc/tests/test_edge_integration.py
import time
from backend.smc.smc_edge_core import get_edge
from backend.smc.trap_detector import detect_trap


def make_bar(price=1.1000, vol=100):
    t = time.time()
    o = price - 0.0001
    c = price + 0.0001
    h = price + 0.0002
    l = price - 0.0002
    return {"ts": t, "o": o, "h": h, "l": l, "c": c, "v": vol, "symbol": "EURUSD"}


def test_trap_detect_small_history():
    bars = [make_bar(price=1.1 + (i * 0.0001)) for i in range(10)]
    trap = detect_trap(bars)
    assert "trap_score" in trap


def test_edge_ingest_creates_idea_or_none():
    eng = get_edge("EURUSD")
    for i in range(20):
        b = make_bar(price=1.1000 + (i * 0.00005), vol=100 + i * 5)
        idea = eng.ingest_bar(b, recent_ticks=None)
    # Should not raise. idea may be None if not confirmed
    assert hasattr(eng, "memory")
