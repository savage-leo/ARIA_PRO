#!/usr/bin/env python3
"""
Test script for SMC Edge Core
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import time
from backend.smc.trap_detector import detect_trap
from backend.smc.smc_edge_core import get_edge


def make_test_bar(price=1.1000, vol=100):
    """Create a test bar"""
    t = time.time()
    o = price - 0.0001
    c = price + 0.0001
    h = price + 0.0002
    l = price - 0.0002
    return {"ts": t, "o": o, "h": h, "l": l, "c": c, "v": vol, "symbol": "EURUSD"}


def test_trap_detector():
    """Test trap detector"""
    print("Testing trap detector...")
    bars = [make_test_bar(price=1.1 + (i * 0.0001)) for i in range(10)]
    trap = detect_trap(bars)
    print(f"Trap result: {trap}")
    assert "trap_score" in trap
    print("✓ Trap detector test passed")


def test_edge_engine():
    """Test edge engine"""
    print("Testing edge engine...")
    eng = get_edge("EURUSD")
    for i in range(20):
        b = make_test_bar(price=1.1000 + (i * 0.00005), vol=100 + i * 5)
        idea = eng.ingest_bar(b, recent_ticks=None)
    print(f"Engine history length: {len(eng.history)}")
    print(f"Memory entries: {len(eng.memory.list_recent())}")
    print("✓ Edge engine test passed")


if __name__ == "__main__":
    print("Running SMC Edge Core tests...")
    test_trap_detector()
    test_edge_engine()
    print("All tests passed!")
