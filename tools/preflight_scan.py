# tools/preflight_scan.py
"""
Scans repo for heavy frameworks or banned import patterns.
Exits non-zero if heavy libs are detected.
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
BANNED = ["torch", "tensorflow", "transformers", "diffusers", "accelerate", "bitsandbytes"]

def scan_files():
    hits = []
    for p in BASE.rglob("*.py"):
        try:
            txt = p.read_text()
        except Exception:
            continue
        for b in BANNED:
            if f"import {b}" in txt or f"from {b}" in txt:
                hits.append((str(p), b))
    return hits

if __name__ == "__main__":
    hits = scan_files()
    if hits:
        print("Heavy libs detected in repository:")
        for f,b in hits:
            print(f" - {b} in {f}")
        print("Remove or replace heavy libs before live execution.")
        sys.exit(5)
    print("Preflight OK: no banned heavy libs found.")
