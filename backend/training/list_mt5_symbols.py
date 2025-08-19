"""
List MetaTrader 5 symbols (filterable) to find broker-specific names like XAUUSD.i / GOLD / XAUUSDm.
"""

from __future__ import annotations
import argparse
import os
import pathlib
from typing import Optional

try:
    import MetaTrader5 as mt5  # type: ignore
except Exception:
    mt5 = None  # type: ignore

PROJECT = pathlib.Path(__file__).resolve().parents[2]


def ensure_mt5_initialized() -> None:
    if mt5 is None:
        raise RuntimeError(
            "MetaTrader5 package is not installed. `pip install MetaTrader5`."
        )
    if not mt5.initialize():
        code, details = mt5.last_error()
        raise RuntimeError(f"Failed to initialize MT5 terminal: {code} {details}")


def main():
    p = argparse.ArgumentParser(description="List MT5 symbols (filtered)")
    p.add_argument(
        "--filter", default="XAU", help="Substring filter (case-insensitive)"
    )
    p.add_argument("--limit", type=int, default=2000, help="Max symbols to inspect")
    args = p.parse_args()

    ensure_mt5_initialized()

    try:
        symbols = mt5.symbols_get() or []
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass

    filt = args.filter.lower()
    rows = []
    for s in symbols[: args.limit]:
        name = s.name
        if filt in name.lower():
            rows.append(
                {
                    "name": s.name,
                    "path": getattr(s, "path", ""),
                    "visible": getattr(s, "visible", False),
                    "trade_mode": getattr(s, "trade_mode", None),
                    "digits": getattr(s, "digits", None),
                    "point": getattr(s, "point", None),
                }
            )

    if not rows:
        print(
            f"No symbols matched filter '{args.filter}'. Try a different filter like GOLD, XAUUSD, .i, .m"
        )
        return

    print(f"Found {len(rows)} symbols matching '{args.filter}':")
    for r in rows:
        print(
            f"  {r['name']:<15} visible={r['visible']:<5} digits={r['digits']} point={r['point']} path={r['path']}"
        )


if __name__ == "__main__":
    main()
