"""
Download real M1 OHLC data via MetaTrader5 and save as parquet for the training pipeline.
Output schema: ts, bid, ask, mid, vol

Requirements:
- MetaTrader 5 terminal installed and logged-in (or provide login/server via args)
- Python package: MetaTrader5, pandas, pyarrow
"""

from __future__ import annotations
import argparse
import os
import pathlib
from datetime import datetime
from typing import Optional

import pandas as pd

try:
    import MetaTrader5 as mt5  # type: ignore
except Exception as e:  # pragma: no cover
    mt5 = None  # type: ignore

PROJECT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.getenv("ARIA_DATA_ROOT", PROJECT / "data"))


def ensure_mt5_initialized(
    login: Optional[int], password: Optional[str], server: Optional[str]
) -> None:
    if mt5 is None:
        raise RuntimeError(
            "MetaTrader5 package is not installed. Please `pip install MetaTrader5`."
        )

    if not mt5.initialize():
        code, details = mt5.last_error()
        raise RuntimeError(
            f"Failed to initialize MT5 terminal: {code} {details}. Is the terminal installed and logged-in?"
        )

    # Optionally switch/login to specific account if provided
    if login and password and server:
        if not mt5.login(login, password=password, server=server):
            code, details = mt5.last_error()
            raise RuntimeError(
                f"MT5 login failed for {login}@{server}: {code} {details}"
            )


def download_m1(symbol: str, start: str, end: str) -> pd.DataFrame:
    # Normalize dates as naive UTC datetimes (MT5 expects UTC, naive is safest)
    start_dt = pd.Timestamp(start, tz="UTC").tz_convert(None).to_pydatetime()
    end_dt = pd.Timestamp(end, tz="UTC").tz_convert(None).to_pydatetime()

    # Ensure symbol visible
    si = mt5.symbol_info(symbol)
    if si is None:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(
                f"Symbol {symbol} not found or cannot be selected in MT5 terminal."
            )
        si = mt5.symbol_info(symbol)
    elif not si.visible:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(
                f"Symbol {symbol} cannot be made visible in Market Watch."
            )
        si = mt5.symbol_info(symbol)

    # Fetch bars
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start_dt, end_dt)
    if rates is None or len(rates) == 0:
        # Retry with end boundary minus 1 minute to avoid exclusive-boundary edge cases
        from datetime import timedelta

        retry_end = end_dt - timedelta(minutes=1)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start_dt, retry_end)
    if rates is None or len(rates) == 0:
        code, details = mt5.last_error()
        si2 = mt5.symbol_info(symbol)
        vis = si2.visible if si2 else None
        raise RuntimeError(
            f"No M1 data returned for {symbol} in range {start} .. {end}. "
            f"MT5 last_error=({code}, {details}). Symbol visible={vis}. "
            f"Tip: verify the exact broker symbol name (e.g., XAUUSD.i / GOLD / XAUUSDm), and try a shorter recent range."
        )

    df = pd.DataFrame(rates)
    # Convert timestamps to naive UTC (use Series.dt methods)
    ts = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)

    # Derive bid/ask from close and spread, if possible
    point = si.point if si else 0.0001
    if "spread" in df.columns:
        spr = df["spread"].astype(float) * float(point)
    else:
        # Fallback fixed minimal spread in price units
        spr = pd.Series([1.5 * point] * len(df))

    close = df["close"].astype(float)
    bid = close - spr / 2.0
    ask = close + spr / 2.0
    mid = close

    # Volume: prefer real_volume if present otherwise tick_volume
    if "real_volume" in df.columns and df["real_volume"].sum() > 0:
        vol = df["real_volume"].astype(int)
    else:
        vol = df.get("tick_volume", pd.Series([0] * len(df))).astype(int)

    out = pd.DataFrame(
        {
            "ts": ts,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "vol": vol,
        }
    )

    # Basic sanity: drop duplicates and sort
    out = out.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return out


def download_last_n(symbol: str, last: int) -> pd.DataFrame:
    # Ensure symbol visible
    si = mt5.symbol_info(symbol)
    if si is None or not si.visible:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(
                f"Symbol {symbol} not found or cannot be selected in MT5 terminal."
            )
        si = mt5.symbol_info(symbol)

    # First attempt: copy_rates_from (worked in diagnostics)
    safe_cap = min(int(last), 100000)
    from datetime import datetime, timedelta

    now_dt = datetime.utcnow()
    rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, now_dt, safe_cap)
    if rates is None or len(rates) == 0:
        # Second attempt: copy_rates_from_pos with a safe cap
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, safe_cap)
    if rates is None or len(rates) == 0:
        # Fallback: use a recent time window via copy_rates_range
        end_dt = now_dt
        start_dt = end_dt - timedelta(minutes=last + 5)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start_dt, end_dt)
        if rates is None or len(rates) == 0:
            code, details = mt5.last_error()
            raise RuntimeError(
                f"No M1 data returned for {symbol} using fallback time window last={last} minutes. "
                f"MT5 last_error=({code}, {details})."
            )

    df = pd.DataFrame(rates)
    ts = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
    point = si.point if si else 0.0001
    spr = (
        (df.get("spread", 0).astype(float) * float(point))
        if "spread" in df.columns
        else pd.Series([1.5 * point] * len(df))
    )
    close = df["close"].astype(float)
    bid = close - spr / 2.0
    ask = close + spr / 2.0
    mid = close
    if "real_volume" in df.columns and df["real_volume"].sum() > 0:
        vol = df["real_volume"].astype(int)
    else:
        vol = df.get("tick_volume", pd.Series([0] * len(df))).astype(int)

    out = (
        pd.DataFrame(
            {
                "ts": ts,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "vol": vol,
            }
        )
        .drop_duplicates(subset=["ts"])
        .sort_values("ts")
        .reset_index(drop=True)
    )
    return out


def main():
    p = argparse.ArgumentParser(
        description="Download M1 OHLC from MT5 to parquet (ts,bid,ask,mid,vol)"
    )
    p.add_argument("--symbol", required=True)
    p.add_argument("--start", help="YYYY-MM-DD (UTC)")
    p.add_argument("--end", help="YYYY-MM-DD (UTC, exclusive)")
    p.add_argument(
        "--last",
        type=int,
        default=None,
        help="If set, download the latest N M1 bars instead of a date range",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output parquet path. Default: ARIA_PRO/data/parquet/<symbol>/<symbol>_m1.parquet",
    )
    p.add_argument("--login", type=int, default=None)
    p.add_argument("--password", type=str, default=None)
    p.add_argument("--server", type=str, default=None)
    args = p.parse_args()

    ensure_mt5_initialized(args.login, args.password, args.server)

    try:
        if args.last is not None:
            df = download_last_n(args.symbol, args.last)
        else:
            if not args.start or not args.end:
                raise SystemExit(
                    "Either provide --last N or both --start and --end dates."
                )
            df = download_m1(args.symbol, args.start, args.end)
    finally:
        # Cleanly shutdown MT5
        try:
            mt5.shutdown()
        except Exception:
            pass

    if args.out:
        out_path = pathlib.Path(args.out)
    else:
        out_path = DATA_ROOT / "parquet" / args.symbol / f"{args.symbol}_m1.parquet"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"Saved {len(df):,} M1 bars to {out_path}")
    print(f"Date range: {df['ts'].min()} .. {df['ts'].max()}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
