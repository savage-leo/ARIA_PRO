# data_connector.py
# ARIA Institutional Data Connector
# Streams XAUUSD/M1/M5/etc. directly from Dukascopy, zero local storage

import aiohttp
import asyncio
import lzma
import struct
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, AsyncGenerator
import logging
from contextlib import suppress
from backend.core.data_sanity import normalize_timestamps, normalize_volume

logger = logging.getLogger(__name__)

BASE_URL = "https://datafeed.dukascopy.com/datafeed"

# Dukascopy timeframe map (ms per bar)
TF_MAP = {
    "M1": 60_000,
    "M5": 300_000,
    "M15": 900_000,
    "M30": 1_800_000,
    "H1": 3_600_000,
    "H4": 14_400_000,
    "D1": 86_400_000,
}

# Resample strings for pandas
RESAMPLE_MAP = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1H",
    "H4": "4H",
    "D1": "1D",
}


def _price_divisor_for_symbol(symbol: str) -> float:
    """Return appropriate price divisor for a symbol to convert integer ticks to price."""
    s = (symbol or "").upper()
    if s.endswith("JPY"):
        return 1e3
    if s == "XAUUSD":
        return 1e3
    # Default FX pairs
    return 1e5


def _decode_bi5(raw_bytes: bytes) -> np.ndarray:
    """Decode Dukascopy .bi5 compressed tick data into numpy array."""
    if not raw_bytes:
        return np.empty((0, 5))

    try:
        # Dukascopy BI5 = LZMA compressed, big-endian 20-byte chunks
        decompressed = lzma.decompress(raw_bytes)

        # Each tick is 20 bytes: time(4), ask(4), bid(4), ask_vol(4), bid_vol(4)
        # All stored as big-endian integers
        n_ticks = len(decompressed) // 20

        if n_ticks == 0:
            return np.empty((0, 5))

        # Read as big-endian integers
        data = np.frombuffer(decompressed, dtype=">i4").reshape(-1, 5)

        # First column is time in milliseconds (keep as is)
        # Other columns are prices/volumes as integers (will scale later)
        return data

    except lzma.LZMAError:
        # Try alternative: raw uncompressed format
        try:
            data = np.frombuffer(raw_bytes, dtype=">i4").reshape(-1, 5)
            return data
        except:
            pass
    except Exception as e:
        logger.debug(f"BI5 decode error: {e}")

    return np.empty((0, 5))


async def fetch_bars(
    symbol: str = "XAUUSD",
    timeframe: str = "M1",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV bars for given symbol + timeframe, directly from Dukascopy.
    Returns pandas DataFrame (in RAM only).

    Args:
        symbol: Trading symbol (e.g., "XAUUSD", "EURUSD")
        timeframe: Timeframe string (M1, M5, H1, etc.)
        start: Start datetime (UTC)
        end: End datetime (UTC)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if timeframe not in TF_MAP:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}. Use one of {list(TF_MAP.keys())}"
        )

    if end is None:
        end = datetime.utcnow()
    if start is None:
        start = end - timedelta(hours=1)  # default 1h back

    frames = []

    logger.info(f"Fetching {symbol} {timeframe} bars from {start} to {end}")

    async with aiohttp.ClientSession() as session:
        cur = start
        while cur < end:
            # Dukascopy URL format: symbol/year/month-1/day/hour
            url = f"{BASE_URL}/{symbol}/{cur.year}/{cur.month-1:02d}/{cur.day:02d}/{cur.hour:02d}h_ticks.bi5"

            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        raw = await resp.read()
                        ticks = _decode_bi5(raw)
                        if ticks.size > 0:
                            # ticks: [time(ms_from_hour_start), ask, bid, ask_vol, bid_vol]
                            # Convert to absolute epoch ms using the hour base
                            base_dt = datetime(
                                cur.year,
                                cur.month,
                                cur.day,
                                cur.hour,
                                tzinfo=timezone.utc,
                            )
                            base_epoch_ms = int(base_dt.timestamp() * 1000)
                            ts_ms = base_epoch_ms + ticks[:, 0].astype(np.int64)
                            ts = pd.to_datetime(ts_ms, unit="ms", utc=True)

                            div = _price_divisor_for_symbol(symbol)
                            df_hour = pd.DataFrame(
                                {
                                    "timestamp": ts,
                                    "ask": ticks[:, 1] / div,
                                    "bid": ticks[:, 2] / div,
                                    "ask_vol": ticks[:, 3],
                                    "bid_vol": ticks[:, 4],
                                }
                            )
                            frames.append(df_hour)
                            logger.debug(f"Fetched {len(df_hour)} ticks for {cur}")
                    elif resp.status == 404:
                        logger.debug(f"No data available for {cur}")
                    else:
                        logger.warning(f"HTTP {resp.status} for {url}")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching {url}")
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")

            cur += timedelta(hours=1)

    if not frames:
        logger.warning(f"No tick data found for {symbol} between {start} and {end}")
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    # Combine all ticks into a single DataFrame
    df_ticks = pd.concat(frames, ignore_index=True)

    # Use mid price for OHLC
    df_ticks["mid"] = (df_ticks["ask"] + df_ticks["bid"]) / 2
    df_ticks["volume"] = df_ticks["ask_vol"] + df_ticks["bid_vol"]

    # Convert to OHLCV bars
    resample_str = RESAMPLE_MAP.get(timeframe, "1min")
    df_bars = (
        df_ticks.set_index("timestamp")
        .resample(resample_str)
        .agg({"mid": ["first", "max", "min", "last"], "volume": "sum"})
    )
    df_bars.columns = ["open", "high", "low", "close", "volume"]
    df_bars = df_bars.dropna().reset_index()

    # Normalize volumes heuristically if absurdly large
    df_bars, vscale = normalize_volume(df_bars, "volume")
    if vscale != 1.0:
        logger.debug(
            f"Applied volume scale 1/{int(vscale)} to {symbol} {timeframe} bars"
        )

    # Normalize timestamps (defensive)
    df_bars = normalize_timestamps(df_bars, "timestamp")

    logger.info(f"Generated {len(df_bars)} {timeframe} bars for {symbol}")
    return df_bars


async def stream_bars(
    symbol: str = "XAUUSD", timeframe: str = "M1", window_minutes: int = 60
) -> AsyncGenerator[pd.DataFrame, None]:
    """
    Async generator yielding rolling OHLCV DataFrames from Dukascopy.
    Example: iterate forever and train batch-by-batch.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe string
        window_minutes: Rolling window size in minutes

    Yields:
        DataFrame with latest bars
    """
    logger.info(
        f"Starting streaming {symbol} {timeframe} with {window_minutes}min window"
    )

    while True:
        end = datetime.utcnow()
        start = end - timedelta(minutes=window_minutes)

        try:
            df = await fetch_bars(symbol, timeframe, start, end)
            if not df.empty:
                # Memory guard: cap the in-memory bars
                try:
                    max_bars = int(os.getenv("ARIA_MAX_IN_MEMORY_BARS", "50000"))
                except Exception:
                    max_bars = 50000
                if len(df) > max_bars:
                    logger.debug(
                        f"Capping bars from {len(df)} to last {max_bars} for {symbol} {timeframe}"
                    )
                    df = df.tail(max_bars).reset_index(drop=True)
                yield df
            else:
                logger.warning(f"Empty dataframe for {symbol} {timeframe}")
        except Exception as e:
            logger.error(f"Error in stream_bars: {e}")

        # Wait for next bar
        wait_ms = TF_MAP[timeframe]
        await asyncio.sleep(wait_ms / 1000)


async def stream_bars_backpressured(
    symbol: str = "XAUUSD",
    timeframe: str = "M1",
    window_minutes: int = 60,
    queue_maxsize: Optional[int] = None,
    drop_oldest: Optional[bool] = None,
) -> AsyncGenerator[pd.DataFrame, None]:
    """
    Async generator like stream_bars(), but with a bounded internal queue to cap memory.

    - When drop_oldest=True (default), if the queue is full, the oldest batch is dropped to
      make room for the newest. This keeps data fresh and bounds memory.
    - When drop_oldest=False, the producer will backpressure (await) until the consumer catches up.

    Env overrides:
      ARIA_STREAM_QUEUE_MAX (int, default=4)
      ARIA_STREAM_DROP_OLDEST (0/1, default=1)
    """
    if queue_maxsize is None:
        try:
            queue_maxsize = int(os.getenv("ARIA_STREAM_QUEUE_MAX", "4"))
        except Exception:
            queue_maxsize = 4
    if drop_oldest is None:
        try:
            drop_oldest = bool(int(os.getenv("ARIA_STREAM_DROP_OLDEST", "1")))
        except Exception:
            drop_oldest = True

    q: asyncio.Queue[pd.DataFrame] = asyncio.Queue(maxsize=max(1, queue_maxsize))
    stop_event = asyncio.Event()

    async def _producer() -> None:
        try:
            while not stop_event.is_set():
                end = datetime.utcnow()
                start = end - timedelta(minutes=window_minutes)
                try:
                    df = await fetch_bars(symbol, timeframe, start, end)
                    if not df.empty:
                        if drop_oldest:
                            try:
                                q.put_nowait(df)
                            except asyncio.QueueFull:
                                # Drop oldest to keep memory bounded and data fresh
                                with suppress(Exception):
                                    _ = q.get_nowait()
                                    q.task_done()
                                with suppress(Exception):
                                    q.put_nowait(df)
                        else:
                            # Apply backpressure (await) when queue is full
                            await q.put(df)
                    else:
                        logger.debug(
                            f"Backpressured stream: empty dataframe for {symbol} {timeframe}"
                        )
                except Exception as e:
                    logger.error(f"Error in backpressured producer: {e}")

                wait_ms = TF_MAP[timeframe]
                await asyncio.sleep(wait_ms / 1000)
        except asyncio.CancelledError:
            pass

    producer_task = asyncio.create_task(_producer())

    try:
        while True:
            df = await q.get()
            try:
                yield df
            finally:
                q.task_done()
    except asyncio.CancelledError:
        raise
    finally:
        stop_event.set()
        producer_task.cancel()
        with suppress(Exception):
            await producer_task


async def fetch_training_data(
    symbol: str = "XAUUSD", timeframe: str = "M5", days_back: int = 3
) -> pd.DataFrame:
    """
    Convenience function to fetch historical data for training.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe string
        days_back: Number of days of historical data

    Returns:
        DataFrame with historical bars
    """
    end = datetime.utcnow()
    start = end - timedelta(days=days_back)

    logger.info(f"Fetching {days_back} days of {symbol} {timeframe} for training")
    return await fetch_bars(symbol, timeframe, start, end)


# Example usage for ARIA training loop
async def run_training_example():
    """Example of how to use the connector in ARIA training loop."""
    # One-shot historical fetch
    df_historical = await fetch_training_data("XAUUSD", "M5", days_back=7)
    print(f"Historical data shape: {df_historical.shape}")
    print(df_historical.tail())

    # Streaming for online learning
    stream_count = 0
    async for df in stream_bars("XAUUSD", "M5", window_minutes=120):
        print(f"\nStreaming batch {stream_count}: {df.shape}")
        print(df.tail(3))

        # Here you would plug df directly into your LSTM/CNN/PPO
        # The data stays in RAM only - no disk writes

        stream_count += 1
        if stream_count >= 3:  # Stop after 3 iterations for demo
            break


if __name__ == "__main__":
    # Test the connector
    asyncio.run(run_training_example())
