"""
Create sample M1 EURUSD data for training pipeline testing
"""

import os
import pathlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

PROJECT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.getenv("ARIA_DATA_ROOT", PROJECT / "data"))


def generate_synthetic_forex_data(
    symbol: str, start_date: str, end_date: str, initial_price: float = 1.1000
):
    """Generate synthetic M1 forex data with realistic patterns"""

    # Create date range
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Generate minute-by-minute timestamps (skip weekends)
    timestamps = []
    current = start

    while current < end:
        # Skip weekends (Saturday=5, Sunday=6)
        if current.weekday() < 5:
            # Add trading hours (00:00 to 23:59 UTC for forex)
            for hour in range(24):
                for minute in range(60):
                    ts = current.replace(
                        hour=hour, minute=minute, second=0, microsecond=0
                    )
                    if ts < end:
                        timestamps.append(ts)
        current += timedelta(days=1)

    n_bars = len(timestamps)
    print(f"Generating {n_bars:,} M1 bars for {symbol}")

    # Generate realistic price movements
    np.random.seed(42)  # Deterministic

    # Base parameters
    volatility = 0.0001  # 1 pip volatility per minute
    drift = 0.0  # No long-term drift

    # Generate returns with realistic patterns
    returns = np.random.normal(drift, volatility, n_bars)

    # Add some trending periods
    trend_periods = np.random.choice(n_bars, size=n_bars // 100, replace=False)
    for period in trend_periods:
        trend_length = np.random.randint(60, 240)  # 1-4 hour trends
        trend_strength = np.random.choice([-1, 1]) * np.random.uniform(0.00005, 0.0001)
        end_period = min(period + trend_length, n_bars)
        returns[period:end_period] += trend_strength

    # Add volatility clustering
    vol_periods = np.random.choice(n_bars, size=n_bars // 200, replace=False)
    for period in vol_periods:
        vol_length = np.random.randint(30, 120)  # 30min-2hour vol spikes
        vol_multiplier = np.random.uniform(1.5, 3.0)  # Reduced multiplier
        end_period = min(period + vol_length, n_bars)
        returns[period:end_period] *= vol_multiplier

    # Generate prices with bounds checking
    log_prices = np.log(initial_price) + np.cumsum(returns)

    # Clamp to reasonable forex range (0.5 to 2.0 for EURUSD)
    log_prices = np.clip(log_prices, np.log(0.5), np.log(2.0))
    mid_prices = np.exp(log_prices)

    # Generate bid/ask spread (0.5-2 pips)
    spreads = np.random.uniform(0.00005, 0.0002, n_bars)
    bid_prices = mid_prices - spreads / 2
    ask_prices = mid_prices + spreads / 2

    # Generate volume (random but realistic)
    volumes = np.random.exponential(100, n_bars).astype(int)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "ts": timestamps,
            "bid": bid_prices,
            "ask": ask_prices,
            "mid": mid_prices,
            "vol": volumes,
        }
    )

    # Add some data quality issues (realistic)
    # Random missing data (0.1%)
    missing_idx = np.random.choice(n_bars, size=int(n_bars * 0.001), replace=False)
    df.loc[missing_idx, ["bid", "ask"]] = np.nan

    # Forward fill missing values
    df[["bid", "ask", "mid"]] = df[["bid", "ask", "mid"]].ffill()

    return df


def main():
    # Create data directory
    symbol = "EURUSD"
    data_dir = DATA_ROOT / "parquet" / symbol
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate 2 years of data for training
    df = generate_synthetic_forex_data(
        symbol=symbol,
        start_date="2022-01-01",
        end_date="2024-01-01",
        initial_price=1.1000,
    )

    # Save as parquet
    output_path = data_dir / f"{symbol}_m1.parquet"
    df.to_parquet(output_path, index=False)

    print(f"Saved {len(df):,} bars to {output_path}")
    print(f"Date range: {df['ts'].min()} to {df['ts'].max()}")
    print(f"Price range: {df['mid'].min():.5f} to {df['mid'].max():.5f}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
