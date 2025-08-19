"""
M15 Data Preparation with Feature Engineering and Labeling
CPU-optimized for T470 with deterministic processing
"""

import os
import pathlib
import numpy as np
import pandas as pd
from datetime import timedelta
import argparse
import hashlib
import json

PROJECT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.getenv("ARIA_DATA_ROOT", PROJECT / "data"))


def agg_m1_to_m15(symbol: str, m1_path: pathlib.Path):
    """Aggregate M1 to M15 bars with OHLCV"""
    df = pd.read_parquet(m1_path)
    df = df.sort_values("ts")

    # Ensure mid price exists
    if "mid" not in df.columns:
        df["mid"] = (df["bid"] + df["ask"]) / 2.0

    # Resample to 15-minute bars
    df["ts"] = pd.to_datetime(df["ts"])  # ensure datetime
    resampler = df.set_index("ts").resample("15min")
    agg_map = {"mid": ["first", "max", "min", "last"]}
    has_vol = "vol" in df.columns
    if has_vol:
        agg_map["vol"] = "sum"
    df = resampler.agg(agg_map)
    # Flatten columns
    flat_cols = ["open", "high", "low", "close"] + (["volume"] if has_vol else [])
    df.columns = flat_cols
    df = df.reset_index()
    if not has_vol:
        df["volume"] = 0

    # Remove bars with no data
    df = df.dropna(subset=["open", "close"])

    # Compute log returns
    df["r"] = np.log(df["close"] / df["open"])
    df["r"] = df["r"].replace([np.inf, -np.inf], 0)

    return df


def label_direction(df, horizon_bars=8, ret_thresh=0.0001):
    """Create directional labels based on future returns"""
    df = df.copy()

    # Future return at horizon
    df["future_close"] = df["close"].shift(-horizon_bars)
    df["fret"] = np.log(df["future_close"] / df["close"])
    df["fret"] = df["fret"].replace([np.inf, -np.inf], 0)

    # Binary label: 1 for long, 0 for short
    df["label"] = (df["fret"] > ret_thresh).astype(int)

    # Store actual return for RR estimation
    df["target_return"] = df["fret"]

    return df


def extract_features(df, symbol="EURUSD"):
    """Extract technical features for ML models"""
    df = df.copy()

    # Core return features
    df["r"] = np.log(df["close"] / df["open"])
    df["r"] = df["r"].replace([np.inf, -np.inf], 0).fillna(0)
    df["abs_r"] = df["r"].abs()

    # EWMA volatility (halflife=96 bars = 24 hours)
    lam = 0.5 ** (1.0 / 96.0)
    df["sq"] = df["r"].pow(2)
    sigma2_vals = []
    sigma2 = None
    for v in df["sq"].values:
        if sigma2 is None:
            sigma2 = v if not np.isnan(v) else 0
        else:
            sigma2 = lam * sigma2 + (1 - lam) * v
        sigma2_vals.append(sigma2)
    df["ewma_sig2"] = sigma2_vals
    df["ewma_sig"] = np.sqrt(df["ewma_sig2"])

    # ATR(14) for risk sizing
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1)),
        ),
    )
    df["atr14"] = df["tr"].rolling(14, min_periods=1).mean()

    # RSI(14) momentum
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Bollinger Band position
    sma20 = df["close"].rolling(20, min_periods=1).mean()
    std20 = df["close"].rolling(20, min_periods=1).std()
    df["bb_pos"] = (df["close"] - sma20) / (std20 + 1e-10)

    # Volume features
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20, min_periods=1).mean()
    df["vol_ratio"] = df["vol_ratio"].fillna(1)

    # Price momentum
    df["mom5"] = np.log(df["close"] / df["close"].shift(5))
    df["mom10"] = np.log(df["close"] / df["close"].shift(10))
    df["mom20"] = np.log(df["close"] / df["close"].shift(20))

    # Clean infinities and NaNs
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32]:
            df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)

    return df


def compute_regime_states(df):
    """Simple regime detection based on volatility and trend"""
    df = df.copy()

    # Volatility buckets
    vol_percentiles = df["ewma_sig"].quantile([0.33, 0.67])
    conditions = [
        df["ewma_sig"] <= vol_percentiles.iloc[0],
        df["ewma_sig"] <= vol_percentiles.iloc[1],
        df["ewma_sig"] > vol_percentiles.iloc[1],
    ]
    df["vol_bucket"] = np.select(conditions, ["Low", "Med", "High"], default="Med")

    # Trend state (simplified)
    sma50 = df["close"].rolling(50, min_periods=20).mean()
    sma200 = df["close"].rolling(200, min_periods=50).mean()

    trend_conditions = [
        (df["close"] > sma50) & (sma50 > sma200),  # Trending up
        (df["close"] < sma50) & (sma50 < sma200),  # Trending down
        True,  # Ranging
    ]
    df["state"] = np.select(trend_conditions, ["T", "T", "R"], default="R")

    return df


def add_session_info(df):
    """Add trading session information"""
    df = df.copy()

    # Extract hour from timestamp
    df["ts"] = pd.to_datetime(df["ts"])
    hrs = df["ts"].dt.hour

    # Define sessions (UTC times)
    conditions = [
        (hrs >= 22) | (hrs < 7),  # Asia
        (hrs >= 7) & (hrs < 15),  # Europe
        (hrs >= 15) & (hrs < 22),  # US
    ]
    df["session"] = np.select(conditions, ["ASIA", "EU", "US"], default="EU")

    return df


def export_npz(symbol, df, out_path, metadata=None):
    """Export arrays for training and calibration"""
    arrs = {}

    # Labels and targets
    arrs["label"] = df["label"].fillna(0).astype(np.int8).to_numpy()
    arrs["target_return"] = df["target_return"].fillna(0).astype(np.float32).to_numpy()

    # Regime information
    arrs["state"] = df["state"].fillna("R").to_numpy().astype("U1")
    arrs["vol_bucket"] = df["vol_bucket"].fillna("Med").to_numpy().astype("U4")
    arrs["session"] = df["session"].fillna("EU").to_numpy().astype("U4")

    # Core features for models
    arrs["r"] = df["r"].fillna(0).astype(np.float32).to_numpy()
    arrs["abs_r"] = df["abs_r"].fillna(0).astype(np.float32).to_numpy()
    arrs["ewma_sig"] = df["ewma_sig"].fillna(0).astype(np.float32).to_numpy()
    arrs["atr14"] = df["atr14"].fillna(1e-6).astype(np.float32).to_numpy()
    arrs["rsi"] = df["rsi"].fillna(50).astype(np.float32).to_numpy()
    arrs["bb_pos"] = df["bb_pos"].fillna(0).astype(np.float32).to_numpy()
    arrs["vol_ratio"] = df["vol_ratio"].fillna(1).astype(np.float32).to_numpy()
    arrs["mom5"] = df["mom5"].fillna(0).astype(np.float32).to_numpy()
    arrs["mom10"] = df["mom10"].fillna(0).astype(np.float32).to_numpy()
    arrs["mom20"] = df["mom20"].fillna(0).astype(np.float32).to_numpy()

    # Spread estimation (if bid/ask available)
    if "ask" in df.columns and "bid" in df.columns:
        spread = (df["ask"] - df["bid"]) / df["atr14"]
        arrs["spread_z"] = spread.fillna(0).astype(np.float32).to_numpy()
    else:
        # Estimate based on symbol
        typical_spread = 0.00001 if "JPY" not in symbol else 0.001
        arrs["spread_z"] = (
            (typical_spread / df["atr14"]).fillna(0).astype(np.float32).to_numpy()
        )

    # Timestamps for walk-forward
    arrs["timestamp"] = df["ts"].astype(np.int64).to_numpy() // 10**9  # Unix seconds

    # Placeholders for model scores (filled during training)
    n_samples = len(df)
    arrs["s_lstm"] = np.zeros(n_samples, dtype=np.float32)
    arrs["s_xgb"] = np.zeros(n_samples, dtype=np.float32)
    arrs["s_cnn"] = np.zeros(n_samples, dtype=np.float32)
    arrs["s_ppo"] = np.zeros(n_samples, dtype=np.float32)

    # Save metadata
    if metadata is None:
        metadata = {}
    metadata.update(
        {
            "symbol": symbol,
            "n_samples": n_samples,
            "horizon_bars": 8,
            "features": list(arrs.keys()),
            "data_hash": hashlib.md5(str(arrs).encode()).hexdigest()[:8],
        }
    )

    # Create output directory
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save NPZ
    np.savez_compressed(out_path, **arrs)

    # Save metadata
    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {out_path} ({n_samples} samples)")
    print(f"Metadata: {meta_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Prepare M15 training data")
    parser.add_argument("--symbol", required=True, help="Symbol to process")
    parser.add_argument("--m1", required=True, help="Path to M1 parquet file")
    parser.add_argument("--out", default=None, help="Output NPZ path")
    parser.add_argument(
        "--horizon", type=int, default=8, help="Prediction horizon in bars"
    )
    parser.add_argument(
        "--thresh", type=float, default=0.0001, help="Return threshold for labeling"
    )
    args = parser.parse_args()

    # Set deterministic environment
    os.environ["PYTHONHASHSEED"] = "0"
    np.random.seed(0)

    # Output path
    if args.out:
        out_path = pathlib.Path(args.out)
    else:
        out_path = DATA_ROOT / "features_cache" / args.symbol / "train_m15.npz"

    # Process data
    print(f"Processing {args.symbol} from {args.m1}")
    df = agg_m1_to_m15(args.symbol, pathlib.Path(args.m1))
    df = label_direction(df, horizon_bars=args.horizon, ret_thresh=args.thresh)
    df = extract_features(df, symbol=args.symbol)
    df = compute_regime_states(df)
    df = add_session_info(df)

    # Export
    metadata = {
        "source_file": str(args.m1),
        "horizon_bars": args.horizon,
        "return_threshold": args.thresh,
    }
    export_npz(args.symbol, df, out_path, metadata)


if __name__ == "__main__":
    main()
