from typing import Tuple
import pandas as pd
from pandas.api.types import (
    is_integer_dtype,
    is_datetime64_any_dtype,
)


def normalize_timestamps(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Ensure timestamp column is tz-aware and reasonable (not 1970).
    Accepts: pandas datetime-like or int epoch (seconds or milliseconds).
    """
    if df is None or df.empty or ts_col not in df.columns:
        return df

    col = df[ts_col]
    if is_integer_dtype(col):
        sample = int(col.iloc[0])
        if sample > 10**12:
            df[ts_col] = pd.to_datetime(col, unit="ms", utc=True)
        elif sample > 10**9:
            df[ts_col] = pd.to_datetime(col, unit="s", utc=True)
        else:
            df[ts_col] = pd.to_datetime(col, unit="s", utc=True)
        # Salvage tiny/invalid epochs
        mask_invalid = df[ts_col].dt.year < 2000
        if mask_invalid.any():
            try:
                salvage = pd.to_datetime((col // 1000), unit="s", utc=True)
                df.loc[mask_invalid, ts_col] = salvage.loc[mask_invalid]
            except Exception:
                pass
    elif is_datetime64_any_dtype(col):
        # If tz-aware, convert to UTC; else localize to UTC
        try:
            tz = getattr(col.dt, "tz", None)
        except Exception:
            tz = None
        if tz is not None:
            try:
                df[ts_col] = col.dt.tz_convert("UTC")
            except Exception:
                df[ts_col] = col
        else:
            try:
                df[ts_col] = col.dt.tz_localize("UTC")
            except TypeError:
                # Already tz-aware or mixed; fallback to parse
                df[ts_col] = pd.to_datetime(col, utc=True, errors="coerce")
    else:
        df[ts_col] = pd.to_datetime(col, utc=True, errors="coerce")

    return df


def normalize_volume(
    df: pd.DataFrame, vol_col: str = "volume"
) -> Tuple[pd.DataFrame, float]:
    """
    If volume units look absurdly large, scale down by 1e3/1e6 heuristically.
    Returns (df, scale_applied)
    """
    if df is None or df.empty or vol_col not in df.columns:
        return df, 1.0

    try:
        max_v = float(pd.to_numeric(df[vol_col], errors="coerce").abs().max())
    except Exception:
        return df, 1.0

    scale = 1.0
    # thresholds
    if max_v > 1e12:
        scale = 1e6
    elif max_v > 1e9:
        scale = 1e3
    elif max_v > 1e6:
        scale = 1.0

    # Apply scale if changed
    if scale != 1.0:
        df[vol_col] = df[vol_col] / scale

    return df, scale
