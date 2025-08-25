from typing import Tuple
import pandas as pd
import logging
from pandas.api.types import (
    is_integer_dtype,
    is_datetime64_any_dtype,
)

logger = logging.getLogger(__name__)


def normalize_timestamps(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Ensure timestamp column is tz-aware and reasonable (not 1970).
    Accepts: pandas datetime-like or int epoch (seconds or milliseconds).
    Returns a copy of the DataFrame to avoid mutations.
    """
    if df is None or df.empty or ts_col not in df.columns:
        return df
    
    # Work on a copy to avoid mutations
    df = df.copy()
    col = df[ts_col]
    
    # Handle all-NaN columns
    if col.isna().all():
        logger.warning(f"Column {ts_col} contains only NaN values")
        return df
    
    if is_integer_dtype(col):
        # Get first non-NaN value safely
        non_nan = col.dropna()
        if non_nan.empty:
            logger.warning(f"Column {ts_col} has no valid integer values")
            return df
        sample = int(non_nan.iloc[0])
        if sample > 10**12:
            df[ts_col] = pd.to_datetime(col, unit="ms", utc=True)
        elif sample > 10**9:
            df[ts_col] = pd.to_datetime(col, unit="s", utc=True)
        else:
            df[ts_col] = pd.to_datetime(col, unit="s", utc=True)
        # Salvage tiny/invalid epochs (validate they're positive)
        mask_invalid = df[ts_col].dt.year < 2000
        if mask_invalid.any():
            try:
                # Only salvage positive values
                positive_mask = col > 0
                if (mask_invalid & positive_mask).any():
                    salvage = pd.to_datetime((col // 1000), unit="s", utc=True)
                    df.loc[mask_invalid & positive_mask, ts_col] = salvage.loc[mask_invalid & positive_mask]
            except Exception as e:
                logger.warning(f"Failed to salvage invalid timestamps: {e}")
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
    Returns (df_copy, scale_applied)
    Validates volumes are positive.
    """
    if df is None or df.empty or vol_col not in df.columns:
        return df, 1.0
    
    # Work on a copy
    df = df.copy()
    
    try:
        numeric_vol = pd.to_numeric(df[vol_col], errors="coerce")
        
        # Check for negative or zero volumes
        if (numeric_vol <= 0).any():
            neg_count = (numeric_vol < 0).sum()
            zero_count = (numeric_vol == 0).sum()
            if neg_count > 0:
                logger.error(f"Found {neg_count} negative volume values - taking absolute value")
                numeric_vol = numeric_vol.abs()
            if zero_count > 0:
                logger.warning(f"Found {zero_count} zero volume values")
        
        max_v = float(numeric_vol.max())
        
        # Sanity check
        if max_v <= 0:
            raise ValueError(f"Maximum volume is {max_v}, all volumes are invalid")
            
    except Exception as e:
        logger.error(f"Volume normalization failed: {e}")
        return df, 1.0

    scale = 1.0
    # thresholds
    if max_v > 1e12:
        scale = 1e6
        logger.info(f"Scaling volume by 1e6 (max: {max_v:.2e})")
    elif max_v > 1e9:
        scale = 1e3
        logger.info(f"Scaling volume by 1e3 (max: {max_v:.2e})")
    elif max_v > 1e6:
        scale = 1.0
    elif max_v < 1:
        logger.warning(f"Unusually small max volume: {max_v}")

    # Apply scale if changed
    if scale != 1.0:
        df[vol_col] = df[vol_col] / scale

    return df, scale
