import os
import logging
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("ARIA.AV")

BASE_URL = "https://www.alphavantage.co/query"


def parse_fx_symbol(symbol: str) -> Tuple[str, str]:
    """Parse a Forex symbol like 'EURUSD' or 'EUR/USD' to (from_symbol, to_symbol)."""
    s = (symbol or "").strip().upper().replace("/", "").replace("-", "")
    if len(s) == 6:
        return s[:3], s[3:]
    if "_" in s:
        parts = s.split("_", 1)
        if len(parts) == 2 and len(parts[0]) == 3 and len(parts[1]) == 3:
            return parts[0], parts[1]
    raise ValueError(f"Unrecognized FX symbol format: {symbol}")


class AlphaVantageClient:
    def __init__(self, api_key: Optional[str] = None, timeout_s: float = 10.0):
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Alpha Vantage API key not set. Set ALPHA_VANTAGE_API_KEY env var."
            )
        self.timeout_s = timeout_s

    def _get(self, params: Dict[str, str]) -> Optional[Dict]:
        try:
            r = requests.get(BASE_URL, params=params, timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and ("Note" in data or "Information" in data):
                logger.warning(
                    f"Alpha Vantage notice: {data.get('Note') or data.get('Information')}"
                )
            if isinstance(data, dict) and "Error Message" in data:
                logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return None
            return data
        except Exception as e:
            logger.error(f"Alpha Vantage request failed: {e}")
            return None

    def fetch_fx_daily(
        self, from_symbol: str, to_symbol: str, outputsize: str = "compact"
    ) -> Optional[Dict]:
        params = {
            "function": "FX_DAILY",
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
            "outputsize": outputsize,
            "apikey": self.api_key,
        }
        return self._get(params)

    def fetch_fx_intraday(
        self,
        from_symbol: str,
        to_symbol: str,
        interval: str = "5min",
        outputsize: str = "compact",
    ) -> Optional[Dict]:
        params = {
            "function": "FX_INTRADAY",
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
        }
        return self._get(params)

    def get_recent_closes(
        self, symbol: str, n: int = 50, use_intraday: bool = False
    ) -> List[float]:
        """Return last n close prices as ascending list."""
        try:
            frm, to = parse_fx_symbol(symbol)
        except Exception as e:
            logger.error(e)
            return []

        data = (
            self.fetch_fx_intraday(frm, to, interval="5min", outputsize="compact")
            if use_intraday
            else self.fetch_fx_daily(frm, to, outputsize="compact")
        )
        if not data or not isinstance(data, dict):
            return []

        # Determine the correct time series key
        ts_key = None
        for k in data.keys():
            if "Time Series FX" in k:
                ts_key = k
                break
        if not ts_key:
            return []

        series = data.get(ts_key, {})
        # Sort by time ascending
        try:
            items = sorted(series.items(), key=lambda kv: kv[0])
        except Exception:
            items = list(series.items())
        closes: List[float] = []
        for _, row in items[-n:]:
            try:
                closes.append(float(row.get("4. close")))
            except Exception:
                continue
        return closes

    def get_recent_ohlcv(
        self, symbol: str, n: int = 50, use_intraday: bool = False
    ) -> List[List[float]]:
        """Return last n OHLCV rows as ascending list of [open, high, low, close, volume].
        Volume may be 0 if not provided by Alpha Vantage endpoint.
        """
        try:
            frm, to = parse_fx_symbol(symbol)
        except Exception as e:
            logger.error(e)
            return []

        data = (
            self.fetch_fx_intraday(frm, to, interval="5min", outputsize="compact")
            if use_intraday
            else self.fetch_fx_daily(frm, to, outputsize="compact")
        )
        if not data or not isinstance(data, dict):
            return []

        ts_key = None
        for k in data.keys():
            if "Time Series FX" in k:
                ts_key = k
                break
        if not ts_key:
            return []

        series = data.get(ts_key, {})
        try:
            items = sorted(series.items(), key=lambda kv: kv[0])
        except Exception:
            items = list(series.items())

        out: List[List[float]] = []
        for _, row in items[-n:]:
            try:
                o = float(row.get("1. open"))
                h = float(row.get("2. high"))
                l = float(row.get("3. low"))
                c = float(row.get("4. close"))
                v = float(row.get("5. volume", 0.0)) if isinstance(row, dict) else 0.0
                out.append([o, h, l, c, v])
            except Exception:
                continue
        return out
