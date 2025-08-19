# backend/core/regime.py
"""
Lightweight regime detection suitable for CPU-only, low-latency use.

Implements:
- EWMA volatility bucket (low/medium/high)
- 3-state regime classifier: TREND, RANGE, BREAKOUT

Inputs: list of bars [{"o","h","l","c","v","ts"}] ascending.
Outputs: (regime: Regime, metrics: RegimeMetrics)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import os
import json
from enum import Enum
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone

import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)


class Regime(Enum):
    TREND = "trend"
    RANGE = "range"
    BREAKOUT = "breakout"


@dataclass
class RegimeMetrics:
    vol_ewma: float
    vol_bucket: str  # "low" | "medium" | "high"
    trend_strength: float  # [-1,1]
    breakout_score: float  # [0, +inf) typical [0,2]


class RegimeDetector:
    @staticmethod
    def _returns_from_bars(bars: List[Dict]) -> np.ndarray:
        closes = np.array([float(b.get("c", 0.0) or 0.0) for b in bars], dtype=float)
        if len(closes) < 2:
            return np.zeros(0, dtype=float)
        rets = np.diff(np.log(np.clip(closes, 1e-9, None)))
        return rets

    @staticmethod
    def _ewma_volatility(returns: np.ndarray, lam: float = 0.94) -> float:
        if returns.size == 0:
            return 0.0
        v = 0.0
        for r in returns[-256:]:  # limit to recent window for speed
            v = lam * v + (1.0 - lam) * (r * r)
        return math.sqrt(max(v, 0.0))

    @staticmethod
    def _trend_strength(closes: np.ndarray, lookback: int = 48) -> float:
        if closes.size < max(10, lookback):
            return 0.0
        x = np.arange(lookback, dtype=float)
        y = closes[-lookback:]
        y = (y - y.mean()) / (y.std() + 1e-9)
        # simple linear regression slope normalized
        slope = np.polyfit(x, y, 1)[0]
        # squash to [-1,1]
        return float(np.tanh(0.75 * slope))

    @staticmethod
    def _atr(bars: List[Dict], period: int = 14) -> float:
        if not bars or len(bars) < period + 1:
            return 0.0
        trs: List[float] = []
        prev_close = float(bars[-period - 1]["c"])
        for b in bars[-period:]:
            h = float(b.get("h", 0.0) or 0.0)
            l = float(b.get("l", 0.0) or 0.0)
            c = float(b.get("c", 0.0) or 0.0)
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
            trs.append(tr)
            prev_close = c
        return float(sum(trs) / float(period)) if trs else 0.0

    @staticmethod
    def _breakout_score(bars: List[Dict], atr: float, lookback: int = 20) -> float:
        if not bars or len(bars) < lookback:
            return 0.0
        window = bars[-lookback:]
        highs = [float(b.get("h", 0.0) or 0.0) for b in window]
        lows = [float(b.get("l", 0.0) or 0.0) for b in window]
        last_close = float(window[-1].get("c", 0.0) or 0.0)
        hh = max(highs)
        ll = min(lows)
        rng = max(hh - ll, 1e-9)
        # breakout measured as distance beyond band normalized by ATR and range
        up = max(0.0, last_close - hh)
        dn = max(0.0, ll - last_close)
        base = up + dn
        if base <= 0.0:
            return 0.0
        # Normalize: how far beyond band vs ATR and range
        score = (base / (atr + 1e-9)) * min(1.5, rng / (atr + 1e-9))
        return float(score)

    @staticmethod
    def detect(bars: List[Dict]) -> Tuple[Regime, RegimeMetrics]:
        if not bars or len(bars) < 10:
            return Regime.RANGE, RegimeMetrics(0.0, "low", 0.0, 0.0)

        closes = np.array([float(b.get("c", 0.0) or 0.0) for b in bars], dtype=float)
        rets = RegimeDetector._returns_from_bars(bars)

        # EWMA vol and bucket. Prefer 180D per-symbol thresholds if available.
        vol_ewma = RegimeDetector._ewma_volatility(rets, lam=0.94)
        symbol = (
            str(bars[-1].get("symbol", "GLOBAL"))
            if isinstance(bars[-1], dict)
            else "GLOBAL"
        )
        thr = _load_vol_thresholds(symbol)
        if thr is not None:
            q33, q67 = thr
            if vol_ewma <= q33:
                vol_bucket = "low"
            elif vol_ewma <= q67:
                vol_bucket = "medium"
            else:
                vol_bucket = "high"
        else:
            # Fallback to recent in-window percentiles
            abs_rets = np.abs(rets[-256:]) if rets.size > 0 else np.array([0.0])
            p33 = float(np.percentile(abs_rets, 33)) if abs_rets.size > 0 else 0.0
            p66 = float(np.percentile(abs_rets, 66)) if abs_rets.size > 0 else 0.0
            if vol_ewma <= p33:
                vol_bucket = "low"
            elif vol_ewma <= p66:
                vol_bucket = "medium"
            else:
                vol_bucket = "high"

        # Trend strength and breakout
        trend_strength = RegimeDetector._trend_strength(
            closes, lookback=min(64, len(closes))
        )
        atr = RegimeDetector._atr(bars, period=14)
        breakout_score = RegimeDetector._breakout_score(bars, atr, lookback=20)

        # Decision rules (naive)
        if vol_bucket == "high" and breakout_score > 0.8:
            naive_regime = Regime.BREAKOUT
        elif abs(trend_strength) > 0.7 and vol_bucket in ("medium", "high"):
            naive_regime = Regime.TREND
        else:
            naive_regime = Regime.RANGE

        metrics = RegimeMetrics(
            vol_ewma=float(vol_ewma),
            vol_bucket=vol_bucket,
            trend_strength=float(trend_strength),
            breakout_score=float(breakout_score),
        )

        # Optional HMM/Viterbi smoothing overlay with persistence bias
        use_hmm = os.environ.get("ARIA_USE_HMM", "0") == "1"
        if use_hmm:
            algo = str(os.environ.get("ARIA_HMM_ALGO", "viterbi")).strip().lower()
            r15 = _m15_return_from_bars(bars)
            if algo == "viterbi":
                regime = _ViterbiRegimeSmoother.instance().smooth(
                    symbol, naive_regime, metrics, r15
                )
                return regime, metrics
            else:
                regime = _HMMRegimeSmoother.instance().smooth(
                    symbol, naive_regime, metrics, r15
                )
                return regime, metrics
        else:
            return naive_regime, metrics


# -------------------- HMM-like Smoother (lightweight) --------------------


class _HMMRegimeSmoother:
    """
    Lightweight 3-state HMM-like smoother.
    - Fixed transition matrix with strong self-persistence.
    - Emission likelihoods derived from RegimeMetrics.
    - Keeps per-symbol last state in-memory.

    This is not a full EM-trained HMM; it's a persistence-aware filter to reduce
    whipsaws while remaining CPU-light.
    """

    _INST: "_HMMRegimeSmoother" | None = None

    @classmethod
    def instance(cls) -> "_HMMRegimeSmoother":
        if cls._INST is None:
            cls._INST = cls()
        return cls._INST

    def __init__(self) -> None:
        # State order: RANGE, TREND, BREAKOUT
        self.states = [Regime.RANGE, Regime.TREND, Regime.BREAKOUT]
        p = float(os.environ.get("ARIA_HMM_PERSIST", "0.92"))
        q = (1.0 - p) / 2.0
        self.A = np.array(
            [
                [p, q, q],  # from RANGE
                [q, p, q],  # from TREND
                [q, q, p],  # from BREAKOUT
            ],
            dtype=float,
        )
        self.last_state: dict[str, Regime] = {}

    def smooth(
        self, symbol: str, naive: Regime, m: RegimeMetrics, r15: Optional[float] = None
    ) -> Regime:
        prev = self.last_state.get(symbol, naive)
        # Emission likelihoods from metrics
        like = self._emission_likelihoods(m)
        i_prev = self.states.index(prev)
        # Compute posterior for current step (proportional only)
        post = self.A[i_prev, :] * like
        idx = int(np.argmax(post))
        st = self.states[idx]
        self.last_state[symbol] = st
        return st

    def _emission_likelihoods(self, m: RegimeMetrics) -> np.ndarray:
        # Map vol bucket to numeric
        vol_map = {"low": 0.0, "medium": 0.5, "high": 1.0}
        v = float(vol_map.get(m.vol_bucket, 0.5))
        t = float(abs(m.trend_strength))
        b = float(max(0.0, min(2.0, m.breakout_score)))
        # Heuristic likelihoods in [0,1]
        # RANGE: low t, low v
        l_range = (1.0 - t) * (1.0 - 0.6 * v)
        # TREND: high t, med/high v
        l_trend = t * (0.4 + 0.6 * v)
        # BREAKOUT: requires high v and high breakout score
        l_break = (0.5 + 0.5 * v) * min(1.0, b / 1.0)
        arr = np.array([l_range, l_trend, l_break], dtype=float)
        s = float(arr.sum())
        if s <= 0:
            return np.array([1.0, 1.0, 1.0], dtype=float) / 3.0
        return arr / s


# -------------------- Full Viterbi Smoother --------------------


class _ViterbiRegimeSmoother:
    """
    Persistence-biased 3-state Viterbi smoother over a short rolling window.

    - States: RANGE, TREND, BREAKOUT
    - Transition matrix A uses persistence p (ARIA_HMM_PERSIST) and symmetric spillover
    - Emission likelihoods derived from `RegimeMetrics` (same heuristics as lightweight smoother)
    - Runs Viterbi over the last W steps of emissions per symbol (ARIA_HMM_WINDOW, default 20)
    - Hysteresis via min dwell (ARIA_HMM_MIN_DWELL) and switch margin (ARIA_HMM_SWITCH_MARGIN)
    """

    _INST: "_ViterbiRegimeSmoother" | None = None

    @classmethod
    def instance(cls) -> "_ViterbiRegimeSmoother":
        if cls._INST is None:
            cls._INST = cls()
        return cls._INST

    def __init__(self) -> None:
        # State order: RANGE, TREND, BREAKOUT
        self.states = [Regime.RANGE, Regime.TREND, Regime.BREAKOUT]
        p = float(os.environ.get("ARIA_HMM_PERSIST", "0.92"))
        q = (1.0 - p) / 2.0
        self.A = np.array(
            [
                [p, q, q],
                [q, p, q],
                [q, q, p],
            ],
            dtype=float,
        )
        # Sticky transitions: A' = (1-eta)A + eta I
        self.sticky_eta = float(os.environ.get("ARIA_HMM_STICKY", "0.05"))
        A_sticky = (1.0 - self.sticky_eta) * self.A + self.sticky_eta * np.eye(
            3, dtype=float
        )
        # Log-space for numerical stability
        self.logA = np.log(A_sticky + 1e-12)
        self.window = int(os.environ.get("ARIA_HMM_WINDOW", "20"))
        self.min_dwell = int(os.environ.get("ARIA_HMM_MIN_DWELL", "3"))
        self.switch_margin = float(os.environ.get("ARIA_HMM_SWITCH_MARGIN", "0.2"))

        # Per-symbol buffers/state
        self._emissions: Dict[str, deque] = {}
        self._last_state: Dict[str, Regime] = {}
        self._dwell_counts: Dict[str, int] = {}
        self._logA_by_symbol: Dict[str, np.ndarray] = {}

    def smooth(
        self, symbol: str, naive: Regime, m: RegimeMetrics, r15: float | None = None
    ) -> Regime:
        # Update emissions buffer (store log-likelihoods). Prefer Gaussian HMM if params available.
        gauss_log_like = self._gaussian_emission_loglike(symbol, r15)
        if gauss_log_like is not None:
            log_like = gauss_log_like
        else:
            like = self._emission_likelihoods(m)
            log_like = np.log(like + 1e-12)
        buf = self._emissions.get(symbol)
        if buf is None:
            buf = deque(maxlen=self.window)
            self._emissions[symbol] = buf
        buf.append(log_like)

        # If not enough history, fall back to lightweight one-step smoothing
        if len(buf) <= 1:
            prev = self._last_state.get(symbol, naive)
            i_prev = self.states.index(prev)
            post = np.exp(self.logA[i_prev, :]) * np.exp(log_like)
            st = self.states[int(np.argmax(post))]
            self._update_state(symbol, st)
            return st

        # Run Viterbi over the buffer
        T = len(buf)
        S = len(self.states)
        delta = np.full((T, S), -np.inf, dtype=float)
        psi = np.zeros((T, S), dtype=int)

        prev_state = self._last_state.get(symbol, naive)
        try:
            i_prev = self.states.index(prev_state)
        except ValueError:
            i_prev = 0

        # Initial distribution: concentrate on previous state
        pi = np.full(S, -np.inf, dtype=float)
        pi[i_prev] = 0.0

        # t=0
        first = buf[0]
        delta[0, :] = pi + first
        psi[0, :].fill(0)

        # t=1..T-1
        for t in range(1, T):
            obs = buf[t]
            for j in range(S):
                logA = self._logA_by_symbol.get(symbol)
                if logA is None:
                    # If symbol-specific HMM A exists, use it with stickiness; else default
                    hp = _load_hmm_params(symbol)
                    if hp and isinstance(hp.get("A"), list):
                        try:
                            A_sym = np.array(hp["A"], dtype=float)
                            A_sym = (
                                1.0 - self.sticky_eta
                            ) * A_sym + self.sticky_eta * np.eye(3, dtype=float)
                            logA = np.log(A_sym + 1e-12)
                        except Exception:
                            logA = self.logA
                    else:
                        logA = self.logA
                    self._logA_by_symbol[symbol] = logA
                vals = delta[t - 1, :] + logA[:, j]
                arg = int(np.argmax(vals))
                delta[t, j] = vals[arg] + obs[j]
                psi[t, j] = arg

        # Backtrace to last state
        last_idx = int(np.argmax(delta[T - 1, :]))
        cand_state = self.states[last_idx]

        # Hysteresis: require advantage and/or dwell before switching
        adv = float(delta[T - 1, last_idx] - delta[T - 1, i_prev])
        if cand_state != prev_state:
            if (
                self._dwell_counts.get(symbol, 1) < self.min_dwell
                or adv < self.switch_margin
            ):
                cand_state = prev_state

        self._update_state(symbol, cand_state)
        return cand_state

    def _update_state(self, symbol: str, new_state: Regime) -> None:
        prev = self._last_state.get(symbol)
        if prev is None or new_state == prev:
            self._dwell_counts[symbol] = self._dwell_counts.get(symbol, 0) + 1
        else:
            # state switch committed
            self._dwell_counts[symbol] = 1
        self._last_state[symbol] = new_state

    def _emission_likelihoods(self, m: RegimeMetrics) -> np.ndarray:
        # Same heuristic mapping as _HMMRegimeSmoother
        vol_map = {"low": 0.0, "medium": 0.5, "high": 1.0}
        v = float(vol_map.get(m.vol_bucket, 0.5))
        t = float(abs(m.trend_strength))
        b = float(max(0.0, min(2.0, m.breakout_score)))
        l_range = (1.0 - t) * (1.0 - 0.6 * v)
        l_trend = t * (0.4 + 0.6 * v)
        l_break = (0.5 + 0.5 * v) * min(1.0, b / 1.0)
        arr = np.array([l_range, l_trend, l_break], dtype=float)
        s = float(arr.sum())
        if s <= 0:
            return np.array([1.0, 1.0, 1.0], dtype=float) / 3.0
        return arr / s

    def _gaussian_emission_loglike(
        self, symbol: str, r15: Optional[float]
    ) -> Optional[np.ndarray]:
        """Return log-likelihoods for states [RANGE, TREND, BREAKOUT] using Gaussian HMM params.
        Falls back to None if params are missing or r15 is None.
        """
        hp = _load_hmm_params(symbol)
        if not hp or r15 is None:
            return None
        try:
            mu = np.asarray(hp.get("mu", []), dtype=float).reshape(-1)
            sg = np.asarray(hp.get("sigma", []), dtype=float).reshape(-1)
            if mu.size != 3 or sg.size != 3:
                return None
            var_floor = float(os.environ.get("ARIA_HMM_VAR_FLOOR", "1e-8"))
            var = np.maximum(sg * sg, var_floor)
            # Map HMM states -> [RANGE, TREND, BREAKOUT] by variance rank: low->TREND, mid->RANGE, high->BREAKOUT
            order = np.argsort(var)
            idx_trend = int(order[0])
            idx_range = int(order[1])
            idx_break = int(order[2])
            # Per-state log N(r|mu,var)
            two_pi = 2.0 * math.pi
            ll = -0.5 * (np.log(two_pi * var) + ((r15 - mu) ** 2) / var)
            # Reorder to [RANGE, TREND, BREAKOUT]
            out = np.array([ll[idx_range], ll[idx_trend], ll[idx_break]], dtype=float)
            return out
        except Exception:
            return None


# -------------------- Helpers: loaders and session --------------------

_VOL_THRESH_CACHE: Dict[str, Tuple[float, float]] = {}
_HMM_PARAM_CACHE: Dict[str, Dict[str, object]] = {}


def _load_vol_thresholds(symbol: str) -> Tuple[float, float] | None:
    """Load per-symbol 180D volatility thresholds q33/q67 if available.
    Expected path: backend/calibration/vol_buckets/{symbol}.json with keys {"q33","q67"}.
    """
    if symbol in _VOL_THRESH_CACHE:
        return _VOL_THRESH_CACHE[symbol]
    base = os.path.join("backend", "calibration", "vol_buckets")
    path = os.path.join(base, f"{symbol}.json")
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                blob = json.load(f)
            q33 = float(blob.get("q33"))
            q67 = float(blob.get("q67"))
            _VOL_THRESH_CACHE[symbol] = (q33, q67)
            return _VOL_THRESH_CACHE[symbol]
    except Exception:
        pass
    return None


def _load_hmm_params(symbol: str) -> Dict[str, object] | None:
    """Load per-symbol HMM params if available.
    Expected path: backend/calibration/regime/{symbol}_hmm.json with keys {"mu","sigma","A"}.
    """
    if symbol in _HMM_PARAM_CACHE:
        return _HMM_PARAM_CACHE[symbol]
    base = os.path.join("backend", "calibration", "regime")
    path = os.path.join(base, f"{symbol}_hmm.json")
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                blob = json.load(f)
            if isinstance(blob, dict):
                _HMM_PARAM_CACHE[symbol] = blob
                return blob
    except Exception:
        pass
    return None


def _session_from_ts(ts: float) -> str:
    """Basic session classifier from UTC hour. Not used in API, kept for future meta wiring."""
    try:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
        h = dt.hour
    except Exception:
        return "unknown"
    # Rough session buckets (UTC): Asia 0-7, London 7-13, NY 13-22, else off
    if 0 <= h < 7:
        return "asia"
    if 7 <= h < 13:
        return "london"
    if 13 <= h < 22:
        return "ny"
    return "off"


def _m15_return_from_bars(bars: List[Dict]) -> Optional[float]:
    """Approximate 15-minute return using last 16 closes.
    Assumes input bars are around 1-minute granularity; falls back if insufficient length.
    """
    try:
        closes = [float(b.get("c", 0.0) or 0.0) for b in bars]
        if len(closes) >= 16 and closes[-1] > 0 and closes[-16] > 0:
            return math.log(closes[-1]) - math.log(closes[-16])
    except Exception:
        return None
    return None


def fit_hmm_on_m15_returns(
    symbol: str, m15_returns: List[float], window_years: int = 5
) -> Optional[Dict[str, object]]:
    """Fit 3-state HMM with diagonal covariance on M15 returns.

    Args:
        symbol: Trading symbol
        m15_returns: List of M15 log returns
        window_years: Lookback window in years (default 5)

    Returns:
        Dict with fitted HMM parameters or None on failure
    """
    try:
        if not m15_returns or len(m15_returns) < 50:  # Minimum data requirement
            logger.warning(f"Insufficient M15 returns for {symbol}: {len(m15_returns)}")
            return None

        # Convert to numpy array
        returns = np.array(m15_returns, dtype=float)

        # Filter out extreme outliers (>5 std devs)
        std = np.std(returns)
        if std > 0:
            median = np.median(returns)
            returns = returns[np.abs(returns - median) <= 5 * std]

        if len(returns) < 50:
            logger.warning(
                f"Insufficient filtered returns for {symbol}: {len(returns)}"
            )
            return None

        # Initialize HMM parameters
        # State mapping: 0=TREND, 1=RANGE, 2=BREAKOUT (by variance)
        mu = np.array([0.0, 0.0, 0.0], dtype=float)  # Mean returns
        sigma = np.array(
            [0.0001, 0.0002, 0.0003], dtype=float
        )  # Volatilities (initial guess)

        # Transition matrix (sticky transitions)
        p = 0.98  # High persistence
        q = (1.0 - p) / 2.0
        A = np.array([[p, q, q], [q, p, q], [q, q, p]], dtype=float)

        # EM algorithm
        max_iter = 50
        tol = 1e-6
        prev_loglik = -np.inf

        # Initialize state probabilities
        T = len(returns)
        gamma = np.full((T, 3), 1.0 / 3.0, dtype=float)  # Posterior state probabilities

        for iteration in range(max_iter):
            # E-step: Compute posterior probabilities
            # This is a simplified version - full implementation would use forward-backward

            # M-step: Update parameters
            # Update means
            for i in range(3):
                if np.sum(gamma[:, i]) > 1e-10:
                    mu[i] = np.sum(gamma[:, i] * returns) / np.sum(gamma[:, i])

            # Update variances (diagonal covariance)
            var_floor = float(os.environ.get("ARIA_HMM_VAR_FLOOR", "1e-8"))
            for i in range(3):
                if np.sum(gamma[:, i]) > 1e-10:
                    diff = returns - mu[i]
                    sigma_sq = np.sum(gamma[:, i] * diff * diff) / np.sum(gamma[:, i])
                    sigma[i] = np.sqrt(max(sigma_sq, var_floor))

            # Update transition matrix
            # Simplified: Use fixed sticky transitions

            # Compute log-likelihood
            loglik = 0.0
            for t in range(T):
                pdf = np.exp(-0.5 * ((returns[t] - mu) / sigma) ** 2) / (
                    sigma * np.sqrt(2 * np.pi)
                )
                likelihood = np.sum(pdf)
                if likelihood > 0:
                    loglik += np.log(likelihood)

            # Check convergence
            if abs(loglik - prev_loglik) < tol:
                logger.info(
                    f"HMM fitting converged for {symbol} after {iteration+1} iterations"
                )
                break

            prev_loglik = loglik

        # Sort states by variance (low->TREND, mid->RANGE, high->BREAKOUT)
        var_order = np.argsort(sigma**2)
        state_mapping = {var_order[0]: 0, var_order[1]: 1, var_order[2]: 2}

        # Reorder parameters to match [RANGE, TREND, BREAKOUT] order
        mu_reordered = np.array(
            [mu[state_mapping[1]], mu[state_mapping[0]], mu[state_mapping[2]]],
            dtype=float,
        )
        sigma_reordered = np.array(
            [sigma[state_mapping[1]], sigma[state_mapping[0]], sigma[state_mapping[2]]],
            dtype=float,
        )

        # Create result dict
        result = {
            "symbol": symbol,
            "mu": mu_reordered.tolist(),
            "sigma": sigma_reordered.tolist(),
            "A": A.tolist(),
            "fitted_at": datetime.now(timezone.utc).isoformat(),
            "n_samples": len(returns),
            "log_likelihood": float(loglik),
        }

        # Save to calibration directory
        calib_dir = os.path.join("backend", "calibration", "regime")
        os.makedirs(calib_dir, exist_ok=True)
        filepath = os.path.join(calib_dir, f"{symbol}_hmm.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Saved HMM parameters for {symbol} to {filepath}")
        return result

    except Exception as e:
        logger.exception(f"Failed to fit HMM for {symbol}: {e}")
        return None
