# -*- coding: utf-8 -*-
"""
Regime Online API: EWMA volatility + 3-state HMM with Viterbi + dwell/hysteresis
CPU-optimized for real-time inference on T470
"""
from __future__ import annotations
import os, json, time, math, pathlib, threading
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.getenv("ARIA_DATA_ROOT", PROJECT_ROOT / "data"))


class EWMAVolatility:
    """Exponentially Weighted Moving Average volatility with configurable half-life"""

    def __init__(self, half_life_bars: float = 96.0):
        self.half_life = half_life_bars
        self.lambda_decay = 0.5 ** (1.0 / half_life_bars)
        self.variance = None
        self.count = 0

    def update(self, return_pct: float) -> float:
        """Update with M15 return, return current volatility (per-period)"""
        r_sq = float(return_pct) ** 2

        if self.variance is None:
            self.variance = r_sq
        else:
            self.variance = (
                self.lambda_decay * self.variance + (1.0 - self.lambda_decay) * r_sq
            )

        self.count += 1
        return math.sqrt(max(1e-8, self.variance))

    def get_volatility(self) -> float:
        """Get current volatility (per-period)"""
        if self.variance is None:
            return 0.01  # default 1% per period
        return math.sqrt(max(1e-8, self.variance))

    def annualized(self, periods_per_year: float = 35040.0) -> float:
        """Convert to annualized volatility (M15 = 35040 periods/year)"""
        return self.get_volatility() * math.sqrt(periods_per_year)


class VolatilityBucket:
    """Rolling percentile-based volatility bucketing"""

    def __init__(self, window_periods: int = 17280):  # ~6 months of M15 bars
        self.window = window_periods
        self.vol_history = deque(maxlen=window_periods)
        self.p33 = 0.01
        self.p67 = 0.02

    def update(self, volatility: float) -> str:
        """Update with new volatility, return bucket (Low/Med/High)"""
        self.vol_history.append(float(volatility))

        if len(self.vol_history) >= 100:  # need enough samples
            sorted_vols = sorted(self.vol_history)
            n = len(sorted_vols)
            self.p33 = sorted_vols[int(n * 0.33)]
            self.p67 = sorted_vols[int(n * 0.67)]

        vol = float(volatility)
        if vol <= self.p33:
            return "Low"
        elif vol <= self.p67:
            return "Med"
        else:
            return "High"

    def get_percentiles(self) -> Tuple[float, float]:
        """Get current P33 and P67 thresholds"""
        return self.p33, self.p67


class HMM3State:
    """3-state HMM for regime detection with Gaussian emissions"""

    def __init__(self):
        # Initial state probabilities
        self.pi = np.array([1 / 3, 1 / 3, 1 / 3])

        # Transition matrix (states: T=0, R=1, B=2)
        self.A = np.array(
            [
                [0.85, 0.10, 0.05],  # T -> T, R, B
                [0.10, 0.85, 0.05],  # R -> T, R, B
                [0.15, 0.15, 0.70],  # B -> T, R, B
            ]
        )

        # Emission parameters: means and covariances for [return, abs_return]
        self.means = np.array(
            [
                [0.0002, 0.008],  # T: small positive drift, medium volatility
                [0.0000, 0.005],  # R: zero drift, low volatility
                [0.0000, 0.015],  # B: zero drift, high volatility
            ]
        )

        self.covs = np.array(
            [
                [[1e-5, 0], [0, 1e-4]],  # T: medium variance
                [[5e-6, 0], [0, 5e-5]],  # R: low variance
                [[2e-5, 0], [0, 4e-4]],  # B: high variance
            ]
        )

        self.state_names = ["T", "R", "B"]

    def _gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
        """Multivariate Gaussian PDF"""
        try:
            diff = x - mean
            inv_cov = np.linalg.inv(cov + 1e-8 * np.eye(len(cov)))
            exponent = -0.5 * np.dot(diff, np.dot(inv_cov, diff))
            normalizer = 1.0 / math.sqrt(
                (2 * math.pi) ** len(x) * np.linalg.det(cov + 1e-8 * np.eye(len(cov)))
            )
            return normalizer * math.exp(exponent)
        except:
            return 1e-10

    def emission_probs(self, observation: List[float]) -> np.ndarray:
        """Compute emission probabilities for observation [return, abs_return]"""
        x = np.array(observation)
        probs = np.zeros(3)

        for state in range(3):
            probs[state] = self._gaussian_pdf(x, self.means[state], self.covs[state])

        # Normalize and add small epsilon
        probs = probs + 1e-10
        return probs / np.sum(probs)

    def viterbi_step(
        self, observation: List[float], prev_probs: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Single Viterbi step, returns (state_probs, most_likely_state)"""
        emission = self.emission_probs(observation)

        # Viterbi forward step
        new_probs = np.zeros(3)
        for j in range(3):
            transition_scores = prev_probs + np.log(self.A[:, j] + 1e-10)
            new_probs[j] = np.max(transition_scores) + math.log(emission[j] + 1e-10)

        # Normalize
        max_prob = np.max(new_probs)
        new_probs = new_probs - max_prob
        new_probs = np.exp(new_probs)
        new_probs = new_probs / np.sum(new_probs)

        most_likely = int(np.argmax(new_probs))
        return new_probs, most_likely


class RegimeDetector:
    """Online regime detection with persistence bias and dwell time"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ewma = EWMAVolatility(half_life_bars=96.0)
        self.vol_bucket = VolatilityBucket()
        self.hmm = HMM3State()

        # Persistence and dwell parameters
        env = os.environ
        self.persistence = float(env.get("ARIA_HMM_PERSIST", 0.6))
        self.min_dwell = int(env.get("ARIA_HMM_MIN_DWELL", 4))
        self.switch_margin = float(env.get("ARIA_HMM_SWITCH_MARGIN", 0.2))

        # State tracking
        self.current_state = 0  # T
        self.state_probs = np.array([1 / 3, 1 / 3, 1 / 3])
        self.dwell_count = 0
        self.returns_buffer = deque(maxlen=100)

        # Thread safety
        self.lock = threading.Lock()

    def update(self, price: float, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """Update with new price, return regime info"""
        with self.lock:
            if len(self.returns_buffer) == 0:
                self.returns_buffer.append(price)
                return self._get_state_info()

            # Calculate M15 return
            prev_price = self.returns_buffer[-1]
            if prev_price <= 0:
                return self._get_state_info()

            ret = math.log(price / prev_price)
            abs_ret = abs(ret)

            self.returns_buffer.append(price)

            # Update volatility
            vol = self.ewma.update(ret)
            vol_bucket = self.vol_bucket.update(vol)

            # HMM inference with persistence
            observation = [ret, abs_ret]
            new_probs, most_likely = self.hmm.viterbi_step(
                observation, np.log(self.state_probs + 1e-10)
            )

            # Apply persistence bias
            if self.dwell_count < self.min_dwell:
                # Force current state during minimum dwell
                most_likely = self.current_state
                self.dwell_count += 1
            else:
                # Allow switch only with sufficient margin
                current_prob = new_probs[self.current_state]
                new_prob = new_probs[most_likely]

                if most_likely != self.current_state:
                    if new_prob > current_prob + self.switch_margin:
                        # Switch allowed
                        self.current_state = most_likely
                        self.dwell_count = 1
                    else:
                        # Stay in current state (persistence)
                        most_likely = self.current_state
                        self.dwell_count += 1
                else:
                    self.dwell_count += 1

            self.state_probs = new_probs

            return {
                "symbol": self.symbol,
                "state": self.hmm.state_names[self.current_state],
                "state_probs": {
                    "T": float(new_probs[0]),
                    "R": float(new_probs[1]),
                    "B": float(new_probs[2]),
                },
                "volatility": float(vol),
                "volatility_annualized": float(self.ewma.annualized()),
                "vol_bucket": vol_bucket,
                "vol_percentiles": self.vol_bucket.get_percentiles(),
                "dwell_count": self.dwell_count,
                "timestamp": timestamp or time.time(),
                "observation": observation,
            }

    def _get_state_info(self) -> Dict[str, Any]:
        """Get current state without update"""
        return {
            "symbol": self.symbol,
            "state": self.hmm.state_names[self.current_state],
            "state_probs": {
                "T": float(self.state_probs[0]),
                "R": float(self.state_probs[1]),
                "B": float(self.state_probs[2]),
            },
            "volatility": float(self.ewma.get_volatility()),
            "volatility_annualized": float(self.ewma.annualized()),
            "vol_bucket": "Med",
            "vol_percentiles": self.vol_bucket.get_percentiles(),
            "dwell_count": self.dwell_count,
            "timestamp": time.time(),
            "observation": [0.0, 0.0],
        }


class RegimeManager:
    """Manager for multiple symbol regime detectors"""

    def __init__(self):
        self.detectors: Dict[str, RegimeDetector] = {}
        self.lock = threading.Lock()

    def get_detector(self, symbol: str) -> RegimeDetector:
        """Get or create detector for symbol"""
        with self.lock:
            if symbol not in self.detectors:
                self.detectors[symbol] = RegimeDetector(symbol)
            return self.detectors[symbol]

    def update_symbol(
        self, symbol: str, price: float, timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """Update regime for symbol"""
        detector = self.get_detector(symbol)
        return detector.update(price, timestamp)

    def get_state(self, symbol: str) -> Dict[str, Any]:
        """Get current regime state for symbol"""
        detector = self.get_detector(symbol)
        return detector._get_state_info()

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get regime states for all symbols"""
        with self.lock:
            return {
                symbol: detector._get_state_info()
                for symbol, detector in self.detectors.items()
            }


# Global instance
regime_manager = RegimeManager()
