"""
Concept drift detectors: ADWIN (Adaptive Windowing) and DDM (Drift Detection
Method), applied to a model's rolling prediction-error stream to detect when
a player's underlying strategy has changed mid-match.
"""

import math

import numpy as np


class ADWIN:
    """
    Simplified ADWIN: maintains a window of recent error values; splits the
    window and compares sub-window means. If they differ beyond a Hoeffding-
    bound threshold, flags drift and shrinks the window to the most recent data.
    """

    def __init__(self, delta: float = 0.05, min_window: int = 10):
        self.delta = delta
        self.min_window = min_window
        self.window: list[int] = []

    def update(self, error: int) -> bool:
        """Adds a new error observation (0 or 1). Returns True if drift detected."""
        self.window.append(error)
        if len(self.window) < self.min_window * 2:
            return False

        drift_detected = self._check_drift()
        return drift_detected

    def _check_drift(self) -> bool:
        n = len(self.window)
        for split in range(self.min_window, n - self.min_window):
            w0 = self.window[:split]
            w1 = self.window[split:]
            n0, n1 = len(w0), len(w1)
            mean0, mean1 = np.mean(w0), np.mean(w1)

            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            epsilon = math.sqrt((1.0 / (2 * m)) * math.log(4 / self.delta))

            if abs(mean0 - mean1) > epsilon:
                # Drift detected: shrink window to the more recent sub-window
                self.window = w1
                return True
        return False

    def reset(self) -> None:
        self.window = []


class DDM:
    """
    Drift Detection Method: tracks the running error rate and its standard
    deviation. Flags a WARNING when error rate exceeds min + 2*std, and DRIFT
    when it exceeds min + 3*std. Requires a minimum sample size before
    evaluating thresholds, to avoid false triggers from early-stream noise
    (a well-documented DDM edge case).
    """

    def __init__(self, min_samples: int = 30):
        self.min_samples = min_samples
        self.n = 0
        self.error_sum = 0
        self.p_min = float("inf")
        self.s_min = float("inf")

    def update(self, error: int) -> str:
        """Returns 'drift', 'warning', or 'stable' based on the new observation."""
        self.n += 1
        self.error_sum += error

        p = self.error_sum / self.n
        s = math.sqrt(p * (1 - p) / self.n) if self.n > 0 else 0

        # Don't evaluate or update min-tracking until we have enough samples
        # to avoid early-stream noise collapsing p_min/s_min toward zero.
        if self.n < self.min_samples:
            return "stable"

        if p + s < self.p_min + self.s_min:
            self.p_min = p
            self.s_min = s

        if p + s > self.p_min + 3 * self.s_min:
            self.reset()
            return "drift"
        elif p + s > self.p_min + 2 * self.s_min:
            return "warning"
        return "stable"

    def reset(self) -> None:
        self.n = 0
        self.error_sum = 0
        self.p_min = float("inf")
        self.s_min = float("inf")
