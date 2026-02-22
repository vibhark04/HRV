"""
hrv_features.py - HRV features from rPPG pulse waveform.

Detects peaks with scipy.find_peaks, computes RR intervals,
then Heart Rate (BPM), SDNN, and RMSSD.
Uses only: numpy, scipy. Assumes 30 FPS.
"""

import numpy as np
from scipy import signal as scipy_signal


DEFAULT_FPS = 30.0


class HRVFeatures:
    """
    Computes heart rate and HRV metrics from 1D pulse waveform.
    """

    def __init__(self, fps: float = DEFAULT_FPS, min_peak_distance: float = 0.4):
        """
        Args:
            fps: Sampling rate of the waveform (default 30).
            min_peak_distance: Minimum time (seconds) between peaks to avoid double detection.
        """
        self.fps = float(fps)
        self.min_peak_distance = min_peak_distance

    def get_peaks(self, waveform: np.ndarray) -> np.ndarray:
        """
        Detect systolic peaks in the pulse waveform.

        Args:
            waveform: 1D filtered rPPG signal.

        Returns:
            Indices of detected peaks (frame indices).
        """
        # Minimum number of samples between peaks (~0.4 s at 30 fps = 12 samples)
        min_samples = max(1, int(self.min_peak_distance * self.fps))
        peaks, _ = scipy_signal.find_peaks(
            waveform,
            distance=min_samples,
            prominence=np.std(waveform) * 0.3,
            height=np.median(waveform),
        )
        return peaks

    def compute(self, waveform: np.ndarray) -> dict:
        """
        Get peaks, RR intervals, then BPM, SDNN, RMSSD.

        Args:
            waveform: 1D filtered pulse from rppg_signal.

        Returns:
            Dictionary with keys: 'heart_rate_bpm', 'sdnn_ms', 'rmssd_ms',
            'peaks', 'rr_intervals_ms', 'num_beats'.
            SDNN/RMSSD are NaN if fewer than 2 RR intervals.
        """
        peaks = self.get_peaks(waveform)
        result = {
            "peaks": peaks,
            "rr_intervals_ms": None,
            "heart_rate_bpm": np.nan,
            "sdnn_ms": np.nan,
            "rmssd_ms": np.nan,
            "num_beats": len(peaks),
        }

        if len(peaks) < 2:
            result["rr_intervals_ms"] = np.array([])
            return result

        # RR intervals in seconds (difference of peak indices / fps)
        rr_sec = np.diff(peaks) / self.fps
        rr_ms = rr_sec * 1000.0
        result["rr_intervals_ms"] = rr_ms

        # Mean RR (ms) and heart rate (BPM)
        mean_rr_ms = np.mean(rr_ms)
        result["heart_rate_bpm"] = 60_000.0 / mean_rr_ms

        # SDNN: standard deviation of NN (RR) intervals, in ms
        result["sdnn_ms"] = float(np.std(rr_ms))

        # RMSSD: root mean square of successive differences of RR, in ms
        successive_diff = np.diff(rr_ms)
        result["rmssd_ms"] = float(np.sqrt(np.mean(successive_diff ** 2)))

        return result


# ---------------------------------------------------------------------------
# Example usage (run as script)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Demo: synthetic pulse with known ~72 BPM (1.2 Hz)
    np.random.seed(42)
    fps = DEFAULT_FPS
    duration_sec = 30
    n = int(fps * duration_sec)
    t = np.arange(n) / fps
    # Peaks every ~0.833 s => 72 bpm
    waveform = np.sin(2 * np.pi * 1.2 * t) + 0.2 * np.random.randn(n)
    hrv = HRVFeatures(fps=fps)
    out = hrv.compute(waveform)
    print("HR (BPM):", round(out["heart_rate_bpm"], 2))
    print("SDNN (ms):", round(out["sdnn_ms"], 2))
    print("RMSSD (ms):", round(out["rmssd_ms"], 2))
    print("Num beats:", out["num_beats"])
