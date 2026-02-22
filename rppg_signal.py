"""
rppg_signal.py - rPPG signal extraction from RGB means.

Takes mean RGB per frame, uses green channel, detrends, and applies
Butterworth bandpass (0.7–3 Hz) to obtain pulse waveform.
Uses only: numpy, scipy. Assumes 30 FPS.
"""

import numpy as np
from scipy import signal as scipy_signal


# Default FPS for UBFC/PURE-style videos
DEFAULT_FPS = 30.0


class RPPGSignal:
    """
    Extracts pulse waveform from RGB temporal signals via green channel
    and bandpass filtering.
    """

    def __init__(self, fps: float = DEFAULT_FPS, low_hz: float = 0.7, high_hz: float = 3.0, order: int = 4):
        """
        Args:
            fps: Video frame rate (default 30).
            low_hz: Bandpass low cutoff in Hz (default 0.7 ~ 42 bpm).
            high_hz: Bandpass high cutoff in Hz (default 3.0 ~ 180 bpm).
            order: Butterworth filter order (default 4).
        """
        self.fps = float(fps)
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.order = order

    def extract_green(self, rgb_means: np.ndarray) -> np.ndarray:
        """
        Extract green channel from (N, 3) RGB array.
        Green typically has strongest PPG component in ambient light.

        Args:
            rgb_means: Shape (num_frames, 3), columns [R, G, B].

        Returns:
            1D array of length num_frames (green channel).
        """
        return np.asarray(rgb_means[:, 1], dtype=np.float64)

    def bandpass(self, x: np.ndarray) -> np.ndarray:
        """
        Apply Butterworth bandpass filter (0.7–3 Hz by default).
        Uses forward-backward filtfilt to avoid phase shift.

        Args:
            x: 1D signal (e.g. green channel).

        Returns:
            Filtered 1D signal, same length as x.
        """
        nyq = 0.5 * self.fps
        low = max(0.01, self.low_hz / nyq)
        high = min(0.99, self.high_hz / nyq)
        if low >= high:
            return x
        b, a = scipy_signal.butter(self.order, [low, high], btype="band")
        return scipy_signal.filtfilt(b, a, x)

    def process(self, rgb_means: np.ndarray) -> np.ndarray:
        """
        Full pipeline: green channel -> detrend -> bandpass.

        Args:
            rgb_means: Shape (num_frames, 3) from face_roi.

        Returns:
            Filtered pulse waveform, 1D, same length as num_frames.
        """
        green = self.extract_green(rgb_means)
        # Detrend: remove linear trend to avoid low-freq drift
        detrended = scipy_signal.detrend(green, type="linear")
        filtered = self.bandpass(detrended)
        return filtered


# ---------------------------------------------------------------------------
# Example usage (run as script)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Demo: synthetic RGB means
    np.random.seed(42)
    n_frames = 300
    t = np.arange(n_frames) / DEFAULT_FPS
    # Fake pulse ~1.2 Hz
    pulse = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(n_frames)
    rgb_demo = np.column_stack([
        np.ones(n_frames) * 100 + 2 * pulse + np.random.randn(n_frames),
        np.ones(n_frames) * 120 + 5 * pulse + np.random.randn(n_frames),
        np.ones(n_frames) * 90 + 1.5 * pulse + np.random.randn(n_frames),
    ])
    rppg = RPPGSignal(fps=DEFAULT_FPS)
    waveform = rppg.process(rgb_demo)
    print("Processed waveform length:", len(waveform))
    print("First 5 values:", waveform[:5].round(4))
