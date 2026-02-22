"""
face_roi.py - Face and ROI extraction for rPPG pipeline.

Reads video, detects face with Haar cascade, extracts forehead/cheek ROI,
and returns mean RGB per frame. Suitable for UBFC/PURE datasets.
Uses only: opencv-python, numpy.
"""

import cv2
import numpy as np
from pathlib import Path


class FaceROI:
    """
    Detects face and extracts mean RGB from forehead/cheek ROI per frame.
    """

    def __init__(self, cascade_path: str = None):
        """
        Initialize face detector. Uses OpenCV's bundled Haar cascade if path not given.
        """
        if cascade_path is None:
            # OpenCV data path (haarcascade_frontalface_default.xml)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise FileNotFoundError(f"Haar cascade not found: {cascade_path}")

    def get_roi(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect face in frame and return mean RGB [R, G, B] of forehead/cheek ROI.
        If no face found, returns mean of central crop as fallback.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            Shape (3,) array [mean_R, mean_G, mean_B] in 0-255 range.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            # Fallback: use center 50% of frame
            h, w = frame.shape[:2]
            x1, y1 = int(w * 0.25), int(h * 0.25)
            x2, y2 = int(w * 0.75), int(h * 0.75)
            roi = frame[y1:y2, x1:x2]
        else:
            # Use first (largest) face
            x, y, fw, fh = faces[0]
            # Forehead: upper 1/4 of face; cheek: middle 1/2 horizontally, lower 1/2 vertically
            # Combined ROI: upper-middle part of face (forehead + upper cheek)
            roi_y1 = y + int(0.1 * fh)
            roi_y2 = y + int(0.6 * fh)
            roi_x1 = x + int(0.2 * fw)
            roi_x2 = x + int(0.8 * fw)
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        if roi.size == 0:
            roi = frame

        # Mean RGB (OpenCV is BGR)
        mean_bgr = np.mean(roi, axis=(0, 1))
        # Return as RGB order for downstream rPPG
        mean_rgb = np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=np.float64)
        return mean_rgb


def extract_roi_from_video(video_path: str, face_detector: FaceROI = None) -> np.ndarray:
    """
    Read video, extract mean RGB from face ROI per frame.

    Args:
        video_path: Path to video file (e.g. UBFC or PURE format).
        face_detector: Optional FaceROI instance; created internally if None.

    Returns:
        numpy array of shape (num_frames, 3) with columns [R, G, B].
    """
    video_path = str(Path(video_path).resolve())
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    if face_detector is None:
        face_detector = FaceROI()

    rgb_means = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mean_rgb = face_detector.get_roi(frame)
        rgb_means.append(mean_rgb)

    cap.release()
    return np.array(rgb_means, dtype=np.float64)


# ---------------------------------------------------------------------------
# Example usage (run as script)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python face_roi.py <video_path>")
        sys.exit(1)
    rgb = extract_roi_from_video(sys.argv[1])
    print(f"Frames: {rgb.shape[0]}, shape: (N, 3) RGB means")
    print("First 3 frames RGB:", rgb[:3].round(2))
