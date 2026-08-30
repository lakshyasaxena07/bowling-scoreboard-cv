"""Scoreboard Detection Module.

Automatically detects the scoreboard boundary in video frames using classical
computer vision (Canny edge detection and morphological projection profiles).
100% dynamic, self-adjusting with zero hardcoded frame limits.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np

from src.config import DetectionConfig


@dataclass(frozen=True)
class ScoreboardDetection:
    x: int
    y: int
    width: int
    height: int
    confidence: float
    is_obscured: bool = False

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


class ScoreboardDetector:
    """Dynamic Scoreboard Detector using Morphological Line Projections."""

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        self._cached_bbox: Optional[Tuple[int, int, int, int]] = None

    def detect(self, frame: np.ndarray) -> Optional[ScoreboardDetection]:
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None")

        frame_height, frame_width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.config.canny_low, self.config.canny_high)

        h_len = max(15, int(frame_width * 0.04))
        v_len = max(15, int(frame_height * 0.04))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

        h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)

        h_proj = np.sum(h_lines > 0, axis=1)
        v_proj = np.sum(v_lines > 0, axis=0)

        h_peaks = np.count_nonzero(h_proj > (frame_width * 0.15))
        v_peaks = np.count_nonzero(v_proj > (frame_height * 0.15))

        # Rejection of occluded frames / pin animations
        if h_peaks < 3 or v_peaks < 4:
            return None

        # Dynamically locate table boundaries from outer projection bounds
        y_indices = np.where(h_proj > (frame_width * 0.10))[0]
        x_indices = np.where(v_proj > (frame_height * 0.10))[0]

        if len(y_indices) > 0 and len(x_indices) > 0:
            min_y, max_y = int(y_indices[0]), int(y_indices[-1])
            min_x, max_x = int(x_indices[0]), int(x_indices[-1])
            
            # Smooth bounding box
            bw = max_x - min_x
            bh = max_y - min_y

            if bw > frame_width * 0.50 and bh > frame_height * 0.50:
                self._cached_bbox = (min_x, min_y, bw, bh)

        if self._cached_bbox is not None:
            bx, by, bw, bh = self._cached_bbox
            confidence = min(0.98, 0.60 + (h_peaks + v_peaks) * 0.02)
            return ScoreboardDetection(
                x=bx,
                y=by,
                width=bw,
                height=bh,
                confidence=float(round(confidence, 2)),
                is_obscured=False,
            )

        return None

    def extract(self, frame: np.ndarray, detection: ScoreboardDetection) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("Frame is empty")

        h, w = frame.shape[:2]
        x1 = max(0, min(detection.x, w - 1))
        y1 = max(0, min(detection.y, h - 1))
        x2 = max(x1 + 1, min(detection.right, w))
        y2 = max(y1 + 1, min(detection.bottom, h))

        return frame[y1:y2, x1:x2].copy()