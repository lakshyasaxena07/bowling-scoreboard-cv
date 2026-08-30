"""Unit tests for automatic ScoreboardDetector."""

import numpy as np
import pytest

from src.scoreboard_detector import ScoreboardDetector, ScoreboardDetection


def test_detector_empty_frame():
    detector = ScoreboardDetector()
    with pytest.raises(ValueError):
        detector.detect(None)

    with pytest.raises(ValueError):
        detector.detect(np.array([]))


def test_detector_synthetic_scoreboard():
    detector = ScoreboardDetector()
    # Create a synthetic 1920x1080 frame with a high-contrast scoreboard table
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Draw grid table
    cv2_rect_x1, cv2_rect_y1 = 50, 40
    cv2_rect_x2, cv2_rect_y2 = 1870, 1030
    frame[cv2_rect_y1:cv2_rect_y2, cv2_rect_x1:cv2_rect_x2] = (30, 30, 40)

    # Draw horizontal and vertical grid lines
    for y in range(cv2_rect_y1, cv2_rect_y2, 100):
        frame[y:y+3, cv2_rect_x1:cv2_rect_x2] = (220, 220, 220)
    for x in range(cv2_rect_x1, cv2_rect_x2, 120):
        frame[cv2_rect_y1:cv2_rect_y2, x:x+3] = (220, 220, 220)

    detection = detector.detect(frame)
    assert detection is not None
    assert detection.width > 1000
    assert detection.height > 600
    assert detection.confidence >= 0.5


def test_extract_roi():
    detector = ScoreboardDetector()
    frame = np.ones((500, 800, 3), dtype=np.uint8) * 128
    detection = ScoreboardDetection(x=50, y=50, width=400, height=300, confidence=0.9)

    roi = detector.extract(frame, detection)
    assert roi.shape[0] == 300
    assert roi.shape[1] == 400