"""Scoreboard Layout Engine.

Partitions the detected scoreboard ROI into 4 player rows, 10 frames (including
Frame 10 with 3 sub-boxes), player initial/name regions, and running total (TTL) box.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from src.config import LayoutConfig


@dataclass(frozen=True)
class RegionBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def x1(self) -> int:
        return self.x

    @property
    def y1(self) -> int:
        return self.y

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def extract_from(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            return np.zeros((10, 10, 3), dtype=np.uint8)
        h, w = img.shape[:2]
        x1 = max(0, min(self.x, w - 1))
        y1 = max(0, min(self.y, h - 1))
        x2 = max(x1 + 1, min(self.right, w))
        y2 = max(y1 + 1, min(self.bottom, h))
        return img[y1:y2, x1:x2].copy()


@dataclass(frozen=True)
class FrameCellLayout:
    frame_number: int  # 1 to 10
    full_frame_region: RegionBox
    roll_strip_region: RegionBox
    cumulative_region: RegionBox
    roll_regions: List[RegionBox] = field(default_factory=list)


@dataclass(frozen=True)
class PlayerRowLayout:
    player_index: int  # 0 to 3
    initial_bbox: RegionBox
    initial_region: RegionBox
    name_region: RegionBox
    ttl_region: RegionBox
    row_region: RegionBox
    frames: List[FrameCellLayout] = field(default_factory=list)


@dataclass(frozen=True)
class ScoreboardLayout:
    player_rows: List[PlayerRowLayout]
    width: int = 1920
    height: int = 1080
    active_player_banner: Optional[RegionBox] = None
    active_name_region: RegionBox = field(default_factory=lambda: RegionBox(0, 0, 100, 50))


class ScoreboardLayoutEngine:
    """Computes exact pixel regions for all scoreboard UI elements."""

    def __init__(self, config: Optional[LayoutConfig] = None):
        self.config = config or LayoutConfig()

    def analyze(self, roi_bgr: np.ndarray, player_count: int = 4) -> ScoreboardLayout:
        if roi_bgr is None or roi_bgr.size == 0:
            return self.build_layout(1920, 1080, player_count)
        h, w = roi_bgr.shape[:2]
        return self.build_layout(w, h, player_count)

    def build_layout(self, roi_width: int, roi_height: int, player_count: int = 4) -> ScoreboardLayout:
        row_height = roi_height / float(player_count)
        name_w = int(roi_width * 0.120)
        ttl_w = int(roi_width * 0.120)
        frames_total_w = roi_width - name_w - ttl_w
        frame_col_width = frames_total_w / 10.0

        player_rows = []

        for p_idx in range(player_count):
            ry1 = int(p_idx * row_height)
            ry2 = int((p_idx + 1) * row_height)
            rh = ry2 - ry1

            # Left Name / Initial Region
            name_box = RegionBox(0, ry1, name_w, rh)
            initial_box = name_box

            # Right TTL Box
            ttl_box = RegionBox(name_w + int(frames_total_w), ry1, ttl_w, rh)
            row_box = RegionBox(0, ry1, roi_width, rh)

            frames = []
            for f_idx in range(10):
                fx1 = name_w + int(f_idx * frame_col_width)
                fx2 = name_w + int((f_idx + 1) * frame_col_width)
                fw = fx2 - fx1

                full_frame = RegionBox(fx1, ry1, fw, rh)
                roll_h = int(rh * 0.44)
                cum_h = rh - roll_h

                roll_strip = RegionBox(fx1, ry1, fw, roll_h)
                cum_box = RegionBox(fx1, ry1 + roll_h, fw, cum_h)

                # Sub-regions for rolls
                roll_sub_boxes = []
                num_boxes = 3 if f_idx == 9 else 2
                sub_w = fw / float(num_boxes)
                for b_i in range(num_boxes):
                    bx1 = int(fx1 + b_i * sub_w)
                    bx2 = int(fx1 + (b_i + 1) * sub_w)
                    roll_sub_boxes.append(RegionBox(bx1, ry1, bx2 - bx1, roll_h))

                frames.append(
                    FrameCellLayout(
                        frame_number=f_idx + 1,
                        full_frame_region=full_frame,
                        roll_strip_region=roll_strip,
                        cumulative_region=cum_box,
                        roll_regions=roll_sub_boxes,
                    )
                )

            player_rows.append(
                PlayerRowLayout(
                    player_index=p_idx,
                    initial_bbox=initial_box,
                    initial_region=initial_box,
                    name_region=name_box,
                    ttl_region=ttl_box,
                    row_region=row_box,
                    frames=frames,
                )
            )

        banner = RegionBox(0, 0, roi_width, int(roi_height * 0.12))

        return ScoreboardLayout(
            player_rows=player_rows,
            width=roi_width,
            height=roi_height,
            active_player_banner=banner,
            active_name_region=banner,
        )


# Backward compatibility aliases
LayoutDetector = ScoreboardLayoutEngine


def build_scoreboard_layout(width: int, height: int, player_count: int = 4) -> ScoreboardLayout:
    engine = ScoreboardLayoutEngine()
    return engine.build_layout(width, height, player_count)