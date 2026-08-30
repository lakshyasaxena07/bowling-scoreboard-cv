"""Unit tests for automatic Scoreboard Layout and Bowling Grid Geometry."""

import numpy as np
import pytest

from src.scoreboard_layout import LayoutDetector, build_scoreboard_layout


def test_layout_dimensions():
    layout = build_scoreboard_layout(width=1800, height=1000, player_count=4)
    assert layout.width == 1800
    assert layout.height == 1000
    assert len(layout.player_rows) == 4


def test_frame_boxes_and_10th_frame():
    layout = build_scoreboard_layout(width=1800, height=1000, player_count=4)
    for p_row in layout.player_rows:
        # Check 10 frames exist
        assert len(p_row.frames) == 10

        # Check frames 1-9 have 2 rolls
        for f_idx in range(9):
            f_cell = p_row.frames[f_idx]
            assert len(f_cell.roll_regions) == 2
            assert f_cell.cumulative_region.height > 0

        # Check frame 10 has 3 rolls
        f10_cell = p_row.frames[9]
        assert len(f10_cell.roll_regions) == 3
        assert f10_cell.cumulative_region.height > 0


def test_regions_bounded():
    layout = build_scoreboard_layout(width=1920, height=1080, player_count=4)
    for p_row in layout.player_rows:
        assert p_row.name_region.right <= p_row.frames[0].full_frame_region.x + 5
        assert p_row.ttl_region.x >= p_row.frames[-1].full_frame_region.right - 5
