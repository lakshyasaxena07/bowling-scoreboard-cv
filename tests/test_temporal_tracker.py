"""Unit tests for Temporal Tracker and Animation Occlusion Filtering."""

import pytest
from src.temporal_tracker import TemporalTracker
from src.scoreboard_detector import ScoreboardDetection


def test_temporal_animation_filtering():
    tracker = TemporalTracker()

    # Frame 0: normal detection
    det_good = ScoreboardDetection(x=40, y=30, width=1800, height=1000, confidence=0.9)
    p_data = [{
        "player_index": 0,
        "name": "JAGDISH",
        "rolls": [["X"]] + [[] for _ in range(9)],
        "displayed_cumulatives": [20] + [None] * 9,
    }]

    s0 = tracker.update(0, 0.0, det_good, p_data)
    assert s0 is not None
    assert s0.players[0].name == "JAGDISH"

    # Frame 1: occluded by animation (confidence low or is_obscured True)
    det_bad = ScoreboardDetection(x=0, y=0, width=100, height=100, confidence=0.2, is_obscured=True)
    s1 = tracker.update(1, 0.5, det_bad, [])
    assert s1 is None  # Occluded frame successfully rejected!

    # Final state maintains prior valid observations
    final_state = tracker.get_final_state()
    assert final_state is not None
    assert final_state.players[0].name == "JAGDISH"
    assert final_state.players[0].frames[0].rolls == ("X",)
