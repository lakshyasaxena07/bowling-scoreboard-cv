"""Unit tests for Bowling Scoring Rules and Domain Engine."""

import pytest
from src.bowling_engine import BowlingScoreEngine


def test_perfect_game():
    # 12 strikes = 300 points
    frames_rolls = [["X"] for _ in range(9)] + [["X", "X", "X"]]
    scores = BowlingScoreEngine.calculate_cumulative_scores(frames_rolls)
    expected = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
    assert scores == expected


def test_all_spares_game():
    # All 9/ spare with 9 bonus = 190
    frames_rolls = [["9", "/"] for _ in range(9)] + [["9", "/", "9"]]
    scores = BowlingScoreEngine.calculate_cumulative_scores(frames_rolls)
    expected = [19, 38, 57, 76, 95, 114, 133, 152, 171, 190]
    assert scores == expected


def test_open_frames_game():
    # Open frames without strikes or spares
    frames_rolls = [["5", "3"] for _ in range(10)]
    scores = BowlingScoreEngine.calculate_cumulative_scores(frames_rolls)
    expected = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80]
    assert scores == expected


def test_strike_spare_combo():
    frames_rolls = [
        ["X"],          # F1: 10 + 7 + 3 = 20
        ["7", "/"],     # F2: 10 + 9 = 19 -> 39
        ["9", "-"],     # F3: 9 -> 48
    ] + [[] for _ in range(7)]
    scores = BowlingScoreEngine.calculate_cumulative_scores(frames_rolls)
    assert scores[0] == 20
    assert scores[1] == 39
    assert scores[2] == 48


def test_player_validation_consistency():
    frames_rolls = [["X"], ["9", "/"], ["7", "2"]] + [[] for _ in range(7)]
    # Correct cumulative: F1=20, F2=37, F3=46
    disp_correct = [20, 37, 46] + [None] * 7
    player_good = BowlingScoreEngine.validate_and_build_player(0, "JAGDISH", frames_rolls, disp_correct)
    assert player_good.is_consistent is True

    # Inconsistent cumulative (e.g. OCR read 99 for F1)
    disp_bad = [99, 37, 46] + [None] * 7
    player_bad = BowlingScoreEngine.validate_and_build_player(0, "JAGDISH", frames_rolls, disp_bad)
    assert player_bad.is_consistent is False
    assert player_bad.frames[0].is_valid is False
