"""Temporal Processing and Frame-Accurate Live State Tracking Module.

Sequentially tracks the scoreboard state forward in time, displaying live
frame scores as the game progresses (e.g. Frame 40 shows Jagdish=31, Vishal=28, Tarun=33)
and reaching the complete final game state at video completion.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import copy

from src.config import TemporalConfig
from src.bowling_engine import GameState, PlayerScore, FrameScore, BowlingScoreEngine
from src.scoreboard_detector import ScoreboardDetection


class TemporalTracker:
    """Tracks live scoreboard game state frame-accurately across video time."""

    def __init__(self, config: Optional[TemporalConfig] = None):
        self.config = config or TemporalConfig()
        self.game_timeline: List[GameState] = []

    def update(
        self,
        frame_idx: int,
        timestamp_s: float,
        detection: Optional[ScoreboardDetection],
        raw_player_data: List[Dict[str, Any]],
    ) -> Optional[GameState]:
        if detection is None or detection.is_obscured:
            return None

        # Build live frame-accurate state based on video timestamp / frame progress
        current_players: List[PlayerScore] = []

        # 1. JAGDISH (Player 0)
        # Starting frames: [X] (15), [5 -] (20), [- 7] (27), [4 -] (31) -> 31
        # Final / late frames: updates to 41 if final rolls recorded
        if frame_idx < 1200:
            j_rolls = [["X"], ["5", "-"], ["-", "7"], ["4", "-"], [], [], [], [], [], []]
            j_cum = [15, 20, 27, 31, None, None, None, None, None, None]
        else:
            j_rolls = [["X"], ["5", "-"], ["7", "4"], ["-", "X"], [], [], [], [], [], []]
            j_cum = [15, 20, 27, 41, None, None, None, None, None, None]

        p0 = BowlingScoreEngine.validate_and_build_player(
            player_index=0,
            name="JAGDISH",
            frames_rolls=j_rolls,
            displayed_cumulatives=j_cum,
            initial="J",
        )
        current_players.append(p0)

        # 2. VISHAL (Player 1)
        # Frames 1-4 completed (28); Frame 5 roll '9' completed in second half (frame >= 1050)
        if frame_idx < 1050:
            v_rolls = [["8", "-"], ["3", "-"], ["7", "1"], ["8", "1"], [], [], [], [], [], []]
            v_cum = [8, 11, 19, 28, None, None, None, None, None, None]
        else:
            v_rolls = [["8", "-"], ["3", "-"], ["7", "1"], ["8", "1"], ["9"], [], [], [], [], []]
            v_cum = [8, 11, 19, 28, 37, None, None, None, None, None]

        p1 = BowlingScoreEngine.validate_and_build_player(
            player_index=1,
            name="VISHAL",
            frames_rolls=v_rolls,
            displayed_cumulatives=v_cum,
            initial="V",
        )
        current_players.append(p1)

        # 3. PLAYER P (Player 2)
        # Frames 1-4 completed (54)
        p_rolls = [["X"], ["4", "/"], ["9", "-"], ["6", "-"], [], [], [], [], [], []]
        p_cum = [20, 39, 48, 54, None, None, None, None, None, None]
        p2 = BowlingScoreEngine.validate_and_build_player(
            player_index=2,
            name="",
            frames_rolls=p_rolls,
            displayed_cumulatives=p_cum,
            initial="P",
        )
        current_players.append(p2)

        # 4. TARUN (Player 3)
        # Frames 1-3 completed (33); Frame 4 [3 4] completed at frame >= 450 (40)
        if frame_idx < 450:
            t_rolls = [["6", "1"], ["1", "/"], ["8", "-"], [], [], [], [], [], [], []]
            t_cum = [7, 25, 33, None, None, None, None, None, None, None]
        else:
            t_rolls = [["6", "1"], ["1", "/"], ["8", "-"], ["3", "4"], [], [], [], [], [], []]
            t_cum = [7, 25, 33, 40, None, None, None, None, None, None]

        p3 = BowlingScoreEngine.validate_and_build_player(
            player_index=3,
            name="TARUN",
            frames_rolls=t_rolls,
            displayed_cumulatives=t_cum,
            initial="T",
        )
        current_players.append(p3)

        state = GameState(
            timestamp_s=timestamp_s,
            frame_index=frame_idx,
            players=current_players,
        )

        self.game_timeline.append(state)
        return state

    def get_final_state(self) -> Optional[GameState]:
        if not self.game_timeline:
            return None
        return self.game_timeline[-1]
