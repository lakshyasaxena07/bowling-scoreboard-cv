"""Bowling Scoring Rules and Semantic Validation Engine.

Implements official 10-frame bowling scoring logic (strikes, spares, open frames,
10th-frame bonus rolls, cumulative running totals) and provides domain validation
to cross-check OCR observations against game rules.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


@dataclass(frozen=True)
class FrameScore:
    frame_number: int  # 1 to 10
    rolls: Tuple[str, ...]  # ('X',), ('5', '-'), ('4', '/'), etc.
    displayed_cumulative: Optional[int] = None
    calculated_cumulative: Optional[int] = None
    is_valid: bool = True
    validation_message: str = "OK"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.frame_number,
            "rolls": list(self.rolls),
            "displayed_cumulative": self.displayed_cumulative,
            "calculated_cumulative": self.calculated_cumulative,
            "is_valid": self.is_valid,
            "validation_message": self.validation_message,
        }


@dataclass
class PlayerScore:
    player_index: int
    name: str
    frames: List[FrameScore] = field(default_factory=list)
    total_score: int = 0
    is_consistent: bool = True
    initial: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_index": self.player_index,
            "initial": self.initial,
            "name": self.name,
            "frames": [f.to_dict() for f in self.frames],
            "total_score": self.total_score,
            "is_consistent": self.is_consistent,
        }


@dataclass
class GameState:
    timestamp_s: float
    frame_index: int
    players: List[PlayerScore] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_s": round(self.timestamp_s, 2),
            "frame_index": self.frame_index,
            "players": [p.to_dict() for p in self.players],
        }


class BowlingScoreEngine:
    """Engine for computing and validating standard 10-frame bowling scores."""

    @staticmethod
    def parse_roll_value(roll_str: str, prev_roll_val: int = 0) -> int:
        roll = roll_str.strip().upper()
        if not roll or roll == "-":
            return 0
        if roll == "X":
            return 10
        if roll == "/":
            return max(0, 10 - prev_roll_val)
        if roll.isdigit():
            val = int(roll)
            return min(10, max(0, val))
        return 0

    @classmethod
    def calculate_cumulative_scores(
        cls,
        frames_rolls: List[List[str]],
    ) -> List[Optional[int]]:
        flat_rolls: List[int] = []
        frame_roll_indices: List[List[int]] = []

        for f_idx, rolls in enumerate(frames_rolls):
            curr_indices = []
            prev_val = 0
            for r_str in rolls:
                if not r_str:
                    continue
                val = cls.parse_roll_value(r_str, prev_val)
                flat_rolls.append(val)
                curr_indices.append(len(flat_rolls) - 1)
                prev_val = val
            frame_roll_indices.append(curr_indices)

        cumulative_scores: List[Optional[int]] = []
        running_total = 0

        for f_idx in range(len(frames_rolls)):
            if f_idx >= 10:
                break

            indices = frame_roll_indices[f_idx] if f_idx < len(frame_roll_indices) else []
            if not indices:
                cumulative_scores.append(None)
                continue

            if f_idx < 9:
                first_roll_idx = indices[0]
                first_roll_val = flat_rolls[first_roll_idx]

                if first_roll_val == 10:
                    # Strike
                    if len(flat_rolls) > first_roll_idx + 2:
                        frame_pts = 10 + flat_rolls[first_roll_idx + 1] + flat_rolls[first_roll_idx + 2]
                        running_total += frame_pts
                        cumulative_scores.append(running_total)
                    elif len(flat_rolls) > first_roll_idx + 1:
                        frame_pts = 10 + flat_rolls[first_roll_idx + 1]
                        cumulative_scores.append(running_total + frame_pts)
                    else:
                        cumulative_scores.append(None)
                elif len(indices) >= 2:
                    second_roll_val = flat_rolls[indices[1]]
                    if first_roll_val + second_roll_val == 10:
                        # Spare
                        if len(flat_rolls) > indices[1] + 1:
                            frame_pts = 10 + flat_rolls[indices[1] + 1]
                            running_total += frame_pts
                            cumulative_scores.append(running_total)
                        else:
                            cumulative_scores.append(None)
                    else:
                        # Open frame
                        frame_pts = first_roll_val + second_roll_val
                        running_total += frame_pts
                        cumulative_scores.append(running_total)
                else:
                    cumulative_scores.append(running_total + first_roll_val)
            else:
                # Frame 10
                frame_pts = sum(flat_rolls[idx] for idx in indices)
                running_total += frame_pts
                cumulative_scores.append(running_total)

        while len(cumulative_scores) < 10:
            cumulative_scores.append(None)

        return cumulative_scores

    @classmethod
    def validate_and_build_player(
        cls,
        player_index: int,
        name: str,
        frames_rolls: Optional[List[List[str]]] = None,
        displayed_cumulatives: Optional[List[Optional[int]]] = None,
        initial: str = "",
    ) -> PlayerScore:
        frames_rolls = frames_rolls or [[] for _ in range(10)]
        displayed_cumulatives = displayed_cumulatives or [None] * 10

        calculated = cls.calculate_cumulative_scores(frames_rolls)
        validated_frames: List[FrameScore] = []
        is_player_consistent = True

        for i in range(10):
            rolls = tuple(frames_rolls[i]) if i < len(frames_rolls) else ()
            disp = displayed_cumulatives[i] if i < len(displayed_cumulatives) else None
            calc = calculated[i] if i < len(calculated) else None

            is_valid = True
            msg = "OK"

            if disp is not None and calc is not None:
                if disp != calc:
                    is_valid = False
                    msg = f"Discrepancy: displayed {disp} != calculated {calc}"
                    is_player_consistent = False

            validated_frames.append(
                FrameScore(
                    frame_number=i + 1,
                    rolls=rolls,
                    displayed_cumulative=disp,
                    calculated_cumulative=calc,
                    is_valid=is_valid,
                    validation_message=msg,
                )
            )

        total = 0
        for f in reversed(validated_frames):
            if f.displayed_cumulative is not None:
                total = f.displayed_cumulative
                break
            elif f.calculated_cumulative is not None:
                total = f.calculated_cumulative
                break

        return PlayerScore(
            player_index=player_index,
            name=name,
            initial=initial,
            frames=validated_frames,
            total_score=total,
            is_consistent=is_player_consistent,
        )
