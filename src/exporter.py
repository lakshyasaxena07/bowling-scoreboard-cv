"""Data Exporter Module.

Exports extracted bowling scoreboard game records into structured JSON and CSV formats
matching the exact assignment specification.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import csv

from src.bowling_engine import GameState


class ScoreboardExporter:
    """Exports GameState data to JSON and CSV files."""

    @staticmethod
    def export_json(
        game_state: GameState,
        output_path: Path,
        video_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Export game state to a well-formatted JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "video_metadata": video_metadata or {},
            "final_game_state": game_state.to_dict() if game_state else {},
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def export_csv(
        game_state: GameState,
        output_path: Path,
    ) -> None:
        """Export player score summary to CSV matching exact assignment format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        headers = [
            "Player_Initial",
            "Player_Name",
            "F1_B1", "F1_B2", "F1_Total",
            "F2_B1", "F2_B2", "F2_Total",
            "F3_B1", "F3_B2", "F3_Total",
            "F4_B1", "F4_B2", "F4_Total",
            "F5_B1", "F5_B2", "F5_Total",
            "TTL",
        ]

        rows = []
        if game_state:
            for p in game_state.players:
                row = [p.initial, p.name]
                for f_idx in range(5):
                    if f_idx < len(p.frames):
                        frame = p.frames[f_idx]
                        b1 = frame.rolls[0] if len(frame.rolls) > 0 else ""
                        b2 = frame.rolls[1] if len(frame.rolls) > 1 else ""
                        score_val = frame.displayed_cumulative if frame.displayed_cumulative is not None else frame.calculated_cumulative
                        score_str = str(score_val) if score_val is not None and score_val > 0 else ""
                    else:
                        b1 = ""
                        b2 = ""
                        score_str = ""
                    row.extend([b1, b2, score_str])
                
                ttl_str = str(p.total_score) if p.total_score > 0 else ""
                row.append(ttl_str)
                rows.append(row)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
