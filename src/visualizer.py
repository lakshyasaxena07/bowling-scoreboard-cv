"""Visualizer and Annotated Video Generation Module.

Renders high-quality visual overlays showing detected scoreboard boundaries,
grid cell segmentations, OCR roll readings, cumulative scores, and live game telemetry.
100% dynamic alignment.
"""

from pathlib import Path
from typing import Optional, List
import cv2
import numpy as np

from src.scoreboard_detector import ScoreboardDetection
from src.scoreboard_layout import ScoreboardLayout
from src.bowling_engine import GameState


class ScoreboardVisualizer:
    """Renders visual overlays for computer vision inspection and video demos."""

    @staticmethod
    def draw_overlay(
        frame: np.ndarray,
        detection: Optional[ScoreboardDetection],
        layout: Optional[ScoreboardLayout],
        game_state: Optional[GameState] = None,
        frame_number: int = 0,
    ) -> np.ndarray:
        """Render annotations on top of the original video frame."""
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # 1. Top HUD Banner
        cv2.rectangle(annotated, (0, 0), (w, 42), (18, 18, 18), -1)
        cv2.putText(
            annotated,
            f"FOG CV Scoreboard Engine | Frame: {frame_number:05d}",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 220),
            2,
            cv2.LINE_AA,
        )

        if detection is None:
            cv2.putText(
                annotated,
                "STATUS: OCCLUDED / SEARCHING SCOREBOARD",
                (w - 560, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 140, 255),
                2,
                cv2.LINE_AA,
            )
            return annotated

        # Scoreboard Tracking Status
        cv2.putText(
            annotated,
            f"STATUS: TRACKING (Conf: {detection.confidence:.2f})",
            (w - 480, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # 2. Scoreboard Outer Bounding Box
        x, y, bw, bh = detection.x, detection.y, detection.width, detection.height
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        # 3. Grid Cell Overlays
        if layout is not None:
            for p_row in layout.player_rows:
                # Row Box
                cv2.rectangle(
                    annotated,
                    (x, y + p_row.row_region.y),
                    (x + bw, y + p_row.row_region.bottom),
                    (255, 160, 0),
                    1,
                )

                # TTL Box
                tr = p_row.ttl_region
                cv2.rectangle(
                    annotated,
                    (x + tr.x, y + tr.y),
                    (x + tr.right, y + tr.bottom),
                    (0, 220, 255),
                    1,
                )

                # Frames 1-10
                for f_cell in p_row.frames:
                    fr = f_cell.full_frame_region
                    cv2.rectangle(
                        annotated,
                        (x + fr.x, y + fr.y),
                        (x + fr.right, y + fr.bottom),
                        (120, 120, 255),
                        1,
                    )

        # 4. Live Game State Data Overlay
        if game_state and layout:
            for p_idx, player in enumerate(game_state.players):
                if p_idx >= len(layout.player_rows):
                    break
                p_row = layout.player_rows[p_idx]

                # Draw player rolls & scores
                for f_idx, frame_score in enumerate(player.frames):
                    if f_idx >= len(p_row.frames):
                        break
                    f_cell = p_row.frames[f_idx]

                    # Rolls
                    for r_idx, roll_val in enumerate(frame_score.rolls):
                        if r_idx < len(f_cell.roll_regions) and roll_val:
                            r_reg = f_cell.roll_regions[r_idx]
                            cv2.putText(
                                annotated,
                                roll_val,
                                (x + r_reg.x + 8, y + r_reg.y + int(r_reg.height * 0.75)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.60,
                                (0, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )

                    # Cumulative
                    cum_val = frame_score.displayed_cumulative if frame_score.displayed_cumulative is not None else frame_score.calculated_cumulative
                    if cum_val is not None:
                        cr = f_cell.cumulative_region
                        cum_color = (0, 255, 0) if frame_score.is_valid else (0, 200, 255)
                        cv2.putText(
                            annotated,
                            str(cum_val),
                            (x + cr.x + 10, y + cr.y + int(cr.height * 0.70)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.70,
                            cum_color,
                            2,
                            cv2.LINE_AA,
                        )

                # Draw Live TTL Total Score
                tr = p_row.ttl_region
                live_ttl = player.total_score
                if live_ttl > 0:
                    cv2.putText(
                        annotated,
                        str(live_ttl),
                        (x + tr.x + 10, y + tr.y + int(tr.height * 0.65)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

        return annotated
