"""Main CLI Pipeline for Bowling Scoreboard Computer Vision Extraction."""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np

from src.config import (
    DEFAULT_VIDEO_PATH,
    OUTPUT_DIR,
    DEBUG_DIR,
    DEFAULT_JSON_PATH,
    DEFAULT_CSV_PATH,
    DEFAULT_ANNOTATED_VIDEO_PATH,
)
from src.video_reader import VideoReader
from src.scoreboard_detector import ScoreboardDetector, ScoreboardDetection
from src.scoreboard_layout import LayoutDetector, ScoreboardLayout
from src.ocr_engine import ScoreboardOCREngine
from src.temporal_tracker import TemporalTracker
from src.exporter import ScoreboardExporter
from src.visualizer import ScoreboardVisualizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BowlingCV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production Bowling Scoreboard CV Extraction Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help="Path to input bowling scoreboard video",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="Root directory for extracted outputs",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=10,
        help="Process every Nth video frame",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Render and export full annotated MP4 video",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate debug visualization artifacts",
    )

    return parser.parse_args()


def process_video_pipeline(
    video_path: Path,
    output_dir: Path,
    sample_every: int = 10,
    save_video: bool = False,
    debug: bool = False,
) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"
    if debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "scoreboard_data.json"
    csv_path = output_dir / "scoreboard_summary.csv"
    annotated_video_path = output_dir / "annotated_bowling_scoreboard.mp4"

    logger.info("Initializing CV Pipeline on %s", video_path)
    start_time = time.time()

    detector = ScoreboardDetector()
    layout_analyzer = LayoutDetector()
    ocr_engine = ScoreboardOCREngine()
    temporal_tracker = TemporalTracker()
    visualizer = ScoreboardVisualizer()

    video_writer: cv2.VideoWriter = None

    with VideoReader(video_path) as reader:
        video_info = reader.get_info()
        logger.info(
            "Video Loaded: %dx%d @ %.1f FPS | Total Frames: %d (%.1fs)",
            video_info.width,
            video_info.height,
            video_info.fps,
            video_info.frame_count,
            video_info.duration_seconds,
        )

        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_fps = max(1.0, video_info.fps / sample_every)
            video_writer = cv2.VideoWriter(
                str(annotated_video_path),
                fourcc,
                out_fps,
                (video_info.width, video_info.height),
            )

        processed_count = 0
        cached_layout: ScoreboardLayout = None

        for frame_idx, frame in reader.frames(sample_every=sample_every):
            timestamp_s = frame_idx / video_info.fps if video_info.fps > 0 else 0.0

            # 1. Scoreboard Detection
            detection = detector.detect(frame)
            raw_players: List[Dict[str, Any]] = []

            if detection is not None:
                scoreboard_roi = detector.extract(frame, detection)

                # 2. Analyze Grid Layout
                layout = layout_analyzer.analyze(scoreboard_roi)
                cached_layout = layout

                active_banner_crop = layout.active_name_region.extract_from(scoreboard_roi)
                active_name = ocr_engine.recognize_active_name(active_banner_crop)

                # 3. Extract Cells & Perform OCR
                for p_idx, p_row in enumerate(layout.player_rows):
                    init_crop = p_row.initial_region.extract_from(scoreboard_roi)
                    initial_letter = ocr_engine.recognize_initial(init_crop, row_idx=p_idx)

                    if p_idx == 0:
                        player_name = f"{initial_letter} - {active_name}"
                    else:
                        player_name = f"PLAYER {initial_letter}"

                    player_rolls: List[List[str]] = []
                    player_cum: List[Optional[int]] = []

                    for f_cell in p_row.frames:
                        # Extract rolls from roll strip
                        roll_strip_crop = f_cell.roll_strip_region.extract_from(scoreboard_roi)
                        rolls = ocr_engine.extract_rolls_from_strip(roll_strip_crop)
                        player_rolls.append(rolls)

                        # Extract cumulative number from cumulative box
                        cum_crop = f_cell.cumulative_region.extract_from(scoreboard_roi)
                        cum_num = ocr_engine.extract_cumulative_number(cum_crop)
                        player_cum.append(cum_num)

                    raw_players.append({
                        "player_index": p_idx,
                        "name": player_name,
                        "rolls": player_rolls,
                        "displayed_cumulatives": player_cum,
                    })

            # 4. Temporal Aggregation & Animation Filtering
            current_game_state = temporal_tracker.update(
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                detection=detection,
                raw_player_data=raw_players,
            )

            # 5. Visual Overlay & Video Writing
            if save_video or (debug and processed_count % 15 == 0):
                annotated_frame = visualizer.draw_overlay(
                    frame=frame,
                    detection=detection,
                    layout=cached_layout,
                    game_state=current_game_state,
                    frame_number=frame_idx,
                )

                if video_writer:
                    video_writer.write(annotated_frame)

                if debug and processed_count % 15 == 0:
                    debug_path = debug_dir / f"frame_{frame_idx:06d}_debug.jpg"
                    cv2.imwrite(str(debug_path), annotated_frame)

            processed_count += 1
            if processed_count % 20 == 0:
                logger.info(
                    "Processed %d/%d frames (%.1f%%)",
                    frame_idx,
                    video_info.frame_count,
                    (frame_idx / video_info.frame_count) * 100.0,
                )

    if video_writer:
        video_writer.release()
        logger.info("Saved annotated video to %s", annotated_video_path)

    # 6. Export Final Results
    final_state = temporal_tracker.get_final_state()

    meta = {
        "source_video": str(video_path.name),
        "fps": video_info.fps,
        "width": video_info.width,
        "height": video_info.height,
        "total_frames": video_info.frame_count,
        "processed_samples": processed_count,
        "duration_seconds": video_info.duration_seconds,
        "processing_time_s": round(time.time() - start_time, 2),
    }

    ScoreboardExporter.export_json(final_state, json_path, video_metadata=meta)
    ScoreboardExporter.export_csv(final_state, csv_path)

    logger.info("Extraction complete in %.2fs!", time.time() - start_time)
    logger.info("Structured JSON saved to: %s", json_path)
    logger.info("Summary CSV saved to: %s", csv_path)


def main() -> None:
    args = parse_args()
    try:
        process_video_pipeline(
            video_path=args.video,
            output_dir=args.output,
            sample_every=args.sample_every,
            save_video=args.save_video,
            debug=args.debug,
        )
    except Exception as e:
        logger.exception("Pipeline failed with error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()