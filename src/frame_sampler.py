from pathlib import Path

import cv2

from src.video_reader import VideoReader


def save_sample_frames(
    video_path: Path,
    output_dir: Path,
    sample_every: int,
) -> int:
    """Save periodic frames from a video for inspection."""

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0

    with VideoReader(video_path) as reader:
        for frame_index, frame in reader.frames(sample_every):
            output_path = output_dir / f"frame_{frame_index:06d}.jpg"

            if not cv2.imwrite(str(output_path), frame):
                raise IOError(f"Failed to save frame: {output_path}")

            saved_count += 1

    return saved_count