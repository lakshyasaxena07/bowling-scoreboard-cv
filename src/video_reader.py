from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


class VideoReader:
    def __init__(self, video_path: Path):
        self.video_path = Path(video_path)
        self.capture = cv2.VideoCapture(str(self.video_path))

        if not self.capture.isOpened():
            raise FileNotFoundError(
                f"Unable to open video: {self.video_path}"
            )

    def get_info(self) -> VideoInfo:
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

        duration = frame_count / fps if fps > 0 else 0.0

        return VideoInfo(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
        )

    def frames(
        self,
        sample_every: int = 1,
    ) -> Iterator[tuple[int, np.ndarray]]:
        if sample_every < 1:
            raise ValueError("sample_every must be at least 1")

        frame_index = 0

        while True:
            success, frame = self.capture.read()

            if not success:
                break

            if frame_index % sample_every == 0:
                yield frame_index, frame

            frame_index += 1

    def close(self) -> None:
        self.capture.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()