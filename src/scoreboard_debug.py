from pathlib import Path

import cv2

from src.scoreboard_detector import ScoreboardDetector


def save_detected_scoreboard(
    image_path: Path,
    output_path: Path,
) -> None:
    frame = cv2.imread(str(image_path))

    if frame is None:
        raise FileNotFoundError(
            f"Unable to read image: {image_path}"
        )

    detector = ScoreboardDetector()
    detection = detector.detect(frame)

    if detection is None:
        raise RuntimeError(
            f"Scoreboard was not detected in {image_path}"
        )

    scoreboard = detector.extract(frame, detection)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), scoreboard):
        raise IOError(
            f"Failed to write scoreboard image: {output_path}"
        )


def main() -> None:
    sample_dir = Path("output/samples")

    save_detected_scoreboard(
        sample_dir / "frame_000300.jpg",
        Path("output/scoreboard_000300.jpg"),
    )

    save_detected_scoreboard(
        sample_dir / "frame_000900.jpg",
        Path("output/scoreboard_000900.jpg"),
    )

    print("Scoreboard ROI images created.")


if __name__ == "__main__":
    main()