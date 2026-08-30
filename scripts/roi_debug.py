from pathlib import Path

import cv2


# Initial ROI estimate based on visual inspection.
# These values will be validated against multiple frames.
SCOREBOARD_ROI = (35, 25, 1850, 1040)


def draw_roi(image_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    x, y, width, height = SCOREBOARD_ROI

    x2 = x + width
    y2 = y + height

    cv2.rectangle(
        image,
        (x, y),
        (x2, y2),
        (0, 255, 0),
        4,
    )

    cv2.putText(
        image,
        f"ROI: x={x}, y={y}, w={width}, h={height}",
        (x, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), image):
        raise IOError(f"Failed to write {output_path}")


if __name__ == "__main__":
    sample_dir = Path("output/samples")

    draw_roi(
        sample_dir / "frame_000300.jpg",
        Path("output/roi_frame_000300.jpg"),
    )

    draw_roi(
        sample_dir / "frame_000900.jpg",
        Path("output/roi_frame_000900.jpg"),
    )

    print("ROI debug images created.")