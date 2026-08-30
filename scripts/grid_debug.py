from pathlib import Path

import cv2
import numpy as np


def detect_grid_lines(image_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Unable to read image: {image_path}"
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(
        gray,
        threshold1=50,
        threshold2=150,
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, 25),
    )

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (25, 1),
    )

    vertical_lines = cv2.morphologyEx(
        edges,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )

    horizontal_lines = cv2.morphologyEx(
        edges,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    debug_image = image.copy()

    vertical_projection = np.sum(
        vertical_lines > 0,
        axis=0,
    )

    horizontal_projection = np.sum(
        horizontal_lines > 0,
        axis=1,
    )

    vertical_threshold = image.shape[0] * 0.15
    horizontal_threshold = image.shape[1] * 0.15

    for x, value in enumerate(vertical_projection):
        if value > vertical_threshold:
            cv2.line(
                debug_image,
                (x, 0),
                (x, image.shape[0]),
                (0, 255, 0),
                2,
            )

    for y, value in enumerate(horizontal_projection):
        if value > horizontal_threshold:
            cv2.line(
                debug_image,
                (0, y),
                (image.shape[1], y),
                (255, 0, 0),
                2,
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(output_path),
        debug_image,
    ):
        raise IOError(
            f"Failed to write {output_path}"
        )


def main() -> None:
    sample_dir = Path("output")

    detect_grid_lines(
        sample_dir / "scoreboard_000900.jpg",
        sample_dir / "grid_debug_000900.jpg",
    )

    detect_grid_lines(
        sample_dir / "scoreboard_000300.jpg",
        sample_dir / "grid_debug_000300.jpg",
    )

    print("Grid debug images created.")


if __name__ == "__main__":
    main()