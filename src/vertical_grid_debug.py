from pathlib import Path

import cv2
import numpy as np


def find_vertical_candidates(
    image_path: Path,
) -> list[int]:
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

    height, _ = gray.shape

    y_start = int(height * 0.10)
    y_end = int(height * 0.45)

    score_area = edges[y_start:y_end, :]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, 15),
    )

    vertical_lines = cv2.morphologyEx(
        score_area,
        cv2.MORPH_OPEN,
        kernel,
    )

    projection = np.sum(
        vertical_lines > 0,
        axis=0,
    )

    threshold = (y_end - y_start) * 0.20

    candidates = np.where(
        projection > threshold
    )[0]

    return candidates.tolist()


def merge_nearby_positions(
    positions: list[int],
    max_gap: int = 8,
) -> list[int]:
    if not positions:
        return []

    groups = []
    current_group = [positions[0]]

    for position in positions[1:]:
        if position - current_group[-1] <= max_gap:
            current_group.append(position)
        else:
            groups.append(current_group)
            current_group = [position]

    groups.append(current_group)

    return [
        int(round(sum(group) / len(group)))
        for group in groups
    ]


def main() -> None:
    image_path = Path(
        "output/scoreboard_000900.jpg"
    )

    candidates = find_vertical_candidates(
        image_path
    )

    boundaries = merge_nearby_positions(
        candidates
    )

    print("Merged vertical boundaries:")

    for index, x in enumerate(boundaries):
        print(f"{index:02d}: x={x}")

    if len(boundaries) > 1:
        print("\nBoundary spacing:")

        for left, right in zip(
            boundaries,
            boundaries[1:],
        ):
            print(
                f"{left:4d} -> {right:4d} : "
                f"{right - left:3d}px"
            )


if __name__ == "__main__":
    main()