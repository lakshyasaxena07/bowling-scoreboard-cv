from pathlib import Path

import cv2

from src.scoreboard_layout import build_scoreboard_layout


def main() -> None:
    image_path = Path(
        "output/scoreboard_000900.jpg"
    )

    output_path = Path(
        "output/frame_regions_debug.jpg"
    )

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Unable to read image: {image_path}"
        )

    height, width = image.shape[:2]

    layout = build_scoreboard_layout(
        width=width,
        height=height,
    )

    debug_image = image.copy()

    for index, region in enumerate(
        layout.frame_regions,
        start=1,
    ):
        cv2.rectangle(
            debug_image,
            (region.x, region.y),
            (region.right, region.bottom),
            (255, 0, 0),
            2,
        )

        cv2.putText(
            debug_image,
            f"F{index}",
            (region.x + 5, region.y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
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

    print(
        f"Saved frame region debug image to {output_path}"
    )


if __name__ == "__main__":
    main()