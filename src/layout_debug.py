from pathlib import Path

import cv2

from src.scoreboard_layout import build_scoreboard_layout


def draw_region(
    image,
    region,
    label,
    color,
):
    x1 = region.x
    y1 = region.y
    x2 = region.right
    y2 = region.bottom

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    cv2.putText(
        image,
        label,
        (x1 + 3, y1 + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def create_layout_debug(
    image_path: Path,
    output_path: Path,
) -> None:
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    height, width = image.shape[:2]

    layout = build_scoreboard_layout(
        width=width,
        height=height,
    )

    debug_image = image.copy()

    draw_region(
        debug_image,
        layout.player_region,
        "PLAYER",
        (0, 255, 0),
    )

    draw_region(
        debug_image,
        layout.ttl_region,
        "TTL",
        (0, 0, 255),
    )

    for index, region in enumerate(layout.frame_regions, start=1):
        draw_region(
            debug_image,
            region,
            f"F{index}",
            (255, 0, 0),
        )

    for index, player_row in enumerate(layout.player_rows,start=1):
        draw_region(
            debug_image,
            player_row.row,
            f"ROW {index}",
            (0, 255, 255),
        )

        draw_region(
            debug_image,
            player_row.result,
            f"R{index}",
            (255, 255, 0),
        )

        draw_region(
            debug_image,
            player_row.cumulative,
            f"C{index}",
            (255, 0, 255),
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(output_path),
        debug_image,
    ):
        raise IOError(f"Failed to write {output_path}")


def main() -> None:
    create_layout_debug(
        Path("output/scoreboard_000900.jpg"),
        Path("output/layout_debug_000900.jpg"),
    )


if __name__ == "__main__":
    main()
