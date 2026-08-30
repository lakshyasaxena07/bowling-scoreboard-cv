from pathlib import Path
import math

import cv2
import numpy as np


def create_contact_sheet(
    image_dir: Path,
    output_path: Path,
    columns: int = 5,
) -> None:
    """Create a grid image containing sampled video frames."""

    image_paths = sorted(image_dir.glob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError(
            f"No sample frames found in {image_dir}"
        )

    images = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        image = cv2.resize(image, (384, 216))

        label = image_path.stem

        cv2.putText(
            image,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        images.append(image)

    rows = math.ceil(len(images) / columns)

    cell_height, cell_width = images[0].shape[:2]

    sheet = np.zeros(
        (
            rows * cell_height,
            columns * cell_width,
            3,
        ),
        dtype=np.uint8,
    )

    for index, image in enumerate(images):
        row = index // columns
        column = index % columns

        y = row * cell_height
        x = column * cell_width

        sheet[
            y:y + cell_height,
            x:x + cell_width
        ] = image

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), sheet):
        raise IOError(f"Failed to save contact sheet: {output_path}")