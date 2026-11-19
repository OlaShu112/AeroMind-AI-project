
"""Run AeroMind AI object detection on a test image."""

import json
from pathlib import Path

import cv2

from ai.object_detection import ObjectDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
    PROJECT_ROOT / "simulation" / "test_image.jpg"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "simulation" / "detected_image.jpg"
)


def main() -> None:
    frame = cv2.imread(str(IMAGE_PATH))

    if frame is None:
        raise FileNotFoundError(
            f"Cannot load image: {IMAGE_PATH}"
        )

    detector = ObjectDetector()

    try:
        result = detector.detect(frame)

        print(
            json.dumps(
                result.to_dict(),
                indent=2,
            )
        )

        annotated = detector.annotate_frame(
            frame,
            result,
        )

        if not cv2.imwrite(
            str(OUTPUT_PATH),
            annotated,
        ):
            raise RuntimeError(
                "Failed to save annotated image."
            )

        print(f"Annotated image saved: {OUTPUT_PATH}")

    finally:
        detector.close()


if __name__ == "__main__":
    main()
