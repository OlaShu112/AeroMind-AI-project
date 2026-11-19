
"""
AeroMind AI - Object Detection and Tracking Test.

Demonstrates integration between the YOLO object detector
and the multi-object tracker using a test image.

This is a development script, not a flight-control service.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2

from ai import ObjectDetector, ObjectTracker


LOGGER = logging.getLogger("aeromind.scripts.tracking")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = PROJECT_ROOT / "simulation" / "test_image.jpg"
OUTPUT_PATH = PROJECT_ROOT / "simulation" / "tracked_image.jpg"


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not IMAGE_PATH.is_file():
        LOGGER.error("Test image not found: %s", IMAGE_PATH)
        return 1

    frame = cv2.imread(str(IMAGE_PATH))

    if frame is None:
        LOGGER.error("Unable to read test image.")
        return 1

    detector = ObjectDetector()
    tracker = ObjectTracker()

    try:
        # Detect objects in the image.
        detection_result = detector.detect(frame)

        # Associate detections with tracked objects.
        tracking_result = tracker.update(detection_result)

        LOGGER.info(
            "Objects detected: %d",
            detection_result.object_count,
        )

        for tracked_object in tracking_result.tracks:
            LOGGER.info(
                "Track ID: %d | Class: %s | "
                "Status: %s | Centre: %s",
                tracked_object.track_id,
                tracked_object.label,
                tracked_object.status.value,
                tracked_object.centre,
            )

        annotated = detector.annotate_frame(
            frame,
            detection_result,
        )

        if not cv2.imwrite(str(OUTPUT_PATH), annotated):
            LOGGER.error("Unable to save output image.")
            return 1

        LOGGER.info("Output saved: %s", OUTPUT_PATH)
        return 0

    except Exception:
        LOGGER.exception(
            "Detection and tracking test failed."
        )
        return 1

    finally:
        detector.close()


if __name__ == "__main__":
    raise SystemExit(main())
