
"""Tests for AeroMind AI object detection."""

import numpy as np
import pytest

from ai.object_detection import (
    BoundingBox,
    Detection,
    DetectionResult,
    DetectorConfig,
    InvalidFrameError,
    ObjectDetector,
)


def test_detector_initialisation():
    detector = ObjectDetector()

    assert detector.config.model_path == "yolo11n.pt"
    assert detector._model is None


def test_invalid_frame_rejected():
    detector = ObjectDetector()

    with pytest.raises(InvalidFrameError):
        detector.detect(
            np.zeros((100, 100), dtype=np.uint8)
        )


def test_bounding_box_centre():
    box = BoundingBox(
        x1=10,
        y1=20,
        x2=110,
        y2=120,
    )

    assert box.width == 100
    assert box.height == 100
    assert box.centre == (60, 70)


def test_detection_result_filtering():
    result = DetectionResult(
        detections=(
            Detection(
                class_id=0,
                label="person",
                confidence=0.95,
                bounding_box=BoundingBox(
                    10, 20, 100, 200
                ),
            ),
        ),
        frame_width=640,
        frame_height=480,
        inference_time_ms=15.0,
        model_name="yolo11n.pt",
    )

    assert result.object_count == 1
    assert len(result.filter_by_label("person")) == 1
    assert len(result.filter_by_label("car")) == 0


def test_invalid_detector_configuration():
    with pytest.raises(ValueError):
        DetectorConfig(
            confidence_threshold=1.5
        )
