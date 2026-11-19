
"""Unit tests for AeroMind AI object tracking."""

from datetime import datetime, timedelta, timezone

from ai.object_detection import (
    BoundingBox,
    Detection,
    DetectionResult,
)

from ai.object_tracking import (
    ObjectTracker,
    TrackerConfig,
    TrackStatus,
    calculate_iou,
)


BASE_TIME = datetime(
    2026, 1, 1, tzinfo=timezone.utc
)


def make_result(
    timestamp,
    x_offset=0,
    include_object=True,
):
    detections = ()

    if include_object:
        detections = (
            Detection(
                class_id=0,
                label="person",
                confidence=0.95,
                bounding_box=BoundingBox(
                    10 + x_offset,
                    20,
                    110 + x_offset,
                    120,
                ),
            ),
        )

    return DetectionResult(
        detections=detections,
        frame_width=640,
        frame_height=480,
        inference_time_ms=10.0,
        model_name="yolo11n.pt",
        timestamp=timestamp,
    )


def test_iou_identical_boxes():
    box = BoundingBox(0, 0, 100, 100)

    assert calculate_iou(box, box) == 1.0


def test_tracker_creates_track():
    tracker = ObjectTracker()

    result = tracker.update(
        make_result(BASE_TIME)
    )

    assert result.active_count == 1
    assert result.tracks[0].track_id == 1
    assert result.tracks[0].status == TrackStatus.TENTATIVE


def test_tracker_preserves_identity():
    tracker = ObjectTracker()

    first = tracker.update(
        make_result(BASE_TIME)
    )

    second = tracker.update(
        make_result(
            BASE_TIME + timedelta(seconds=1),
            x_offset=5,
        )
    )

    assert (
        first.tracks[0].track_id
        == second.tracks[0].track_id
    )

    assert second.tracks[0].velocity_x_px_s == 5.0


def test_tracker_confirms_object():
    tracker = ObjectTracker(
        TrackerConfig(min_confirmed_hits=2)
    )

    tracker.update(
        make_result(BASE_TIME)
    )

    result = tracker.update(
        make_result(
            BASE_TIME + timedelta(seconds=1)
        )
    )

    assert result.confirmed_count == 1
    assert result.tracks[0].status == TrackStatus.CONFIRMED


def test_tracker_expires_missing_object():
    tracker = ObjectTracker(
        TrackerConfig(max_missing_frames=1)
    )

    tracker.update(
        make_result(BASE_TIME)
    )

    tracker.update(
        make_result(
            BASE_TIME + timedelta(seconds=1),
            include_object=False,
        )
    )

    result = tracker.update(
        make_result(
            BASE_TIME + timedelta(seconds=2),
            include_object=False,
        )
    )

    assert result.active_count == 0


def test_tracker_reset():
    tracker = ObjectTracker()

    tracker.update(
        make_result(BASE_TIME)
    )

    tracker.reset()

    result = tracker.update(
        make_result(
            BASE_TIME + timedelta(seconds=1)
        )
    )

    assert result.tracks[0].track_id == 1
