
"""
AeroMind AI - Multi-Object Tracking Module.

Tracks objects across consecutive camera frames using
bounding-box Intersection over Union (IoU).

Features:
    - Persistent object IDs
    - Configurable IoU matching
    - One-to-one detection association
    - Object appearance and disappearance handling
    - Image-plane velocity estimation
    - Track lifecycle management
    - Structured tracking results
    - Thread-safe tracker state
    - Logging and input validation

Limitations:
    - This is a basic IoU tracker, not ByteTrack or DeepSORT.
    - Occlusion and rapid camera motion can cause ID switches.
    - Pixel velocity is not physical velocity in metres/second.
    - Tracking alone cannot determine collision risk.

No flight commands are issued by this module.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from ai.object_detection import (
    BoundingBox,
    Detection,
    DetectionResult,
)


LOGGER = logging.getLogger("aeromind.ai.object_tracking")


# ============================================================
# EXCEPTIONS
# ============================================================

class ObjectTrackingError(RuntimeError):
    """Base exception for tracking failures."""


class InvalidTrackingInput(ObjectTrackingError):
    """Raised when tracking input is invalid."""


# ============================================================
# TRACK STATES
# ============================================================

class TrackStatus(str, Enum):
    """Lifecycle status of a tracked object."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """Configuration for multi-object tracking."""

    iou_threshold: float = 0.3
    max_missing_frames: int = 10
    min_confirmed_hits: int = 3
    max_tracks: int = 100
    match_same_class_only: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.iou_threshold, bool)
            or not isinstance(self.iou_threshold, (int, float))
            or not math.isfinite(self.iou_threshold)
            or not 0 < self.iou_threshold <= 1
        ):
            raise ValueError(
                "iou_threshold must be between 0 and 1."
            )

        for name in (
            "max_missing_frames",
            "min_confirmed_hits",
            "max_tracks",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(
                    f"{name} must be a positive integer."
                )


# ============================================================
# TRACKED OBJECT
# ============================================================

@dataclass(frozen=True, slots=True)
class TrackedObject:
    """Public immutable snapshot of a tracked object."""

    track_id: int
    class_id: int
    label: str
    confidence: float
    bounding_box: BoundingBox

    status: TrackStatus

    age_frames: int
    hit_count: int
    missing_frames: int

    velocity_x_px_s: float
    velocity_y_px_s: float

    last_seen: datetime

    @property
    def centre(self) -> tuple[float, float]:
        return self.bounding_box.centre

    @property
    def speed_px_s(self) -> float:
        return math.hypot(
            self.velocity_x_px_s,
            self.velocity_y_px_s,
        )


@dataclass(frozen=True, slots=True)
class TrackingResult:
    """Tracking results for one processed frame."""

    frame_timestamp: datetime
    tracks: tuple[TrackedObject, ...]
    active_count: int
    confirmed_count: int
    frame_width: int
    frame_height: int

    def to_dict(self) -> dict[str, Any]:
        """Convert tracking results to serialisable data."""

        return {
            "timestamp": self.frame_timestamp.isoformat(),
            "active_count": self.active_count,
            "confirmed_count": self.confirmed_count,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "tracks": [
                {
                    "track_id": track.track_id,
                    "class_id": track.class_id,
                    "label": track.label,
                    "confidence": track.confidence,
                    "status": track.status.value,
                    "age_frames": track.age_frames,
                    "hit_count": track.hit_count,
                    "missing_frames": track.missing_frames,
                    "centre": list(track.centre),
                    "velocity_px_s": {
                        "x": track.velocity_x_px_s,
                        "y": track.velocity_y_px_s,
                    },
                    "bounding_box": {
                        "x1": track.bounding_box.x1,
                        "y1": track.bounding_box.y1,
                        "x2": track.bounding_box.x2,
                        "y2": track.bounding_box.y2,
                    },
                }
                for track in self.tracks
            ],
        }


# ============================================================
# INTERNAL MUTABLE TRACK
# ============================================================

@dataclass(slots=True)
class _Track:
    """Internal state maintained between frames."""

    track_id: int
    detection: Detection

    first_seen: datetime
    last_seen: datetime

    age_frames: int = 1
    hit_count: int = 1
    missing_frames: int = 0

    velocity_x_px_s: float = 0.0
    velocity_y_px_s: float = 0.0


# ============================================================
# TRACKING UTILITIES
# ============================================================

def calculate_iou(
    first: BoundingBox,
    second: BoundingBox,
) -> float:
    """
    Calculate Intersection over Union for two boxes.

    Returns a value between 0.0 and 1.0.
    """

    intersection_x1 = max(first.x1, second.x1)
    intersection_y1 = max(first.y1, second.y1)
    intersection_x2 = min(first.x2, second.x2)
    intersection_y2 = min(first.y2, second.y2)

    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1,
    )

    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1,
    )

    intersection_area = (
        intersection_width * intersection_height
    )

    first_area = first.width * first.height
    second_area = second.width * second.height

    union_area = (
        first_area + second_area - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


# ============================================================
# OBJECT TRACKER
# ============================================================

class ObjectTracker:
    """
    Stateful multi-object tracker.

    Tracks detections across sequential frames using
    greedy, one-to-one IoU association.

    The tracker must receive frames in chronological order.
    """

    def __init__(
        self,
        config: TrackerConfig | None = None,
    ) -> None:

        self.config = config or TrackerConfig()

        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1
        self._last_timestamp: datetime | None = None

        self._lock = threading.RLock()

        LOGGER.info(
            "Object tracker initialised: IoU=%.2f",
            self.config.iou_threshold,
        )

    def update(
        self,
        result: DetectionResult,
    ) -> TrackingResult:
        """
        Update tracks using detections from a new frame.

        Args:
            result:
                Detection result from ObjectDetector.

        Returns:
            TrackingResult containing current tracks.
        """

        if not isinstance(result, DetectionResult):
            raise InvalidTrackingInput(
                "Expected a DetectionResult instance."
            )

        timestamp = result.timestamp

        if (
            timestamp.tzinfo is None
            or timestamp.utcoffset() is None
        ):
            raise InvalidTrackingInput(
                "Frame timestamp must be timezone-aware."
            )

        with self._lock:

            if (
                self._last_timestamp is not None
                and timestamp <= self._last_timestamp
            ):
                raise InvalidTrackingInput(
                    "Frame timestamps must increase."
                )

            self._last_timestamp = timestamp

            detections = list(result.detections)

            # Age existing tracks.
            for track in self._tracks.values():
                track.age_frames += 1

            # Build eligible track/detection matches.
            candidates: list[
                tuple[float, int, int]
            ] = []

            for track_id, track in self._tracks.items():
                for detection_index, detection in enumerate(
                    detections
                ):

                    if (
                        self.config.match_same_class_only
                        and track.detection.class_id
                        != detection.class_id
                    ):
                        continue

                    iou = calculate_iou(
                        track.detection.bounding_box,
                        detection.bounding_box,
                    )

                    if iou >= self.config.iou_threshold:
                        candidates.append(
                            (
                                iou,
                                track_id,
                                detection_index,
                            )
                        )

            # Match highest IoU pairs first.
            candidates.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            matched_tracks: set[int] = set()
            matched_detections: set[int] = set()

            for _, track_id, detection_index in candidates:

                if track_id in matched_tracks:
                    continue

                if detection_index in matched_detections:
                    continue

                track = self._tracks[track_id]
                detection = detections[detection_index]

                elapsed_seconds = (
                    timestamp - track.last_seen
                ).total_seconds()

                old_x, old_y = track.detection.centre
                new_x, new_y = detection.centre

                if elapsed_seconds > 0:
                    track.velocity_x_px_s = (
                        new_x - old_x
                    ) / elapsed_seconds

                    track.velocity_y_px_s = (
                        new_y - old_y
                    ) / elapsed_seconds

                track.detection = detection
                track.last_seen = timestamp
                track.hit_count += 1
                track.missing_frames = 0

                matched_tracks.add(track_id)
                matched_detections.add(detection_index)

            # Mark unmatched tracks as missing.
            for track_id, track in self._tracks.items():

                if track_id not in matched_tracks:
                    track.missing_frames += 1

            # Remove expired tracks.
            expired_ids = [
                track_id
                for track_id, track in self._tracks.items()
                if track.missing_frames
                > self.config.max_missing_frames
            ]

            for track_id in expired_ids:
                del self._tracks[track_id]

                LOGGER.debug(
                    "Removed expired track %d.",
                    track_id,
                )

            # Register unmatched detections.
            for index, detection in enumerate(detections):

                if index in matched_detections:
                    continue

                if len(self._tracks) >= self.config.max_tracks:
                    LOGGER.warning(
                        "Maximum number of tracks reached."
                    )
                    break

                track_id = self._next_track_id
                self._next_track_id += 1

                self._tracks[track_id] = _Track(
                    track_id=track_id,
                    detection=detection,
                    first_seen=timestamp,
                    last_seen=timestamp,
                )

            snapshots: list[TrackedObject] = []

            for track in self._tracks.values():

                if track.missing_frames > 0:
                    status = TrackStatus.LOST

                elif (
                    track.hit_count
                    >= self.config.min_confirmed_hits
                ):
                    status = TrackStatus.CONFIRMED

                else:
                    status = TrackStatus.TENTATIVE

                snapshots.append(
                    TrackedObject(
                        track_id=track.track_id,
                        class_id=track.detection.class_id,
                        label=track.detection.label,
                        confidence=track.detection.confidence,
                        bounding_box=track.detection.bounding_box,
                        status=status,
                        age_frames=track.age_frames,
                        hit_count=track.hit_count,
                        missing_frames=track.missing_frames,
                        velocity_x_px_s=track.velocity_x_px_s,
                        velocity_y_px_s=track.velocity_y_px_s,
                        last_seen=track.last_seen,
                    )
                )

            confirmed_count = sum(
                track.status == TrackStatus.CONFIRMED
                for track in snapshots
            )

            tracking_result = TrackingResult(
                frame_timestamp=timestamp,
                tracks=tuple(snapshots),
                active_count=len(snapshots),
                confirmed_count=confirmed_count,
                frame_width=result.frame_width,
                frame_height=result.frame_height,
            )

            LOGGER.debug(
                "Tracking updated: active=%d confirmed=%d",
                tracking_result.active_count,
                tracking_result.confirmed_count,
            )

            return tracking_result

    def reset(self) -> None:
        """Clear all tracking state."""

        with self._lock:
            self._tracks.clear()
            self._next_track_id = 1
            self._last_timestamp = None

        LOGGER.info("Object tracker state reset.")
