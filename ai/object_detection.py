
"""
AeroMind AI - Object Detection Module.

YOLO-based object detection for simulated drone perception.

Features:
    - Lazy YOLO model loading
    - Configurable detection thresholds
    - NumPy/OpenCV image validation
    - Structured and immutable detection results
    - Bounding-box validation
    - Frame-level inference metadata
    - Optional annotated image output
    - Logging and error handling

Important:
    Object detection alone does not provide reliable
    obstacle distance or collision avoidance.

    This module does not issue drone flight commands.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


LOGGER = logging.getLogger("aeromind.ai.object_detection")


# ============================================================
# EXCEPTIONS
# ============================================================

class ObjectDetectionError(RuntimeError):
    """Base exception for object detection failures."""


class ModelLoadingError(ObjectDetectionError):
    """Raised when the YOLO model cannot be loaded."""


class InvalidFrameError(ObjectDetectionError):
    """Raised when an image frame is invalid."""


class InferenceError(ObjectDetectionError):
    """Raised when object detection inference fails."""


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Configuration for YOLO inference."""

    model_path: str = "yolo11n.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    image_size: int = 640
    device: str = "cpu"
    max_detections: int = 100

    def __post_init__(self) -> None:
        if not self.model_path.strip():
            raise ValueError("model_path cannot be empty.")

        for name, value in (
            ("confidence_threshold", self.confidence_threshold),
            ("iou_threshold", self.iou_threshold),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(
                    f"{name} must be a finite value between 0 and 1."
                )

        if (
            isinstance(self.image_size, bool)
            or not isinstance(self.image_size, int)
            or self.image_size <= 0
            or self.image_size % 32 != 0
        ):
            raise ValueError(
                "image_size must be a positive multiple of 32."
            )

        if (
            isinstance(self.max_detections, bool)
            or not isinstance(self.max_detections, int)
            or self.max_detections <= 0
        ):
            raise ValueError(
                "max_detections must be a positive integer."
            )

        if not self.device.strip():
            raise ValueError("device cannot be empty.")


# ============================================================
# DETECTION DATA MODELS
# ============================================================

@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Bounding box in original image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)

        if not all(math.isfinite(value) for value in values):
            raise ValueError(
                "Bounding-box coordinates must be finite."
            )

        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(
                "Bounding-box coordinates are invalid."
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def centre(self) -> tuple[float, float]:
        return (
            (self.x1 + self.x2) / 2,
            (self.y1 + self.y2) / 2,
        )


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected object in an image."""

    class_id: int
    label: str
    confidence: float
    bounding_box: BoundingBox

    @property
    def centre(self) -> tuple[float, float]:
        return self.bounding_box.centre


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Complete detection result for a single image."""

    detections: tuple[Detection, ...]
    frame_width: int
    frame_height: int
    inference_time_ms: float
    model_name: str

    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def object_count(self) -> int:
        return len(self.detections)

    def filter_by_label(
        self,
        label: str,
    ) -> tuple[Detection, ...]:
        """Return detections matching an object label."""

        return tuple(
            detection
            for detection in self.detections
            if detection.label.casefold() == label.casefold()
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert detection results to serialisable data."""

        return {
            "timestamp": self.timestamp.isoformat(),
            "model_name": self.model_name,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "inference_time_ms": self.inference_time_ms,
            "object_count": self.object_count,
            "detections": [
                {
                    "class_id": item.class_id,
                    "label": item.label,
                    "confidence": item.confidence,
                    "bounding_box": {
                        "x1": item.bounding_box.x1,
                        "y1": item.bounding_box.y1,
                        "x2": item.bounding_box.x2,
                        "y2": item.bounding_box.y2,
                    },
                }
                for item in self.detections
            ],
        }


# ============================================================
# OBJECT DETECTOR
# ============================================================

class ObjectDetector:
    """
    YOLO object detection service.

    The model is loaded on first inference, rather than
    during package import or object construction.
    """

    def __init__(
        self,
        config: DetectorConfig | None = None,
    ) -> None:
        self.config = config or DetectorConfig()
        self._model: Any | None = None

        LOGGER.info(
            "Object detector configured: model=%s device=%s",
            self.config.model_path,
            self.config.device,
        )

    def load_model(self) -> None:
        """Load the configured YOLO model."""

        if self._model is not None:
            return

        try:
            from ultralytics import YOLO

            LOGGER.info(
                "Loading YOLO model: %s",
                self.config.model_path,
            )

            self._model = YOLO(self.config.model_path)

            LOGGER.info("YOLO model loaded successfully.")

        except Exception as exc:
            LOGGER.exception("Failed to load YOLO model.")
            raise ModelLoadingError(
                f"Unable to load model: {self.config.model_path}"
            ) from exc

    @staticmethod
    def validate_frame(frame: np.ndarray) -> None:
        """Validate a BGR image before inference."""

        if not isinstance(frame, np.ndarray):
            raise InvalidFrameError(
                "Frame must be a NumPy array."
            )

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise InvalidFrameError(
                "Frame must have shape (height, width, 3)."
            )

        if frame.shape[0] == 0 or frame.shape[1] == 0:
            raise InvalidFrameError(
                "Frame dimensions must be non-zero."
            )

        if frame.dtype != np.uint8:
            raise InvalidFrameError(
                "Frame must use uint8 pixel values."
            )

    def detect(
        self,
        frame: np.ndarray,
    ) -> DetectionResult:
        """
        Detect objects in one BGR image.

        Args:
            frame:
                OpenCV-compatible BGR image.

        Returns:
            Structured detection result.

        Raises:
            InvalidFrameError:
                Invalid input image.

            ModelLoadingError:
                YOLO model could not be loaded.

            InferenceError:
                Model inference or result parsing failed.
        """

        self.validate_frame(frame)
        self.load_model()

        start_time = time.perf_counter()

        try:
            predictions = self._model.predict(
                source=frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                imgsz=self.config.image_size,
                device=self.config.device,
                max_det=self.config.max_detections,
                verbose=False,
            )

            if len(predictions) != 1:
                raise InferenceError(
                    "Expected exactly one image result."
                )

            prediction = predictions[0]
            detections: list[Detection] = []

            boxes = prediction.boxes

            if boxes is not None:
                class_names = prediction.names

                for box in boxes:
                    coordinates = (
                        box.xyxy[0]
                        .detach()
                        .cpu()
                        .tolist()
                    )

                    class_id = int(
                        box.cls[0].item()
                    )

                    confidence = float(
                        box.conf[0].item()
                    )

                    if isinstance(class_names, Mapping):
                        label = str(
                            class_names.get(
                                class_id,
                                f"class_{class_id}",
                            )
                        )
                    else:
                        label = str(class_names[class_id])

                    detection = Detection(
                        class_id=class_id,
                        label=label,
                        confidence=confidence,
                        bounding_box=BoundingBox(
                            x1=float(coordinates[0]),
                            y1=float(coordinates[1]),
                            x2=float(coordinates[2]),
                            y2=float(coordinates[3]),
                        ),
                    )

                    detections.append(detection)

            elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000

            result = DetectionResult(
                detections=tuple(detections),
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
                inference_time_ms=round(elapsed_ms, 2),
                model_name=Path(
                    self.config.model_path
                ).name,
            )

            LOGGER.debug(
                "Detected %d objects in %.2f ms.",
                result.object_count,
                result.inference_time_ms,
            )

            return result

        except ObjectDetectionError:
            raise

        except Exception as exc:
            LOGGER.exception(
                "Object detection inference failed."
            )
            raise InferenceError(
                "Unable to process image frame."
            ) from exc

    def annotate_frame(
        self,
        frame: np.ndarray,
        result: DetectionResult,
    ) -> np.ndarray:
        """
        Draw detection boxes and labels on a copy of the frame.

        The original input image is not modified.
        """

        self.validate_frame(frame)

        annotated = frame.copy()

        for detection in result.detections:
            box = detection.bounding_box

            x1 = int(box.x1)
            y1 = int(box.y1)
            x2 = int(box.x2)
            y2 = int(box.y2)

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = (
                f"{detection.label} "
                f"{detection.confidence:.2f}"
            )

            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return annotated

    def close(self) -> None:
        """Release the model reference."""

        self._model = None

        LOGGER.info(
            "Object detector resources released."
        )
