
"""
AeroMind AI - Artificial Intelligence Package.

Provides the artificial intelligence capabilities for the
AeroMind autonomous drone platform.

Modules:
    object_detection:
        Computer vision and object recognition.

    object_tracking:
        Tracking detected objects across video frames.

    decision_engine:
        AI-assisted decision support for navigation and
        mission planning.

Design principles:
    - Modular architecture
    - Lazy loading of AI components
    - Separation of perception and flight control
    - Explicit public API
    - Simulation-first development

Safety:
    AI decisions must not directly bypass flight-control
    safety checks or autopilot failsafe mechanisms.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

__author__ = "AeroMind AI Development Team"

__all__ = [
    "ObjectDetector",
    "ObjectTracker",
    "DecisionEngine",
]

_COMPONENTS = {
    "ObjectDetector": (
        ".object_detection",
        "ObjectDetector",
    ),
    "ObjectTracker": (
        ".object_tracking",
        "ObjectTracker",
    ),
    "DecisionEngine": (
        ".decision_engine",
        "DecisionEngine",
    ),
}


if TYPE_CHECKING:
    from .decision_engine import DecisionEngine
    from .object_detection import ObjectDetector
    from .object_tracking import ObjectTracker


def __getattr__(name: str) -> Any:
    """
    Lazily import public AI components.

    Components are imported only when explicitly accessed,
    reducing unnecessary startup overhead.

    Args:
        name:
            Name of the requested public component.

    Returns:
        The requested AI class.

    Raises:
        AttributeError:
            If the requested component is not exported.
    """

    if name not in _COMPONENTS:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

    module_name, class_name = _COMPONENTS[name]

    module = import_module(
        module_name,
        package=__name__,
    )

    component = getattr(module, class_name)

    globals()[name] = component

    return component


def __dir__() -> list[str]:
    """
    Return discoverable package attributes.

    Supports IDE inspection and interactive development.
    """

    return sorted(
        set(globals()) | set(__all__)
    )
