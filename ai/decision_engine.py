
"""
AeroMind AI - Autonomous Decision Engine.

Provides deterministic, safety-oriented decision support
for simulated autonomous drone missions.

Responsibilities:
    - Evaluate drone telemetry and sensor health.
    - Assess obstacles and navigation conditions.
    - Recommend mission actions.
    - Explain decisions with reason codes.
    - Avoid issuing flight commands directly.

Important:
    This module is not a certified flight safety system.
    PX4 failsafes and independent safety checks remain
    authoritative.

Supported environment:
    Python 3.12
    PX4 SITL / Gazebo simulation
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence


LOGGER = logging.getLogger("aeromind.ai.decision_engine")


# ============================================================
# ENUMERATIONS
# ============================================================

class DecisionAction(str, Enum):
    """Actions recommended by the decision engine."""

    CONTINUE = "continue"
    HOLD = "hold"
    REPLAN = "replan"
    RETURN_TO_LAUNCH = "return_to_launch"
    LAND = "land"
    REQUEST_OPERATOR = "request_operator"


class DecisionPriority(str, Enum):
    """Decision urgency classification."""

    NORMAL = "normal"
    CAUTION = "caution"
    HIGH = "high"
    CRITICAL = "critical"


class FlightPhase(str, Enum):
    """Current drone flight phase."""

    GROUNDED = "grounded"
    TAKEOFF = "takeoff"
    MISSION = "mission"
    RETURNING = "returning"
    LANDING = "landing"


# ============================================================
# INPUT DATA MODELS
# ============================================================

@dataclass(frozen=True, slots=True)
class DroneState:
    """Validated snapshot of drone telemetry."""

    connected: bool
    armed: bool
    in_air: bool

    battery_percent: float
    altitude_m: float

    gps_healthy: bool
    position_valid: bool

    flight_phase: FlightPhase = FlightPhase.GROUNDED

    def __post_init__(self) -> None:
        if not isinstance(self.flight_phase, FlightPhase):
            raise ValueError("Invalid flight phase.")

        _validate_number(
            self.battery_percent,
            "battery_percent",
            minimum=0,
            maximum=100,
        )

        _validate_number(
            self.altitude_m,
            "altitude_m",
        )


@dataclass(frozen=True, slots=True)
class Obstacle:
    """Obstacle observation from a perception system."""

    identifier: str
    distance_m: float
    confidence: float

    is_on_flight_path: bool = True

    def __post_init__(self) -> None:
        if not self.identifier.strip():
            raise ValueError(
                "Obstacle identifier cannot be empty."
            )

        _validate_number(
            self.distance_m,
            "distance_m",
            minimum=0,
        )

        _validate_number(
            self.confidence,
            "confidence",
            minimum=0,
            maximum=1,
        )


@dataclass(frozen=True, slots=True)
class NavigationState:
    """Current navigation and route-planning status."""

    mission_active: bool
    route_available: bool
    destination_reached: bool = False

    geofence_breached: bool = False

    navigation_healthy: bool = True


@dataclass(frozen=True, slots=True)
class SensorState:
    """Health of systems required for autonomous operation."""

    telemetry_fresh: bool = True
    obstacle_sensor_healthy: bool = True
    navigation_sensor_healthy: bool = True


# ============================================================
# DECISION OUTPUT
# ============================================================

@dataclass(frozen=True, slots=True)
class Decision:
    """Immutable decision recommendation."""

    action: DecisionAction
    priority: DecisionPriority
    reason_code: str
    message: str

    requires_operator_approval: bool = False

    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True, slots=True)
class DecisionEngineConfig:
    """Thresholds used by the decision engine."""

    battery_warning_percent: float = 30.0
    battery_critical_percent: float = 20.0

    obstacle_clearance_m: float = 3.0
    emergency_obstacle_distance_m: float = 1.5

    minimum_detection_confidence: float = 0.5

    maximum_altitude_m: float = 10.0

    require_operator_approval: bool = True

    def __post_init__(self) -> None:
        _validate_number(
            self.battery_warning_percent,
            "battery_warning_percent",
            minimum=0,
            maximum=100,
        )

        _validate_number(
            self.battery_critical_percent,
            "battery_critical_percent",
            minimum=0,
            maximum=100,
        )

        if (
            self.battery_critical_percent
            >= self.battery_warning_percent
        ):
            raise ValueError(
                "Critical battery threshold must be lower "
                "than warning threshold."
            )

        _validate_number(
            self.obstacle_clearance_m,
            "obstacle_clearance_m",
            minimum=0,
        )

        _validate_number(
            self.emergency_obstacle_distance_m,
            "emergency_obstacle_distance_m",
            minimum=0,
        )

        if (
            self.emergency_obstacle_distance_m
            >= self.obstacle_clearance_m
        ):
            raise ValueError(
                "Emergency obstacle distance must be "
                "smaller than clearance distance."
            )

        _validate_number(
            self.minimum_detection_confidence,
            "minimum_detection_confidence",
            minimum=0,
            maximum=1,
        )

        _validate_number(
            self.maximum_altitude_m,
            "maximum_altitude_m",
            minimum=0,
        )


def _validate_number(
    value: float,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    """Reject invalid or non-finite numeric values."""

    if isinstance(value, bool) or not isinstance(
        value, (int, float)
    ):
        raise ValueError(f"{name} must be numeric.")

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    if minimum is not None and value < minimum:
        raise ValueError(
            f"{name} must be at least {minimum}."
        )

    if maximum is not None and value > maximum:
        raise ValueError(
            f"{name} must not exceed {maximum}."
        )


# ============================================================
# DECISION ENGINE
# ============================================================

class DecisionEngine:
    """
    Evaluate autonomous drone conditions.

    The engine returns recommendations only.

    It does not arm, take off, land, or control the drone.
    """

    def __init__(
        self,
        config: DecisionEngineConfig | None = None,
    ) -> None:

        self.config = config or DecisionEngineConfig()

        LOGGER.info(
            "AeroMind decision engine initialised."
        )

    def evaluate(
        self,
        drone: DroneState,
        navigation: NavigationState,
        sensors: SensorState,
        obstacles: Sequence[Obstacle] = (),
    ) -> Decision:
        """
        Evaluate system conditions in priority order.

        Args:
            drone:
                Current drone telemetry.

            navigation:
                Mission and navigation status.

            sensors:
                Sensor and telemetry health.

            obstacles:
                Current obstacle observations.

        Returns:
            A decision recommendation.
        """

        # 1. Connection and telemetry validity
        if not drone.connected or not sensors.telemetry_fresh:
            return self._decision(
                DecisionAction.REQUEST_OPERATOR,
                DecisionPriority.CRITICAL,
                "TELEMETRY_UNAVAILABLE",
                "Drone connection or telemetry is unavailable.",
            )

        # 2. Critical battery condition
        if (
            drone.battery_percent
            <= self.config.battery_critical_percent
        ):
            return self._decision(
                DecisionAction.LAND if drone.in_air
                else DecisionAction.HOLD,
                DecisionPriority.CRITICAL,
                "BATTERY_CRITICAL",
                "Battery level is critically low.",
                battery_percent=drone.battery_percent,
            )

        # 3. Geofence breach
        if navigation.geofence_breached:
            return self._decision(
                DecisionAction.RETURN_TO_LAUNCH
                if drone.in_air
                else DecisionAction.HOLD,
                DecisionPriority.CRITICAL,
                "GEOFENCE_BREACH",
                "Drone has exceeded its configured geofence.",
            )

        # 4. Altitude limit
        if drone.altitude_m > self.config.maximum_altitude_m:
            return self._decision(
                DecisionAction.REQUEST_OPERATOR,
                DecisionPriority.CRITICAL,
                "ALTITUDE_LIMIT_EXCEEDED",
                "Drone altitude exceeds the configured limit.",
                altitude_m=drone.altitude_m,
            )

        # 5. Sensor health
        if (
            not sensors.obstacle_sensor_healthy
            or not sensors.navigation_sensor_healthy
            or not navigation.navigation_healthy
        ):
            return self._decision(
                DecisionAction.REQUEST_OPERATOR,
                DecisionPriority.HIGH,
                "SENSOR_HEALTH_FAILURE",
                "Required autonomous navigation sensors "
                "are unhealthy.",
            )

        # 6. Position validity
        if not drone.position_valid or not drone.gps_healthy:
            return self._decision(
                DecisionAction.REQUEST_OPERATOR,
                DecisionPriority.HIGH,
                "POSITION_UNAVAILABLE",
                "Reliable positioning is unavailable.",
            )

        # 7. Emergency obstacle proximity
        relevant_obstacles = [
            obstacle
            for obstacle in obstacles
            if obstacle.is_on_flight_path
            and obstacle.confidence
            >= self.config.minimum_detection_confidence
        ]

        if relevant_obstacles:
            closest = min(
                relevant_obstacles,
                key=lambda item: item.distance_m,
            )

            if (
                closest.distance_m
                <= self.config.emergency_obstacle_distance_m
            ):
                return self._decision(
                    DecisionAction.HOLD,
                    DecisionPriority.CRITICAL,
                    "OBSTACLE_EMERGENCY",
                    "Obstacle detected within emergency "
                    "clearance distance.",
                    obstacle_id=closest.identifier,
                    distance_m=closest.distance_m,
                )

            if (
                closest.distance_m
                <= self.config.obstacle_clearance_m
            ):
                return self._decision(
                    DecisionAction.REPLAN,
                    DecisionPriority.HIGH,
                    "OBSTACLE_DETECTED",
                    "Obstacle detected along the flight path.",
                    obstacle_id=closest.identifier,
                    distance_m=closest.distance_m,
                )

        # 8. Battery warning
        if (
            drone.battery_percent
            <= self.config.battery_warning_percent
        ):
            return self._decision(
                DecisionAction.RETURN_TO_LAUNCH
                if drone.in_air
                else DecisionAction.HOLD,
                DecisionPriority.HIGH,
                "BATTERY_WARNING",
                "Battery has reached the warning threshold.",
                battery_percent=drone.battery_percent,
            )

        # 9. Grounded state
        if not drone.in_air:
            return self._decision(
                DecisionAction.HOLD,
                DecisionPriority.NORMAL,
                "DRONE_GROUNDED",
                "Drone is on the ground.",
            )

        # 10. Mission completion
        if navigation.destination_reached:
            return self._decision(
                DecisionAction.HOLD,
                DecisionPriority.NORMAL,
                "DESTINATION_REACHED",
                "The configured destination has been reached.",
            )

        # 11. Navigation availability
        if (
            navigation.mission_active
            and not navigation.route_available
        ):
            return self._decision(
                DecisionAction.HOLD,
                DecisionPriority.HIGH,
                "ROUTE_UNAVAILABLE",
                "No validated navigation route is available.",
            )

        # 12. Mission continuation
        if navigation.mission_active:
            return self._decision(
                DecisionAction.CONTINUE,
                DecisionPriority.NORMAL,
                "MISSION_CLEAR",
                "Current mission conditions permit continuation.",
            )

        # 13. Default safe recommendation
        return self._decision(
            DecisionAction.HOLD,
            DecisionPriority.NORMAL,
            "NO_ACTIVE_MISSION",
            "No autonomous mission is currently active.",
        )

    def _decision(
        self,
        action: DecisionAction,
        priority: DecisionPriority,
        reason_code: str,
        message: str,
        **metadata: Any,
    ) -> Decision:
        """Create and log a decision recommendation."""

        decision = Decision(
            action=action,
            priority=priority,
            reason_code=reason_code,
            message=message,
            requires_operator_approval=(
                self.config.require_operator_approval
            ),
            metadata=metadata,
        )

        LOGGER.info(
            "Decision: action=%s priority=%s reason=%s",
            decision.action.value,
            decision.priority.value,
            decision.reason_code,
        )

        return decision
