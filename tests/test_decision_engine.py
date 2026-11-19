
"""Unit tests for AeroMind AI decision engine."""

from ai.decision_engine import (
    DecisionAction,
    DecisionEngine,
    DroneState,
    FlightPhase,
    NavigationState,
    Obstacle,
    SensorState,
)


def make_drone(
    battery_percent: float = 80.0,
) -> DroneState:
    return DroneState(
        connected=True,
        armed=True,
        in_air=True,
        battery_percent=battery_percent,
        altitude_m=3.0,
        gps_healthy=True,
        position_valid=True,
        flight_phase=FlightPhase.MISSION,
    )


def make_navigation() -> NavigationState:
    return NavigationState(
        mission_active=True,
        route_available=True,
    )


def test_normal_mission_continues():
    engine = DecisionEngine()

    result = engine.evaluate(
        drone=make_drone(),
        navigation=make_navigation(),
        sensors=SensorState(),
    )

    assert result.action == DecisionAction.CONTINUE


def test_critical_battery_recommends_landing():
    engine = DecisionEngine()

    result = engine.evaluate(
        drone=make_drone(battery_percent=15),
        navigation=make_navigation(),
        sensors=SensorState(),
    )

    assert result.action == DecisionAction.LAND


def test_obstacle_requests_replanning():
    engine = DecisionEngine()

    result = engine.evaluate(
        drone=make_drone(),
        navigation=make_navigation(),
        sensors=SensorState(),
        obstacles=[
            Obstacle(
                identifier="obstacle-001",
                distance_m=2.5,
                confidence=0.95,
            )
        ],
    )

    assert result.action == DecisionAction.REPLAN


def test_emergency_obstacle_recommends_hold():
    engine = DecisionEngine()

    result = engine.evaluate(
        drone=make_drone(),
        navigation=make_navigation(),
        sensors=SensorState(),
        obstacles=[
            Obstacle(
                identifier="obstacle-002",
                distance_m=1.0,
                confidence=0.99,
            )
        ],
    )

    assert result.action == DecisionAction.HOLD


def test_missing_telemetry_requests_operator():
    engine = DecisionEngine()

    result = engine.evaluate(
        drone=make_drone(),
        navigation=make_navigation(),
        sensors=SensorState(
            telemetry_fresh=False
        ),
    )

    assert result.action == DecisionAction.REQUEST_OPERATOR
