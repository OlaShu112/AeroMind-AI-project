
"""
AeroMind AI - Drone Flight Controller.

Asynchronous PX4 flight-control interface using MAVSDK.

Responsibilities:
    - Establish and monitor autopilot connectivity
    - Validate telemetry before flight
    - Arm and disarm the vehicle
    - Execute takeoff and landing commands
    - Monitor flight-state transitions
    - Handle connection and command failures
    - Provide structured logging

Designed initially for PX4 SITL with Gazebo.

Important:
    This controller is not a complete flight-safety system.
    Real-world deployment requires independent safety checks,
    operational limits, hardware testing and operator oversight.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Callable, TypeVar

from mavsdk import System
from mavsdk.action import ActionError


LOGGER = logging.getLogger("aeromind.flight_control.drone_controller")

T = TypeVar("T")


# ============================================================
# EXCEPTIONS
# ============================================================

class DroneControllerError(RuntimeError):
    """Base exception for drone-controller failures."""


class DroneConnectionError(DroneControllerError):
    """Raised when the autopilot connection fails."""


class DronePreflightError(DroneControllerError):
    """Raised when required preflight conditions are not met."""


class DroneCommandError(DroneControllerError):
    """Raised when a flight command fails."""


class DroneTimeoutError(DroneControllerError):
    """Raised when an operation exceeds its timeout."""


# ============================================================
# FLIGHT STATES
# ============================================================

class FlightState(str, Enum):
    """High-level controller flight states."""

    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    READY = "ready"
    ARMED = "armed"
    TAKING_OFF = "taking_off"
    IN_AIR = "in_air"
    LANDING = "landing"
    LANDED = "landed"
    ERROR = "error"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True, slots=True)
class DroneControllerConfig:
    """Flight-controller configuration."""

    system_address: str = "udpin://0.0.0.0:14540"

    takeoff_altitude_m: float = 3.0
    maximum_takeoff_altitude_m: float = 10.0

    connection_timeout_s: float = 30.0
    health_timeout_s: float = 60.0
    command_timeout_s: float = 15.0
    takeoff_timeout_s: float = 45.0
    landing_timeout_s: float = 90.0

    minimum_battery_percent: float = 30.0

    require_global_position: bool = False
    require_home_position: bool = False

    simulation_only: bool = True

    def __post_init__(self) -> None:
        if not self.system_address.strip():
            raise ValueError("system_address cannot be empty.")

        positive_values = (
            "takeoff_altitude_m",
            "maximum_takeoff_altitude_m",
            "connection_timeout_s",
            "health_timeout_s",
            "command_timeout_s",
            "takeoff_timeout_s",
            "landing_timeout_s",
        )

        for name in positive_values:
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a finite positive number."
                )

        if (
            self.takeoff_altitude_m
            > self.maximum_takeoff_altitude_m
        ):
            raise ValueError(
                "Takeoff altitude exceeds configured maximum."
            )

        if not (
            0 <= self.minimum_battery_percent <= 100
        ):
            raise ValueError(
                "minimum_battery_percent must be 0-100."
            )


# ============================================================
# DRONE CONTROLLER
# ============================================================

class DroneController:
    """
    Asynchronous MAVSDK drone controller.

    Intended for controlled PX4 simulation workflows.

    Flight operations are serialised using an asyncio lock
    to prevent overlapping takeoff/landing commands.
    """

    def __init__(
        self,
        config: DroneControllerConfig | None = None,
        drone: System | None = None,
    ) -> None:

        self.config = config or DroneControllerConfig()

        self.drone = drone or System()

        self._state = FlightState.DISCONNECTED
        self._connected = False

        self._operation_lock = asyncio.Lock()

        LOGGER.info(
            "Drone controller initialised for %s",
            self.config.system_address,
        )

    @property
    def state(self) -> FlightState:
        return self._state

    @property
    def is_connected(self) -> bool:
        return self._connected

    # --------------------------------------------------------
    # TELEMETRY HELPERS
    # --------------------------------------------------------

    async def _wait_for(
        self,
        stream_factory: Callable[[], AsyncIterator[T]],
        predicate: Callable[[T], bool],
        timeout_s: float,
        description: str,
    ) -> T:
        """
        Wait for a telemetry stream value satisfying a condition.

        Uses an independent stream subscription per call.
        """

        async def consume() -> T:
            async for item in stream_factory():
                if predicate(item):
                    return item

            raise DroneControllerError(
                f"Telemetry stream ended: {description}"
            )

        try:
            return await asyncio.wait_for(
                consume(),
                timeout=timeout_s,
            )

        except asyncio.TimeoutError as exc:
            raise DroneTimeoutError(
                f"Timed out waiting for {description}."
            ) from exc

    async def _is_armed(self) -> bool:
        result = await self._wait_for(
            self.drone.telemetry.armed,
            lambda _: True,
            self.config.command_timeout_s,
            "armed telemetry",
        )
        return bool(result)

    async def _is_in_air(self) -> bool:
        result = await self._wait_for(
            self.drone.telemetry.in_air,
            lambda _: True,
            self.config.command_timeout_s,
            "in-air telemetry",
        )
        return bool(result)

    # --------------------------------------------------------
    # CONNECTION
    # --------------------------------------------------------

    async def connect(self) -> None:
        """Connect to PX4 and wait for connection confirmation."""

        if self._connected:
            LOGGER.info("Drone already connected.")
            return

        LOGGER.info(
            "Connecting to PX4: %s",
            self.config.system_address,
        )

        try:
            await asyncio.wait_for(
                self.drone.connect(
                    system_address=self.config.system_address
                ),
                timeout=self.config.connection_timeout_s,
            )

            await self._wait_for(
                self.drone.core.connection_state,
                lambda state: state.is_connected,
                self.config.connection_timeout_s,
                "autopilot connection",
            )

            self._connected = True
            self._state = FlightState.CONNECTED

            LOGGER.info("PX4 connection established.")

        except Exception as exc:
            self._connected = False
            self._state = FlightState.ERROR

            LOGGER.exception("PX4 connection failed.")

            raise DroneConnectionError(
                "Unable to establish PX4 connection."
            ) from exc

    # --------------------------------------------------------
    # PREFLIGHT VALIDATION
    # --------------------------------------------------------

    async def wait_until_ready(self) -> None:
        """Wait until the configured position checks pass."""

        self._require_connection()

        LOGGER.info("Waiting for healthy position telemetry.")

        def is_ready(health: Any) -> bool:
            if not health.is_local_position_ok:
                return False

            if (
                self.config.require_global_position
                and not health.is_global_position_ok
            ):
                return False

            if (
                self.config.require_home_position
                and not health.is_home_position_ok
            ):
                return False

            return True

        try:
            await self._wait_for(
                self.drone.telemetry.health,
                is_ready,
                self.config.health_timeout_s,
                "healthy position estimate",
            )

            self._state = FlightState.READY

            LOGGER.info("Position health checks passed.")

        except DroneTimeoutError as exc:
            raise DronePreflightError(
                "Position estimate did not become healthy."
            ) from exc

    async def check_battery(self) -> None:
        """Check the battery level before arming."""

        self._require_connection()

        battery = await self._wait_for(
            self.drone.telemetry.battery,
            lambda _: True,
            self.config.command_timeout_s,
            "battery telemetry",
        )

        battery_percent = (
            float(battery.remaining_percent) * 100
        )

        if not math.isfinite(battery_percent):
            raise DronePreflightError(
                "Battery percentage is invalid."
            )

        LOGGER.info(
            "Battery remaining: %.1f%%",
            battery_percent,
        )

        if (
            battery_percent
            < self.config.minimum_battery_percent
        ):
            raise DronePreflightError(
                "Battery level is below the minimum "
                "required for takeoff."
            )

    # --------------------------------------------------------
    # ARMING
    # --------------------------------------------------------

    async def arm(self) -> None:
        """Arm the drone after required checks."""

        async with self._operation_lock:
            await self._arm_unlocked()

    async def _arm_unlocked(self) -> None:
        self._require_connection()

        if await self._is_in_air():
            raise DroneCommandError(
                "Cannot arm: vehicle is already airborne."
            )

        if await self._is_armed():
            LOGGER.info("Vehicle is already armed.")
            self._state = FlightState.ARMED
            return

        await self.wait_until_ready()
        await self.check_battery()

        LOGGER.info("Arming vehicle.")

        try:
            await asyncio.wait_for(
                self.drone.action.arm(),
                timeout=self.config.command_timeout_s,
            )

            await self._wait_for(
                self.drone.telemetry.armed,
                lambda armed: armed is True,
                self.config.command_timeout_s,
                "armed confirmation",
            )

            self._state = FlightState.ARMED

            LOGGER.info("Vehicle armed successfully.")

        except (ActionError, asyncio.TimeoutError) as exc:
            self._state = FlightState.ERROR

            raise DroneCommandError(
                "Vehicle arming failed."
            ) from exc

    # --------------------------------------------------------
    # TAKEOFF
    # --------------------------------------------------------

    async def takeoff(
        self,
        altitude_m: float | None = None,
    ) -> None:
        """
        Arm and take off to the configured altitude.

        A takeoff command is issued only after
        connection and preflight checks.
        """

        async with self._operation_lock:
            self._require_connection()

            target_altitude = (
                self.config.takeoff_altitude_m
                if altitude_m is None
                else altitude_m
            )

            if (
                not math.isfinite(target_altitude)
                or target_altitude <= 0
                or target_altitude
                > self.config.maximum_takeoff_altitude_m
            ):
                raise DroneCommandError(
                    "Requested takeoff altitude is invalid."
                )

            if await self._is_in_air():
                raise DroneCommandError(
                    "Takeoff rejected: vehicle already airborne."
                )

            await self._arm_unlocked()

            LOGGER.info(
                "Setting takeoff altitude to %.2f m.",
                target_altitude,
            )

            try:
                await asyncio.wait_for(
                    self.drone.action.set_takeoff_altitude(
                        target_altitude
                    ),
                    timeout=self.config.command_timeout_s,
                )

                self._state = FlightState.TAKING_OFF

                LOGGER.info("Sending takeoff command.")

                await asyncio.wait_for(
                    self.drone.action.takeoff(),
                    timeout=self.config.command_timeout_s,
                )

                await self._wait_for(
                    self.drone.telemetry.in_air,
                    lambda in_air: in_air is True,
                    self.config.takeoff_timeout_s,
                    "airborne confirmation",
                )

                self._state = FlightState.IN_AIR

                LOGGER.info(
                    "Vehicle confirmed airborne."
                )

            except (
                ActionError,
                DroneTimeoutError,
                asyncio.TimeoutError,
            ) as exc:
                self._state = FlightState.ERROR

                LOGGER.exception(
                    "Takeoff command or confirmation failed."
                )

                raise DroneCommandError(
                    "Takeoff could not be confirmed. "
                    "Check vehicle telemetry before "
                    "issuing further commands."
                ) from exc

    # --------------------------------------------------------
    # LANDING
    # --------------------------------------------------------

    async def land(self) -> None:
        """Command landing and wait for landed confirmation."""

        async with self._operation_lock:
            self._require_connection()

            if not await self._is_in_air():
                LOGGER.info(
                    "Vehicle is already on the ground."
                )
                self._state = FlightState.LANDED
                return

            LOGGER.info("Sending landing command.")

            try:
                self._state = FlightState.LANDING

                await asyncio.wait_for(
                    self.drone.action.land(),
                    timeout=self.config.command_timeout_s,
                )

                await self._wait_for(
                    self.drone.telemetry.in_air,
                    lambda in_air: in_air is False,
                    self.config.landing_timeout_s,
                    "landed confirmation",
                )

                self._state = FlightState.LANDED

                LOGGER.info(
                    "Vehicle confirmed landed."
                )

            except (
                ActionError,
                DroneTimeoutError,
                asyncio.TimeoutError,
            ) as exc:
                self._state = FlightState.ERROR

                LOGGER.exception(
                    "Landing failed or could not be confirmed."
                )

                raise DroneCommandError(
                    "Landing could not be confirmed. "
                    "Inspect telemetry and PX4 state."
                ) from exc

    # --------------------------------------------------------
    # DISARMING
    # --------------------------------------------------------

    async def disarm(self) -> None:
        """Disarm the vehicle only when confirmed on the ground."""

        async with self._operation_lock:
            self._require_connection()

            if await self._is_in_air():
                raise DroneCommandError(
                    "Disarming while airborne is prohibited."
                )

            if not await self._is_armed():
                LOGGER.info("Vehicle already disarmed.")
                return

            try:
                await asyncio.wait_for(
                    self.drone.action.disarm(),
                    timeout=self.config.command_timeout_s,
                )

                await self._wait_for(
                    self.drone.telemetry.armed,
                    lambda armed: armed is False,
                    self.config.command_timeout_s,
                    "disarmed confirmation",
                )

                self._state = FlightState.LANDED

                LOGGER.info("Vehicle disarmed.")

            except (
                ActionError,
                DroneTimeoutError,
                asyncio.TimeoutError,
            ) as exc:
                raise DroneCommandError(
                    "Vehicle disarming failed."
                ) from exc

    # --------------------------------------------------------
    # INTERNAL VALIDATION
    # --------------------------------------------------------

    def _require_connection(self) -> None:
        if not self._connected:
            raise DroneConnectionError(
                "Drone is not connected. "
                "Call connect() first."
            )
