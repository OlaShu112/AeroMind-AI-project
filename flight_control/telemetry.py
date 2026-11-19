
"""
AeroMind AI - Flight Telemetry Module
=====================================

Module:
    flight_control.telemetry

Description:
    Provides asynchronous telemetry monitoring for the AeroMind AI
    autonomous drone simulation platform.

    Integrates with MAVSDK-Python and PX4 SITL to collect, validate,
    and expose flight telemetry to other system components.

Supported telemetry:
    - GPS latitude and longitude
    - Absolute and relative altitude
    - Ground speed and vertical velocity
    - Battery percentage
    - Armed and in-air status
    - MAVSDK connection state
    - Telemetry update timestamps

Architecture:
    PX4 SITL
        |
        v
    MAVSDK System
        |
        v
    DroneTelemetry
        |
        +-- Position monitoring
        +-- Velocity monitoring
        +-- Battery monitoring
        +-- Armed-state monitoring
        +-- In-air monitoring
        +-- Connection monitoring
        |
        v
    TelemetryData
        |
        +-- Safety Manager
        +-- Navigation System
        +-- AI Decision Engine
        +-- Monitoring Dashboard

Notes:
    - This module is intended for simulation development.
    - Telemetry collection does not command the aircraft.
    - Missing telemetry values are represented by None.
    - A connected vehicle does not necessarily have valid GPS data.
"""

from __future__ import annotations

import asyncio
import logging
import math

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from mavsdk import System


LOGGER = logging.getLogger(__name__)


@dataclass
class TelemetryData:
    """
    Represents the latest available drone telemetry.

    Optional fields remain None until their corresponding
    MAVSDK telemetry stream provides a reading.
    """

    # Geographic position
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # Altitude measurements
    absolute_altitude_m: Optional[float] = None
    relative_altitude_m: Optional[float] = None

    # Velocity measurements
    ground_speed_m_s: Optional[float] = None
    vertical_velocity_m_s: Optional[float] = None

    # Battery status
    battery_percent: Optional[float] = None

    # Flight state
    armed: Optional[bool] = None
    in_air: Optional[bool] = None

    # Connection state
    connected: bool = False

    # Update tracking
    last_update: Optional[str] = None

    # Per-stream timestamps for freshness checks
    stream_updates: dict[str, str] = field(
        default_factory=dict
    )


class DroneTelemetry:
    """
    Manages asynchronous telemetry subscriptions for a MAVSDK drone.

    Responsibilities:
        1. Subscribe to PX4 telemetry streams.
        2. Maintain the latest telemetry snapshot.
        3. Monitor connection status.
        4. Handle stream failures and cancellations.
        5. Provide telemetry data to other modules.
        6. Release background tasks during shutdown.

    Example:
        drone = System()
        await drone.connect(
            system_address="udpin://0.0.0.0:14540"
        )

        telemetry = DroneTelemetry(drone)
        await telemetry.start()

        snapshot = telemetry.get_data()

        await telemetry.stop()
    """

    def __init__(self, drone: System) -> None:
        if drone is None:
            raise ValueError(
                "A valid MAVSDK System instance is required."
            )

        self.drone = drone
        self.data = TelemetryData()

        self._tasks: list[asyncio.Task[None]] = []
        self._running = False
        self._stopping = False

        # All subscriptions run on the same asyncio event loop.
        self._lock = asyncio.Lock()

        LOGGER.info(
            "AeroMind AI telemetry manager initialised."
        )

    @staticmethod
    def _timestamp() -> str:
        """Return the current UTC timestamp in ISO 8601 format."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _finite(value: float) -> Optional[float]:
        """
        Return a finite floating-point value.

        NaN and infinity are treated as unavailable telemetry.
        """
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return None

        return result if math.isfinite(result) else None

    async def _update(
        self,
        stream: str,
        **values: Any,
    ) -> None:
        """
        Update telemetry fields atomically within the event loop.

        Also record when the associated stream last produced data.
        """
        timestamp = self._timestamp()

        async with self._lock:
            for name, value in values.items():
                if not hasattr(self.data, name):
                    raise AttributeError(
                        f"Unknown telemetry field: {name}"
                    )

                setattr(self.data, name, value)

            self.data.last_update = timestamp
            self.data.stream_updates[stream] = timestamp

    async def _read_position(self) -> None:
        """Subscribe to geographic position telemetry."""
        async for position in self.drone.telemetry.position():
            latitude = self._finite(position.latitude_deg)
            longitude = self._finite(position.longitude_deg)

            if latitude is not None and not -90 <= latitude <= 90:
                latitude = None

            if longitude is not None and not -180 <= longitude <= 180:
                longitude = None

            await self._update(
                "position",
                latitude=latitude,
                longitude=longitude,
                absolute_altitude_m=self._finite(
                    position.absolute_altitude_m
                ),
                relative_altitude_m=self._finite(
                    position.relative_altitude_m
                ),
            )

    async def _read_velocity(self) -> None:
        """
        Subscribe to velocity in the North-East-Down frame.

        Ground speed is calculated from horizontal components.
        Positive vertical velocity represents upward motion.
        """
        async for velocity in self.drone.telemetry.velocity_ned():
            north = self._finite(velocity.north_m_s)
            east = self._finite(velocity.east_m_s)
            down = self._finite(velocity.down_m_s)

            ground_speed = None

            if north is not None and east is not None:
                ground_speed = math.hypot(north, east)

            vertical_velocity = (
                -down if down is not None else None
            )

            await self._update(
                "velocity",
                ground_speed_m_s=ground_speed,
                vertical_velocity_m_s=vertical_velocity,
            )

    async def _read_battery(self) -> None:
        """Subscribe to battery state telemetry."""
        async for battery in self.drone.telemetry.battery():
            remaining = self._finite(
                battery.remaining_percent
            )

            percentage = None

            if remaining is not None and 0.0 <= remaining <= 1.0:
                percentage = remaining * 100.0

            await self._update(
                "battery",
                battery_percent=percentage,
            )

    async def _read_armed(self) -> None:
        """Monitor whether the vehicle is armed."""
        async for armed in self.drone.telemetry.armed():
            await self._update(
                "armed",
                armed=bool(armed),
            )

    async def _read_in_air(self) -> None:
        """Monitor whether the vehicle is airborne."""
        async for in_air in self.drone.telemetry.in_air():
            await self._update(
                "in_air",
                in_air=bool(in_air),
            )

    async def _read_connection(self) -> None:
        """Monitor MAVSDK vehicle connection state."""
        async for state in self.drone.core.connection_state():
            connected = bool(state.is_connected)

            await self._update(
                "connection",
                connected=connected,
            )

            if connected:
                LOGGER.info(
                    "AeroMind AI connected to PX4."
                )
            else:
                LOGGER.warning(
                    "AeroMind AI vehicle connection unavailable."
                )

    async def _run_stream(
        self,
        name: str,
        reader: Callable[[], Awaitable[None]],
    ) -> None:
        """
        Supervise a telemetry subscription.

        Unexpected failures are logged and retried after a
        short delay. Cancellation stops the subscription.
        """
        retry_delay = 2.0

        while self._running:
            try:
                LOGGER.debug(
                    "Starting telemetry stream: %s",
                    name,
                )

                await reader()

                if self._running:
                    LOGGER.warning(
                        "Telemetry stream ended: %s",
                        name,
                    )

            except asyncio.CancelledError:
                LOGGER.debug(
                    "Telemetry stream cancelled: %s",
                    name,
                )
                raise

            except Exception:
                LOGGER.exception(
                    "Telemetry stream failed: %s",
                    name,
                )

            if self._running:
                await asyncio.sleep(retry_delay)

    async def start(self) -> None:
        """
        Start all telemetry subscriptions.

        Calling start repeatedly while running has no effect.
        """
        if self._stopping:
            raise RuntimeError(
                "Telemetry shutdown is in progress."
            )

        if self._running:
            LOGGER.warning(
                "Telemetry monitoring is already running."
            )
            return

        self._running = True

        streams = {
            "position": self._read_position,
            "velocity": self._read_velocity,
            "battery": self._read_battery,
            "armed": self._read_armed,
            "in_air": self._read_in_air,
            "connection": self._read_connection,
        }

        self._tasks = [
            asyncio.create_task(
                self._run_stream(name, reader),
                name=f"aeromind_telemetry_{name}",
            )
            for name, reader in streams.items()
        ]

        LOGGER.info(
            "AeroMind AI telemetry monitoring started."
        )

    async def stop(self) -> None:
        """
        Cancel subscriptions and release background tasks.

        Safe to call when monitoring is already stopped.
        """
        if self._stopping:
            return

        self._stopping = True
        self._running = False

        try:
            tasks = list(self._tasks)

            for task in tasks:
                task.cancel()

            if tasks:
                await asyncio.gather(
                    *tasks,
                    return_exceptions=True,
                )

            self._tasks.clear()

            LOGGER.info(
                "AeroMind AI telemetry monitoring stopped."
            )

        finally:
            self._stopping = False

    def get_snapshot(self) -> TelemetryData:
        """
        Return an independent copy of current telemetry.

        The returned snapshot can be read without modifying
        the telemetry manager's internal state.

        Call from the same event-loop thread as the manager.
        """
        snapshot = replace(self.data)

        snapshot.stream_updates = dict(
            self.data.stream_updates
        )

        return snapshot

    def get_data(self) -> dict[str, Any]:
        """
        Return telemetry as a serialisable dictionary.

        Maintains compatibility with existing callers that
        expect DroneTelemetry.get_data().
        """
        return asdict(self.get_snapshot())

    def is_running(self) -> bool:
        """Return whether monitoring is active."""
        return self._running

    def is_connected(self) -> bool:
        """Return the most recently reported connection state."""
        return self.data.connected

    def get_battery_level(self) -> Optional[float]:
        """Return the latest battery percentage."""
        return self.data.battery_percent

    def get_altitude(self) -> Optional[float]:
        """Return the latest relative altitude in metres."""
        return self.data.relative_altitude_m

    def get_ground_speed(self) -> Optional[float]:
        """Return the latest horizontal speed in metres/second."""
        return self.data.ground_speed_m_s

    def get_position(
        self,
    ) -> tuple[Optional[float], Optional[float]]:
        """Return the latest latitude and longitude."""
        return (
            self.data.latitude,
            self.data.longitude,
        )

    def is_armed(self) -> Optional[bool]:
        """Return the latest armed status, if available."""
        return self.data.armed

    def is_in_air(self) -> Optional[bool]:
        """Return the latest airborne status, if available."""
        return self.data.in_air
