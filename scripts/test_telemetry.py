
"""
AeroMind AI - Telemetry Integration Test
========================================

Module:
    scripts.test_telemetry

Description:
    Connects to the PX4 Software-In-The-Loop (SITL)
    simulator through MAVSDK and validates live
    telemetry collection.

Features:
    - PX4 simulator connection
    - Connection timeout handling
    - Continuous telemetry monitoring
    - GPS position reporting
    - Altitude and velocity reporting
    - Battery status monitoring
    - Armed and airborne state reporting
    - Structured logging
    - Graceful shutdown

Requirements:
    - Python 3.12+
    - MAVSDK-Python 3.x
    - PX4 SITL running
    - AeroMind AI DroneTelemetry module

Usage:
    python -m scripts.test_telemetry

Note:
    This script only monitors simulated telemetry.
    It does not arm, take off, or control the drone.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Any

from mavsdk import System

from flight_control.telemetry import DroneTelemetry


# ============================================================
# Configuration
# ============================================================

SYSTEM_ADDRESS = "udpin://0.0.0.0:14540"

CONNECTION_TIMEOUT_SECONDS = 30.0

TELEMETRY_INTERVAL_SECONDS = 2.0

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)


logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGER = logging.getLogger("AeroMindAI.TelemetryTest")


# ============================================================
# Formatting Utilities
# ============================================================

def format_number(
    value: Any,
    decimal_places: int = 2,
    unit: str = "",
) -> str:
    """
    Format a numeric telemetry value.

    Returns N/A when telemetry has not yet been received.
    """
    if value is None:
        return "N/A"

    try:
        formatted = f"{float(value):.{decimal_places}f}"
        return f"{formatted} {unit}".strip()

    except (TypeError, ValueError, OverflowError):
        return "N/A"


def format_status(value: Any) -> str:
    """
    Convert a boolean telemetry value into a readable status.
    """
    if value is None:
        return "UNKNOWN"

    return "YES" if value else "NO"


def display_telemetry(data: dict[str, Any]) -> None:
    """
    Display the latest drone telemetry snapshot.
    """
    separator = "=" * 58

    print(f"\n{separator}")
    print("AEROMIND AI | PX4 SIMULATION TELEMETRY")
    print(separator)

    print("\n[CONNECTION]")
    print(
        f"Connected       : "
        f"{format_status(data.get('connected'))}"
    )

    print("\n[GPS POSITION]")
    print(
        f"Latitude        : "
        f"{format_number(data.get('latitude'), 7)}"
    )
    print(
        f"Longitude       : "
        f"{format_number(data.get('longitude'), 7)}"
    )

    print("\n[ALTITUDE]")
    print(
        f"Absolute        : "
        f"{format_number(data.get('absolute_altitude_m'), 2, 'm')}"
    )
    print(
        f"Relative        : "
        f"{format_number(data.get('relative_altitude_m'), 2, 'm')}"
    )

    print("\n[VELOCITY]")
    print(
        f"Ground Speed    : "
        f"{format_number(data.get('ground_speed_m_s'), 2, 'm/s')}"
    )
    print(
        f"Vertical Speed  : "
        f"{format_number(data.get('vertical_velocity_m_s'), 2, 'm/s')}"
    )

    print("\n[BATTERY]")
    print(
        f"Battery Level   : "
        f"{format_number(data.get('battery_percent'), 1, '%')}"
    )

    print("\n[FLIGHT STATUS]")
    print(
        f"Armed           : "
        f"{format_status(data.get('armed'))}"
    )
    print(
        f"In Air          : "
        f"{format_status(data.get('in_air'))}"
    )

    print("\n[SYSTEM]")
    print(
        f"Last Update     : "
        f"{data.get('last_update') or 'N/A'}"
    )

    print(separator, flush=True)


# ============================================================
# PX4 Connection
# ============================================================

async def connect_to_px4(drone: System) -> None:
    """
    Connect to PX4 SITL and wait for a valid MAVSDK connection.

    Raises:
        TimeoutError:
            If the simulator is not discovered within
            the configured timeout.
    """
    LOGGER.info(
        "Connecting to PX4 simulator at %s",
        SYSTEM_ADDRESS,
    )

    await asyncio.wait_for(
        drone.connect(system_address=SYSTEM_ADDRESS),
        timeout=CONNECTION_TIMEOUT_SECONDS,
    )

    async def wait_for_connection() -> None:
        async for state in drone.core.connection_state():
            if state.is_connected:
                return

        raise ConnectionError(
            "MAVSDK connection-state stream ended "
            "before a vehicle was discovered."
        )

    await asyncio.wait_for(
        wait_for_connection(),
        timeout=CONNECTION_TIMEOUT_SECONDS,
    )

    LOGGER.info(
        "Successfully connected to PX4 SITL."
    )


# ============================================================
# Telemetry Monitoring
# ============================================================

async def monitor_telemetry(
    telemetry: DroneTelemetry,
    stop_event: asyncio.Event,
) -> None:
    """
    Periodically retrieve and display telemetry snapshots.

    Monitoring continues until a shutdown signal is received.
    """
    LOGGER.info(
        "Telemetry monitoring started. Interval: %.1f seconds.",
        TELEMETRY_INTERVAL_SECONDS,
    )

    while not stop_event.is_set():
        snapshot = telemetry.get_data()

        display_telemetry(snapshot)

        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=TELEMETRY_INTERVAL_SECONDS,
            )

        except asyncio.TimeoutError:
            # Expected: continue monitoring after the interval.
            pass


# ============================================================
# Application Entry Point
# ============================================================

async def main() -> None:
    """
    Run the AeroMind AI telemetry integration test.

    Lifecycle:
        1. Initialise MAVSDK.
        2. Connect to PX4 SITL.
        3. Start telemetry subscriptions.
        4. Display live telemetry.
        5. Stop monitoring on shutdown.
    """
    LOGGER.info(
        "Starting AeroMind AI telemetry integration test."
    )

    stop_event = asyncio.Event()

    loop = asyncio.get_running_loop()

    # Signal handlers are supported in Ubuntu / WSL2.
    for shutdown_signal in (
        signal.SIGINT,
        signal.SIGTERM,
    ):
        try:
            loop.add_signal_handler(
                shutdown_signal,
                stop_event.set,
            )

        except NotImplementedError:
            LOGGER.warning(
                "Signal handler unavailable: %s",
                shutdown_signal,
            )

    drone = System()
    telemetry = DroneTelemetry(drone)

    try:
        await connect_to_px4(drone)

        await telemetry.start()

        LOGGER.info(
            "Telemetry subscriptions activated."
        )

        await monitor_telemetry(
            telemetry,
            stop_event,
        )

    except asyncio.TimeoutError:
        LOGGER.error(
            "Connection timed out after %.0f seconds. "
            "Verify that PX4 SITL is running.",
            CONNECTION_TIMEOUT_SECONDS,
        )

    except ConnectionError as error:
        LOGGER.error(
            "PX4 connection failed: %s",
            error,
        )

    except asyncio.CancelledError:
        LOGGER.info(
            "Telemetry test cancelled."
        )
        raise

    except Exception:
        LOGGER.exception(
            "Unexpected telemetry integration test failure."
        )

    finally:
        LOGGER.info(
            "Stopping AeroMind AI telemetry monitoring..."
        )

        await telemetry.stop()

        LOGGER.info(
            "Telemetry integration test finished."
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        LOGGER.info(
            "Telemetry test interrupted by user."
        )
