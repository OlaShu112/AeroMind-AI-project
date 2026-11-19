"""AeroMind AI - First Flight Simulation.

Supervised PX4 SITL / Gazebo X500 flight: connect, verify readiness,
confirm with operator, take off to 3 m, hover for 10 s, and land.
Uses the existing flight_control.drone_controller unchanged.
"""
from __future__ import annotations

import asyncio
import logging
import math

from flight_control.drone_controller import (
    DroneController,
    DroneControllerError,
)

HOVER_DURATION_SECONDS = 10
TARGET_ALTITUDE_M = 3.0
ALTITUDE_TOLERANCE_M = 0.6
CLIMB_TIMEOUT_SECONDS = 30.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("aeromind.first_flight")


async def wait_for_altitude(controller: DroneController) -> None:
    """Wait for the drone to reach near the requested relative altitude."""
    async def observe() -> None:
        async for position in controller.drone.telemetry.position():
            altitude = position.relative_altitude_m
            if math.isfinite(altitude):
                logger.info("Climbing: relative altitude %.2f m", altitude)
                if altitude >= TARGET_ALTITUDE_M - ALTITUDE_TOLERANCE_M:
                    return
        raise DroneControllerError("Position telemetry stream ended during climb.")

    await asyncio.wait_for(observe(), timeout=CLIMB_TIMEOUT_SECONDS)


async def report_flight_status(controller: DroneController) -> None:
    """Display altitude once per second during the hover period."""
    async def sample() -> None:
        async for position in controller.drone.telemetry.position():
            logger.info("Hover altitude: %.2f m", position.relative_altitude_m)
            return

    for _ in range(HOVER_DURATION_SECONDS):
        await asyncio.wait_for(sample(), timeout=5.0)
        await asyncio.sleep(1)


async def main() -> None:
    controller = DroneController()
    takeoff_attempted = False

    try:
        logger.info("Connecting to PX4 simulator...")
        await controller.connect()

        logger.info("Checking position readiness...")
        await controller.wait_until_ready()

        logger.info("Checking battery telemetry and minimum charge...")
        await controller.check_battery()

        confirmation = await asyncio.to_thread(
            input,
            "Gazebo X500 ready? Type TAKEOFF to fly: ",
        )
        if confirmation.strip() != "TAKEOFF":
            logger.info("Flight cancelled by operator.")
            return

        takeoff_attempted = True
        logger.info("Initiating takeoff to %.1f m...", TARGET_ALTITUDE_M)
        await controller.takeoff(altitude_m=TARGET_ALTITUDE_M)

        logger.info("Airborne confirmed; waiting to reach target altitude...")
        await wait_for_altitude(controller)

        logger.info("Hovering for %d seconds...", HOVER_DURATION_SECONDS)
        await report_flight_status(controller)
        logger.info("Hover complete.")

    except (DroneControllerError, asyncio.TimeoutError) as exc:
        logger.error("Flight did not complete: %s", exc)
        if isinstance(exc, asyncio.TimeoutError):
            logger.error("Altitude or telemetry wait timed out.")

    except asyncio.CancelledError:
        logger.warning("Flight operation cancelled.")
        raise

    except Exception:
        logger.exception("Unexpected flight error.")

    finally:
        if takeoff_attempted and controller.is_connected:
            try:
                logger.info("Checking whether landing is necessary...")
                if await controller._is_in_air():
                    logger.info("Requesting controlled landing...")
                    await controller.land()
                    logger.info("Landing confirmed by telemetry.")
                else:
                    logger.info("Vehicle reports it is not airborne.")
            except Exception:
                logger.exception(
                    "Landing could not be confirmed. Inspect PX4/Gazebo immediately."
                )
        logger.info("First flight sequence finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Interrupted by operator. Check drone status in Gazebo.")
