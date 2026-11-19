
"""
AeroMind AI - First Flight Simulation
-------------------------------------

Executes a basic flight sequence using MAVSDK and PX4 SITL.

Flight sequence:
    1. Connect to the PX4 simulator.
    2. Perform takeoff.
    3. Wait for 10 seconds while PX4 controls the vehicle.
    4. Request landing.
    5. Confirm completion.

Environment:
    PX4 SITL + Gazebo X500

Warning:
    Intended exclusively for supervised simulation testing.
"""

from __future__ import annotations

import asyncio
import logging

from flight_control.drone_controller import (
    DroneController,
    DroneControllerError,
)


# ============================================================
# CONFIGURATION
# ============================================================

HOVER_DURATION_SECONDS = 10

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)-8s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("aeromind.first_flight")


# ============================================================
# FIRST FLIGHT
# ============================================================

async def main() -> None:
    """Execute AeroMind AI's first simulated flight."""

    controller = DroneController()
    takeoff_attempted = False

    try:
        # Step 1: Connect to PX4
        logger.info("Connecting to PX4 simulator...")
        await controller.connect()

        # Step 2: Request operator confirmation
        confirmation = await asyncio.to_thread(
            input,
            "Confirm PX4 SITL is running. Type TAKEOFF: ",
        )

        if confirmation.strip() != "TAKEOFF":
            logger.info("Flight cancelled by operator.")
            return

        # Step 3: Take off
        logger.info("Initiating takeoff...")

        takeoff_attempted = True
        await controller.takeoff()

        # Step 4: Wait while the vehicle is airborne
        logger.info(
            "Vehicle airborne. Waiting for %d seconds.",
            HOVER_DURATION_SECONDS,
        )

        await asyncio.sleep(HOVER_DURATION_SECONDS)

        logger.info("Flight duration completed.")

    except DroneControllerError as exc:
        logger.error("Flight controller error: %s", exc)

    except asyncio.CancelledError:
        logger.warning("Flight operation cancelled.")
        raise

    except Exception:
        logger.exception("Unexpected flight error.")

    finally:
        # Step 5: Request landing if takeoff was attempted
        if takeoff_attempted:
            try:
                logger.info("Checking airborne status...")

                if await controller._is_in_air():
                    logger.info("Requesting landing...")
                    await controller.land()
                    logger.info("Landing confirmed.")
                else:
                    logger.info(
                        "Vehicle reports it is not airborne."
                    )

            except Exception:
                logger.exception(
                    "Unable to confirm landing. "
                    "Inspect PX4 and Gazebo immediately."
                )

        logger.info("First flight sequence finished.")


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Flight interrupted by operator.")
