
"""
AeroMind AI - PX4 SITL Controller Test.

Connects to PX4 and performs an operator-authorised
takeoff and landing sequence in simulation only.
"""

import asyncio
import logging

from flight_control.drone_controller import (
    DroneController,
    DroneControllerConfig,
)


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    controller = DroneController(
        DroneControllerConfig(
            system_address="udpin://0.0.0.0:14540",
            takeoff_altitude_m=3.0,
        )
    )

    await controller.connect()
    await controller.wait_until_ready()

    print("PX4 connected and position checks passed.")

    confirmation = input(
        "Confirm PX4 SITL simulation is running "
        "and type TAKEOFF to continue: "
    )

    if confirmation != "TAKEOFF":
        print("Takeoff cancelled.")
        return

    try:
        await controller.takeoff()

        print("Airborne. Holding for 5 seconds.")
        await asyncio.sleep(5)

    finally:
        # Only issue landing if telemetry confirms airborne.
        if await controller._is_in_air():
            await controller.land()

    print("Simulation test completed.")


if __name__ == "__main__":
    asyncio.run(main())
