
"""
AeroMind AI
Autonomous Intelligent Drone System

Application entry point.

Responsibilities:
    - Load and validate configuration
    - Initialise application logging
    - Manage application lifecycle
    - Coordinate system startup and shutdown

Future integrations:
    - PX4 / MAVSDK flight control
    - ROS 2 robotics communication
    - AI object detection
    - Autonomous navigation
    - Safety monitoring
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "settings.yaml"
LOG_DIR = BASE_DIR / "logs"

LOGGER = logging.getLogger("aeromind")


class ConfigurationError(Exception):
    """Raised when application configuration is invalid."""


class AeroMindApplication:
    """Main application lifecycle controller."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config: dict[str, Any] = {}
        self.running = False

    def load_configuration(self) -> None:
        """Load and validate YAML configuration."""

        if not self.config_path.is_file():
            raise ConfigurationError(
                f"Configuration file not found: "
                f"{self.config_path}"
            )

        try:
            with self.config_path.open(
                "r", encoding="utf-8"
            ) as file:
                data = yaml.safe_load(file)

        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Invalid YAML configuration: {exc}"
            ) from exc

        except OSError as exc:
            raise ConfigurationError(
                f"Cannot read configuration: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ConfigurationError(
                "Configuration must be a YAML mapping."
            )

        for section in ("project", "drone", "safety"):
            if not isinstance(data.get(section), dict):
                raise ConfigurationError(
                    f"Missing or invalid section: {section}"
                )

        project = data["project"]

        if not isinstance(project.get("name"), str):
            raise ConfigurationError(
                "project.name must be a string."
            )

        if project.get("environment") != "simulation":
            raise ConfigurationError(
                "Only simulation mode is currently supported."
            )

        self.config = data

        LOGGER.info("Configuration loaded successfully.")

    def initialise_system(self) -> None:
        """Initialise currently implemented services."""

        LOGGER.info("Initialising AeroMind AI...")

        LOGGER.info(
            "Project: %s",
            self.config["project"]["name"]
        )

        LOGGER.info(
            "Environment: %s",
            self.config["project"]["environment"]
        )

        LOGGER.info(
            "Flight controller integration: pending"
        )

        LOGGER.info(
            "AI perception integration: pending"
        )

        LOGGER.info(
            "Navigation integration: pending"
        )

        LOGGER.info(
            "Safety manager integration: pending"
        )

    def start(self) -> None:
        """Start the application."""

        self.load_configuration()
        self.initialise_system()

        self.running = True

        LOGGER.info(
            "AeroMind AI application initialised."
        )

        LOGGER.info(
            "No flight commands will be issued."
        )

    def shutdown(self) -> None:
        """Shut down application services."""

        if not self.running:
            return

        LOGGER.info("Shutting down AeroMind AI...")

        # Future: stop missions and close connections.
        # Actual flight failsafes remain PX4's responsibility.

        self.running = False

        LOGGER.info("Application shutdown complete.")


def configure_logging() -> None:
    """Configure console and rotating file logging."""

    from logging.handlers import RotatingFileHandler

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | "
        "%(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_DIR / "aeromind.log",
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.addHandler(console_handler)
    LOGGER.addHandler(file_handler)
    LOGGER.propagate = False


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "AeroMind AI Autonomous Drone System"
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to the YAML configuration file",
    )

    return parser.parse_args()


def main() -> int:
    """Application entry point."""

    configure_logging()
    args = parse_arguments()

    application = AeroMindApplication(
        config_path=args.config
    )

    try:
        LOGGER.info("=" * 50)
        LOGGER.info("AEROMIND AI")
        LOGGER.info("Autonomous Intelligent Drone System")
        LOGGER.info("=" * 50)

        application.start()
        return 0

    except ConfigurationError as exc:
        LOGGER.error("Configuration error: %s", exc)
        return 1

    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user.")
        return 130

    except Exception:
        LOGGER.exception(
            "Unexpected application failure."
        )
        return 1

    finally:
        application.shutdown()


if __name__ == "__main__":
    sys.exit(main())
