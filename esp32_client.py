"""
esp32_client.py – Sends robot commands to the ESP32 via HTTP GET requests.

ESP32 firmware API (expected on the microcontroller side)
---------------------------------------------------------
GET /command?action=FORWARD&speed=60
GET /command?action=LEFT&speed=50
GET /command?action=RIGHT&speed=50
GET /command?action=STOP&speed=0

The ESP32 should respond with HTTP 200 on success.

IRL translation notes
---------------------
- If the ESP32 is in Access-Point mode, connect the PC running this code to
  the ESP32's WiFi network and use the default ESP32_BASE_URL = "http://192.168.4.1".
- If both devices are on the same local WiFi (Station mode), update ESP32_BASE_URL
  in config.py to the ESP32's DHCP-assigned IP address.
- ESP32_TIMEOUT (config.py) is deliberately short (0.5 s) so that a slow/missed
  request doesn't block the control loop.  Increase it if your network is lossy.
- Commands are deduplicated: the same command is NOT re-sent every frame to avoid
  flooding the ESP32.  A new HTTP request is only triggered when the action or
  speed changes.
"""

import logging
from typing import Optional

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

import config
from robot_controller import RobotCommand, CMD_STOP

logger = logging.getLogger(__name__)


class Esp32Client:
    """
    Thin HTTP client for the ESP32 robot controller.

    Parameters
    ----------
    base_url : str
        Base URL of the ESP32 (e.g. "http://192.168.4.1").
    timeout : float
        HTTP request timeout in seconds.
    dry_run : bool
        If True, commands are logged but NOT actually sent (useful for testing
        without hardware).
    """

    def __init__(
        self,
        base_url: str = config.ESP32_BASE_URL,
        timeout: float = config.ESP32_TIMEOUT,
        dry_run: bool = False,
    ):
        if not _REQUESTS_AVAILABLE and not dry_run:
            raise ImportError(
                "The 'requests' library is required.  Install it with: pip install requests"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self.dry_run  = dry_run
        self._last_command: Optional[RobotCommand] = None

    def send(self, command: RobotCommand) -> bool:
        """
        Send *command* to the ESP32.

        Returns True if the command was sent (or acknowledged in dry-run mode),
        False if skipped (duplicate) or if the request failed.

        Duplicate suppression: if the new command is identical to the last
        successfully sent command, the request is skipped.
        """
        # Deduplicate
        if command == self._last_command:
            return False

        url = (
            f"{self.base_url}{config.ESP32_COMMAND_ENDPOINT}"
            f"?action={command.action}&speed={command.speed}"
        )

        if self.dry_run:
            logger.info("[DRY-RUN] Would send: %s", url)
            self._last_command = command
            return True

        try:
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 200:
                logger.debug("ESP32 OK: %s", url)
                self._last_command = command
                return True
            else:
                logger.warning(
                    "ESP32 returned HTTP %d for %s", response.status_code, url
                )
        except requests.exceptions.Timeout:
            logger.warning("ESP32 request timed out: %s", url)
        except requests.exceptions.ConnectionError as exc:
            logger.warning("ESP32 connection error: %s", exc)

        return False

    def stop(self) -> bool:
        """Convenience method: send a STOP command unconditionally."""
        # Bypass deduplication for STOP (safety)
        self._last_command = None
        return self.send(RobotCommand(CMD_STOP, config.SPEED_STOP))
