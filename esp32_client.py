"""
esp32_client.py – Sends robot commands to the ESP32 via HTTP GET requests.

Real ESP32_script.ino API
--------------------------
The firmware exposes one endpoint per movement plus a separate speed endpoint:

  GET /forward              – both motors forward
  GET /left                 – motor 1 backward, motor 2 forward  (turn left)
  GET /right                – motor 1 forward,  motor 2 backward (turn right)
  GET /stop                 – all motor pins LOW (coast to stop)
  GET /reverse              – both motors backward
  GET /speed?value=<0-100>  – set PWM duty cycle (slider steps: 0,25,50,75,100)
                              0  → PWM 0   (motors off via enable pins)
                              25 → PWM 200
                              50 → PWM ~220
                              75 → PWM ~237
                              100→ PWM 255

Speed and direction are independent on the ESP32.  This client sends a
/speed request whenever the speed changes and a direction request whenever
the action changes, keeping both in sync with minimal HTTP traffic.

IRL notes
---------
- The ESP32 is in Station (STA) mode – it connects to your WiFi router and
  gets a DHCP address.  Update ESP32_BASE_URL in config.py to that address
  (it's printed on the Serial monitor on every boot).
- ESP32_TIMEOUT is deliberately short (0.5 s) so a missed request doesn't
  stall the control loop.  Increase it on a noisy WiFi link.
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

# Map command tokens → firmware endpoint paths
_ACTION_ENDPOINT = {
    "FORWARD": config.ESP32_EP_FORWARD,
    "LEFT":    config.ESP32_EP_LEFT,
    "RIGHT":   config.ESP32_EP_RIGHT,
    "STOP":    config.ESP32_EP_STOP,
    "REVERSE": config.ESP32_EP_REVERSE,
}


class Esp32Client:
    """
    HTTP client for the ESP32 robot (ESP32_script.ino).

    Speed and direction are tracked independently:
    - A /speed request is only sent when the speed value changes.
    - A direction request (/forward, /left, etc.) is only sent when the
      action changes.
    This keeps the two HTTP calls per command-change, not per frame.

    Parameters
    ----------
    base_url : str
        Base URL of the ESP32 (e.g. "http://192.168.1.42").
    timeout : float
        HTTP request timeout in seconds.
    dry_run : bool
        If True, commands are logged but NOT actually sent (useful for
        testing without hardware).
    """

    def __init__(
        self,
        base_url: str = config.ESP32_BASE_URL,
        timeout: float = config.ESP32_TIMEOUT,
        dry_run: bool = False,
    ):
        if not _REQUESTS_AVAILABLE and not dry_run:
            raise ImportError(
                "The 'requests' library is required.  "
                "Install it with: pip install requests"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self.dry_run  = dry_run
        self._last_action: Optional[str] = None
        self._last_speed:  Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, command: RobotCommand) -> bool:
        """
        Send *command* to the ESP32.

        Returns True if at least one HTTP request was issued (or logged in
        dry-run mode).  Returns False if the command was fully deduplicated
        (nothing new to send) or if all requests failed.
        """
        if command.action == self._last_action and command.speed == self._last_speed:
            return False  # nothing changed

        sent_anything = False

        # 1. Update speed if it changed
        if command.speed != self._last_speed:
            if self._request(f"{config.ESP32_EP_SPEED}?value={command.speed}"):
                self._last_speed = command.speed
                sent_anything = True

        # 2. Update direction if it changed
        if command.action != self._last_action:
            endpoint = _ACTION_ENDPOINT.get(command.action)
            if endpoint is None:
                logger.warning("Unknown action: %s", command.action)
            elif self._request(endpoint):
                self._last_action = command.action
                sent_anything = True

        return sent_anything

    def stop(self) -> bool:
        """Send STOP unconditionally (bypasses deduplication)."""
        self._last_action = None
        self._last_speed  = None
        return self.send(RobotCommand(CMD_STOP, config.SPEED_STOP))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(self, path: str) -> bool:
        """Issue GET <base_url><path>.  Returns True on HTTP 200."""
        url = self.base_url + path
        if self.dry_run:
            logger.info("[DRY-RUN] GET %s", url)
            return True
        try:
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 200:
                logger.debug("ESP32 OK: %s", url)
                return True
            logger.warning("ESP32 HTTP %d: %s", response.status_code, url)
        except requests.exceptions.Timeout:
            logger.warning("ESP32 timeout: %s", url)
        except requests.exceptions.ConnectionError as exc:
            logger.warning("ESP32 connection error (%s): %s", exc, url)
        return False
