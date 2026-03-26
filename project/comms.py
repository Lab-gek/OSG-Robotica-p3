"""HTTP command sender to ESP32."""

from __future__ import annotations

import warnings

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False
    warnings.warn("'requests' not installed – HTTP commands will be disabled.", stacklevel=1)
import config

# Dry-run mode: when True, HTTP requests are not sent and are logged instead.
DRY_RUN: bool = False

# Default base URL; overridden by main.py at runtime.  Use config defaults when available.
ESP32_IP: str = getattr(config, "ESP32_IP", getattr(config, "ESP32_BASE_URL", "http://192.168.1.42"))
_TIMEOUT: float = getattr(config, "ESP32_TIMEOUT", 0.3)  # seconds – keep short so we don't block the frame loop

_last_speed_pct: int | None = None  # avoid redundant /speed calls


def _get(url: str) -> None:
    """Fire-and-forget GET with graceful error handling."""
    # In dry-run mode we don't attempt network calls, but log them instead.
    if DRY_RUN:
        print(f"[comms] [DRY-RUN] GET {url}")
        return

    if not _REQUESTS_AVAILABLE:
        return

    try:
        _requests.get(url, timeout=_TIMEOUT)
    except Exception as exc:  # network unreachable, timeout, etc.
        print(f"[comms] WARNING: {exc}")


def send_motors(left: int, right: int) -> None:
    """Translate left/right PWM values into ESP32 endpoint calls.

    Parameters
    ----------
    left, right:
        Motor PWM values in [-255, 255].  Negative values mean reverse.
    """
    global _last_speed_pct

    avg = (left + right) / 2.0
    avg_abs = (abs(left) + abs(right)) / 2.0
    speed_pct = int(round(avg_abs / 255.0 * 100))

    # Determine direction endpoint.
    if avg_abs < 10:
        direction = "/stop"
    elif left < 0 and right < 0:
        direction = "/reverse"
    elif (left - right) > 30:
        direction = "/right"
    elif (right - left) > 30:
        direction = "/left"
    else:
        direction = "/forward"

    _get(f"{ESP32_IP}{direction}")

    # Only send /speed when the value changes to reduce chatter.
    if speed_pct != _last_speed_pct:
        _get(f"{ESP32_IP}/speed?value={speed_pct}")
        _last_speed_pct = speed_pct


def stop() -> None:
    """Immediately stop the robot."""
    global _last_speed_pct
    _get(f"{ESP32_IP}/stop")
    _get(f"{ESP32_IP}/speed?value=0")
    _last_speed_pct = 0
