"""ArUco marker detection and heading computation.

Heading convention (matching the spec):
    0°   = marker pointing right
    90°  = marker pointing down
    180° = marker pointing left
    270° = marker pointing up

All angles returned are wrapped into [0, 360).
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
import config

# ---------------------------------------------------------------------------
# Attempt to load the ArUco module from opencv-contrib-python.
# ---------------------------------------------------------------------------
try:
    _aruco = cv2.aruco  # type: ignore[attr-defined]
    # Allow dictionary selection from `project/config.py` (e.g. "DICT_4X4_50").
    try:
        dict_attr = getattr(_aruco, config.ARUCO_DICT_ID)
        _DICT = _aruco.getPredefinedDictionary(dict_attr)
    except Exception:
        _DICT = _aruco.getPredefinedDictionary(_aruco.DICT_4X4_50)
    _PARAMS = _aruco.DetectorParameters()
    try:
        _DETECTOR = _aruco.ArucoDetector(_DICT, _PARAMS)
        _USE_DETECTOR_CLASS = True
    except AttributeError:
        _DETECTOR = None
        _USE_DETECTOR_CLASS = False
except AttributeError:
    _aruco = None
    _DICT = None
    _PARAMS = None
    _DETECTOR = None
    _USE_DETECTOR_CLASS = False


def _wrap360(angle: float) -> float:
    return angle % 360.0


def detect(frame: np.ndarray) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Detect an ArUco marker in *frame* and return ``(rx, ry, heading)``.

    * **rx, ry** – pixel coordinates of the marker centre (mean of corners).
    * **heading** – degrees in [0, 360) using the spec convention.

    Returns ``(None, None, None)`` when no marker is found.
    """
    if _aruco is None:
        return None, None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if _USE_DETECTOR_CLASS and _DETECTOR is not None:
        corners, ids, _ = _DETECTOR.detectMarkers(gray)
    else:
        corners, ids, _ = _aruco.detectMarkers(gray, _DICT, parameters=_PARAMS)

    if ids is None or len(ids) == 0:
        return None, None, None

    # Use the first detected marker.
    c = corners[0][0]  # shape (4, 2): TL, TR, BR, BL

    rx = float(np.mean(c[:, 0]))
    ry = float(np.mean(c[:, 1]))

    # Direction vector from left-mid to right-mid of marker.
    top_mid = (c[0] + c[1]) / 2.0
    bot_mid = (c[3] + c[2]) / 2.0
    right_mid = (c[1] + c[2]) / 2.0
    left_mid = (c[0] + c[3]) / 2.0

    dx = float(right_mid[0] - left_mid[0])
    dy = float(right_mid[1] - left_mid[1])

    # atan2 gives angle from positive-x axis in math convention (CCW).
    # In image coords y increases downward, so we negate dy to stay consistent
    # with the spec's visual convention, then map to [0, 360).
    heading_rad = math.atan2(dy, dx)
    heading_deg = math.degrees(heading_rad)
    heading_deg = _wrap360(heading_deg)

    return rx, ry, heading_deg


def draw_marker(frame: np.ndarray, rx: float, ry: float, heading: float) -> None:
    """Draw a cross and heading annotation on *frame* (in-place)."""
    x, y = int(rx), int(ry)
    cv2.drawMarker(frame, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(
        frame,
        f"hdg={heading:.1f}",
        (x + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )
