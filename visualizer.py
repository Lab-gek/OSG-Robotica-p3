"""
visualizer.py – Builds a composite debug window that shows everything the
control system "sees" during each frame.

Layout (single window)
-----------------------

  ┌────────────────────────┬────────────────────────┐
  │  Camera + overlays     │  HSV map               │
  │  (line contour,        │  (V channel + detected │
  │   ArUco marker,        │   pixels in green,     │
  │   heading arrow,       │   HSV range shown)     │
  │   nearest line point,  │                        │
  │   local line heading)  │                        │
  └────────────────────────┴────────────────────────┘
  │  Status bar: CMD | Speed | FPS | Line error | ArUco heading | Local heading │
  └──────────────────────────────────────────────────────────────────────────────┘

Pass the two detector results and the current command to
``DebugVisualizer.build_frame()`` to get a single BGR image ready for
``cv2.imshow()``.
"""

import math
from typing import Optional

import cv2
import numpy as np

import config
from line_detector import LineDetectionResult, LineDetector
from aruco_tracker import ArucoDetectionResult, ArucoTracker
from robot_controller import RobotCommand

# Colour palette (BGR)
_WHITE  = (255, 255, 255)
_GREEN  = (0, 220, 0)
_ORANGE = (0, 140, 255)
_RED    = (0, 0, 220)
_PURPLE = (255, 0, 255)
_CYAN   = (255, 220, 0)
_DARK   = (30, 30, 30)
_GREY   = (100, 100, 100)


class DebugVisualizer:
    """
    Composes a rich debug frame that shows both the annotated camera view
    and the raw binary line-detection mask side-by-side, plus a status bar.

    Parameters
    ----------
    scale : float
        Scale factor applied to the full composite image before display.
        Useful if the composite (2× frame width) is too large for your screen;
        set to 0.6 or 0.5 in config.py (DEBUG_WINDOW_SCALE).
    """

    WINDOW_TITLE = "Robot Vision – Debug"

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def build_frame(
        self,
        frame: np.ndarray,
        line_result: LineDetectionResult,
        aruco_result: ArucoDetectionResult,
        command: RobotCommand,
        fps: float = 0.0,
    ) -> np.ndarray:
        """
        Build and return a composite BGR debug frame.

        Parameters
        ----------
        frame : np.ndarray
            Raw BGR camera frame.
        line_result : LineDetectionResult
            Output of ``LineDetector.detect()``.
        aruco_result : ArucoDetectionResult
            Output of ``ArucoTracker.detect()``.
        command : RobotCommand
            The command computed for this frame.
        fps : float
            Current frames-per-second (shown in the status bar).

        Returns
        -------
        np.ndarray
            Composite BGR image (twice the original width + status strip).
        """
        h, w = frame.shape[:2]

        # Left panel: annotated camera view
        annotated = LineDetector.draw_debug(frame, line_result)
        annotated = ArucoTracker.draw_debug(annotated, aruco_result)

        # Compute nearest contour point and local line heading when both
        # the ArUco marker and line are visible.  These are drawn on the
        # annotated view so the operator can see how the controller is
        # interpreting corners.
        local_heading: Optional[float] = None
        if (
            aruco_result.found
            and line_result.found
            and line_result.contour is not None
            and aruco_result.centre_x is not None
            and aruco_result.centre_y is not None
            and aruco_result.heading_deg is not None
        ):
            nearest_x, nearest_y = LineDetector.nearest_point_on_contour(
                line_result.contour, aruco_result.centre_x, aruco_result.centre_y
            )
            local_heading = LineDetector.local_line_heading(
                line_result.contour, nearest_x, nearest_y, aruco_result.heading_deg
            )
            self._draw_nearest_point(
                annotated,
                nearest_x, nearest_y,
                local_heading,
                aruco_result.centre_x, aruco_result.centre_y,
            )

        # Right panel: HSV map (Value channel + detection overlay + range info)
        mask_panel = self._build_hsv_map_panel(frame, line_result, h, w)

        # Combine side-by-side
        combined = np.hstack([annotated, mask_panel])

        # Status bar below
        status = self._build_status_bar(
            combined.shape[1], command, fps, line_result, aruco_result, local_heading
        )
        composite = np.vstack([combined, status])

        # Optional scale-down for smaller screens
        if self.scale != 1.0:
            new_w = int(composite.shape[1] * self.scale)
            new_h = int(composite.shape[0] * self.scale)
            composite = cv2.resize(composite, (new_w, new_h))

        return composite

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_hsv_map_panel(
        frame: np.ndarray,
        line_result: LineDetectionResult,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Build the right-hand HSV-map panel (h × w, BGR).

        Shows the Value (brightness) channel of the current frame so the
        operator can see which pixels are dark enough to be the black line.
        The detected line pixels are highlighted in green and the active
        HSV threshold range is printed at the top of the panel.
        """
        # Derive the Value channel – low V = dark = black line
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        panel = cv2.cvtColor(v_channel, cv2.COLOR_GRAY2BGR)

        # Overlay detected pixels in green
        if line_result.mask is not None:
            panel[line_result.mask > 0] = _GREEN

        # Panel title
        cv2.putText(
            panel, "HSV Map (V channel + line mask)",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1,
        )

        # Active HSV threshold range
        lo = config.LINE_HSV_LOWER
        hi = config.LINE_HSV_UPPER
        cv2.putText(
            panel,
            f"H:{lo[0]}-{hi[0]}  S:{lo[1]}-{hi[1]}  V:{lo[2]}-{hi[2]}",
            (8, 38),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, _CYAN, 1,
        )

        # Centroid cross-hair (if line was found)
        if line_result.found and line_result.centroid_x is not None:
            cx, cy = line_result.centroid_x, line_result.centroid_y
            cv2.circle(panel, (cx, cy), 8, _CYAN, 2)
            cv2.line(panel, (cx - 14, cy), (cx + 14, cy), _CYAN, 1)
            cv2.line(panel, (cx, cy - 14), (cx, cy + 14), _CYAN, 1)

        # Vertical dashed line at frame centre (steering reference)
        cx_ref = w // 2
        for y in range(0, h, 14):
            cv2.line(panel, (cx_ref, y), (cx_ref, min(y + 7, h - 1)), _GREY, 1)

        return panel

    @staticmethod
    def _draw_nearest_point(
        vis: np.ndarray,
        nearest_x: float,
        nearest_y: float,
        local_heading: float,
        robot_x: float,
        robot_y: float,
    ) -> None:
        """Draw the nearest line point and local line heading on the annotated view.

        Draws:
        - A diamond marker at the nearest contour point.
        - A dashed line from the robot centre to that point (shows lateral
          displacement clearly).
        - An arrow at the nearest point in the direction of the local line
          heading (shows the controller's corner-aware target direction).
        """
        nx, ny = int(nearest_x), int(nearest_y)
        rx, ry = int(robot_x), int(robot_y)

        # Line from robot centre to nearest contour point
        cv2.line(vis, (rx, ry), (nx, ny), _ORANGE, 1, cv2.LINE_AA)

        # Diamond marker at the nearest point
        cv2.drawMarker(vis, (nx, ny), _ORANGE, cv2.MARKER_DIAMOND, 12, 2)

        # Local line heading arrow at the nearest point
        arrow_len = 40
        heading_rad = math.radians(local_heading)
        ax = int(nx + arrow_len * math.cos(heading_rad))
        ay = int(ny - arrow_len * math.sin(heading_rad))
        cv2.arrowedLine(vis, (nx, ny), (ax, ay), _ORANGE, 2, tipLength=0.3)

    @staticmethod
    def _build_status_bar(
        full_width: int,
        command: RobotCommand,
        fps: float,
        line_result: LineDetectionResult,
        aruco_result: ArucoDetectionResult,
        local_heading: Optional[float] = None,
    ) -> np.ndarray:
        """Build a status bar strip (40 px tall × full_width, BGR)."""
        bar_h = 40
        bar = np.full((bar_h, full_width, 3), _DARK, dtype=np.uint8)

        action = command.action

        # Colour for the command token
        if action == "FORWARD":
            cmd_col = _GREEN
        elif action == "STOP":
            cmd_col = _RED
        else:
            cmd_col = _ORANGE

        line_err_str = f"{line_result.error_x:+d}px" if line_result.found else "N/A"
        aruco_str = (
            f"hdg:{aruco_result.heading_deg:.1f}deg"
            if aruco_result.found else "No ArUco"
        )
        aruco_col = _PURPLE if aruco_result.found else _RED
        line_col  = _GREEN  if line_result.found  else _RED

        parts = [
            (f"CMD: {action}",            cmd_col),
            (f"Spd: {command.speed}",     _WHITE),
            (f"FPS: {fps:.1f}",           _WHITE),
            (f"Line err: {line_err_str}", line_col),
            (aruco_str,                   aruco_col),
        ]

        # Local line heading (corner-aware target direction)
        if local_heading is not None:
            parts.append((f"Loc.hdg:{local_heading:.1f}", _ORANGE))

        x = 10
        y = bar_h - 10
        for text, colour in parts:
            cv2.putText(bar, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            x += tw + 25

        # Steering arrow on the right edge
        DebugVisualizer._draw_steering_arrow(bar, action, bar_h, full_width)
        return bar

    @staticmethod
    def _draw_steering_arrow(
        bar: np.ndarray,
        action: str,
        bar_h: int,
        full_width: int,
    ) -> None:
        """Draw a small directional arrow on the right side of the status bar."""
        cx = full_width - 30
        cy = bar_h // 2
        if action == "FORWARD":
            cv2.arrowedLine(bar, (cx, cy + 10), (cx, cy - 10), _GREEN, 2, tipLength=0.4)
        elif action == "REVERSE":
            cv2.arrowedLine(bar, (cx, cy - 10), (cx, cy + 10), _ORANGE, 2, tipLength=0.4)
        elif action == "LEFT":
            cv2.arrowedLine(bar, (cx + 12, cy), (cx - 12, cy), _ORANGE, 2, tipLength=0.4)
        elif action == "RIGHT":
            cv2.arrowedLine(bar, (cx - 12, cy), (cx + 12, cy), _ORANGE, 2, tipLength=0.4)
        else:  # STOP
            cv2.rectangle(bar, (cx - 8, cy - 8), (cx + 8, cy + 8), _RED, 2)
