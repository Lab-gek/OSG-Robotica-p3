"""
robot_controller.py – Decision logic: maps sensor data → robot command.

The controller fuses line-detection and ArUco tracking to decide what the robot
should do next.  Commands are simple string tokens that the ESP32 understands.

Command vocabulary
------------------
FORWARD  – drive straight ahead
LEFT     – turn left (pivot or arc)
RIGHT    – turn right
STOP     – halt motors

IRL translation notes
---------------------
- CENTRE_TOLERANCE (config.py) is the main parameter to tune on real hardware.
  Start wide (50 px) and tighten as detection becomes reliable.
- If the robot overshoots turns (oscillates left/right), increase tolerance or
  reduce SPEED_TURN.
- The heading-based correction (ArUco) adds a secondary signal.  If the robot
  body is angled to the line, a correction is applied even if the line is
  centred — this prevents the robot from drifting off after a straight section.
"""

from dataclasses import dataclass
from typing import Optional

import config
from line_detector import LineDetectionResult
from aruco_tracker import ArucoDetectionResult


# Valid command tokens (must match ESP32 firmware expectations)
CMD_FORWARD = "FORWARD"
CMD_LEFT    = "LEFT"
CMD_RIGHT   = "RIGHT"
CMD_STOP    = "STOP"


@dataclass
class RobotCommand:
    """Encapsulates the command to be sent to the ESP32."""
    action: str = CMD_STOP
    speed: int = config.SPEED_STOP

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RobotCommand):
            return NotImplemented
        return self.action == other.action and self.speed == other.speed

    def __repr__(self) -> str:
        return f"RobotCommand({self.action}, speed={self.speed})"


class RobotController:
    """
    Stateless (per-frame) controller that maps sensor readings to a RobotCommand.

    Parameters
    ----------
    centre_tolerance : int
        Pixel half-width of the "dead zone" around the frame centre.
        Within this band the robot goes FORWARD; outside it turns.
    heading_tolerance : float
        Heading dead-zone in degrees.  If the robot's heading (from ArUco)
        differs from the line direction by less than this, no heading
        correction is applied.
    """

    def __init__(
        self,
        centre_tolerance: int = config.CENTRE_TOLERANCE,
        heading_tolerance: float = 15.0,
    ):
        self.centre_tolerance = centre_tolerance
        self.heading_tolerance = heading_tolerance

    def compute(
        self,
        line: LineDetectionResult,
        aruco: ArucoDetectionResult,
    ) -> RobotCommand:
        """
        Compute the robot command for the current frame.

        Priority order:
        1. If neither line nor robot is detected → STOP.
        2. If line is detected → steer toward line centre.
        3. If only ArUco is available (line lost) → STOP (safe default).
        """

        # -- Safety: nothing detected ----------------------------------------
        if not line.found:
            return RobotCommand(CMD_STOP, config.SPEED_STOP)

        # -- Primary: line-position error ------------------------------------
        error = line.error_x   # negative = line left of centre, positive = right

        if abs(error) <= self.centre_tolerance:
            action = CMD_FORWARD
            speed  = config.SPEED_FORWARD
        elif error < 0:
            # Line is to the LEFT → robot must steer LEFT
            action = CMD_LEFT
            speed  = config.SPEED_TURN
        else:
            # Line is to the RIGHT → robot must steer RIGHT
            action = CMD_RIGHT
            speed  = config.SPEED_TURN

        # -- Secondary: heading correction (only when going roughly straight) --
        # If the robot is angled relative to the line even when centred, apply
        # a gentle correction so it doesn't drift off at the next section.
        if action == CMD_FORWARD and aruco.found and aruco.heading_deg is not None:
            # We assume the robot should be pointing "up" (90°) relative to the
            # overhead camera.  A positive heading error means robot is rotated
            # clockwise → needs to turn LEFT to correct.
            heading_error = aruco.heading_deg - 90.0
            # Normalise to [-180, 180]
            while heading_error >  180: heading_error -= 360
            while heading_error < -180: heading_error += 360

            if abs(heading_error) > self.heading_tolerance:
                if heading_error > 0:
                    action = CMD_LEFT
                else:
                    action = CMD_RIGHT
                speed = config.SPEED_TURN

        return RobotCommand(action, speed)
