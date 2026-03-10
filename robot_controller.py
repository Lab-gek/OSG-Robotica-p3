"""
robot_controller.py – Decision logic: maps sensor data → robot command.

The camera is mounted **overhead** (fixed, above the track).  The controller
fuses ArUco tracking and line detection to steer the robot along the black
line.  The robot travels from the **right** side of the camera frame to the
**left** side.

Steering algorithm
------------------
When the ArUco marker (robot position) is visible:

  1. Compute the *lateral error* – how far the robot centre is displaced from
     the line centroid, measured perpendicular to the robot's current heading.

       lateral_err = Δx · sin(θ) + Δy · cos(θ)

     where (Δx, Δy) = robot_pos − line_centroid and θ is the robot heading in
     radians.  Positive lateral_err means the robot is to the RIGHT of the
     line; negative means it is to the LEFT.

  2. If |lateral_err| ≤ CENTRE_TOLERANCE → FORWARD (robot is on the line).
  3. If lateral_err > 0 → LEFT  (robot drifted right of the line).
  4. If lateral_err < 0 → RIGHT (robot drifted left of the line).

  5. Heading correction (applied on top of lateral control when the robot is
     centred): if the robot's heading deviates from ROBOT_DESIRED_HEADING by
     more than heading_tolerance, a corrective turn is issued so the robot
     stays aligned with the line direction.

When the ArUco marker is NOT visible (fallback):

  The controller falls back to using the line's horizontal position relative
  to the frame centre (line.error_x) as the steering signal.

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
- ROBOT_DESIRED_HEADING defaults to 180° (pointing left) because the robot
  travels from the right side of the frame to the left side.  Change it in
  config.py if the camera or track orientation is different.
"""

import math
from dataclasses import dataclass
from typing import Optional

import config
from line_detector import LineDetectionResult
from aruco_tracker import ArucoDetectionResult


# Valid command tokens (must match ESP32 firmware endpoints)
CMD_FORWARD = "FORWARD"
CMD_LEFT    = "LEFT"
CMD_RIGHT   = "RIGHT"
CMD_STOP    = "STOP"
CMD_REVERSE = "REVERSE"


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
        Pixel half-width of the lateral dead-zone.
        While the robot's displacement from the line is within this band it
        drives FORWARD; outside it turns.
    heading_tolerance : float
        Heading dead-zone in degrees.  If the robot's heading deviates from
        ROBOT_DESIRED_HEADING by less than this value, no heading correction
        is applied.
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
        1. If no line detected → STOP (safe default).
        2. If both ArUco and line are detected → overhead-camera steering using
           the lateral error between the robot position and the line centroid.
        3. If only line detected (no ArUco) → fallback: steer toward the line
           using the line's horizontal offset from the frame centre.
        """

        # -- Safety: nothing detected ----------------------------------------
        if not line.found:
            return RobotCommand(CMD_STOP, config.SPEED_STOP)

        # -- Primary: overhead-camera mode (ArUco available) -----------------
        if aruco.found and aruco.heading_deg is not None \
                and aruco.centre_x is not None and aruco.centre_y is not None \
                and line.centroid_x is not None and line.centroid_y is not None:

            # Vector from line centroid to robot centre (image coordinates)
            dx = aruco.centre_x - line.centroid_x
            dy = aruco.centre_y - line.centroid_y

            # Project displacement onto the robot's lateral (right) axis.
            # In image coordinates: heading 0° = right, 90° = up (−Y), 180° = left.
            #   forward vector : (cos θ,  −sin θ)
            #   right   vector : (sin θ,   cos θ)   ← lateral axis
            heading_rad = math.radians(aruco.heading_deg)
            lateral_err = dx * math.sin(heading_rad) + dy * math.cos(heading_rad)
            # positive lateral_err → robot is to the RIGHT of the line
            # negative lateral_err → robot is to the LEFT  of the line

            if abs(lateral_err) <= self.centre_tolerance:
                action = CMD_FORWARD
                speed  = config.SPEED_FORWARD
            elif lateral_err > 0:
                # Robot to the RIGHT of line → steer LEFT to return to line
                action = CMD_LEFT
                speed  = config.SPEED_TURN
            else:
                # Robot to the LEFT of line → steer RIGHT to return to line
                action = CMD_RIGHT
                speed  = config.SPEED_TURN

            # -- Heading correction (only when laterally on track) ------------
            # Ensure the robot is facing the correct direction (ROBOT_DESIRED_HEADING).
            # A positive heading_error means the robot has rotated clockwise from
            # the desired heading → turn LEFT to correct.
            if action == CMD_FORWARD:
                heading_error = aruco.heading_deg - config.ROBOT_DESIRED_HEADING
                # Normalise to [-180, 180]
                while heading_error >  180: heading_error -= 360
                while heading_error < -180: heading_error += 360

                if abs(heading_error) > self.heading_tolerance:
                    if heading_error > 0:
                        action = CMD_LEFT
                    else:
                        action = CMD_RIGHT
                    speed = config.SPEED_TURN

        # -- Fallback: no ArUco → use line position relative to frame centre --
        else:
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

        return RobotCommand(action, speed)
