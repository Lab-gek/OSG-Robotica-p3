"""
robot_controller.py – Decision logic: maps sensor data → robot command.

The camera is mounted **overhead** (fixed, above the track).  The controller
fuses ArUco tracking and line detection to steer the robot along the black
line.  The robot travels from the **right** side of the camera frame to the
**left** side.

Steering algorithm
------------------
When the ArUco marker (robot position) is visible:

  1. Find the point on the detected line **contour** that is closest to the
     robot (``LineDetector.nearest_point_on_contour``).  Using the nearest
     point — rather than the overall line centroid — is critical for corners:
     the centroid of an L-shaped contour is biased toward the inside of the
     bend, whereas the nearest contour point is always on the segment of the
     line that is adjacent to the robot.

  2. Compute the *lateral error* – how far the robot centre is displaced from
     the nearest line point, measured perpendicular to the robot's current
     heading.

       lateral_err = Δx · sin(θ) + Δy · cos(θ)

     where (Δx, Δy) = robot_pos − nearest_line_point and θ is the robot
     heading in radians.  Positive lateral_err means the robot is to the
     RIGHT of the line; negative means it is to the LEFT.

  3. If |lateral_err| ≤ CENTRE_TOLERANCE → FORWARD (robot is on the line).
  4. If lateral_err > 0 → LEFT  (robot drifted right of the line).
  5. If lateral_err < 0 → RIGHT (robot drifted left of the line).

  6. Heading correction (applied on top of lateral control when the robot is
     centred): the *local* line direction at the nearest point is estimated
     via ``LineDetector.local_line_heading`` and used as the target heading.
     On a straight segment this matches ``ROBOT_DESIRED_HEADING``; at a
     corner it adapts to the turn so the robot aligns with the new direction
     instead of fighting it.

When the ArUco marker is NOT visible:

  The controller stops the robot.  With a fixed overhead camera the line
  centroid position relative to the frame centre carries no information about
  where the robot is, so there is no valid steering signal without ArUco.

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
- LINE_LOOKAHEAD_RADIUS (config.py) controls how far around the nearest point
  the controller looks to estimate the local line direction.  80 px works well
  for a 640×480 frame; decrease it if the robot starts turning too early at
  corners.
- ROBOT_DESIRED_HEADING is now only a fallback (used when no contour is
  available).  Normal operation uses the dynamic local line heading.
"""

import math
from dataclasses import dataclass
from typing import Optional

import config
from line_detector import LineDetectionResult, LineDetector
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
           the lateral error between the robot position and the *nearest point
           on the line contour* (handles corners correctly).
        3. If only line detected (no ArUco) → STOP.  Without the robot's pixel
           position the frame gives no usable steering signal.
        """

        # -- Safety: nothing detected ----------------------------------------
        if not line.found:
            return RobotCommand(CMD_STOP, config.SPEED_STOP)

        # -- Primary: overhead-camera mode (ArUco available) -----------------
        if aruco.found and aruco.heading_deg is not None \
                and aruco.centre_x is not None and aruco.centre_y is not None \
                and line.centroid_x is not None and line.centroid_y is not None:

            # Find the nearest point on the line contour to the robot.
            # Using the nearest point instead of the overall centroid ensures
            # correct lateral error at corners: the centroid of an L-shaped
            # contour is biased toward the inside of the bend, which would
            # steer the robot away from the actual line segment ahead.
            if line.contour is not None:
                nearest_x, nearest_y = LineDetector.nearest_point_on_contour(
                    line.contour, aruco.centre_x, aruco.centre_y
                )
            else:
                nearest_x, nearest_y = float(line.centroid_x), float(line.centroid_y)

            # Vector from nearest line point to robot centre (image coordinates)
            dx = aruco.centre_x - nearest_x
            dy = aruco.centre_y - nearest_y

            # Project displacement onto the robot's lateral (right) axis.
            # In image coordinates: heading 0° = right, 90° = up (-Y), 180° = left.
            #   forward vector : (cos θ,  -sin θ)
            #   right   vector : (sin θ,   cos θ)   <- lateral axis
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
            # Use the LOCAL line direction at the nearest point as the target
            # heading.  On straight segments this matches ROBOT_DESIRED_HEADING;
            # at a corner the local direction adapts to the turn so the robot
            # aligns with the new segment instead of fighting the bend.
            if action == CMD_FORWARD:
                if line.contour is not None:
                    target_heading = LineDetector.local_line_heading(
                        line.contour, nearest_x, nearest_y, aruco.heading_deg
                    )
                else:
                    target_heading = config.ROBOT_DESIRED_HEADING

                heading_error = aruco.heading_deg - target_heading
                # Normalise to [-180, 180]
                while heading_error >  180: heading_error -= 360
                while heading_error < -180: heading_error += 360

                if abs(heading_error) > self.heading_tolerance:
                    if heading_error > 0:
                        action = CMD_RIGHT
                    else:
                        action = CMD_LEFT
                    speed = config.SPEED_TURN

        # -- No ArUco: robot position unknown → stop safely ------------------
        # With a fixed overhead camera the line centroid relative to frame
        # centre carries no information about the robot's actual position.
        else:
            action = CMD_STOP
            speed  = config.SPEED_STOP

        return RobotCommand(action, speed)
