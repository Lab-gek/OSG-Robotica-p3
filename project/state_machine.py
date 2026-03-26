"""Junction state machine for the overhead-camera line follower.

States
------
FOLLOW_LINE  – PID steers on lateral error; detects junction by blob-width spike.
JUNCTION_TURN – rotate in place to a target heading; transitions when within tolerance.
REALIGN      – creep forward until line is reacquired.
STOP         – send stop and exit loop.

Heading convention (0°=right, 90°=down, 180°=left, 270°=up).
Junction targets: [90, 180] (junction 1 → down; junction 2 → left).
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Optional

import config


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

# ---------------------------------------------------------------------------
# Runtime-adjustable parameters (modified by GUI via module attributes).
# Defaults are sourced from `project/config.py`.
# ---------------------------------------------------------------------------
JUNCTION_TARGETS: list[int] = getattr(config, "JUNCTION_TARGETS", [90, 180])
HEADING_TOLERANCE: float = getattr(config, "HEADING_TOLERANCE", 10.0)   # degrees
TURN_SPEED: int = getattr(config, "TURN_SPEED", 120)             # PWM for in-place rotation
BASE_SPEED: int = getattr(config, "BASE_SPEED", 180)             # forward PWM
NO_LINE_LIMIT: int = getattr(config, "NO_LINE_LIMIT", 5)            # consecutive no-line frames before REALIGN


class State(Enum):
    FOLLOW_LINE = auto()
    JUNCTION_TURN = auto()
    REALIGN = auto()
    STOP = auto()


def heading_error(current: float, target: float) -> float:
    """Shortest signed angular distance from *current* to *target* (degrees).

    Returns a value in (-180, 180].
    """
    err = (target - current) % 360.0
    if err > 180.0:
        err -= 360.0
    return err


class StateMachine:
    """Manages robot state and produces (left_speed, right_speed) each frame."""

    def __init__(self) -> None:
        self.state: State = State.FOLLOW_LINE
        self.junctions_done: int = 0
        self._no_line_count: int = 0
        self._target_heading: Optional[float] = None

    def reset(self) -> None:
        self.state = State.FOLLOW_LINE
        self.junctions_done = 0
        self._no_line_count = 0
        self._target_heading = None

    @property
    def running(self) -> bool:
        return self.state != State.STOP

    def update(
        self,
        line_cx: Optional[int],
        frame_width: int,
        at_junction: bool,
        heading: Optional[float],
        pid_correction: float,
    ) -> tuple[int, int]:
        """Compute (left_speed, right_speed) for the current frame.

        Parameters
        ----------
        line_cx      : centroid x in the ROI, or None if no line.
        frame_width  : width of the full frame (used to compute lateral error).
        at_junction  : True when blob_width spike detected.
        heading      : current robot heading in degrees, or None.
        pid_correction: PID correction value (already computed externally).
        """
        left = right = 0

        # -------------------------------------------------------------------
        if self.state == State.FOLLOW_LINE:
            if line_cx is None:
                self._no_line_count += 1
                if self._no_line_count >= NO_LINE_LIMIT:
                    self.state = State.REALIGN
                    self._no_line_count = 0
                # Hold last command (zero for now).
                left = right = 0
            else:
                self._no_line_count = 0
                left = int(_clamp(BASE_SPEED + pid_correction, -255, 255))
                right = int(_clamp(BASE_SPEED - pid_correction, -255, 255))

                if at_junction and self.junctions_done < len(JUNCTION_TARGETS):
                    self._target_heading = float(JUNCTION_TARGETS[self.junctions_done])
                    self.state = State.JUNCTION_TURN
                    left = right = 0

        # -------------------------------------------------------------------
        elif self.state == State.JUNCTION_TURN:
            if heading is None or self._target_heading is None:
                # Can't control without heading; creep to reacquire.
                left, right = TURN_SPEED // 2, TURN_SPEED // 2
            else:
                err = heading_error(heading, self._target_heading)
                if abs(err) <= HEADING_TOLERANCE:
                    self.junctions_done += 1
                    self._target_heading = None
                    if self.junctions_done >= len(JUNCTION_TARGETS):
                        self.state = State.STOP
                    else:
                        self.state = State.REALIGN
                    left = right = 0
                else:
                    # Rotate in place: err > 0 → need to turn right (CW in image).
                    if err > 0:
                        left, right = TURN_SPEED, -TURN_SPEED
                    else:
                        left, right = -TURN_SPEED, TURN_SPEED

        # -------------------------------------------------------------------
        elif self.state == State.REALIGN:
            if line_cx is not None:
                self.state = State.FOLLOW_LINE
                left = right = BASE_SPEED
            else:
                # Creep forward.
                left = right = BASE_SPEED // 2

        # -------------------------------------------------------------------
        elif self.state == State.STOP:
            left = right = 0

        return left, right
