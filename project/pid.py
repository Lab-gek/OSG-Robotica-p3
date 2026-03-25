"""PID controller for line-following lateral error."""


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class PID:
    """Simple discrete PID controller."""

    def __init__(self, kp: float = 0.4, ki: float = 0.001, kd: float = 0.1) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self._integral: float = 0.0
        self._prev_error: float = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0

    def update(self, error: float) -> float:
        """Return PID correction for *error*.  Call once per frame."""
        self._integral += error
        derivative = error - self._prev_error
        correction = self.kp * error + self.ki * self._integral + self.kd * derivative
        self._prev_error = error
        return correction

    def compute_speeds(
        self,
        error: float,
        base_speed: int = 180,
    ) -> tuple[int, int]:
        """Return (left_speed, right_speed) for the given lateral *error*.

        A positive error means the line is to the right of centre, so we
        increase the left motor and decrease the right motor to steer right.
        """
        correction = self.update(error)
        left = int(_clamp(base_speed + correction, -255, 255))
        right = int(_clamp(base_speed - correction, -255, 255))
        return left, right
