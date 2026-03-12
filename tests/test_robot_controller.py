"""
tests/test_robot_controller.py – Unit tests for RobotController decision logic.

These tests run without a camera or ESP32 – they construct synthetic detection
results and verify the controller maps them to the correct command.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import config
from line_detector import LineDetectionResult, LineDetector
from aruco_tracker import ArucoDetectionResult
from robot_controller import RobotController, RobotCommand, CMD_FORWARD, CMD_LEFT, CMD_RIGHT, CMD_STOP


def make_line(found=True, error_x=0, centroid_y=None):
    r = LineDetectionResult()
    r.found = found
    r.error_x = error_x
    r.centroid_x = (config.FRAME_WIDTH // 2) + error_x
    r.centroid_y = centroid_y if centroid_y is not None else config.FRAME_HEIGHT // 2
    return r


def make_aruco(found=True, heading_deg=180.0, centre_x=None, centre_y=None):
    r = ArucoDetectionResult()
    r.found = found
    r.heading_deg = heading_deg
    r.centre_x = centre_x if centre_x is not None else config.FRAME_WIDTH / 2
    r.centre_y = centre_y if centre_y is not None else config.FRAME_HEIGHT / 2
    return r


class TestRobotControllerNoLine:
    def test_stop_when_no_line(self):
        ctrl = RobotController()
        cmd = ctrl.compute(make_line(found=False), make_aruco(found=False))
        assert cmd.action == CMD_STOP

    def test_stop_when_no_line_even_with_aruco(self):
        ctrl = RobotController()
        cmd = ctrl.compute(make_line(found=False), make_aruco(found=True, heading_deg=90))
        assert cmd.action == CMD_STOP


class TestRobotControllerLineOnly:
    def setup_method(self):
        self.ctrl = RobotController(centre_tolerance=30)

    def test_forward_when_centred(self):
        cmd = self.ctrl.compute(make_line(error_x=0), make_aruco(found=False))
        assert cmd.action == CMD_FORWARD
        assert cmd.speed == config.SPEED_FORWARD

    def test_forward_within_tolerance_positive(self):
        cmd = self.ctrl.compute(make_line(error_x=29), make_aruco(found=False))
        assert cmd.action == CMD_FORWARD

    def test_forward_within_tolerance_negative(self):
        cmd = self.ctrl.compute(make_line(error_x=-29), make_aruco(found=False))
        assert cmd.action == CMD_FORWARD

    def test_right_when_line_right_of_centre(self):
        # Line is to the RIGHT → robot steers RIGHT
        cmd = self.ctrl.compute(make_line(error_x=80), make_aruco(found=False))
        assert cmd.action == CMD_RIGHT
        assert cmd.speed == config.SPEED_TURN

    def test_left_when_line_left_of_centre(self):
        # Line is to the LEFT → robot steers LEFT
        cmd = self.ctrl.compute(make_line(error_x=-80), make_aruco(found=False))
        assert cmd.action == CMD_LEFT
        assert cmd.speed == config.SPEED_TURN

    def test_forward_at_exact_tolerance_boundary(self):
        # error_x == centre_tolerance is still within the dead-band (<=)
        cmd = self.ctrl.compute(make_line(error_x=30), make_aruco(found=False))
        assert cmd.action == CMD_FORWARD

    def test_just_outside_tolerance(self):
        cmd = self.ctrl.compute(make_line(error_x=31), make_aruco(found=False))
        assert cmd.action == CMD_RIGHT


class TestRobotControllerHeadingCorrection:
    def setup_method(self):
        self.ctrl = RobotController(centre_tolerance=30, heading_tolerance=15.0)

    def test_no_heading_correction_when_straight(self):
        # Line centred AND robot pointing left (180°, the desired heading) → FORWARD
        cmd = self.ctrl.compute(make_line(error_x=0), make_aruco(found=True, heading_deg=180.0))
        assert cmd.action == CMD_FORWARD

    def test_heading_correction_left_when_rotated_cw(self):
        # Robot rotated clockwise past desired heading (heading > 180°) → needs LEFT correction
        cmd = self.ctrl.compute(make_line(error_x=0), make_aruco(found=True, heading_deg=210.0))
        assert cmd.action == CMD_LEFT

    def test_heading_correction_right_when_rotated_ccw(self):
        # Robot rotated counter-clockwise (heading < 180°) → needs RIGHT correction
        cmd = self.ctrl.compute(make_line(error_x=0), make_aruco(found=True, heading_deg=150.0))
        assert cmd.action == CMD_RIGHT

    def test_no_heading_correction_within_tolerance(self):
        # Heading error of 10° is within the 15° tolerance → FORWARD
        cmd = self.ctrl.compute(make_line(error_x=0), make_aruco(found=True, heading_deg=190.0))
        assert cmd.action == CMD_FORWARD

    def test_heading_correction_only_when_lateral_error_within_tolerance(self):
        # Robot is off the line laterally → lateral error takes priority over heading.
        # line at x=400, robot at x=320, heading=150° (CCW from desired 180°)
        # dx=-80, dy=0 → lateral_err = -80*sin(150°) = -40 → outside tolerance → CMD_RIGHT
        cmd = self.ctrl.compute(make_line(error_x=80), make_aruco(found=True, heading_deg=150.0))
        assert cmd.action == CMD_RIGHT  # from lateral error, not heading correction


class TestRobotControllerOverheadCamera:
    """
    Tests for the overhead-camera mode: both ArUco and line detected.

    The robot travels from RIGHT to LEFT (desired heading ≈ 180°).
    Lateral error is the displacement of the robot from the line, projected
    onto the robot's lateral (right) axis.
    """

    def setup_method(self):
        self.ctrl = RobotController(centre_tolerance=30, heading_tolerance=15.0)

    def test_forward_when_robot_on_line(self):
        # Robot centre coincides with line centroid → zero lateral error → FORWARD
        cmd = self.ctrl.compute(
            make_line(error_x=0),
            make_aruco(found=True, heading_deg=180.0),
        )
        assert cmd.action == CMD_FORWARD
        assert cmd.speed == config.SPEED_FORWARD

    def test_forward_within_lateral_tolerance(self):
        # Robot slightly below the line (dy=10 px) – within 30 px tolerance → FORWARD
        # heading=180°: right-axis points UP → lateral_err = -dy = -10 → within tol
        cmd = self.ctrl.compute(
            make_line(error_x=0),
            make_aruco(found=True, heading_deg=180.0, centre_y=config.FRAME_HEIGHT / 2 + 10),
        )
        assert cmd.action == CMD_FORWARD

    def test_left_when_robot_above_line_heading_left(self):
        # Robot above the line (robot_y < line_y, dy < 0), heading = 180°.
        # lateral_err = −dy > 0 → robot is to the RIGHT of the line → CMD_LEFT.
        cmd = self.ctrl.compute(
            make_line(error_x=0),                                        # line  y=240
            make_aruco(found=True, heading_deg=180.0, centre_y=190.0),   # robot y=190
        )
        assert cmd.action == CMD_LEFT
        assert cmd.speed == config.SPEED_TURN

    def test_right_when_robot_below_line_heading_left(self):
        # Robot below the line (robot_y > line_y, dy > 0), heading = 180°.
        # lateral_err = −dy < 0 → robot is to the LEFT of the line → CMD_RIGHT.
        cmd = self.ctrl.compute(
            make_line(error_x=0),                                        # line  y=240
            make_aruco(found=True, heading_deg=180.0, centre_y=290.0),   # robot y=290
        )
        assert cmd.action == CMD_RIGHT
        assert cmd.speed == config.SPEED_TURN

    def test_fallback_to_line_error_when_no_aruco(self):
        # No ArUco → fall back to line.error_x steering
        cmd = self.ctrl.compute(make_line(error_x=80), make_aruco(found=False))
        assert cmd.action == CMD_RIGHT

    def test_fallback_forward_when_no_aruco_and_centred(self):
        cmd = self.ctrl.compute(make_line(error_x=0), make_aruco(found=False))
        assert cmd.action == CMD_FORWARD

    def test_lateral_error_heading_0_right_of_line(self):
        # Robot heading = 0° (pointing right), robot directly below the line.
        # right-axis for 0° heading is DOWN (increasing Y).
        # dy > 0 → robot is to the robot's right of the line → CMD_LEFT.
        dy = 50.0
        cmd = self.ctrl.compute(
            make_line(error_x=0),
            make_aruco(found=True, heading_deg=0.0, centre_y=config.FRAME_HEIGHT / 2 + dy),
        )
        assert cmd.action == CMD_LEFT

    def test_stop_when_line_lost(self):
        cmd = self.ctrl.compute(make_line(found=False), make_aruco(found=True, heading_deg=180.0))
        assert cmd.action == CMD_STOP


class TestEsp32ClientDryRun:
    def test_dry_run_send_returns_true(self):
        from esp32_client import Esp32Client
        from robot_controller import RobotCommand
        client = Esp32Client(dry_run=True)
        assert client.send(RobotCommand(CMD_FORWARD, 75)) is True

    def test_dry_run_deduplication(self):
        from esp32_client import Esp32Client
        from robot_controller import RobotCommand
        client = Esp32Client(dry_run=True)
        cmd = RobotCommand(CMD_FORWARD, 75)
        client.send(cmd)
        # Second identical command should be deduplicated → False
        assert client.send(cmd) is False

    def test_dry_run_different_action_sent(self):
        from esp32_client import Esp32Client
        from robot_controller import RobotCommand
        client = Esp32Client(dry_run=True)
        client.send(RobotCommand(CMD_FORWARD, 75))
        # Different action, same speed → should send (only direction call)
        assert client.send(RobotCommand(CMD_LEFT, 75)) is True

    def test_dry_run_speed_change_sent(self):
        from esp32_client import Esp32Client
        from robot_controller import RobotCommand
        client = Esp32Client(dry_run=True)
        client.send(RobotCommand(CMD_FORWARD, 75))
        # Same action, different speed → should send (only speed call)
        assert client.send(RobotCommand(CMD_FORWARD, 50)) is True

    def test_stop_bypasses_deduplication(self):
        from esp32_client import Esp32Client
        from robot_controller import RobotCommand
        client = Esp32Client(dry_run=True)
        client.send(RobotCommand(CMD_STOP, 0))
        # stop() resets last state → sends again
        assert client.stop() is True

    def test_last_action_and_speed_tracked_independently(self):
        from esp32_client import Esp32Client
        from robot_controller import RobotCommand
        client = Esp32Client(dry_run=True)
        client.send(RobotCommand(CMD_FORWARD, 75))
        assert client._last_action == CMD_FORWARD
        assert client._last_speed  == 75


class TestLineDetectorLogic:
    """Tests for LineDetector that don't require a real camera."""

    def test_detect_returns_not_found_on_white_frame(self):
        import numpy as np
        from line_detector import LineDetector
        detector = LineDetector()
        # A fully white frame has V=255, far outside the dark-pixel range → no line
        white = np.full((480, 640, 3), 255, dtype=np.uint8)
        result = detector.detect(white)
        assert result.found is False

    def test_detect_finds_black_line_on_yellow(self):
        import numpy as np
        from line_detector import LineDetector
        detector = LineDetector()
        # Create a frame: yellow background with a black vertical stripe in the centre
        frame = np.full((480, 640, 3), (30, 180, 180), dtype=np.uint8)  # yellowish in BGR
        # Draw a black vertical line at x=320
        frame[:, 310:330] = (0, 0, 0)
        result = detector.detect(frame)
        assert result.found is True
        # Centroid should be near the centre of the frame horizontally
        assert abs(result.centroid_x - 320) < 20

    def test_error_x_sign_left(self):
        import numpy as np
        from line_detector import LineDetector
        detector = LineDetector()
        frame = np.full((480, 640, 3), (30, 180, 180), dtype=np.uint8)
        # Line to the LEFT of centre (x=200)
        frame[:, 190:210] = (0, 0, 0)
        result = detector.detect(frame)
        if result.found:
            assert result.error_x < 0

    def test_error_x_sign_right(self):
        import numpy as np
        from line_detector import LineDetector
        detector = LineDetector()
        frame = np.full((480, 640, 3), (30, 180, 180), dtype=np.uint8)
        # Line to the RIGHT of centre (x=480)
        frame[:, 470:490] = (0, 0, 0)
        result = detector.detect(frame)
        if result.found:
            assert result.error_x > 0


class TestDebugVisualizer:
    """Tests for DebugVisualizer.build_frame() – no camera or ESP32 required."""

    def _make_frame(self):
        import numpy as np
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_build_frame_returns_ndarray(self):
        import numpy as np
        from visualizer import DebugVisualizer
        from robot_controller import RobotCommand, CMD_FORWARD
        vis = DebugVisualizer()
        frame = self._make_frame()
        out = vis.build_frame(
            frame,
            make_line(found=False),
            make_aruco(found=False),
            RobotCommand(CMD_FORWARD, 75),
            fps=30.0,
        )
        assert isinstance(out, np.ndarray)
        assert out.ndim == 3

    def test_composite_is_wider_than_input(self):
        from visualizer import DebugVisualizer
        from robot_controller import RobotCommand, CMD_STOP
        vis = DebugVisualizer()
        frame = self._make_frame()
        out = vis.build_frame(
            frame,
            make_line(found=False),
            make_aruco(found=False),
            RobotCommand(CMD_STOP, 0),
        )
        # Side-by-side layout: output must be at least twice as wide
        assert out.shape[1] >= frame.shape[1] * 2

    def test_composite_is_taller_than_input(self):
        from visualizer import DebugVisualizer
        from robot_controller import RobotCommand, CMD_STOP
        vis = DebugVisualizer()
        frame = self._make_frame()
        out = vis.build_frame(
            frame,
            make_line(found=False),
            make_aruco(found=False),
            RobotCommand(CMD_STOP, 0),
        )
        # Status bar adds height
        assert out.shape[0] > frame.shape[0]

    def test_scale_reduces_output_size(self):
        from visualizer import DebugVisualizer
        from robot_controller import RobotCommand, CMD_FORWARD
        vis_full  = DebugVisualizer(scale=1.0)
        vis_half  = DebugVisualizer(scale=0.5)
        frame = self._make_frame()
        out_full = vis_full.build_frame(
            frame, make_line(found=False), make_aruco(found=False),
            RobotCommand(CMD_FORWARD, 75),
        )
        out_half = vis_half.build_frame(
            frame, make_line(found=False), make_aruco(found=False),
            RobotCommand(CMD_FORWARD, 75),
        )
        assert out_half.shape[1] < out_full.shape[1]
        assert out_half.shape[0] < out_full.shape[0]

    def test_mask_panel_shows_detected_pixels(self):
        """When a mask is provided, the right panel should have non-zero green pixels."""
        import numpy as np
        from visualizer import DebugVisualizer
        from robot_controller import RobotCommand, CMD_FORWARD
        from line_detector import LineDetectionResult

        vis = DebugVisualizer()
        frame = self._make_frame()

        line_result = LineDetectionResult()
        line_result.found = False
        # Provide a mask with a bright stripe in the middle
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[200:280, 300:340] = 255
        line_result.mask = mask

        out = vis.build_frame(
            frame, line_result, make_aruco(found=False),
            RobotCommand(CMD_FORWARD, 75),
        )
        # The right panel starts at column 640 in the composite.
        # Some pixels in that region should be green (0, 220, 0).
        right_panel = out[:480, 640:]
        green_pixels = np.all(right_panel == (0, 220, 0), axis=2)
        assert green_pixels.sum() > 0

    def test_window_title_is_string(self):
        from visualizer import DebugVisualizer
        assert isinstance(DebugVisualizer.WINDOW_TITLE, str)
        assert len(DebugVisualizer.WINDOW_TITLE) > 0


# ---------------------------------------------------------------------------
# Corner-handling tests
# ---------------------------------------------------------------------------

class TestNearestPointOnContour:
    """Unit tests for LineDetector.nearest_point_on_contour."""

    def _horizontal_contour(self):
        import numpy as np
        # Horizontal line of contour points at y=240, x=100..400
        pts = [[[x, 240]] for x in range(100, 401, 10)]
        return np.array(pts, dtype=np.int32)

    def test_nearest_to_exact_point(self):
        contour = self._horizontal_contour()
        nx, ny = LineDetector.nearest_point_on_contour(contour, 250.0, 240.0)
        assert nx == 250.0
        assert ny == 240.0

    def test_nearest_when_robot_offset_vertically(self):
        # Robot is 20 px above the horizontal line – nearest should still be on the line
        contour = self._horizontal_contour()
        nx, ny = LineDetector.nearest_point_on_contour(contour, 250.0, 220.0)
        assert ny == 240.0  # must be on the line (not above it)
        assert abs(nx - 250.0) < 15  # close to x=250

    def test_nearest_point_right_end(self):
        import numpy as np
        contour = self._horizontal_contour()
        # Reference point beyond the right end of the contour
        nx, ny = LineDetector.nearest_point_on_contour(contour, 450.0, 240.0)
        assert nx == 400.0  # rightmost point

    def test_nearest_point_four_corner_contour(self):
        import numpy as np
        # Square contour
        contour = np.array(
            [[[10, 10]], [[100, 10]], [[100, 100]], [[10, 100]]], dtype=np.int32
        )
        nx, ny = LineDetector.nearest_point_on_contour(contour, 5.0, 5.0)
        # (10, 10) is the closest corner
        assert nx == 10.0 and ny == 10.0


class TestLocalLineHeading:
    """Unit tests for LineDetector.local_line_heading."""

    def _horizontal_contour(self):
        import numpy as np
        pts = [[[x, 240]] for x in range(50, 451, 10)]
        return np.array(pts, dtype=np.int32)

    def _vertical_contour(self):
        import numpy as np
        pts = [[[320, y]] for y in range(50, 401, 10)]
        return np.array(pts, dtype=np.int32)

    def test_horizontal_line_resolved_to_180(self):
        # Robot heading ~180° (pointing left) → should disambiguate to ~180°
        contour = self._horizontal_contour()
        heading = LineDetector.local_line_heading(contour, 250.0, 240.0, 180.0)
        diff = abs(((heading - 180.0) + 180) % 360 - 180)
        assert diff < 20, f"Expected ~180°, got {heading:.1f}°"

    def test_horizontal_line_resolved_to_0(self):
        # Same contour but robot heading ~0° → should disambiguate to ~0°
        contour = self._horizontal_contour()
        heading = LineDetector.local_line_heading(contour, 250.0, 240.0, 0.0)
        diff = abs(((heading - 0.0) + 180) % 360 - 180)
        assert diff < 20, f"Expected ~0°, got {heading:.1f}°"

    def test_vertical_line_resolved_to_90(self):
        # Vertical contour (pointing up) + robot heading ~90° → should give ~90°
        contour = self._vertical_contour()
        heading = LineDetector.local_line_heading(contour, 320.0, 200.0, 90.0)
        diff = abs(((heading - 90.0) + 180) % 360 - 180)
        assert diff < 20, f"Expected ~90°, got {heading:.1f}°"

    def test_vertical_line_resolved_to_270(self):
        # Vertical contour + robot heading ~270° → disambiguate to ~270°
        contour = self._vertical_contour()
        heading = LineDetector.local_line_heading(contour, 320.0, 200.0, 270.0)
        diff = abs(((heading - 270.0) + 180) % 360 - 180)
        assert diff < 20, f"Expected ~270°, got {heading:.1f}°"

    def test_fallback_when_too_few_points(self):
        import numpy as np
        # Only 2 nearby points → fall back to robot_heading_deg
        contour = np.array([[[100, 100]], [[102, 100]]], dtype=np.int32)
        heading = LineDetector.local_line_heading(contour, 101.0, 100.0, 123.0, search_radius=5.0)
        assert heading == 123.0


class TestCornerHandling:
    """
    Integration tests for corner-aware steering.

    A synthetic L-shaped track (horizontal then vertical) is created in a
    640×480 image and fed through the full detection + control pipeline to
    verify that the robot receives sensible commands near the corner.
    """

    def _detect_l_shape(self):
        """Detect an L-shaped line: horizontal left + vertical up at the junction."""
        import numpy as np
        img = np.full((480, 640, 3), 255, dtype=np.uint8)
        img[228:252, 180:440] = 0   # horizontal segment  y≈240
        img[100:228, 228:252] = 0   # vertical segment    x≈240
        from line_detector import LineDetector
        detector = LineDetector()
        result = detector.detect(img)
        return result

    def test_centroid_biased_toward_bend(self):
        """
        Confirm the known limitation: centroid IS biased toward the bend.
        nearest_x should be closer to the horizontal segment than centroid_x is.
        """
        result = self._detect_l_shape()
        if not result.found or result.contour is None:
            pytest.skip("L-line not detected in this environment")

        # Robot approaching from the right side of the horizontal segment
        robot_x, robot_y = 400.0, 240.0
        nx, ny = LineDetector.nearest_point_on_contour(result.contour, robot_x, robot_y)

        # Nearest point should be close to y=240 (horizontal segment)
        assert abs(ny - 240) < 30, (
            f"Nearest y={ny:.0f} should be near horizontal segment y=240"
        )
        # And closer to the robot than the centroid is
        dist_nearest = ((nx - robot_x) ** 2 + (ny - robot_y) ** 2) ** 0.5
        dist_centroid = (
            (result.centroid_x - robot_x) ** 2 + (result.centroid_y - robot_y) ** 2
        ) ** 0.5
        assert dist_nearest <= dist_centroid, (
            f"nearest ({nx:.0f},{ny:.0f}) should be ≤ centroid "
            f"({result.centroid_x},{result.centroid_y}) distance from robot"
        )

    def test_controller_does_not_stop_at_corner(self):
        """Robot on the horizontal segment with ArUco → must NOT issue STOP."""
        result = self._detect_l_shape()
        if not result.found:
            pytest.skip("L-line not detected in this environment")

        ctrl = RobotController(centre_tolerance=30, heading_tolerance=15.0)
        aruco = make_aruco(found=True, heading_deg=180.0, centre_x=380.0, centre_y=240.0)
        cmd = ctrl.compute(result, aruco)
        assert cmd.action != CMD_STOP

    def test_heading_correction_adapts_after_corner(self):
        """
        Once the robot has turned 90° (heading≈90°) and is on the vertical
        segment, the heading correction should NOT fight the turn.

        With the old code (fixed ROBOT_DESIRED_HEADING=180°) the heading error
        would be -90°, triggering CMD_RIGHT and reversing the completed turn.
        With the new local-heading approach the error should be small → FORWARD.
        """
        import numpy as np
        # Frame where only the vertical segment is visible (robot is past the corner)
        img = np.full((480, 640, 3), 255, dtype=np.uint8)
        img[100:350, 228:252] = 0   # only vertical segment

        from line_detector import LineDetector
        result = LineDetector().detect(img)
        if not result.found:
            pytest.skip("Vertical line not detected in this environment")

        ctrl = RobotController(centre_tolerance=30, heading_tolerance=15.0)
        # Robot is on the vertical segment, heading 90° (pointing up) – correct orientation
        aruco = make_aruco(found=True, heading_deg=90.0, centre_x=240.0, centre_y=220.0)
        cmd = ctrl.compute(result, aruco)
        # With the new local-heading correction this should be FORWARD (not RIGHT)
        assert cmd.action == CMD_FORWARD, (
            f"Expected FORWARD after completed 90° corner turn, got {cmd.action}"
        )
