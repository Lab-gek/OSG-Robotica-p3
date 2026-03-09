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
from line_detector import LineDetectionResult
from aruco_tracker import ArucoDetectionResult
from robot_controller import RobotController, RobotCommand, CMD_FORWARD, CMD_LEFT, CMD_RIGHT, CMD_STOP


def make_line(found=True, error_x=0):
    r = LineDetectionResult()
    r.found = found
    r.error_x = error_x
    r.centroid_x = (config.FRAME_WIDTH // 2) + error_x
    r.centroid_y = config.FRAME_HEIGHT // 2
    return r


def make_aruco(found=True, heading_deg=90.0):
    r = ArucoDetectionResult()
    r.found = found
    r.heading_deg = heading_deg
    r.centre_x = config.FRAME_WIDTH / 2
    r.centre_y = config.FRAME_HEIGHT / 2
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
        # Line centred AND robot pointing straight up (90°) → FORWARD
        cmd = self.ctrl.compute(make_line(error_x=0), make_aruco(found=True, heading_deg=90.0))
        assert cmd.action == CMD_FORWARD

    def test_heading_correction_left_when_rotated_cw(self):
        # Robot angled clockwise (heading > 90°) → needs LEFT correction
        cmd = self.ctrl.compute(make_line(error_x=0), make_aruco(found=True, heading_deg=120.0))
        assert cmd.action == CMD_LEFT

    def test_heading_correction_right_when_rotated_ccw(self):
        # Robot angled counter-clockwise (heading < 90°) → needs RIGHT correction
        cmd = self.ctrl.compute(make_line(error_x=0), make_aruco(found=True, heading_deg=60.0))
        assert cmd.action == CMD_RIGHT

    def test_no_heading_correction_within_tolerance(self):
        # Heading error of 10° is within the 15° tolerance → FORWARD
        cmd = self.ctrl.compute(make_line(error_x=0), make_aruco(found=True, heading_deg=100.0))
        assert cmd.action == CMD_FORWARD

    def test_heading_correction_only_when_line_centred(self):
        # Line is already off-centre → line error takes priority, not heading
        cmd = self.ctrl.compute(make_line(error_x=80), make_aruco(found=True, heading_deg=60.0))
        assert cmd.action == CMD_RIGHT  # from line error, not heading


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
