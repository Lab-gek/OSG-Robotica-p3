"""
tests/test_visualizer.py – Unit tests for DebugVisualizer.

Tests run without a camera: they use synthetic BGR frames and synthetic
detection results to verify that build_frame() and its helpers produce
correctly shaped, valid NumPy images and do not raise exceptions under
all combinations of found / not-found detection results.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math

import cv2
import numpy as np
import pytest

import config
from line_detector import LineDetectionResult
from aruco_tracker import ArucoDetectionResult
from robot_controller import RobotCommand, CMD_FORWARD, CMD_STOP, CMD_LEFT, CMD_RIGHT
from visualizer import DebugVisualizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 120, w: int = 160) -> np.ndarray:
    """Return a small synthetic BGR frame (gradient so V-channel has variety)."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Create a gradient so the Value channel is non-trivial
    for x in range(w):
        frame[:, x, :] = int(x * 255 / w)
    # Add a dark "line" strip in the middle
    frame[h // 2 - 5: h // 2 + 5, :, :] = 10
    return frame


def _make_line(found: bool = True, centroid_x: int = 80, centroid_y: int = 60,
               error_x: int = 0, with_contour: bool = True) -> LineDetectionResult:
    r = LineDetectionResult()
    r.found = found
    if found:
        r.centroid_x = centroid_x
        r.centroid_y = centroid_y
        r.error_x = error_x
        # Simple 2-point contour along the horizontal line
        if with_contour:
            pts = np.array([[[20, centroid_y]], [[140, centroid_y]]], dtype=np.int32)
            r.contour = pts
        # Build a minimal binary mask
        mask = np.zeros((120, 160), dtype=np.uint8)
        mask[centroid_y - 5: centroid_y + 5, :] = 255
        r.mask = mask
    return r


def _make_aruco(found: bool = True, heading_deg: float = 180.0,
                centre_x: float = 80.0, centre_y: float = 60.0) -> ArucoDetectionResult:
    r = ArucoDetectionResult()
    r.found = found
    if found:
        r.marker_id = 0
        r.heading_deg = heading_deg
        r.centre_x = centre_x
        r.centre_y = centre_y
        # Minimal square corners
        r.corners = np.array(
            [[centre_x - 10, centre_y - 10],
             [centre_x + 10, centre_y - 10],
             [centre_x + 10, centre_y + 10],
             [centre_x - 10, centre_y + 10]],
            dtype=np.float32,
        )
    return r


def _make_command(action: str = CMD_FORWARD, speed: int = 75) -> RobotCommand:
    return RobotCommand(action=action, speed=speed)


# ---------------------------------------------------------------------------
# build_frame() output shape / dtype
# ---------------------------------------------------------------------------

class TestBuildFrameShape:
    def setup_method(self):
        self.vis = DebugVisualizer(scale=1.0)

    def test_output_is_bgr_array(self):
        frame = _make_frame()
        out = self.vis.build_frame(
            frame, _make_line(found=False), _make_aruco(found=False), _make_command()
        )
        assert isinstance(out, np.ndarray)
        assert out.ndim == 3
        assert out.shape[2] == 3
        assert out.dtype == np.uint8

    def test_width_is_double_frame_width(self):
        frame = _make_frame(h=120, w=160)
        out = self.vis.build_frame(
            frame, _make_line(found=False), _make_aruco(found=False), _make_command()
        )
        # Status bar (40 px) is added below; width should be 2× frame width
        assert out.shape[1] == 160 * 2

    def test_height_includes_status_bar(self):
        frame = _make_frame(h=120, w=160)
        out = self.vis.build_frame(
            frame, _make_line(found=False), _make_aruco(found=False), _make_command()
        )
        assert out.shape[0] == 120 + 40  # frame height + status bar

    def test_scale_half(self):
        vis_half = DebugVisualizer(scale=0.5)
        frame = _make_frame(h=120, w=160)
        out = vis_half.build_frame(
            frame, _make_line(found=False), _make_aruco(found=False), _make_command()
        )
        assert out.shape[1] == int(160 * 2 * 0.5)
        assert out.shape[0] == int((120 + 40) * 0.5)


# ---------------------------------------------------------------------------
# build_frame() – all detection combinations
# ---------------------------------------------------------------------------

class TestBuildFrameDetectionCombinations:
    def setup_method(self):
        self.vis = DebugVisualizer(scale=1.0)
        self.frame = _make_frame()

    def test_no_line_no_aruco(self):
        out = self.vis.build_frame(
            self.frame,
            _make_line(found=False),
            _make_aruco(found=False),
            _make_command(CMD_STOP, 0),
        )
        assert out.shape == (160, 320, 3)

    def test_line_only(self):
        out = self.vis.build_frame(
            self.frame,
            _make_line(found=True),
            _make_aruco(found=False),
            _make_command(CMD_FORWARD, 75),
        )
        assert out.shape == (160, 320, 3)

    def test_aruco_only(self):
        out = self.vis.build_frame(
            self.frame,
            _make_line(found=False),
            _make_aruco(found=True),
            _make_command(CMD_STOP, 0),
        )
        assert out.shape == (160, 320, 3)

    def test_both_detected(self):
        out = self.vis.build_frame(
            self.frame,
            _make_line(found=True),
            _make_aruco(found=True),
            _make_command(CMD_FORWARD, 75),
        )
        assert out.shape == (160, 320, 3)

    def test_both_detected_without_contour(self):
        """build_frame should not crash when line has no contour array."""
        out = self.vis.build_frame(
            self.frame,
            _make_line(found=True, with_contour=False),
            _make_aruco(found=True),
            _make_command(CMD_FORWARD, 75),
        )
        assert out.shape == (160, 320, 3)


# ---------------------------------------------------------------------------
# _build_hsv_map_panel()
# ---------------------------------------------------------------------------

class TestHsvMapPanel:
    def test_shape_matches_frame(self):
        frame = _make_frame(h=120, w=160)
        panel = DebugVisualizer._build_hsv_map_panel(
            frame, _make_line(found=False), 120, 160
        )
        assert panel.shape == (120, 160, 3)
        assert panel.dtype == np.uint8

    def test_line_pixels_are_green_when_found(self):
        frame = _make_frame(h=120, w=160)
        line = _make_line(found=True, centroid_y=60)
        panel = DebugVisualizer._build_hsv_map_panel(frame, line, 120, 160)
        # The detection mask rows (55–65) should contain green pixels
        mid_row = panel[60, :, :]
        assert np.any(mid_row[:, 1] > 200), "Expected green (high G) on detected line"

    def test_hsv_range_text_present(self):
        """Panel should be non-trivially coloured (not all zeros) when text is drawn."""
        frame = _make_frame(h=120, w=160)
        panel = DebugVisualizer._build_hsv_map_panel(
            frame, _make_line(found=False), 120, 160
        )
        # After drawing text the panel cannot be all black
        assert panel.max() > 0


# ---------------------------------------------------------------------------
# _draw_nearest_point()
# ---------------------------------------------------------------------------

class TestDrawNearestPoint:
    def test_does_not_raise(self):
        vis = np.zeros((120, 160, 3), dtype=np.uint8)
        DebugVisualizer._draw_nearest_point(
            vis, nearest_x=80.0, nearest_y=60.0,
            local_heading=180.0,
            robot_x=100.0, robot_y=60.0,
        )

    def test_pixels_modified(self):
        vis = np.zeros((120, 160, 3), dtype=np.uint8)
        DebugVisualizer._draw_nearest_point(
            vis, nearest_x=80.0, nearest_y=60.0,
            local_heading=180.0,
            robot_x=100.0, robot_y=60.0,
        )
        # Some pixels must have been drawn (not all black)
        assert vis.max() > 0


# ---------------------------------------------------------------------------
# _build_status_bar()
# ---------------------------------------------------------------------------

class TestStatusBar:
    def _bar(self, action=CMD_FORWARD, local_heading=None):
        line = _make_line(found=True)
        aruco = _make_aruco(found=True, heading_deg=180.0)
        cmd = _make_command(action=action, speed=75)
        return DebugVisualizer._build_status_bar(
            640, cmd, 30.0, line, aruco, local_heading
        )

    def test_shape(self):
        bar = self._bar()
        assert bar.shape == (40, 640, 3)

    def test_without_local_heading(self):
        bar = self._bar(local_heading=None)
        assert bar.max() > 0  # text was drawn

    def test_with_local_heading(self):
        bar = self._bar(local_heading=135.5)
        assert bar.max() > 0

    @pytest.mark.parametrize("action", [CMD_FORWARD, CMD_STOP, CMD_LEFT, CMD_RIGHT])
    def test_all_actions(self, action):
        bar = self._bar(action=action)
        assert bar.shape == (40, 640, 3)

    def test_no_line_no_aruco(self):
        line = _make_line(found=False)
        aruco = _make_aruco(found=False)
        cmd = _make_command(CMD_STOP, 0)
        bar = DebugVisualizer._build_status_bar(640, cmd, 0.0, line, aruco, None)
        assert bar.shape == (40, 640, 3)
