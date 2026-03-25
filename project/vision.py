"""Camera capture, preprocessing, contour detection, and junction logic."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Runtime-adjustable parameters (modified by GUI via module attributes).
# ---------------------------------------------------------------------------
THRESHOLD: int = 60           # grayscale threshold for black line (inverted)
JUNCTION_WIDTH_RATIO: float = 2.8  # blob_width > normal_width * ratio → junction


class VisionProcessor:
    """Stateful vision processor for a single camera source."""

    def __init__(self, camera_index: int = 0) -> None:
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_index = camera_index

        # Running average of line width during normal following.
        self.normal_width: float = 0.0
        self._calib_frames: int = 0
        self._calib_limit: int = 30  # frames to average over at startup

        # Latest processed values (for GUI overlays).
        self.line_cx: Optional[int] = None
        self.blob_width: int = 0
        self.at_junction: bool = False
        self.roi_y_start: int = 0
        self.roi_y_end: int = 0
        self.frame_width: int = 0
        self.frame_height: int = 0

    # ------------------------------------------------------------------
    # Camera management
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Open the camera.  Returns True on success."""
        self.cap = cv2.VideoCapture(self.camera_index)
        return self.cap.isOpened()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def read_frame(self) -> Optional[np.ndarray]:
        """Read one frame from the camera.  Returns None on failure."""
        if self.cap is None or not self.cap.isOpened():
            return None
        ok, frame = self.cap.read()
        return frame if ok else None

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> tuple[Optional[np.ndarray], np.ndarray]:
        """Run the full preprocessing pipeline on *frame*.

        Returns
        -------
        mask : binary mask (white = line) in the ROI region, or None on failure.
        roi  : the colour ROI slice for display.
        """
        h, w = frame.shape[:2]
        self.frame_height = h
        self.frame_width = w

        # Middle-third ROI.
        y0 = h // 3
        y1 = 2 * h // 3
        self.roi_y_start = y0
        self.roi_y_end = y1

        roi_color = frame[y0:y1, :]
        gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold: invert so black line becomes white.
        _, mask = cv2.threshold(blurred, THRESHOLD, 255, cv2.THRESH_BINARY_INV)

        # Find largest contour.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.line_cx = None
        self.blob_width = 0
        self.at_junction = False

        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                self.line_cx = int(M["m10"] / M["m00"])

            x, _y, bw, _bh = cv2.boundingRect(largest)
            self.blob_width = bw

            # Calibrate normal_width during first N frames.
            if self._calib_frames < self._calib_limit and bw > 0:
                self._calib_frames += 1
                self.normal_width += (bw - self.normal_width) / self._calib_frames

            if self.normal_width > 0:
                self.at_junction = bw > self.normal_width * JUNCTION_WIDTH_RATIO

        return mask, roi_color

    def recalibrate_width(self) -> None:
        """Reset the running normal_width average (useful for recalibration button)."""
        self.normal_width = 0.0
        self._calib_frames = 0

    # ------------------------------------------------------------------
    # Overlay helpers
    # ------------------------------------------------------------------

    def draw_overlays(self, frame: np.ndarray) -> None:
        """Draw ROI box and centroid marker on *frame* (in-place)."""
        h = self.frame_height or frame.shape[0]
        w = self.frame_width or frame.shape[1]
        y0 = self.roi_y_start
        y1 = self.roi_y_end

        cv2.rectangle(frame, (0, y0), (w, y1), (255, 200, 0), 1)

        if self.line_cx is not None:
            cx_abs = self.line_cx
            cy_abs = (y0 + y1) // 2
            cv2.circle(frame, (cx_abs, cy_abs), 6, (0, 0, 255), -1)

        colour = (0, 0, 255) if self.at_junction else (0, 255, 0)
        label = f"w={self.blob_width}  nw={self.normal_width:.0f}"
        if self.at_junction:
            label += "  JUNCTION"
        cv2.putText(frame, label, (10, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
