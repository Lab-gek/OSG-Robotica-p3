"""
line_detector.py – Detects a black line on a yellowish (wooden-plank) background.

The camera is mounted **overhead** (fixed, looking straight down at the track).
Returns the centroid of the largest black-line contour so that the controller
can compute the robot's lateral deviation from the line using the ArUco pose.

IRL translation notes
---------------------
- Lighting is the #1 enemy.  Run calibration.py first to re-tune LINE_HSV_LOWER /
  LINE_HSV_UPPER for your specific environment.
- With an overhead camera the full frame is valid; ROI_TOP_FRACTION in config.py
  defaults to 0.0 (no skip).  Increase it only if a static obstacle at the top of
  the frame causes false positives.
- If the floor has visible wood-grain or other dark patterns, increase
  LINE_MIN_CONTOUR_AREA to filter them out.
- The contour stored in LineDetectionResult is used by the controller to compute
  nearest_point_on_contour() and local_line_heading(), which are the key helpers
  for correct corner navigation.
"""

import math

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import config


@dataclass
class LineDetectionResult:
    """Holds the result of one frame's line detection."""
    found: bool = False
    # Centroid of the largest detected line contour (pixel coords)
    centroid_x: Optional[int] = None
    centroid_y: Optional[int] = None
    # Horizontal error: positive = line is to the right of frame centre
    error_x: int = 0
    # The binary mask (for debug visualisation)
    mask: Optional[np.ndarray] = None
    # The largest contour found
    contour: Optional[np.ndarray] = None


class LineDetector:
    """
    Detects a dark (black) line on a light-coloured surface using HSV thresholding.

    Parameters
    ----------
    hsv_lower : tuple[int, int, int]
        Lower HSV bound for the line colour.
    hsv_upper : tuple[int, int, int]
        Upper HSV bound for the line colour.
    frame_width, frame_height : int
        Expected frame dimensions (used to compute the centre reference).
    roi_top_fraction : float
        Fraction of the frame height to skip from the top.  With an overhead
        camera this should be 0.0 (full frame); increase only if the top of
        the frame contains a static obstacle that causes false positives.
    """

    def __init__(
        self,
        hsv_lower: tuple = config.LINE_HSV_LOWER,
        hsv_upper: tuple = config.LINE_HSV_UPPER,
        frame_width: int = config.FRAME_WIDTH,
        frame_height: int = config.FRAME_HEIGHT,
        roi_top_fraction: float = config.ROI_TOP_FRACTION,
    ):
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_centre_x = frame_width // 2
        self.roi_top_fraction = roi_top_fraction
        # Skip the top portion of the frame if specified (0.0 = full frame).
        # With an overhead camera there is nothing to skip, so the default is 0.0.
        self.roi_y_start = int(frame_height * roi_top_fraction)

        kernel_size = config.MORPH_KERNEL_SIZE
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        self.morph_iterations = config.MORPH_ITERATIONS
        self.min_contour_area = config.LINE_MIN_CONTOUR_AREA

    def update_hsv_range(self, lower: tuple, upper: tuple) -> None:
        """Allows live recalibration from calibration.py without restarting."""
        self.hsv_lower = np.array(lower, dtype=np.uint8)
        self.hsv_upper = np.array(upper, dtype=np.uint8)

    def detect(self, frame: np.ndarray) -> LineDetectionResult:
        """
        Run detection on a single BGR frame.

        Returns a LineDetectionResult.  The mask inside the result is full-frame
        sized (with zeros in the ROI-excluded area) for easy overlay.
        """
        result = LineDetectionResult()

        frame_height, frame_width = frame.shape[:2]
        frame_centre_x = frame_width // 2
        roi_y_start = int(frame_height * self.roi_top_fraction)
        # Clamp to a valid slice start to avoid empty ROI edge-cases.
        roi_y_start = max(0, min(roi_y_start, frame_height - 1))

        # Crop to region-of-interest (bottom portion of frame)
        roi = frame[roi_y_start:, :]

        # Convert to HSV and threshold
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_roi = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Morphological open (erode then dilate) to remove noise
        mask_roi = cv2.morphologyEx(
            mask_roi, cv2.MORPH_OPEN, self.kernel,
            iterations=self.morph_iterations
        )
        # Morphological close to fill small gaps in the line
        mask_roi = cv2.morphologyEx(
            mask_roi, cv2.MORPH_CLOSE, self.kernel,
            iterations=self.morph_iterations
        )

        # Build a full-frame mask for debug display
        full_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        full_mask[roi_y_start:, :] = mask_roi
        result.mask = full_mask

        # Find contours in the ROI mask
        contours, _ = cv2.findContours(
            mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return result

        # Pick the largest contour above the minimum area threshold
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_contour_area:
            return result

        # Compute centroid via image moments
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return result

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"]) + roi_y_start  # offset back to full frame

        # Adjust contour coordinates back to full-frame space
        adjusted_contour = largest.copy()
        adjusted_contour[:, :, 1] += roi_y_start

        result.found = True
        result.centroid_x = cx
        result.centroid_y = cy
        result.error_x = cx - frame_centre_x
        result.contour = adjusted_contour

        return result

    # ------------------------------------------------------------------
    # Static helpers for corner-aware steering (used by RobotController)
    # ------------------------------------------------------------------

    @staticmethod
    def nearest_point_on_contour(
        contour: np.ndarray,
        ref_x: float,
        ref_y: float,
    ) -> tuple:
        """
        Return the point on *contour* closest to (*ref_x*, *ref_y*).

        This is used by the controller instead of the overall line centroid
        when computing the lateral steering error.  At corners the whole-line
        centroid is biased toward the inside of the bend; the nearest contour
        point is always on the part of the line that is directly adjacent to
        the robot, giving a correct steering signal throughout the turn.

        Parameters
        ----------
        contour : np.ndarray
            Contour array as returned by ``cv2.findContours`` (shape N×1×2).
        ref_x, ref_y : float
            Reference position (typically the robot's ArUco centre).

        Returns
        -------
        tuple[float, float]
            ``(nearest_x, nearest_y)`` pixel coordinates of the nearest
            contour vertex.
        """
        pts = contour[:, 0, :].astype(np.float64)
        dists = np.hypot(pts[:, 0] - ref_x, pts[:, 1] - ref_y)
        idx = int(np.argmin(dists))
        return float(pts[idx, 0]), float(pts[idx, 1])

    @staticmethod
    def local_line_heading(
        contour: np.ndarray,
        near_x: float,
        near_y: float,
        robot_heading_deg: float,
        search_radius: float = config.LINE_LOOKAHEAD_RADIUS,
    ) -> float:
        """
        Estimate the local direction of the line near (*near_x*, *near_y*).

        A straight line is fitted through all contour points that fall within
        *search_radius* pixels of the anchor point.  The result is used by the
        controller for heading correction so that the robot aligns with the
        *actual* line direction — including after a corner — rather than being
        locked to a fixed ``ROBOT_DESIRED_HEADING``.

        Direction ambiguity (a line has two valid headings, 180° apart) is
        resolved by choosing the angle closest to *robot_heading_deg*, so the
        correction is always a small nudge rather than a 180° reversal.

        Falls back to *robot_heading_deg* when fewer than 4 nearby contour
        points are found.

        Parameters
        ----------
        contour : np.ndarray
            Contour array (shape N×1×2).
        near_x, near_y : float
            Anchor point (usually the nearest contour point to the robot).
        robot_heading_deg : float
            Current robot heading (degrees) used to disambiguate direction.
        search_radius : float
            Radius within which contour points are included in the line fit.
            Tune via ``LINE_LOOKAHEAD_RADIUS`` in config.py.

        Returns
        -------
        float
            Estimated local line heading in degrees [0, 360).
        """
        pts = contour[:, 0, :].astype(np.float64)
        dists = np.hypot(pts[:, 0] - near_x, pts[:, 1] - near_y)
        nearby = pts[dists <= search_radius]

        if len(nearby) < 4:
            return robot_heading_deg % 360

        line_params = cv2.fitLine(
            nearby.astype(np.float32).reshape(-1, 1, 2),
            cv2.DIST_L2, 0, 0.01, 0.01,
        )
        params = line_params.flatten()
        vx, vy = float(params[0]), float(params[1])
        # Same heading convention as ArUco: atan2(-vy, vx) because image Y is flipped
        heading = math.degrees(math.atan2(-vy, vx)) % 360

        # Resolve the 180° direction ambiguity: pick the angle closest to
        # robot_heading_deg so the correction is always a small adjustment.
        diff = ((heading - robot_heading_deg + 180) % 360) - 180
        if abs(diff) > 90:
            heading = (heading + 180) % 360

        return heading

    @staticmethod
    def draw_debug(frame: np.ndarray, result: LineDetectionResult) -> np.ndarray:
        """
        Return a copy of *frame* annotated with line-detection debug info.
        """
        vis = frame.copy()
        if result.mask is not None:
            # Tint detected pixels green
            tint = np.zeros_like(vis)
            tint[result.mask > 0] = (0, 200, 0)
            vis = cv2.addWeighted(vis, 0.7, tint, 0.3, 0)

        if result.found:
            if result.contour is not None:
                cv2.drawContours(vis, [result.contour], -1, (0, 255, 0), 2)
            cv2.circle(vis, (result.centroid_x, result.centroid_y), 6, (0, 255, 255), -1)
            cv2.putText(
                vis,
                f"Line err: {result.error_x:+d}px",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )
        else:
            cv2.putText(
                vis, "No line", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )
        return vis
