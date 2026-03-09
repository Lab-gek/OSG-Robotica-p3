"""
aruco_tracker.py – Detects and tracks a single ArUco marker (the robot).

Returns the marker's pixel centroid and heading angle so that the controller
can fuse this with line-detection data to make steering decisions.

IRL translation notes
---------------------
- Marker size on the physical robot: bigger = more reliable at a distance.
  Recommended minimum: 7 × 7 cm when the camera is ~1 m above the floor.
- Camera must be undistorted for accurate pose.  Pass a calibration matrix to
  ArucoTracker if you have one; otherwise pixel-based heading still works well.
- Use DICT_4X4_50 (default) – it's fast and has low false-positive rates.
  Avoid DICT_6X6 under time-constrained conditions.
"""

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import config


# Map config string → OpenCV ArUco constant
_ARUCO_DICT_MAP = {
    "DICT_4X4_50":   cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100":  cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50":   cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50":   cv2.aruco.DICT_6X6_50,
    "DICT_7X7_50":   cv2.aruco.DICT_7X7_50,
}


@dataclass
class ArucoDetectionResult:
    """Result of ArUco detection for a single frame."""
    found: bool = False
    marker_id: Optional[int] = None
    # Centre of the marker in pixel coordinates
    centre_x: Optional[float] = None
    centre_y: Optional[float] = None
    # Heading angle in degrees (0 = pointing right, 90 = pointing up, etc.)
    # Derived from the orientation of the marker corners.
    heading_deg: Optional[float] = None
    # Raw corners array (shape 4×2) for drawing
    corners: Optional[np.ndarray] = None


class ArucoTracker:
    """
    Detects a single ArUco marker on the robot and returns its pose.

    Parameters
    ----------
    dict_id : str
        ArUco dictionary name (must be a key of _ARUCO_DICT_MAP).
    target_marker_id : int
        The marker ID that is mounted on the robot.
    camera_matrix : np.ndarray or None
        3×3 camera intrinsic matrix.  If None, pose estimation is skipped and
        only the pixel-space heading is returned.
    dist_coeffs : np.ndarray or None
        Camera distortion coefficients.  Paired with camera_matrix.
    """

    def __init__(
        self,
        dict_id: str = config.ARUCO_DICT_ID,
        target_marker_id: int = config.ROBOT_MARKER_ID,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
    ):
        dict_const = _ARUCO_DICT_MAP.get(dict_id, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_const)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)

        self.target_id = target_marker_id
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)

    def detect(self, frame: np.ndarray) -> ArucoDetectionResult:
        """
        Detect the target ArUco marker in *frame* (BGR).

        Returns an ArucoDetectionResult.  If the marker is not visible, the
        result has found=False.
        """
        result = ArucoDetectionResult()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = self.detector.detectMarkers(gray)

        if ids is None:
            return result

        # Find our specific marker
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id != self.target_id:
                continue

            corners = corners_list[i][0]  # shape (4, 2)

            # Centre = mean of 4 corners
            cx = float(np.mean(corners[:, 0]))
            cy = float(np.mean(corners[:, 1]))

            # Heading: angle of the vector from corner[0] to corner[1]
            # (corner order: top-left, top-right, bottom-right, bottom-left)
            dx = corners[1][0] - corners[0][0]
            dy = corners[1][1] - corners[0][1]
            heading_deg = math.degrees(math.atan2(-dy, dx))  # -dy because Y is flipped

            result.found = True
            result.marker_id = int(marker_id)
            result.centre_x = cx
            result.centre_y = cy
            result.heading_deg = heading_deg
            result.corners = corners
            break

        return result

    @staticmethod
    def draw_debug(frame: np.ndarray, result: ArucoDetectionResult) -> np.ndarray:
        """Return a copy of *frame* annotated with ArUco tracking debug info."""
        vis = frame.copy()
        if not result.found:
            cv2.putText(
                vis, "No ArUco", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )
            return vis

        # Draw marker outline
        pts = result.corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=(255, 0, 255), thickness=2)

        # Draw centre
        cx, cy = int(result.centre_x), int(result.centre_y)
        cv2.circle(vis, (cx, cy), 6, (255, 0, 255), -1)

        # Draw heading arrow
        arrow_len = 40
        heading_rad = math.radians(result.heading_deg)
        ax = int(cx + arrow_len * math.cos(heading_rad))
        ay = int(cy - arrow_len * math.sin(heading_rad))
        cv2.arrowedLine(vis, (cx, cy), (ax, ay), (0, 255, 255), 2, tipLength=0.3)

        cv2.putText(
            vis,
            f"ID:{result.marker_id} hdg:{result.heading_deg:.1f}deg",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2,
        )
        return vis
