"""
calibration.py – Interactive HSV calibration tool.

Run this script BEFORE using the system in a new environment (different
lighting, camera, or floor surface).  It lets you visually tune the HSV
thresholds for black-line detection by showing live trackbars.

Usage
-----
    python calibration.py [--camera <index>] [--width <px>] [--height <px>]

Controls
--------
  H_min, H_max  – Hue range
  S_min, S_max  – Saturation range
  V_min, V_max  – Value (brightness) range
  ROI_top       – % of frame height to skip from the top (0–80)

When you are happy with the mask (white = line, black = background), note down
the six HSV values and update LINE_HSV_LOWER / LINE_HSV_UPPER in config.py.

Press  q  or  ESC  to quit.
Press  s  to print the current values to stdout (copy-paste them into config.py).
"""

import argparse
import sys
import time

import cv2
import numpy as np


WINDOW_MAIN  = "Calibration – Original + Mask"
WINDOW_BARS  = "HSV Trackbars"


def nothing(_: int) -> None:
    pass


def create_trackbars() -> None:
    cv2.namedWindow(WINDOW_BARS)
    # Black line defaults
    cv2.createTrackbar("H_min", WINDOW_BARS,   0, 180, nothing)
    cv2.createTrackbar("H_max", WINDOW_BARS, 180, 180, nothing)
    cv2.createTrackbar("S_min", WINDOW_BARS,   0, 255, nothing)
    cv2.createTrackbar("S_max", WINDOW_BARS, 255, 255, nothing)
    cv2.createTrackbar("V_min", WINDOW_BARS,   0, 255, nothing)
    cv2.createTrackbar("V_max", WINDOW_BARS,  80, 255, nothing)
    cv2.createTrackbar("ROI_top%", WINDOW_BARS, 40, 80, nothing)


def read_trackbars() -> tuple:
    h_min = cv2.getTrackbarPos("H_min",    WINDOW_BARS)
    h_max = cv2.getTrackbarPos("H_max",    WINDOW_BARS)
    s_min = cv2.getTrackbarPos("S_min",    WINDOW_BARS)
    s_max = cv2.getTrackbarPos("S_max",    WINDOW_BARS)
    v_min = cv2.getTrackbarPos("V_min",    WINDOW_BARS)
    v_max = cv2.getTrackbarPos("V_max",    WINDOW_BARS)
    roi   = cv2.getTrackbarPos("ROI_top%", WINDOW_BARS)
    return (h_min, s_min, v_min), (h_max, s_max, v_max), roi


def run(camera_index: int, width: int, height: int) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_index}", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Camera warm-up: some USB cameras return black frames until the sensor
    # stabilises (same fix as main.py).
    time.sleep(1.0)
    for _ in range(20):
        cap.read()

    create_trackbars()
    print("Calibration running.  Press 's' to print values, 'q'/ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame", file=sys.stderr)
            break

        lower, upper, roi_pct = read_trackbars()
        roi_y = int(height * roi_pct / 100)

        # Apply ROI (blank out top portion)
        display_frame = frame.copy()
        display_frame[:roi_y, :] = 0

        hsv  = cv2.cvtColor(display_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))

        # Morphological cleanup (mirrors line_detector.py)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Overlay mask on frame
        mask_bgr   = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined   = np.hstack([display_frame, mask_bgr])

        # Draw ROI line
        cv2.line(combined, (0, roi_y), (width, roi_y), (0, 255, 255), 1)

        cv2.imshow(WINDOW_MAIN, combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            print("\n--- Copy these into config.py ---")
            print(f"LINE_HSV_LOWER = {lower}")
            print(f"LINE_HSV_UPPER = {upper}")
            print(f"# ROI top fraction ≈ {roi_pct / 100:.2f}  (use in LineDetector roi_top_fraction)")
            print("---------------------------------\n")

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="HSV calibration tool for line detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width",  type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    args = parser.parse_args()
    run(args.camera, args.width, args.height)


if __name__ == "__main__":
    main()
