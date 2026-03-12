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

import config


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
    roi_default = int(max(0.0, min(0.8, config.ROI_TOP_FRACTION)) * 100)
    cv2.createTrackbar("ROI_top%", WINDOW_BARS, roi_default, 80, nothing)


def read_trackbars() -> tuple:
    h_min = cv2.getTrackbarPos("H_min",    WINDOW_BARS)
    h_max = cv2.getTrackbarPos("H_max",    WINDOW_BARS)
    s_min = cv2.getTrackbarPos("S_min",    WINDOW_BARS)
    s_max = cv2.getTrackbarPos("S_max",    WINDOW_BARS)
    v_min = cv2.getTrackbarPos("V_min",    WINDOW_BARS)
    v_max = cv2.getTrackbarPos("V_max",    WINDOW_BARS)
    roi   = cv2.getTrackbarPos("ROI_top%", WINDOW_BARS)
    return (h_min, s_min, v_min), (h_max, s_max, v_max), roi


def _configure_capture(cap: cv2.VideoCapture, width: int, height: int) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, config.FRAME_FPS)


def _probe_capture(cap: cv2.VideoCapture, probe_frames: int = 12) -> tuple[bool, float, tuple[int, int] | None]:
    """Read a few frames and check whether we get a non-black stream."""
    brightness_values: list[float] = []
    shape: tuple[int, int] | None = None
    for _ in range(probe_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_values.append(float(gray.mean()))
        h, w = frame.shape[:2]
        shape = (w, h)

    if not brightness_values:
        return False, 0.0, shape

    avg_brightness = float(sum(brightness_values) / len(brightness_values))
    # Mean near zero across many frames indicates an effectively black stream.
    return avg_brightness > 5.0, avg_brightness, shape


def open_camera(source, width: int, height: int) -> cv2.VideoCapture:
    """Open camera source and print requested vs negotiated stream settings."""
    try:
        source = int(source)
    except (TypeError, ValueError):
        pass

    backends: list[tuple[str, int | None]] = [("ANY", None)]
    if isinstance(source, int):
        backends = [("V4L2", cv2.CAP_V4L2), ("ANY", None)]

    best_cap: cv2.VideoCapture | None = None
    best_brightness = -1.0
    best_name = "UNKNOWN"

    for backend_name, backend_flag in backends:
        cap = cv2.VideoCapture(source) if backend_flag is None else cv2.VideoCapture(source, backend_flag)
        if not cap.isOpened():
            continue

        _configure_capture(cap, width, height)
        ok, brightness, _ = _probe_capture(cap)
        if ok:
            best_cap = cap
            best_brightness = brightness
            best_name = backend_name
            break

        if brightness > best_brightness:
            if best_cap is not None:
                best_cap.release()
            best_cap = cap
            best_brightness = brightness
            best_name = backend_name
        else:
            cap.release()

    if best_cap is None:
        print(f"ERROR: Cannot open camera: {source}", file=sys.stderr)
        sys.exit(1)

    _configure_capture(best_cap, width, height)
    actual_w = int(best_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(best_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = best_cap.get(cv2.CAP_PROP_FPS)
    print(
        "Camera opened: requested "
        f"{width}x{height}@{config.FRAME_FPS}  got {actual_w}x{actual_h}@{actual_fps:.1f}"
    )
    print(f"Capture backend selected: {best_name}  probe_brightness={best_brightness:.1f}")
    return best_cap


def run(camera_source, width: int, height: int) -> None:
    cap = open_camera(camera_source, width, height)

    # Camera warm-up: some USB cameras return black frames until the sensor
    # stabilises (same fix as main.py).
    time.sleep(1.0)
    for _ in range(20):
        cap.read()

    ret_check, check_frame = cap.read()
    if ret_check and check_frame is not None:
        brightness = float(np.mean(cv2.cvtColor(check_frame, cv2.COLOR_BGR2GRAY)))
        print(
            "Post-warmup frame: "
            f"shape={check_frame.shape}  mean_brightness={brightness:.1f}"
        )
    else:
        print(f"WARNING: Post-warmup frame read failed (ret={ret_check})")

    create_trackbars()
    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    print("Calibration running.  Press 's' to print values, 'q'/ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame", file=sys.stderr)
            break

        frame_h, frame_w = frame.shape[:2]
        gray_mean = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))

        lower, upper, roi_pct = read_trackbars()
        roi_y = int(frame_h * roi_pct / 100)

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
        cv2.line(combined, (0, roi_y), (frame_w, roi_y), (0, 255, 255), 1)
        cv2.putText(
            combined,
            f"brightness={gray_mean:.1f}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

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
    parser.add_argument(
        "--camera",
        default=config.CAMERA_INDEX,
        help=f"Camera index or URL (default: {config.CAMERA_INDEX})",
    )
    parser.add_argument("--width",  type=int, default=config.FRAME_WIDTH, help="Frame width")
    parser.add_argument("--height", type=int, default=config.FRAME_HEIGHT, help="Frame height")
    args = parser.parse_args()
    run(args.camera, args.width, args.height)


if __name__ == "__main__":
    main()
