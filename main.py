"""
main.py – Main control loop for the line-following / ArUco-tracking robot.

Architecture
------------

    Camera
      │
      ▼
    LineDetector  ──┐
                    ├──► RobotController ──► Esp32Client ──► ESP32
    ArucoTracker  ──┘

Each iteration:
  1. Capture a frame from the camera.
  2. Detect the black line (LineDetector).
  3. Detect the ArUco marker on the robot (ArucoTracker).
  4. Compute the desired action (RobotController).
  5. Send the action to the ESP32 (Esp32Client).
  6. (Optional) Display a debug window.

Usage
-----
    python main.py [--camera <index>] [--esp32 <url>] [--dry-run] [--no-display]

Options
-------
  --camera    Camera index or RTSP/HTTP URL (default: value in config.py)
  --esp32     ESP32 base URL (default: value in config.py)
  --dry-run   Log commands instead of sending HTTP requests (no hardware needed)
  --no-display  Disable the OpenCV debug window (headless / SSH mode)

Press  q  or  ESC  in the debug window to stop.
Press  Ctrl-C  in the terminal to stop.
"""

import argparse
import logging
import sys
import time

import cv2

import config
from line_detector  import LineDetector
from aruco_tracker  import ArucoTracker
from robot_controller import RobotController
from esp32_client   import Esp32Client
from visualizer     import DebugVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Line-following robot controller with ArUco tracking"
    )
    parser.add_argument(
        "--camera", default=None,
        help=f"Camera index or URL (default: {config.CAMERA_INDEX})"
    )
    parser.add_argument(
        "--esp32", default=None,
        help=f"ESP32 base URL (default: {config.ESP32_BASE_URL})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log commands instead of sending to ESP32"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable OpenCV debug windows (headless mode)"
    )
    return parser.parse_args()


def open_camera(source) -> cv2.VideoCapture:
    """Open camera; source can be an int (index) or a string (URL)."""
    try:
        source = int(source)
    except (TypeError, ValueError):
        pass  # keep as string (URL / device path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open camera: %s", source)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.FRAME_FPS)
    return cap


def main() -> None:
    args = parse_args()

    camera_source = args.camera if args.camera is not None else config.CAMERA_INDEX
    esp32_url     = args.esp32  if args.esp32  is not None else config.ESP32_BASE_URL
    show_display  = not args.no_display and config.SHOW_DEBUG_WINDOW

    # ------------------------------------------------------------------ setup
    cap        = open_camera(camera_source)
    detector   = LineDetector()
    tracker    = ArucoTracker()
    controller = RobotController()
    client     = Esp32Client(base_url=esp32_url, dry_run=args.dry_run)
    visualizer = DebugVisualizer(scale=config.DEBUG_WINDOW_SCALE)

    logger.info("System ready.  Camera=%s  ESP32=%s  dry_run=%s",
                camera_source, esp32_url, args.dry_run)

    fps_timer   = time.time()
    frame_count = 0
    fps_display = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame capture failed – retrying…")
                time.sleep(0.05)
                continue

            # ------------------------------------------ perception
            line_result  = detector.detect(frame)
            aruco_result = tracker.detect(frame)

            # ------------------------------------------ decision
            command = controller.compute(line_result, aruco_result)

            # ------------------------------------------ actuation
            sent = client.send(command)
            if sent:
                logger.info("Sent: %s", command)

            # ------------------------------------------ debug display
            frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps_display = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer   = now

            if show_display:
                vis = visualizer.build_frame(
                    frame, line_result, aruco_result, command, fps_display
                )
                cv2.imshow(DebugVisualizer.WINDOW_TITLE, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    logger.info("Quit key pressed.")
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        logger.info("Sending STOP and shutting down.")
        client.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
