"""Overhead-camera line follower – main entry point with OpenCV GUI.

Usage
-----
    python project/main.py [--camera INDEX] [--esp32 http://IP]

Controls
--------
    q / Esc     Quit (stops motors, flushes log)
    Space       Start / Pause processing
    s           Emergency stop (motors off, stays in STOP state)
    r           Reset state machine (back to FOLLOW_LINE)
    c           Recalibrate normal line width
    m           Toggle mask view overlay
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Project modules
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

import aruco
import comms
import pid as pid_module
import state_machine as sm_module
import vision as vision_module
from state_machine import StateMachine
from vision import VisionProcessor
from pid import PID

# ---------------------------------------------------------------------------
# Default constants (overridden by trackbars at runtime)
# ---------------------------------------------------------------------------
ESP32_IP: str = "http://192.168.1.42"
BASE_SPEED: int = 180
TURN_SPEED: int = 120
THRESHOLD: int = 60
JUNCTION_WIDTH_RATIO: float = 2.8
NO_LINE_LIMIT: int = 5
HEADING_TOLERANCE: float = 10.0
KP: float = 0.4
KI: float = 0.001
KD: float = 0.1

LOG_PATH = PROJECT_DIR / "run_log.csv"
WINDOW_MAIN = "Overhead Follower"
WINDOW_MASK = "Mask / ROI"

# ---------------------------------------------------------------------------
# Trackbar helpers
# ---------------------------------------------------------------------------
_TB_WIN = "Controls"
_PID_SCALE = 1000  # trackbar ints store Kp*1000 etc.
_RATIO_SCALE = 10


def _create_controls() -> None:
    cv2.namedWindow(_TB_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_TB_WIN, 420, 480)

    def _tb(name: str, val: int, maxval: int) -> None:
        cv2.createTrackbar(name, _TB_WIN, val, maxval, lambda _: None)

    _tb("BASE_SPEED", BASE_SPEED, 255)
    _tb("TURN_SPEED", TURN_SPEED, 255)
    _tb("Threshold", THRESHOLD, 255)
    _tb("JunctionRatio*10", int(JUNCTION_WIDTH_RATIO * _RATIO_SCALE), 100)
    _tb("NoLineLimit", NO_LINE_LIMIT, 30)
    _tb("HeadingTol", int(HEADING_TOLERANCE), 45)
    _tb("Kp*1000", int(KP * _PID_SCALE), 2000)
    _tb("Ki*1000", int(KI * _PID_SCALE), 500)
    _tb("Kd*1000", int(KD * _PID_SCALE), 2000)


def _read_controls(controller: PID) -> None:
    """Sync trackbar values into global state and PID object."""
    sm_module.BASE_SPEED = cv2.getTrackbarPos("BASE_SPEED", _TB_WIN)
    sm_module.TURN_SPEED = cv2.getTrackbarPos("TURN_SPEED", _TB_WIN)
    vision_module.THRESHOLD = cv2.getTrackbarPos("Threshold", _TB_WIN)
    vision_module.JUNCTION_WIDTH_RATIO = (
        cv2.getTrackbarPos("JunctionRatio*10", _TB_WIN) / _RATIO_SCALE
    )
    sm_module.NO_LINE_LIMIT = max(1, cv2.getTrackbarPos("NoLineLimit", _TB_WIN))
    sm_module.HEADING_TOLERANCE = float(max(1, cv2.getTrackbarPos("HeadingTol", _TB_WIN)))
    controller.kp = cv2.getTrackbarPos("Kp*1000", _TB_WIN) / _PID_SCALE
    controller.ki = cv2.getTrackbarPos("Ki*1000", _TB_WIN) / _PID_SCALE
    controller.kd = cv2.getTrackbarPos("Kd*1000", _TB_WIN) / _PID_SCALE


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------

def _draw_status(
    frame: np.ndarray,
    state: str,
    junctions_done: int,
    heading: Optional[float],
    left: int,
    right: int,
    fps: float,
    running: bool,
) -> None:
    lines = [
        f"State: {state}  Junctions: {junctions_done}",
        f"Hdg: {heading:.1f}" if heading is not None else "Hdg: --",
        f"L={left}  R={right}",
        f"FPS: {fps:.1f}",
        "PAUSED" if not running else "",
    ]
    y = 24
    for line in lines:
        if line:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            y += 22


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _open_log() -> tuple[object, csv.DictWriter]:
    f = open(LOG_PATH, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        f, fieldnames=["t", "rx", "ry", "heading", "state", "junctions_done", "left", "right"]
    )
    writer.writeheader()
    return f, writer


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Overhead camera line follower")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--esp32", type=str, default=ESP32_IP, help="ESP32 base URL")
    args = parser.parse_args()

    comms.ESP32_IP = args.esp32

    # --- init subsystems ---
    vp = VisionProcessor(camera_index=args.camera)
    if not vp.open():
        print(f"[main] ERROR: Cannot open camera {args.camera}")
        sys.exit(1)

    controller = PID(kp=KP, ki=KI, kd=KD)
    machine = StateMachine()

    log_file, log_writer = _open_log()

    # --- windows & controls ---
    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_MASK, cv2.WINDOW_NORMAL)
    _create_controls()

    running = True        # whether the pipeline is active
    show_mask = True
    t_start = time.time()
    fps_acc = 0.0
    fps_alpha = 0.1       # EMA factor

    print("[main] Starting. Press 'q' or Esc to quit.")
    print(f"[main] Logging to {LOG_PATH}")

    left = right = 0

    try:
        while True:
            t_now = time.time()
            t_elapsed = t_now - t_start

            # --- trackbars ---
            _read_controls(controller)

            # --- grab frame ---
            frame = vp.read_frame()
            if frame is None:
                print("[main] Camera read failed – retrying…")
                time.sleep(0.05)
                continue

            frame_h, frame_w = frame.shape[:2]
            display = frame.copy()

            rx = ry = heading = None
            mask = roi = None

            if running:
                # --- ArUco ---
                rx, ry, heading = aruco.detect(frame)
                if rx is not None:
                    aruco.draw_marker(display, rx, ry, heading)

                # --- Vision ---
                mask, roi = vp.process(frame)
                vp.draw_overlays(display)

                # --- PID ---
                if vp.line_cx is not None:
                    error = vp.line_cx - frame_w // 2
                    correction = controller.update(float(error))
                else:
                    correction = 0.0

                # --- State machine ---
                left, right = machine.update(
                    line_cx=vp.line_cx,
                    frame_width=frame_w,
                    at_junction=vp.at_junction,
                    heading=heading,
                    pid_correction=correction,
                )

                # --- Comms ---
                comms.send_motors(left, right)

                # --- Log ---
                log_writer.writerow(
                    {
                        "t": round(t_elapsed, 4),
                        "rx": round(rx, 2) if rx is not None else "",
                        "ry": round(ry, 2) if ry is not None else "",
                        "heading": round(heading, 2) if heading is not None else "",
                        "state": machine.state.name,
                        "junctions_done": machine.junctions_done,
                        "left": left,
                        "right": right,
                    }
                )

                if not machine.running:
                    print("[main] State machine reached STOP – exiting loop.")
                    comms.stop()
                    running = False

            # --- FPS estimation ---
            dt = max(time.time() - t_now, 1e-6)
            fps_acc = fps_alpha * (1.0 / dt) + (1 - fps_alpha) * fps_acc

            # --- Overlays ---
            _draw_status(
                display,
                state=machine.state.name,
                junctions_done=machine.junctions_done,
                heading=heading,
                left=left,
                right=right,
                fps=fps_acc,
                running=running,
            )

            # Keyboard hints at bottom.
            hint = "q=quit  Space=start/pause  s=estop  r=reset  c=recalib  m=mask"
            cv2.putText(
                display, hint, (10, display.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1,
            )

            cv2.imshow(WINDOW_MAIN, display)

            if show_mask and mask is not None:
                # Stack mask (converted to BGR) and ROI side by side.
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                combined = np.hstack([mask_bgr, roi])
                cv2.imshow(WINDOW_MASK, combined)
            elif show_mask:
                blank = np.zeros((100, 320, 3), dtype=np.uint8)
                cv2.putText(blank, "No mask yet", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
                cv2.imshow(WINDOW_MASK, blank)

            # --- Keyboard ---
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                break
            elif key == ord(" "):
                running = not running
                if not running:
                    comms.stop()
            elif key == ord("s"):
                comms.stop()
                machine.state = sm_module.State.STOP
                running = False
                left = right = 0
            elif key == ord("r"):
                machine.reset()
                controller.reset()
                running = True
            elif key == ord("c"):
                vp.recalibrate_width()
                print("[main] Recalibrating normal line width…")
            elif key == ord("m"):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow(WINDOW_MASK)

    finally:
        # --- Cleanup (always runs, even on exception) ---
        print("[main] Shutting down…")
        comms.stop()
        vp.release()
        log_file.flush()
        log_file.close()
        cv2.destroyAllWindows()
        print(f"[main] Log saved to {LOG_PATH}")


if __name__ == "__main__":
    main()
