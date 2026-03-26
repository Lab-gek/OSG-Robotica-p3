# Overhead-Camera Line Follower – Agent Specification

## Overview

This document describes the PC-side Python pipeline for an overhead-camera robot
line follower.  The robot is an ESP32-based vehicle; all motor commands are sent via
HTTP from the PC.

---

## Architecture

```
project/
├── AGENT.md           # this file
├── main.py            # entry point, GUI, main loop (~30 fps)
├── vision.py          # camera capture, ROI preprocessing, contour/centroid
├── aruco.py           # ArUco detection, pose/heading extraction
├── pid.py             # PID controller
├── state_machine.py   # junction state machine
├── comms.py           # HTTP sender to ESP32
└── run_log.csv        # auto-generated at runtime (do not commit)
```

---

## Heading Convention

| Degrees | Direction |
|---------|-----------|
| 0°      | Right     |
| 90°     | Down      |
| 180°    | Left      |
| 270°    | Up        |

Heading is derived from the ArUco marker's left→right edge vector projected onto
the image plane.

---

## State Machine

| State          | Description |
|----------------|-------------|
| `FOLLOW_LINE`  | PID steers on lateral error; detects junction by blob-width spike. |
| `JUNCTION_TURN`| Rotate in place to a target heading; transitions when within `HEADING_TOLERANCE`. |
| `REALIGN`      | Creep forward until line is reacquired. |
| `STOP`         | Send stop and exit loop. |

### Junction targets (corrected)

```python
JUNCTION_TARGETS = [90, 180]   # junction 1 → down (90°), junction 2 → left (180°)
```

---

## Configuration Constants

| Constant              | Default | Description |
|-----------------------|---------|-------------|
| `ESP32_IP`            | `http://192.168.1.42` | Base URL of the ESP32 |
| `BASE_SPEED`          | 180     | Forward PWM (0–255) |
| `TURN_SPEED`          | 120     | In-place rotation PWM |
| `JUNCTION_WIDTH_RATIO`| 2.8     | blob_width > normal_width × ratio → junction |
| `NO_LINE_LIMIT`       | 5       | Consecutive frames with no line before REALIGN |
| `HEADING_TOLERANCE`   | 10      | Degrees – acceptable error for JUNCTION_TURN |
| `Kp`                  | 0.4     | PID proportional gain |
| `Ki`                  | 0.001   | PID integral gain |
| `Kd`                  | 0.1     | PID derivative gain |
| `THRESHOLD`           | 60      | Grayscale threshold for line detection (inverted) |

All constants are adjustable at runtime via the OpenCV trackbar GUI.

---

## Per-Frame Pipeline

1. Grab frame from camera.
2. Detect ArUco marker → `(rx, ry, heading)`.
3. Preprocess ROI (middle third of frame) → binary mask.
4. Find largest contour → `(line_cx, blob_width)`.
5. Compute PID correction from lateral error (`line_cx - frame_width/2`).
6. State machine step → `(left_speed, right_speed)`.
7. Send HTTP commands via `comms.send_motors(left, right)`.
8. Log frame to `run_log.csv`.

---

## comms.py – ESP32 Endpoint Mapping

| Condition                   | Endpoint  |
|-----------------------------|-----------|
| avg PWM < 10                | `/stop`   |
| both negative               | `/reverse`|
| left − right > 30           | `/right`  |
| right − left > 30           | `/left`   |
| else                        | `/forward`|

Speed percentage: `avg(|left|, |right|) / 255 × 100` → sent to `/speed?value=<0-100>` only when it changes.

---

## vision.py – ROI & Preprocessing

- **ROI**: `frame[h//3 : 2*h//3, :]` (middle third, full width).
- **Preprocessing**: grayscale → Gaussian blur 5×5 → `cv2.threshold(…, THRESHOLD, 255, THRESH_BINARY_INV)`.
- **Centroid**: moments of the largest contour.
- **Junction detection**: `blob_width > normal_width × JUNCTION_WIDTH_RATIO`.  
  `normal_width` is calibrated automatically over the first ~30 frames; a `recalibrate_width()` method resets it.

---

## aruco.py – Heading Computation

Uses `cv2.aruco` (from `opencv-contrib-python`).

```python
dx = right_mid.x - left_mid.x
dy = right_mid.y - left_mid.y
heading = atan2(dy, dx) converted to degrees, wrapped to [0, 360)
```

---

## GUI Controls

| Key      | Action |
|----------|--------|
| `q`/Esc  | Quit (stops motors, flushes log) |
| Space    | Start / Pause processing |
| `s`      | Emergency stop |
| `r`      | Reset state machine |
| `c`      | Recalibrate normal line width |
| `m`      | Toggle mask/ROI window |

Trackbars in the "Controls" window: `BASE_SPEED`, `TURN_SPEED`, `Threshold`,
`JunctionRatio×10`, `NoLineLimit`, `HeadingTol`, `Kp×1000`, `Ki×1000`, `Kd×1000`.

---

## Logging (`run_log.csv`)

CSV header:

```
t, rx, ry, heading, state, junctions_done, left, right
```

`t` is seconds since program start (float, 4 decimal places).

---

## Dependencies

```
opencv-contrib-python
numpy
requests
```

Install: `pip install opencv-contrib-python numpy requests`

---

## Implementation Notes

- `heading_error(current, target)` returns the shortest signed angular distance
  in (−180, 180], enabling minimal-rotation turns.
- Network errors from `requests` are caught and printed as warnings; the main
  loop continues.
- `run_log.csv` is written to `project/run_log.csv` and is listed in `.gitignore`.
