# OSG-Robotica-p3
Make robot move follow line based on openvc

## Project overview

An overhead-camera Python system that:

1. **Detects a black line** on a yellowish / wooden-plank surface with OpenCV (HSV masking).
2. **Tracks the robot** via a single ArUco marker mounted on top of it.
3. **Sends HTTP commands** (`FORWARD` / `LEFT` / `RIGHT` / `STOP`) to an ESP32 so it drives along the line.

```
Camera
  │
  ├─► LineDetector  ──┐
  │                   ├──► RobotController ──► Esp32Client ──► ESP32
  └─► ArucoTracker ──┘
```

---

## File layout

| File | Purpose |
|------|---------|
| `main.py` | Main control loop |
| `config.py` | All tunable parameters (HSV ranges, speeds, ESP32 URL, …) |
| `line_detector.py` | Black-line detection (HSV threshold + morphology) |
| `aruco_tracker.py` | ArUco marker detection & heading |
| `robot_controller.py` | Decision logic → command |
| `esp32_client.py` | HTTP client (sends commands to ESP32) |
| `calibration.py` | Interactive HSV calibration tool |
| `firmware/robot_controller.ino` | Reference ESP32 sketch (HTTP server + motor driver) |
| `requirements.txt` | Python dependencies |
| `tests/` | Unit tests (pytest) |

---

## Quick start

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Calibrate for your lighting environment

Run the calibration tool **before** the first real-world deployment.  
It opens a live camera feed with HSV trackbars so you can visually tune the black-line mask:

```bash
python calibration.py --camera 0
```

When the mask looks clean (white = line, black = background), press **`s`** to print the HSV values and copy them into `config.py`.

### 3. Run the controller

```bash
# With debug window (default)
python main.py

# Headless (no display / SSH)
python main.py --no-display

# Test without hardware (logs commands, doesn't call ESP32)
python main.py --dry-run

# Custom camera and ESP32 address
python main.py --camera 1 --esp32 http://192.168.1.42
```

Press **`q`** or **`ESC`** in the debug window, or **Ctrl-C** in the terminal, to stop.

---

## ESP32 setup

### WiFi modes

| Mode | When to use | `ESP32_BASE_URL` in `config.py` |
|------|------------|--------------------------------|
| **Access-Point (AP)** | Default – ESP32 creates its own WiFi | `http://192.168.4.1` |
| **Station (STA)** | Both on the same router | IP printed on Serial monitor |

### HTTP API (what the Python side sends)

```
GET /command?action=<ACTION>&speed=<SPEED>
```

| `action` | Meaning |
|----------|---------|
| `FORWARD` | Drive straight |
| `LEFT` | Turn left |
| `RIGHT` | Turn right |
| `STOP` | Halt motors |

`speed` is 0–100 (mapped to PWM duty cycle on the ESP32).

> **Note:** `firmware/robot_controller.ino` is a reference sketch that implements this API with an L298N motor driver.  
> Merge the WiFi + WebServer sections into your existing `.ino` if you already have one.

---

## IRL (real-world) calibration tips

The biggest challenge when moving from a lab setup to a real track:

| Issue | Solution |
|-------|---------|
| Line not detected (shadows, glare) | Re-run `calibration.py`, lower `V_max` slider |
| False positives (wood grain detected) | Increase `LINE_MIN_CONTOUR_AREA` in `config.py` |
| Robot oscillates left/right | Increase `CENTRE_TOLERANCE` or lower `SPEED_TURN` |
| ArUco marker lost at certain angles | Use a larger marker; ensure good overhead lighting |
| Commands not reaching ESP32 | Check WiFi connection; increase `ESP32_TIMEOUT` |

---

## Running tests

```bash
python -m pytest tests/ -v
```

All tests run without a camera or ESP32 (synthetic frames + dry-run mode).
