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
| `ESP32_script.ino` | ESP32 firmware (WiFi + WebServer + L298N motor driver) |
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

## ESP32 setup  (`ESP32_script.ino`)

### WiFi – Station (STA) mode

The firmware connects to your local WiFi network (SSID **wojo** by default).  
After uploading the sketch, open the Serial monitor at 115200 baud — the assigned IP address is printed there. Then set it in `config.py`:

```python
ESP32_BASE_URL = "http://<IP_FROM_SERIAL_MONITOR>"
```

### HTTP API

The Python controller makes two types of GET requests:

**1. Set speed** (sent whenever speed changes):
```
GET /speed?value=<0|25|50|75|100>
```
The value is a slider step matching the firmware's PWM map:

| `value` | PWM duty (0–255) |
|---------|-----------------|
| 0       | 0 (motors off)  |
| 25      | 200             |
| 50      | ≈ 220           |
| 75      | ≈ 237           |
| 100     | 255             |

**2. Set direction** (sent whenever action changes):
```
GET /forward    – both motors forward
GET /left       – motor 1 back, motor 2 forward  (arc left)
GET /right      – motor 1 forward, motor 2 back  (arc right)
GET /stop       – all motor pins LOW (coast)
GET /reverse    – both motors backward
```

> **Tip:** `SPEED_FORWARD` and `SPEED_TURN` in `config.py` default to `75` and `50`.  
> Tune them if the robot is too fast / slow on your track.

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
