# OSG-Robotica-p3
Make robot move follow line based on openvc

## Project overview

An overhead-camera Python system that:

1. **Detects a black line** on a yellowish / wooden-plank surface with OpenCV (HSV masking).
2. **Tracks the robot** via a single ArUco marker mounted on top of it.
3. **Sends HTTP commands** (`FORWARD` / `LEFT` / `RIGHT` / `STOP`) to an ESP32 so it drives along the line.

```
Camera (overhead, fixed)
  │
  ├─► LineDetector  ──┐
  │                   ├──► RobotController ──► Esp32Client ──► ESP32 ──► L298N ──► Motors
  └─► ArucoTracker ──┘
```

---

## Required setup

### Hardware

| Component | Details |
|-----------|---------|
| **Overhead camera** | Any USB webcam (index 0 by default). Must be mounted **fixed above the track** looking straight down. Minimum recommended resolution: 640 × 480. |
| **ESP32 development board** | Any 30-pin or 38-pin ESP32 (e.g. ESP32 DevKit V1). Must be on the same WiFi network as the host PC. |
| **L298N motor driver** | Dual H-bridge. Connects to the ESP32 GPIO pins listed in the wiring table below. |
| **Two DC motors** | Standard TT or N20 gear motors. One connected to each L298N output channel. |
| **Robot chassis** | Any two-wheel (differential drive) chassis that holds the ESP32 + L298N. |
| **ArUco marker** | Printed marker ID **0** from dictionary **DICT_4X4_50**, mounted **flat on top** of the robot so the overhead camera can see it. Recommended size: 7 × 7 cm or larger. |
| **Track** | A **black line** (e.g. black tape) on a **light / yellowish surface** (e.g. wooden floor or white paper). |
| **WiFi router / hotspot** | The PC running `main.py` and the ESP32 must be on the same local network. |

### ESP32 wiring (L298N)

| Signal | ESP32 GPIO | L298N pin |
|--------|-----------|-----------|
| Motor 1 – direction A | 27 | IN1 |
| Motor 1 – direction B | 26 | IN2 |
| Motor 1 – PWM enable | 14 | ENA |
| Motor 2 – direction A | 33 | IN3 |
| Motor 2 – direction B | 25 | IN4 |
| Motor 2 – PWM enable | 32 | ENB |
| Ground | GND | GND |

> Power the L298N 12 V input from your battery pack.  
> The 5 V output of the L298N can power the ESP32 (connect to VIN).

### Software requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.9 or newer |
| opencv-python | ≥ 4.8.0 |
| numpy | ≥ 1.24.0 |
| requests | ≥ 2.28.0 |
| Arduino IDE (for ESP32 flash) | 2.x recommended |
| ESP32 board package | ≥ 2.0 (install via Arduino Board Manager) |

Install all Python dependencies at once:

```bash
pip install -r requirements.txt
```

### Generating the ArUco marker

Use the quick one-liner (requires opencv-python to be installed):

```python
python - <<'EOF'
import cv2
marker = cv2.aruco.generateImageMarker(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    id=0, sidePixels=400
)
cv2.imwrite("aruco_marker_id0.png", marker)
print("Saved aruco_marker_id0.png")
EOF
```

Print the saved PNG at roughly **7 × 7 cm** and attach it flat on top of the robot.

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

### 2. (Optional) Find your camera index

If you have more than one webcam, run the helper script to list available cameras and their indices:

```bash
python list_cameras.py
```

Use the index shown for your overhead camera with `--camera <index>` in the commands below (default is `0`).

### 3. Calibrate for your lighting environment

Run the calibration tool **before** the first real-world deployment.  
It opens a live camera feed with HSV trackbars so you can visually tune the black-line mask:

```bash
python calibration.py --camera 0
```

When the mask looks clean (white = line, black = background), press **`s`** to print the HSV values and copy them into `config.py`.

### 4. Run the controller

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

### 1. Edit WiFi credentials

Open `ESP32_script.ino` and replace the placeholders with your network details:

```cpp
const char* ssid     = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
```

### 2. Flash the firmware

1. Open `ESP32_script.ino` in the **Arduino IDE**.
2. In **Tools → Board** select your ESP32 board (e.g. *ESP32 Dev Module*).
3. Select the correct **COM / serial port**.
4. Click **Upload**.
5. After upload, open **Tools → Serial Monitor** at **115 200 baud**.
6. The assigned IP address is printed on boot, e.g. `IP address: 192.168.1.42`.

### 3. Update `config.py`

```python
ESP32_BASE_URL = "http://192.168.1.42"   # ← use the IP from Serial Monitor
```

### WiFi – Station (STA) mode

The firmware connects to your local WiFi network.  
Both the PC running `main.py` and the ESP32 **must be on the same network**.

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
| Robot starts turning too early at corners | Decrease `LINE_LOOKAHEAD_RADIUS` in `config.py` |
| Robot turns too late / misses the corner | Increase `LINE_LOOKAHEAD_RADIUS` in `config.py` |
| ArUco marker lost at certain angles | Use a larger marker; ensure good overhead lighting |
| Commands not reaching ESP32 | Check WiFi connection; increase `ESP32_TIMEOUT` |

---

## Corner handling

Yes — the code is designed to work with corners.

### How the overhead-camera steering handles corners

The controller uses two mechanisms that together enable reliable corner navigation:

**1. Nearest-point lateral error** (`LineDetector.nearest_point_on_contour`)

On a straight segment the "line position" is well-represented by the centroid of the
detected contour.  At a corner, however, the entire L-shaped (or U-shaped) contour is
visible at once — and its **centroid is biased toward the inside of the bend**, not the
part of the line directly ahead of the robot.

To fix this, the controller finds the **nearest point on the contour** to the robot's
ArUco position and uses that as the steering reference.  This always points to the
adjacent part of the line regardless of how much of the corner is visible.

**2. Adaptive heading correction** (`LineDetector.local_line_heading`)

Without this fix, the heading correction always tries to point the robot at
`ROBOT_DESIRED_HEADING` (180° = left by default).  Once the robot has turned 90° around
a corner, this would produce a heading error of −90° and issue `CMD_RIGHT`, **reversing
the completed turn**.

Instead, the controller fits a straight line through all contour points within
`LINE_LOOKAHEAD_RADIUS` pixels of the nearest point (default 80 px) to estimate the
**local line direction**.  The heading correction targets this local direction, so:

- On a straight segment → behaves identically to the old fixed-heading correction.
- Approaching a corner → smoothly transitions the target heading as the robot turns.
- After the corner → locks to the new direction immediately.

### Track design tips for reliable corner navigation

| Recommendation | Reason |
|----------------|--------|
| Corner radius ≥ 10 cm | Tighter corners may need very low `SPEED_TURN` to avoid overshooting |
| Line width 3–5 cm | Wider lines give more contour points → better local-direction estimate |
| Consistent lighting across the corner | Shadows at the bend can break the HSV mask mid-corner |
| `LINE_LOOKAHEAD_RADIUS` = 60–100 px | 80 px (default) works for most 640×480 setups |

---

## Running tests

```bash
python -m pytest tests/ -v
```

All tests run without a camera or ESP32 (synthetic frames + dry-run mode).
