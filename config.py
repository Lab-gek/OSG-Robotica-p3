"""
config.py – Central configuration for the line-following / ArUco tracking system.

All values here are the primary knobs to turn when moving from lab conditions to
real-world (IRL) deployment.  See calibration.py for an interactive tool that
helps you find the correct HSV ranges for your specific lighting environment.
"""

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
# Camera is mounted OVERHEAD (fixed position above the track), NOT on the robot.
# The robot is tracked in the camera view via an ArUco marker placed on top of it.
CAMERA_INDEX = 0          # OpenCV VideoCapture index (0 = default webcam / USB cam)
FRAME_WIDTH  = 640        # Captured frame width  (pixels)
FRAME_HEIGHT = 480        # Captured frame height (pixels)
FRAME_FPS    = 30         # Target frames per second

# ---------------------------------------------------------------------------
# Black-line detection (HSV colour space)
#
# Strategy: the track surface is yellowish wood.  We *invert* the problem:
#   mask OUT the yellow background → what remains is the black line.
#   Alternatively we directly threshold on very-low-value (dark) pixels.
#
# Tune these ranges with calibration.py if the line detection is poor IRL.
# ---------------------------------------------------------------------------

# Direct dark-pixel approach (works well under consistent lighting)
LINE_HSV_LOWER = (0,   0,   0)    # H, S, V  – lower bound for "black"
LINE_HSV_UPPER = (180, 255, 80)   # H, S, V  – upper bound for "black"

# Yellow background range (used to *exclude* the background when needed)
BACKGROUND_HSV_LOWER = (15,  60,  60)
BACKGROUND_HSV_UPPER = (35, 255, 255)

# Morphological cleanup (removes noise in the binary mask)
MORPH_KERNEL_SIZE = 5      # pixels  – odd integer
MORPH_ITERATIONS  = 2

# Minimum contour area to be considered a real line segment (filters noise)
LINE_MIN_CONTOUR_AREA = 500   # pixels²

# Fraction of the frame height to skip from the top when detecting the line.
# With an on-robot camera this was ~0.4 to avoid the robot body in the frame.
# With an overhead (top-down) camera the full frame is valid → set to 0.0.
ROI_TOP_FRACTION = 0.0

# ---------------------------------------------------------------------------
# ArUco marker
# ---------------------------------------------------------------------------
ARUCO_DICT_ID  = "DICT_4X4_50"   # OpenCV ArUco dictionary name
ROBOT_MARKER_ID = 0               # The specific marker ID mounted on the robot

# ---------------------------------------------------------------------------
# Control logic
# ---------------------------------------------------------------------------
# The frame is divided into CONTROL_ZONES vertical zones.
# Zone 0 = far left, zone (CONTROL_ZONES-1) = far right.
# The robot tries to keep the line centred in the middle zone.
CONTROL_ZONES = 5

# Dead-band: if the lateral error between the robot (ArUco) and the line is
# within ±CENTRE_TOLERANCE pixels, the robot is considered "on track" → FORWARD.
CENTRE_TOLERANCE = 30   # pixels

# Desired heading for the robot in degrees (0 = right, 90 = up, 180 = left).
# The robot follows the line from the RIGHT side of the screen to the LEFT,
# so its desired heading is 180° (pointing left in the overhead camera view).
ROBOT_DESIRED_HEADING = 180.0

# Speed levels sent to ESP32.
# The firmware maps slider value 0→25→50→75→100 to PWM duty 0→200→220→237→255.
# Valid values are multiples of 25 in [0, 100].
SPEED_FORWARD = 75      # slider value for straight driving
SPEED_TURN    = 50      # slider value when turning
SPEED_STOP    = 0

# ---------------------------------------------------------------------------
# ESP32 HTTP API  (ESP32_script.ino  –  Station / STA mode)
# ---------------------------------------------------------------------------
# The ESP32 connects to your local WiFi (ssid "wojo" in the firmware).
# Find the assigned IP on the Serial monitor after boot and set it below.
ESP32_BASE_URL = "http://192.168.1.1"   # ← replace with your ESP32's IP address
ESP32_TIMEOUT  = 0.5    # seconds – keep short so the control loop isn't blocked

# Individual action endpoints
ESP32_EP_FORWARD = "/forward"
ESP32_EP_LEFT    = "/left"
ESP32_EP_RIGHT   = "/right"
ESP32_EP_STOP    = "/stop"
ESP32_EP_REVERSE = "/reverse"

# Speed endpoint  –  GET /speed?value=<0-100>
ESP32_EP_SPEED   = "/speed"

# ---------------------------------------------------------------------------
# Debug / visualisation
# ---------------------------------------------------------------------------
SHOW_DEBUG_WINDOW = True    # Display annotated OpenCV windows while running
DEBUG_WINDOW_SCALE = 1.0    # Scale factor for the debug window (e.g. 0.5 for half size)
