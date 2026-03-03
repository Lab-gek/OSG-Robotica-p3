"""
config.py – Central configuration for the line-following / ArUco tracking system.

All values here are the primary knobs to turn when moving from lab conditions to
real-world (IRL) deployment.  See calibration.py for an interactive tool that
helps you find the correct HSV ranges for your specific lighting environment.
"""

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
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

# Dead-band: if the line centre is within ±CENTRE_TOLERANCE pixels of the
# frame centre, the robot is considered "on track" → send FORWARD.
CENTRE_TOLERANCE = 30   # pixels

# Speed levels sent to ESP32 (map to PWM duty-cycle % on the ESP32 side)
SPEED_FORWARD = 60      # % duty cycle when driving straight
SPEED_TURN    = 50      # % duty cycle for the faster wheel when turning
SPEED_STOP    = 0

# ---------------------------------------------------------------------------
# ESP32 HTTP API
# ---------------------------------------------------------------------------
ESP32_BASE_URL = "http://192.168.4.1"   # Default ESP32 AP address; change to
                                         # your network IP if using STA mode.
ESP32_TIMEOUT  = 0.5    # seconds – keep short so the control loop isn't blocked

# Endpoint paths (GET /command?action=<ACTION>&speed=<SPEED>)
ESP32_COMMAND_ENDPOINT = "/command"

# ---------------------------------------------------------------------------
# Debug / visualisation
# ---------------------------------------------------------------------------
SHOW_DEBUG_WINDOW = True    # Display annotated OpenCV windows while running
DEBUG_WINDOW_SCALE = 1.0    # Scale factor for the debug window (e.g. 0.5 for half size)
