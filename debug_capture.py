#!/usr/bin/env python3
"""Debug script - captures frames from camera and saves them to disk."""

import cv2
import time
import numpy as np

print("Opening camera 0...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit(1)

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera opened. Waiting 2 seconds for exposure adjustment...")
time.sleep(2)

# Discard first few frames to let camera adjust
for i in range(10):
    ret, frame = cap.read()
    time.sleep(0.1)

print("\nCapturing and analyzing frames...\n")

for i in range(5):
    ret, frame = cap.read()
    if not ret:
        print(f"Frame {i}: FAILED to capture")
        continue
    
    # Save frame
    filename = f"debug_frame_{i}.jpg"
    cv2.imwrite(filename, frame)
    
    # Analyze frame
    mean_brightness = np.mean(frame)
    min_val = np.min(frame)
    max_val = np.max(frame)
    
    print(f"Frame {i}: Saved as {filename}")
    print(f"  Brightness: mean={mean_brightness:.1f}, min={min_val}, max={max_val}")
    print(f"  Shape: {frame.shape}")
    
    if mean_brightness < 10:
        print("  ⚠️  Very dark image - camera might be covered or room is too dark")
    elif mean_brightness > 245:
        print("  ⚠️  Very bright/white image - camera might be overexposed")
    else:
        print("  ✓ Brightness looks normal")
    
    print()
    time.sleep(0.5)

cap.release()
print("Done! Check the debug_frame_*.jpg files in this directory.")
print("Try: eog debug_frame_0.jpg  (or your preferred image viewer)")
