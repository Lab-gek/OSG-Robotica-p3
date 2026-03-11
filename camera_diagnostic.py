#!/usr/bin/env python3
"""Simple camera viewer with brightness info - run with sudo."""

import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera 0")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera opened. Press 'q' to quit, 's' to save frame")
print("If window is black, try:")
print("  1. Remove any lens cover/privacy shutter")
print("  2. Turn on lights")
print("  3. Try camera 1 instead: sudo python thisscript.py 1")

# Let camera warm up
time.sleep(1)
for _ in range(10):
    cap.read()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        time.sleep(0.1)
        continue
    
    frame_count += 1
    
    # Calculate brightness stats
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    min_val = np.min(gray)
    max_val = np.max(gray)
    
    # Add overlay text
    info = [
        f"Frame: {frame_count}",
        f"Brightness: {mean_brightness:.1f} (0=black, 255=white)",
        f"Min: {min_val}  Max: {max_val}",
        f"Resolution: {frame.shape[1]}x{frame.shape[0]}",
    ]
    
    if mean_brightness < 10:
        info.append("WARNING: Very dark! Check lens cover/lights")
    elif mean_brightness > 245:
        info.append("WARNING: Overexposed")
    else:
        info.append("Brightness OK")
    
    y_offset = 30
    for line in info:
        cv2.putText(frame, line, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 30
    
    cv2.imshow('Camera Diagnostic - Press Q to quit, S to save', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('s'):
        filename = f'saved_frame_{frame_count}.jpg'
        cv2.imwrite(filename, frame)
        print(f"Saved {filename} (brightness: {mean_brightness:.1f})")

cap.release()
cv2.destroyAllWindows()
print(f"\nCaptured {frame_count} frames")
