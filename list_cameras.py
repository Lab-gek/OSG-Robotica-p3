#!/usr/bin/env python3
"""List available cameras without opening display windows."""

import cv2

print("Scanning for available cameras (0-5)...")
available = []

for i in range(6):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"✓ Camera {i}: Working ({w}x{h})")
            available.append(i)
        else:
            print(f"✗ Camera {i}: Opens but can't read frames")
        cap.release()
    else:
        print(f"✗ Camera {i}: Not available")

print(f"\nTotal working cameras: {len(available)}")
if available:
    print(f"Use --camera {available[0]} to use the first available camera")
