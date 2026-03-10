#!/usr/bin/env python3
"""Quick camera test - shows what camera 0 sees."""

import cv2
import sys

print("Testing camera 0...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera 0")
    print("\nTrying to list available cameras...")
    for i in range(10):
        test_cap = cv2.VideoCapture(i)
        if test_cap.isOpened():
            print(f"  Camera {i} is available")
            test_cap.release()
    sys.exit(1)

print("Camera opened successfully!")
print("Press 'q' to quit")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to grab frame {frame_count}")
        break
    
    frame_count += 1
    
    # Add frame counter text
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Camera Test - Press Q to quit', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # q or ESC
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nTotal frames captured: {frame_count}")
