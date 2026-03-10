#!/usr/bin/env python3
"""Test the full visualization pipeline like main.py does."""

import cv2
import time
from line_detector import LineDetector
from aruco_tracker import ArucoTracker
from robot_controller import RobotController
from visualizer import DebugVisualizer

print("Opening camera 0...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit(1)

print("Initializing components...")
detector = LineDetector()
tracker = ArucoTracker()
controller = RobotController()
visualizer = DebugVisualizer(scale=1.0)

print("Starting visualization (press 'q' to quit)...")
time.sleep(1)  # Let camera adjust

frame_count = 0
fps_timer = time.time()
fps_display = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame!")
        break
    
    # Check that frame isn't all black
    mean_brightness = frame.mean()
    print(f"Frame {frame_count}: brightness={mean_brightness:.1f}")
    
    # Process frame through the pipeline
    line_result = detector.detect(frame)
    aruco_result = tracker.detect(frame)
    command = controller.compute(line_result, aruco_result)
    
    # Calculate FPS
    frame_count += 1
    now = time.time()
    if now - fps_timer >= 1.0:
        fps_display = frame_count / (now - fps_timer)
        frame_count = 0
        fps_timer = now
    
    # Build visualization
    vis = visualizer.build_frame(frame, line_result, aruco_result, command, fps_display)
    
    # Show it
    cv2.imshow('Test Visualization', vis)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")
