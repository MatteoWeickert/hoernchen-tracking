import cv2
import numpy as np
import matplotlib.pyplot as plt

# Open video
source = cv2.VideoCapture('.\\data\\20241108_TrepN_04_out.mp4')
if not source.isOpened():
    raise ValueError("Error: Could not open video.")

fps = source.get(cv2.CAP_PROP_FPS)

# Initialization
prev_gray = None
frame_idx = 0
changes = []  # List to store count_t values

# Loop over all frames
while True:
    ret, frame = source.read()
    if not ret:
        break 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:
        # Difference between consecutive frames
        diff = cv2.absdiff(gray, prev_gray)

        # Apply threshold
        tau = 10  
        _, mask = cv2.threshold(diff, tau, 255, cv2.THRESH_BINARY)

        # Count changed pixels
        count_t = np.sum(mask > 0)
        changes.append(count_t)

    prev_gray = gray.copy()
    frame_idx += 1

source.release()
print(f"Done – {frame_idx} frames processed.")

# --- Smoothing (Moving Average) ---
window_size = 10  # Number of frames for the mean
kernel = np.ones(window_size) / window_size
smoothed_changes = np.convolve(changes, kernel, mode='valid')

# --- Choose threshold ---
threshold = np.mean(smoothed_changes) + 0.1 * np.std(smoothed_changes)

# Binary mask: 1 = movement (squirrel present)
movement_mask = smoothed_changes > threshold

# --- Find start and end points ---
movement_regions = []
in_region = False
for i, active in enumerate(movement_mask):
    if active and not in_region:
        start = i
        in_region = True
    elif not active and in_region:
        end = i
        in_region = False
        movement_regions.append((start, end))
# If still active at the end:
if in_region:
    movement_regions.append((start, len(movement_mask)-1))

# --- Output results ---
print("Estimated periods during which the squirrel is visible:")
for (start, end) in movement_regions:
    start_time = start / fps
    end_time = end / fps
    print(f"From frame {start} to {end}  →  approx {start_time:.2f}s to {end_time:.2f}s")

# --- Calculate total duration ---
total_time = sum((end - start) / fps for start, end in movement_regions)
print(f"\nTotal time the squirrel is visible: {total_time:.2f} seconds")

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(changes, label='Raw data', alpha=0.5)
plt.plot(range(len(smoothed_changes)), smoothed_changes, color='red', label=f'Smoothed (window={window_size})')
plt.title("Movement / Change Over Time")
plt.xlabel("Frame Number")
plt.ylabel("Number of Changed Pixels")
plt.legend()
plt.grid(True)
plt.show()