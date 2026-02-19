import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# load video
video_path = ".\\data\\Squirrels_new_cups1.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("Could not open video!")

# target folder for frames
output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

# Background Subtractor 

backSub = cv2.createBackgroundSubtractorKNN(history=250, detectShadows=True)

# Initialize heatmap
heatmap = None

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Resize frames to save RAM
    frame = cv2.resize(frame, (640, 480))

    # Background Subtraction
    fgMask = backSub.apply(frame)

    # Initialize heatmap once size is known
    if heatmap is None:
        heatmap = np.zeros_like(fgMask, dtype=np.float32)

    # Accumulate movement
    heatmap += fgMask / 255  # Normalization: white = 1

    # Save frame 
    frame_name = os.path.join(output_folder, f"frame_{frame_idx:04d}.png")
    cv2.imwrite(frame_name, frame)

    frame_idx += 1
    if frame_idx % 50 == 0:
        print(f"{frame_idx} frames processed...")

cap.release()
print(f"Done! Total {frame_idx} frames processed.")

# Plot
plt.figure(figsize=(10, 6))
plt.imshow(heatmap, cmap='hot')
plt.title("Heatmap: Squirrel Movement")
plt.colorbar(label="Movement Intensity")
plt.show()