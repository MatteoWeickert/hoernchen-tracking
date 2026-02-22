import cv2
import numpy as np
import matplotlib.pyplot as plt

# load video
video_path = ''
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError("Error: Could not open video.")

# Kernel for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorKNN(history=500, detectShadows=True)

# How much of the frame to ignore at the top and bottom (as a fraction)
crop_percent_top = 0.0
crop_percent_bottom = 0.0

# Scale factor for downsampling
scale_percent = 60  # Reduce to 60% of original size

frame_idx = 0
changes = []  # Stores foreground pixel counts per frame

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break  # End of video
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Calculate crop boundaries
    top_crop = int(height * crop_percent_top)
    bottom_crop = int(height * (1 - crop_percent_bottom))
    
    # Crop to the region of interest
    cropped_frame = frame[top_crop:bottom_crop, :]
    
    # Resize the cropped frame
    width_resized = int(cropped_frame.shape[1] * scale_percent / 100)
    height_resized = int(cropped_frame.shape[0] * scale_percent / 100)
    cropped_frame_resized = cv2.resize(cropped_frame, (width_resized, height_resized))
    
    # Apply background subtraction
    fgmask = fgbg.apply(cropped_frame_resized)
    
    # Clean up noise with morphological operations
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    
    # Count foreground pixels
    count_t = np.sum(fgmask > 0)
    changes.append(count_t)
    
    # Show original and mask side by side while processing
    fgmask_display = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    combined = cv2.hconcat([cropped_frame_resized, fgmask_display])
    cv2.imshow('Processing...', combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
print(f"Done – {frame_idx} Frames processed.")

# Smooth data with moving average
window_size = 10
kernel_smooth = np.ones(window_size) / window_size
smoothed_changes = np.convolve(changes, kernel_smooth, mode='valid')

# Plot
plt.figure(figsize=(10, 5))
plt.plot(changes, label='Raw Data', alpha=0.5, color='gray')
plt.plot(range(len(smoothed_changes)), smoothed_changes, color='red', 
         linewidth=2, label=f'Smoothed (Window={window_size})')
plt.title("Movement / Change Over Time (Background Subtraction)")
plt.xlabel("Frame Number")
plt.ylabel("Number of Foreground Pixels")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()