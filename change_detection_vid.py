import cv2
import numpy as np

# Open video
VIDEO_PATH = '.\\data\\20241108_TrepN_04_out.mp4'
THRESHOLD_VALUE = 10      # Threshold for motion detection
RESIZE_FACTOR = 0.7       # Uniform scaling factor (0.7 = 70%)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError(f"Error: Could not open video file at {VIDEO_PATH}")

# inialization
prev_gray = None
frame_idx = 0

# Get original dimensions and FPS
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate delay as integer
delay = int(1000 / fps) if fps > 0 else 30

# Calculate new dimensions based on RESIZE_FACTOR
resized_width = int(original_width * RESIZE_FACTOR)
resized_height = int(original_height * RESIZE_FACTOR)

# Calculate dimensions for combined frame (two images side by side)
combined_width = resized_width * 2
combined_height = resized_height

# VideoWriter to save output video with CORRECT dimensions
output_path = 'output_frame_differencing.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))

# Loop over all frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break  # End of video

    # Resize frame to final size at the beginning
    frame_resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Ensure we have a previous frame for comparison
    if prev_gray is not None:
        # 1. Calculate difference between consecutive frames
        diff = cv2.absdiff(gray, prev_gray)

        # 2. Apply threshold to isolate significant changes
        _, mask = cv2.threshold(diff, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        
        # 3. Convert mask from 1-channel (gray) to 3-channel (BGR) for display
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # 4. Concatenate resized original and mask horizontally
        combined_frame = cv2.hconcat([frame_resized, mask_bgr])
    
        # Write frame to output file (dimensions now match)
        out.write(combined_frame)

        # 5. Display the combined image
        cv2.imshow('Original vs. Change Detection', combined_frame)
    else:
        # For the very first frame, create a placeholder to maintain video dimensions
        # Creates a black image the size of the mask
        black_mask = np.zeros_like(frame_resized)
        combined_frame_first = cv2.hconcat([frame_resized, black_mask])
        out.write(combined_frame_first)

    # Save current grayscale frame for next iteration
    prev_gray = gray.copy()
    frame_idx += 1

    # Wait for key press (abort with 'q')
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        print("Video processing stopped by user.")
        break

print(f"Video successfully saved to: {output_path}")
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Done – {frame_idx} frames processed.")