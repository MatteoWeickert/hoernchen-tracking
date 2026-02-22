import cv2
import numpy as np

# --- CONFIG ---
VIDEO_PATH = 'videos\Squirrels_new_leaf2.mp4'
THRESHOLD_VALUE = 10
RESIZE_FACTOR = 0.2

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError(f"Error: Could not open video file at {VIDEO_PATH}")

prev_gray = None
frame_idx = 0

original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

delay = int(1000 / fps) if fps > 0 else 30

resized_width = int(original_width * RESIZE_FACTOR)
resized_height = int(original_height * RESIZE_FACTOR)

combined_width = resized_width * 2
combined_height = resized_height

output_path = 'output_frame_differencing.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))

# --- FRAME LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break

    frame_resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        _, mask = cv2.threshold(diff, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined_frame = cv2.hconcat([frame_resized, mask_bgr])
    
        out.write(combined_frame)
        cv2.imshow('Original vs. Change Detection', combined_frame)
    else:
        black_mask = np.zeros_like(frame_resized)
        combined_frame_first = cv2.hconcat([frame_resized, black_mask])
        out.write(combined_frame_first)

    prev_gray = gray.copy()
    frame_idx += 1

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        print("Video processing stopped by user.")
        break

# --- CLEANUP ---
print(f"Video saved to: {output_path}")
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Done – {frame_idx} Frames processed.")