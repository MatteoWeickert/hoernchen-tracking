import cv2
import numpy as np

# --- CONFIGURATION ---
VIDEO_PATH = "sam3_outside_bw.mp4"
OUTPUT_PATH = "sam3_with_timer.mp4"

WHITE_THRESHOLD = 200        # Threshold for white pixel detection
MOTION_THRESHOLD = 25        # Threshold for motion detection
WHITE_PIXEL_MIN = 500        # Minimum white pixels to detect squirrel
NO_WHITE_DURATION = 0.5      # Duration in seconds without white pixels to stop timer

# --- OPEN VIDEO ---
cap = cv2.VideoCapture(VIDEO_PATH)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# --- INITIALIZATION ---
NO_WHITE_FRAMES = int(fps * NO_WHITE_DURATION)

timer_running = False
start_frame = 0
current_frame_idx = 0
no_white_counter = 0

# Read first frame for motion detection
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# --- MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect white pixels
    white_pixels = np.sum(gray > WHITE_THRESHOLD)
    
    # Detect motion (only used for start condition)
    diff = cv2.absdiff(gray, prev_gray)
    motion_pixels = np.sum(diff > MOTION_THRESHOLD)
    
    # Start condition: sufficient white pixels and motion detected
    if not timer_running:
        if white_pixels > WHITE_PIXEL_MIN and motion_pixels > 300:
            timer_running = True
            start_frame = current_frame_idx
            print("Timer started")
    
    # Stop condition: track frames without white pixels
    if timer_running:
        if white_pixels < WHITE_PIXEL_MIN:
            no_white_counter += 1
        else:
            no_white_counter = 0
    
    # Calculate elapsed time
    elapsed_time = 0
    if timer_running:
        elapsed_time = (current_frame_idx - start_frame) / fps
    
    # Display timer on frame
    cv2.putText(
        frame,
        f"Time: {elapsed_time:.2f} s",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2
    )
    
    out.write(frame)
    
    # Stop processing if no white pixels detected for specified duration
    if timer_running and no_white_counter >= NO_WHITE_FRAMES:
        print(f"Timer stopped at {elapsed_time:.2f} seconds")
        break
    
    prev_gray = gray
    current_frame_idx += 1

# --- CLEANUP ---
cap.release()
out.release()
print("Processing complete!")