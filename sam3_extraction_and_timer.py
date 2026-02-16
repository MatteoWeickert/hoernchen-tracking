import cv2
import numpy as np

# --- CONFIGURATION ---
VIDEO_PATH = "sam_outside.mp4"
INTERMEDIATE_MASK_PATH = "sam3_outside_bw.mp4"
OUTPUT_PATH = "sam3_with_timer.mp4"

# Pink detection parameters (HSV color space)
# Narrower ranges for more precise pink/magenta detection
LOWER_PINK1 = np.array([145, 100, 100])
UPPER_PINK1 = np.array([165, 255, 255])
LOWER_PINK2 = np.array([165, 100, 100])
UPPER_PINK2 = np.array([175, 255, 255])

# Timer parameters
WHITE_THRESHOLD = 200        # Threshold for white pixel detection
MOTION_THRESHOLD = 25        # Threshold for motion detection
WHITE_PIXEL_MIN = 500        # Minimum white pixels to detect squirrel
NO_WHITE_DURATION = 0.5      # Duration in seconds without white pixels to stop timer

# --- STEP 1: EXTRACT PINK MASK FROM VIDEO ---
print("Step 1: Extracting pink SAM3 overlay...")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError(f"Error: Could not open video file at {VIDEO_PATH}")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video loaded: {width}x{height}, {fps} FPS, {total_frames} frames")

# Setup video writer for mask video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
mask_video_writer = cv2.VideoWriter(INTERMEDIATE_MASK_PATH, fourcc, fps, 
                                   (width, height), isColor=False)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to HSV for pink detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create masks for pink color ranges
    # Pink/Magenta has Hue values around 140-180 in OpenCV
    mask1 = cv2.inRange(hsv, LOWER_PINK1, UPPER_PINK1)
    mask2 = cv2.inRange(hsv, LOWER_PINK2, UPPER_PINK2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Morphological operations for smoothing
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Write to intermediate mask video
    mask_video_writer.write(mask)
    
    frame_count += 1
    
    if frame_count % 30 == 0:
        print(f"Processed: {frame_count}/{total_frames} frames")

cap.release()
mask_video_writer.release()

print(f"✓ Pink mask extraction complete! {frame_count} frames processed.")
print(f"Intermediate mask video saved: {INTERMEDIATE_MASK_PATH}")

# --- STEP 2: APPLY TIMER TO MASK VIDEO ---
print("\nStep 2: Applying timer to mask video...")

cap = cv2.VideoCapture(INTERMEDIATE_MASK_PATH)

# Setup video writer for final output
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# Timer initialization
NO_WHITE_FRAMES = int(fps * NO_WHITE_DURATION)
timer_running = False
start_frame = 0
current_frame_idx = 0
no_white_counter = 0

# Read first frame for motion detection
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_display = frame
    else:
        gray = frame
        frame_display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
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
        frame_display,
        f"Time: {elapsed_time:.2f} s",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2
    )
    
    out.write(frame_display)
    
    # Stop processing if no white pixels detected for specified duration
    if timer_running and no_white_counter >= NO_WHITE_FRAMES:
        print(f"Timer stopped at {elapsed_time:.2f} seconds")
        break
    
    prev_gray = gray
    current_frame_idx += 1

# --- CLEANUP ---
cap.release()
out.release()

print("\n" + "="*50)
print("✓ Processing complete!")
print(f"Intermediate mask video: {INTERMEDIATE_MASK_PATH}")
print(f"Final output video: {OUTPUT_PATH}")
print("="*50)