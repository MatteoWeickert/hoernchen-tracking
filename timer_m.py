import cv2
import numpy as np

# --- CONFIGURATION ---
VIDEO_PATH = '/Users/ankenienaber/Documents/Uni/Master/project/hoernchen-tracking/mp4_snippets/Squirrels_new_cups2.mp4'
VIDEO_ID = 'newCups2'
THRESHOLD_VALUE = 10                       # Threshold for general motion detection
RESIZE_FACTOR = 0.2                        # Resize factor (0.2 = 20%)
ENTRY_EXIT_THRESHOLD_ORIGINAL = 1.5e6      # Threshold for timer start/stop (original resolution)
COOLDOWN_SECONDS = 1.0                     # Cooldown to prevent immediate timer toggle

# --- OPEN VIDEO ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError(f"Error: Could not open video file at {VIDEO_PATH}")

# --- INITIALIZATION ---
prev_gray = None
frame_idx = 0

# Get original dimensions and FPS
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 30

# Calculate new dimensions based on resize factor
resized_width = int(original_width * RESIZE_FACTOR)
resized_height = int(original_height * RESIZE_FACTOR)

# Dimensions for combined frame (original + mask side by side)
combined_width = resized_width * 2
combined_height = resized_height

# Setup video writer
output_path = f'output_frame_differencing_with_timer_{VIDEO_ID}.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))

# --- TIMER INITIALIZATION ---
timer_running = False
start_frame = 0
total_elapsed_frames = 0
cooldown_counter = 0
cooldown_frames_total = int(COOLDOWN_SECONDS * fps) if fps > 0 else 30

# Adjust threshold for resized resolution (pixel count scales quadratically)
adjusted_threshold = ENTRY_EXIT_THRESHOLD_ORIGINAL * (RESIZE_FACTOR ** 2)
print(f"Adjusted threshold for timer: {int(adjusted_threshold)} pixels")

# --- MAIN LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break

    frame_resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    
    combined_frame = None
    
    if prev_gray is not None:
        # Calculate frame difference
        diff = cv2.absdiff(gray, prev_gray)
        _, mask = cv2.threshold(diff, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        
        # --- TIMER LOGIC ---
        changed_pixel_count = cv2.countNonZero(mask)
        
        # Decrease cooldown counter
        if cooldown_counter > 0:
            cooldown_counter -= 1
        
        # Check if threshold exceeded and cooldown expired
        if changed_pixel_count > adjusted_threshold and cooldown_counter == 0:
            if not timer_running:
                # EVENT: Start timer
                timer_running = True
                start_frame = frame_idx
                print(f"Frame {frame_idx}: Timer STARTED (Change: {changed_pixel_count} pixels)")
            else:
                # EVENT: Stop timer
                timer_running = False
                elapsed_this_period = frame_idx - start_frame
                total_elapsed_frames += elapsed_this_period
                duration_sec = elapsed_this_period / fps if fps > 0 else 0
                print(f"Frame {frame_idx}: Timer STOPPED (Change: {changed_pixel_count} pixels). Duration: {duration_sec:.2f}s")
            
            # Restart cooldown after each event
            cooldown_counter = cooldown_frames_total
        
        # Create combined frame (original + mask)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined_frame = cv2.hconcat([frame_resized, mask_bgr])
    
    else:
        # First frame: create black placeholder
        black_mask = np.zeros_like(frame_resized)
        combined_frame = cv2.hconcat([frame_resized, black_mask])
    
    # --- DISPLAY TIMER ON FRAME ---
    # Calculate current time to display
    current_total_frames = total_elapsed_frames
    if timer_running:
        current_total_frames += (frame_idx - start_frame)
    
    # Convert to MM:SS format
    total_seconds = current_total_frames / fps if fps > 0 else 0
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    time_string = f"Time in box: {minutes:02d}:{seconds:02d}"
    
    # Write text on combined frame
    cv2.putText(combined_frame, time_string, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Save and display frame
    out.write(combined_frame)
    cv2.imshow('Original vs. Change Detection', combined_frame)
    
    prev_gray = gray.copy()
    frame_idx += 1
    
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        print("Video processing stopped by user.")
        break

# --- CLEANUP AND RESULTS ---
cap.release()
out.release()
cv2.destroyAllWindows()

final_total_seconds = total_elapsed_frames / fps if fps > 0 else 0
print(f"\nProcessing complete. Estimated total time in box: {final_total_seconds:.2f} seconds.")
print(f"Video successfully saved to: {output_path}")
print(f"Done – {frame_idx} frames processed.")