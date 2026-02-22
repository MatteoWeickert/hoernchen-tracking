import cv2
import numpy as np

# --- CONFIG ---
VIDEO_PATH = 'videos\Squirrels_new_cups2.mp4'
video_id = 'newCups2'
THRESHOLD_VALUE = 10
RESIZE_FACTOR = 0.2
ENTRY_EXIT_THRESHOLD_ORIGINAL = 1.5e6
COOLDOWN_SECONDS = 1.0

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

output_path = f'output_frame_differencing_with_timer_{video_id}.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))


# --- TIMER INIT ---
timer_running = False
start_frame = 0
total_elapsed_frames = 0
cooldown_counter = 0
cooldown_frames_total = int(COOLDOWN_SECONDS * fps) if fps > 0 else 30

# Scale threshold proportionally to resize factor
adjusted_threshold = ENTRY_EXIT_THRESHOLD_ORIGINAL * (RESIZE_FACTOR ** 2)
print(f"Adjusted timer threshold: {int(adjusted_threshold)} pixels")

# --- FRAME LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break

    frame_resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    
    combined_frame = None

    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        _, mask = cv2.threshold(diff, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        
        # --- TIMER LOGIC ---
        changed_pixel_count = cv2.countNonZero(mask)

        if cooldown_counter > 0:
            cooldown_counter -= 1

        if changed_pixel_count > adjusted_threshold and cooldown_counter == 0:
            if not timer_running:
                timer_running = True
                start_frame = frame_idx
                print(f"Frame {frame_idx}: Timer STARTED (changed: {changed_pixel_count} px)")
            else:
                timer_running = False
                elapsed_this_period = frame_idx - start_frame
                total_elapsed_frames += elapsed_this_period
                duration_sec = elapsed_this_period / fps if fps > 0 else 0
                print(f"Frame {frame_idx}: Timer STOPPED (changed: {changed_pixel_count} px). Duration: {duration_sec:.2f}s")
            
            cooldown_counter = cooldown_frames_total
        
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined_frame = cv2.hconcat([frame_resized, mask_bgr])
    
    else:
        black_mask = np.zeros_like(frame_resized)
        combined_frame = cv2.hconcat([frame_resized, black_mask])

    # --- DRAW TIMER ON FRAME ---
    current_total_frames = total_elapsed_frames
    if timer_running:
        current_total_frames += (frame_idx - start_frame)
    
    total_seconds = current_total_frames / fps if fps > 0 else 0
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    time_string = f"Time in box: {minutes:02d}:{seconds:02d}"

    cv2.putText(combined_frame, time_string, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    out.write(combined_frame)
    cv2.imshow('Original vs. Change Detection', combined_frame)

    prev_gray = gray.copy()
    frame_idx += 1

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        print("Video processing stopped by user.")
        break

# --- CLEANUP ---
final_total_seconds = total_elapsed_frames / fps if fps > 0 else 0
print(f"Done. Estimated total time in box: {final_total_seconds:.2f} seconds.")
print(f"Video saved to: {output_path}")

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Done – {frame_idx} Frames processed.")