import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
VIDEO_PATH = 'mp4_snippets/TrepN_04_disco5.mp4'
CROP_PERCENT_TOP = 0.0
CROP_PERCENT_BOTTOM = 0.0
SCALE_PERCENT = 60  # Resize frame to 60%

# Box coordinates in original video resolution
BOX1 = (1830, 198, 2190, 588)  # Entry box
BOX2 = (1536, 66, 2520, 870)   # Interior box

# --- OPEN VIDEO ---
cap = cv2.VideoCapture(VIDEO_PATH)

# --- INITIALIZATION ---
# Kernel for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# Background subtraction model
fgbg = cv2.createBackgroundSubtractorKNN(history=250, detectShadows=True)

# Get original dimensions and FPS
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

resized_width = int(original_width * SCALE_PERCENT / 100)
resized_height = int(original_height * SCALE_PERCENT / 100)

# Setup video writer for combined output
combined_width = resized_width * 2
combined_height = resized_height
output_path = 'output_squirrel_box_detection.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))

# Data collection lists
changed_pixels_per_frame = []
changed_pixels_box1_per_frame = []
changed_pixels_box2_per_frame = []
frame_numbers = []

frame_idx = 0
scale = SCALE_PERCENT / 100

def scale_box(box, top_crop, scale):
    """Scale box coordinates to match resized frame."""
    x1, y1, x2, y2 = box
    y1 -= top_crop
    y2 -= top_crop
    return (int(x1 * scale), int(y1 * scale),
            int(x2 * scale), int(y2 * scale))

# --- MAIN LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Crop frame if needed
    height, width = frame.shape[:2]
    top_crop = int(height * CROP_PERCENT_TOP)
    bottom_crop = int(height * (1 - CROP_PERCENT_BOTTOM))
    cropped_frame = frame[top_crop:bottom_crop, :]

    # Resize frame
    cropped_frame_resized = cv2.resize(cropped_frame, (resized_width, resized_height))

    # Apply background subtraction
    fgmask = fgbg.apply(cropped_frame_resized)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    # Count total changed pixels
    total_pixels = cv2.countNonZero(fgmask)
    changed_pixels_per_frame.append(total_pixels)

    # Scale and analyze Box 1 (entry)
    x1a, y1a, x2a, y2a = scale_box(BOX1, top_crop, scale)
    region1 = fgmask[y1a:y2a, x1a:x2a]
    pix1 = cv2.countNonZero(region1)
    rel1 = pix1 / region1.size
    changed_pixels_box1_per_frame.append(pix1)

    # Scale and analyze Box 2 (interior)
    x1b, y1b, x2b, y2b = scale_box(BOX2, top_crop, scale)
    region2 = fgmask[y1b:y2b, x1b:x2b]
    pix2 = cv2.countNonZero(region2)
    rel2 = pix2 / region2.size
    changed_pixels_box2_per_frame.append(pix2)

    # Classify squirrel position
    if rel2 > 0.1:
        label = "FULLY INSIDE"
        color = (0, 0, 255)
    else:
        if rel1 < 0.1:
            label = "head only"
        elif rel1 < 0.4:
            label = "half body"
        else:
            label = "full body"
        color = (0, 255, 0)

    # Draw boxes on frame
    cv2.rectangle(cropped_frame_resized, (x1a, y1a), (x2a, y2a), (0, 255, 0), 2)
    cv2.rectangle(cropped_frame_resized, (x1b, y1b), (x2b, y2b), (255, 0, 0), 2)
    cv2.putText(cropped_frame_resized, label, (x1a, y1a - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Prepare mask for display
    fgmask_color = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    fgmask_resized = cv2.resize(fgmask_color, (resized_width, resized_height))

    # Draw contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            cv2.drawContours(cropped_frame_resized, [cnt], -1, (0, 0, 255), 2)

    # Create side-by-side view
    combined_frame = cv2.hconcat([cropped_frame_resized, fgmask_resized])
    out.write(combined_frame)
    cv2.imshow("Combined Frame", combined_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    frame_numbers.append(frame_idx)
    frame_idx += 1

# --- CLEANUP ---
cap.release()
out.release()
cv2.destroyAllWindows()
print("Processing complete!")

# --- PLOT RESULTS ---
plt.figure(figsize=(10, 5))
plt.plot(frame_numbers, changed_pixels_box1_per_frame, label='Box 1 (Entry)')
plt.plot(frame_numbers, changed_pixels_box2_per_frame, label='Box 2 (Interior)')
plt.xlabel('Frame Number')
plt.ylabel('Changed Pixels')
plt.title('Pixel Changes in Detection Boxes')
plt.legend()
plt.grid()
plt.show()