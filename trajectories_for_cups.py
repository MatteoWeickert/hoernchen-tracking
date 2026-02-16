import cv2
import numpy as np


# Configuration
VIDEO_PATH = "/Users/ankenienaber/Documents/Uni/Master/project/hoernchen-tracking/mp4_snippets/Squirrels_new_cups1.mp4"
OUTPUT_IMAGE_PATH = "movement_profile_cups1.png"

THRESHOLD_VALUE = 20
RESIZE_FACTOR = 0.7
MIN_CONTOUR_AREA = 300
MAX_STEP_DISTANCE = 800


# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError(f"Could not open video file: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

resized_width = int(original_width * RESIZE_FACTOR)
resized_height = int(original_height * RESIZE_FACTOR)

print(f"Processing video at {fps:.2f} FPS...")


# Tracking initialization
prev_gray = None
frame_idx = 0
trajectory_data = []  # (x, y, frame_index)


# Process all frames for motion tracking
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break

    frame_resized = cv2.resize(
        frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if prev_gray is not None:
        # Frame differencing
        diff = cv2.absdiff(gray, prev_gray)
        _, mask = cv2.threshold(diff, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

        # Morphological noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > MIN_CONTOUR_AREA:
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    trajectory_data.append((cx, cy, frame_idx))

    prev_gray = gray.copy()
    frame_idx += 1

cap.release()


# Extract last frame as background
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
ret, last_frame = cap.read()
cap.release()

if not ret:
    raise ValueError("Could not read last frame for background.")

background_frame = cv2.resize(
    last_frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA
)


# Create movement profile
if len(trajectory_data) < 2:
    print("No valid movement detected.")
    exit()

profile_image = background_frame.copy()

speeds = []
valid_segments = []

# Compute speed between consecutive trajectory points
for i in range(len(trajectory_data) - 1):
    x1, y1, f1 = trajectory_data[i]
    x2, y2, f2 = trajectory_data[i + 1]

    dx = x2 - x1
    dy = y2 - y1
    distance = np.sqrt(dx**2 + dy**2)
    frame_diff = f2 - f1

    speed = distance / frame_diff if frame_diff > 0 else 0
    speeds.append(speed)

    if distance <= MAX_STEP_DISTANCE:
        valid_segments.append(i)

if not valid_segments:
    print("No valid trajectory segments found.")
    exit()

valid_speeds = [speeds[i] for i in valid_segments]
min_speed = min(valid_speeds)
max_speed = max(valid_speeds)


# Draw speed-colored trajectory
for i in valid_segments:
    if max_speed > min_speed:
        normalized = (speeds[i] - min_speed) / (max_speed - min_speed)
    else:
        normalized = 0.5

    progress = 1 - normalized  # red = slow, green = fast

    if progress > 0.5:
        b = 0
        g = int(255 * (1 - (progress - 0.5) * 2))
        r = 255
    else:
        b = 0
        g = 255
        r = int(255 * (progress * 2))

    color = (b, g, r)

    x1, y1, _ = trajectory_data[i]
    x2, y2, _ = trajectory_data[i + 1]

    cv2.line(profile_image, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)


# Legend
legend_y = resized_height - 60

cv2.putText(
    profile_image, "Speed:", (10, legend_y),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
)

bar_width = 100
for x in range(bar_width):
    progress = x / bar_width
    if progress < 0.5:
        color = (0, 255, int(255 * (progress * 2)))
    else:
        color = (0, int(255 * (1 - (progress - 0.5) * 2)), 255)

    cv2.line(
        profile_image,
        (10 + x, legend_y + 10),
        (10 + x, legend_y + 20),
        color,
        2,
    )

cv2.putText(
    profile_image, "Fast",
    (10, legend_y + 35),
    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
)
cv2.putText(
    profile_image, "Slow",
    (70, legend_y + 35),
    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1
)


# Statistics
avg_speed = np.mean(valid_speeds)
min_speed_display = min(valid_speeds)
max_speed_display = max(valid_speeds)
skipped_segments = len(speeds) - len(valid_segments)

cv2.putText(
    profile_image, f"Points: {len(trajectory_data)}",
    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
)
cv2.putText(
    profile_image, f"Avg speed: {avg_speed:.1f} px/frame",
    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
)

cv2.imwrite(OUTPUT_IMAGE_PATH, profile_image)

print("\n" + "=" * 50)
print("MOVEMENT PROFILE CREATED (Speed-based)")
print("=" * 50)
print(f"Saved to: {OUTPUT_IMAGE_PATH}")
print(f"Trajectory points: {len(trajectory_data)}")
print(f"Valid segments: {len(valid_segments)}/{len(speeds)}")
print(f"Filtered segments: {skipped_segments}")
print("\nSpeed statistics:")
print(f"  Min: {min_speed_display:.2f} px/frame")
print(f"  Max: {max_speed_display:.2f} px/frame")
print(f"  Avg: {avg_speed:.2f} px/frame")
print("=" * 50 + "\n")

cv2.imshow("Movement Profile", profile_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
