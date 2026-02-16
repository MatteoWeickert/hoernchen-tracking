import cv2
import numpy as np


# Configuration
VIDEO_PATH = "/Users/ankenienaber/Documents/Uni/Master/project/hoernchen-tracking/mp4_snippets/Squirrels_new_leaf2.mp4"
OUTPUT_IMAGE_PATH = "trajectories_leaf2.png"

THRESHOLD_VALUE = 20
RESIZE_FACTOR = 0.7
MIN_CONTOUR_AREA = 300
MAX_STEP_DISTANCE = 800

# Nut detection (HSV range)
NUT_COLOR_LOWER = np.array([10, 40, 40])
NUT_COLOR_UPPER = np.array([30, 255, 200])
MIN_NUT_AREA = 200
DISTANCE_THRESHOLD = 50


# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError(f"Could not open video file: {VIDEO_PATH}")

original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

resized_width = int(original_width * RESIZE_FACTOR)
resized_height = int(original_height * RESIZE_FACTOR)

print(f"Video contains {total_frames} frames at {fps:.2f} FPS")


# Read first frame (for nut detection)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, first_frame = cap.read()
if not ret:
    raise ValueError("Could not read first frame")

first_frame = cv2.resize(first_frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
first_hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)


# Trajectory tracking using frame differencing
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
prev_gray = None
background_frame = None
frame_idx = 0
trajectory_data = []

print("Computing trajectory...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if background_frame is None:
        background_frame = frame_resized.copy()

    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        _, mask = cv2.threshold(diff, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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


# Read last frame (for nut comparison)
cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
ret, last_frame = cap.read()
if not ret:
    raise ValueError("Could not read last frame")

last_frame = cv2.resize(last_frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
last_hsv = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)

cap.release()


# Nut detection function
def detect_nuts(hsv_image):
    mask = cv2.inRange(hsv_image, NUT_COLOR_LOWER, NUT_COLOR_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    nuts = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= MIN_NUT_AREA:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(contour)
                nuts.append({
                    "center": (cx, cy),
                    "bbox": (x, y, w, h),
                    "area": area
                })
    return nuts


print("Detecting nuts...")
nuts_first = detect_nuts(first_hsv)
nuts_last = detect_nuts(last_hsv)


# Compare nut positions
missing_nuts = []

for nut_first in nuts_first:
    cx1, cy1 = nut_first["center"]
    found_match = False

    for nut_last in nuts_last:
        cx2, cy2 = nut_last["center"]
        distance = np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)

        if distance < DISTANCE_THRESHOLD:
            found_match = True
            break

    if not found_match:
        missing_nuts.append(nut_first)

print(f"Nuts before: {len(nuts_first)}, after: {len(nuts_last)}, missing: {len(missing_nuts)}")


# Create result image
result_image = background_frame.copy()


# Draw trajectory (speed-colored)
if len(trajectory_data) > 1:
    speeds = []
    valid_segments = []

    for i in range(len(trajectory_data) - 1):
        x1, y1, f1 = trajectory_data[i]
        x2, y2, f2 = trajectory_data[i + 1]

        dx, dy = x2 - x1, y2 - y1
        distance = np.sqrt(dx ** 2 + dy ** 2)
        frame_diff = f2 - f1
        speed = distance / frame_diff if frame_diff > 0 else 0

        speeds.append(speed)

        if distance <= MAX_STEP_DISTANCE:
            valid_segments.append(i)

    if valid_segments:
        valid_speeds = [speeds[i] for i in valid_segments]
        min_speed, max_speed = min(valid_speeds), max(valid_speeds)

        for i in valid_segments:
            normalized = (
                (speeds[i] - min_speed) / (max_speed - min_speed)
                if max_speed > min_speed else 0.5
            )

            progress = 1 - normalized  # red = slow, green = fast

            if progress > 0.5:
                color = (0, int(255 * (1 - (progress - 0.5) * 2)), 255)
            else:
                color = (0, 255, int(255 * (progress * 2)))

            x1, y1, _ = trajectory_data[i]
            x2, y2, _ = trajectory_data[i + 1]

            cv2.line(result_image, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)


# Mark missing nuts (bright purple)
PURPLE = (255, 0, 255)

for nut in missing_nuts:
    x, y, w, h = nut["bbox"]
    cv2.rectangle(result_image, (x, y), (x + w, y + h), PURPLE, 5)
    cv2.circle(result_image, nut["center"], 15, PURPLE, -1)
    cv2.circle(result_image, nut["center"], 17, (255, 255, 255), 2)
    cv2.putText(result_image, "NUT MISSING", (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, PURPLE, 3)


# Legend
legend_y = resized_height - 90

cv2.putText(result_image, "Speed:", (10, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

bar_width = 100
for x in range(bar_width):
    progress = x / bar_width
    if progress < 0.5:
        color = (0, 255, int(255 * (progress * 2)))
    else:
        color = (0, int(255 * (1 - (progress - 0.5) * 2)), 255)

    cv2.line(result_image, (10 + x, legend_y + 10),
             (10 + x, legend_y + 20), color, 2)

cv2.putText(result_image, "Fast", (10, legend_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
cv2.putText(result_image, "Slow", (70, legend_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

cv2.circle(result_image, (20, legend_y + 60), 10, PURPLE, -1)
cv2.circle(result_image, (20, legend_y + 60), 12, (255, 255, 255), 2)
cv2.putText(result_image, "= Missing nut", (35, legend_y + 67),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


# Save and display result
cv2.imwrite(OUTPUT_IMAGE_PATH, result_image)

print("\n" + "=" * 50)
print("COMBINED ANALYSIS")
print("=" * 50)
print(f"Saved to: {OUTPUT_IMAGE_PATH}")
print(f"Trajectory points: {len(trajectory_data)}")
print(f"Missing nuts: {len(missing_nuts)}")
for i, nut in enumerate(missing_nuts):
    print(f"  Nut {i+1}: position {nut['center']}, area {int(nut['area'])} px")
print("=" * 50 + "\n")

cv2.imshow("Combined Analysis", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
