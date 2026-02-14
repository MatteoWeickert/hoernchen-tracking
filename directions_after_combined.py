import cv2
import numpy as np

# --- KONFIGURATION ---
VIDEO_PATH = '/Users/ankenienaber/Documents/Uni/Master/project/hoernchen-tracking/mp4_snippets/Squirrels_new_leaf1.mp4'
OUTPUT_IMAGE_PATH = 'combined_analysis.png'
THRESHOLD_VALUE = 20
RESIZE_FACTOR = 0.7
MIN_CONTOUR_AREA = 300
MAX_STEP_DISTANCE = 800

# Nuss-Erkennung
NUT_COLOR_LOWER = np.array([10, 40, 40])
NUT_COLOR_UPPER = np.array([30, 255, 200])
MIN_NUT_AREA = 200

# --- VIDEO ÖFFNEN ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError(f"Error: Could not open video file at {VIDEO_PATH}")

original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

resized_width = int(original_width * RESIZE_FACTOR)
resized_height = int(original_height * RESIZE_FACTOR)

print(f"Video hat {total_frames} Frames, {fps} FPS")

# --- ERSTEN FRAME LESEN ---
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, first_frame = cap.read()
if not ret:
    raise ValueError("Konnte ersten Frame nicht lesen!")

first_frame = cv2.resize(first_frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
first_hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)

# --- TRAJECTORY TRACKING ---
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
prev_gray = None
frame_idx = 0
trajectory_data = []
background_frame = None

print("Berechne Trajectory...")

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
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > MIN_CONTOUR_AREA:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    trajectory_data.append((cx, cy, frame_idx))

    prev_gray = gray.copy()
    frame_idx += 1

# --- LETZTEN FRAME LESEN ---
cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
ret, last_frame = cap.read()
if not ret:
    raise ValueError("Konnte letzten Frame nicht lesen!")

last_frame = cv2.resize(last_frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
last_hsv = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)

cap.release()

# --- NUSS-ERKENNUNG ---
print("Erkenne Nüsse...")

# Nüsse im ersten Frame
nut_mask_first = cv2.inRange(first_hsv, NUT_COLOR_LOWER, NUT_COLOR_UPPER)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
nut_mask_first = cv2.morphologyEx(nut_mask_first, cv2.MORPH_OPEN, kernel)
nut_mask_first = cv2.morphologyEx(nut_mask_first, cv2.MORPH_CLOSE, kernel)

contours_first, _ = cv2.findContours(nut_mask_first, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

nuts_first = []
for contour in contours_first:
    area = cv2.contourArea(contour)
    if area >= MIN_NUT_AREA:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(contour)
            nuts_first.append({
                'center': (cx, cy),
                'bbox': (x, y, w, h),
                'area': area
            })

# Nüsse im letzten Frame
nut_mask_last = cv2.inRange(last_hsv, NUT_COLOR_LOWER, NUT_COLOR_UPPER)
nut_mask_last = cv2.morphologyEx(nut_mask_last, cv2.MORPH_OPEN, kernel)
nut_mask_last = cv2.morphologyEx(nut_mask_last, cv2.MORPH_CLOSE, kernel)

contours_last, _ = cv2.findContours(nut_mask_last, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

nuts_last = []
for contour in contours_last:
    area = cv2.contourArea(contour)
    if area >= MIN_NUT_AREA:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(contour)
            nuts_last.append({
                'center': (cx, cy),
                'bbox': (x, y, w, h),
                'area': area
            })

# Vergleiche Nüsse
missing_nuts = []
DISTANCE_THRESHOLD = 50

for nut_first in nuts_first:
    found_match = False
    cx1, cy1 = nut_first['center']
    
    for nut_last in nuts_last:
        cx2, cy2 = nut_last['center']
        distance = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
        
        if distance < DISTANCE_THRESHOLD:
            found_match = True
            break
    
    if not found_match:
        missing_nuts.append(nut_first)

print(f"Nüsse vorher: {len(nuts_first)}, nachher: {len(nuts_last)}, fehlend: {len(missing_nuts)}")

# --- KOMBINIERTES BILD ERSTELLEN ---
result_image = background_frame.copy()

# 1. TRAJECTORY ZEICHNEN
if len(trajectory_data) > 1:
    speeds = []
    valid_segments = []
    
    for i in range(len(trajectory_data) - 1):
        x1, y1, f1 = trajectory_data[i]
        x2, y2, f2 = trajectory_data[i+1]
        
        dx = x2 - x1
        dy = y2 - y1
        distance = np.sqrt(dx**2 + dy**2)
        frame_diff = f2 - f1
        
        if frame_diff > 0:
            speed = distance / frame_diff
        else:
            speed = 0
        
        speeds.append(speed)
        
        if distance <= MAX_STEP_DISTANCE:
            valid_segments.append(i)
    
    if valid_segments:
        valid_speeds = [speeds[i] for i in valid_segments]
        min_speed = min(valid_speeds)
        max_speed = max(valid_speeds)
        
        for i in valid_segments:
            if max_speed > min_speed:
                normalized = (speeds[i] - min_speed) / (max_speed - min_speed)
            else:
                normalized = 0.5
            
            progress = 1 - normalized
            
            if progress > 0.5:
                b, g, r = 0, int(255 * (1 - (progress - 0.5) * 2)), 255
            else:
                b, g, r = 0, 255, int(255 * (progress * 2))
            
            color = (b, g, r)
            
            x1, y1, _ = trajectory_data[i]
            x2, y2, _ = trajectory_data[i+1]
            
            cv2.line(result_image, (x1, y1), (x2, y2), 
                     color, thickness=3, lineType=cv2.LINE_AA)

# 2. FEHLENDE NÜSSE MARKIEREN (KNALL LILA)
LILA = (255, 0, 255)

for nut in missing_nuts:
    x, y, w, h = nut['bbox']
    # Dicker lila Rahmen
    cv2.rectangle(result_image, (x, y), (x+w, y+h), LILA, 5)
    # Großer lila Kreis
    cv2.circle(result_image, nut['center'], 15, LILA, -1)
    # Weißer Rand um Kreis für bessere Sichtbarkeit
    cv2.circle(result_image, nut['center'], 17, (255, 255, 255), 2)
    # "NUSS WEG!" Text
    cv2.putText(result_image, "NUSS WEG!", (x, y-15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, LILA, 3)

# 3. LEGENDE (nur Geschwindigkeit und Nuss)
legend_y = resized_height - 90

# Geschwindigkeits-Legende
cv2.putText(result_image, "Geschwindigkeit:", 
            (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

bar_width = 100
for x in range(bar_width):
    progress = x / bar_width
    if progress < 0.5:
        b, g, r = 0, 255, int(255 * (progress * 2))
    else:
        b, g, r = 0, int(255 * (1 - (progress - 0.5) * 2)), 255
    cv2.line(result_image, (10 + x, legend_y + 10), 
             (10 + x, legend_y + 20), (b, g, r), 2)

cv2.putText(result_image, "schnell", 
            (10, legend_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
cv2.putText(result_image, "langsam", 
            (70, legend_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

# Nuss-Legende
cv2.circle(result_image, (20, legend_y + 60), 10, LILA, -1)
cv2.circle(result_image, (20, legend_y + 60), 12, (255, 255, 255), 2)
cv2.putText(result_image, "= Nuss verschwunden", (35, legend_y + 67), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# --- SPEICHERN ---
cv2.imwrite(OUTPUT_IMAGE_PATH, result_image)

print(f"\n{'='*50}")
print(f"KOMBINIERTE ANALYSE:")
print(f"{'='*50}")
print(f"Bild gespeichert unter: {OUTPUT_IMAGE_PATH}")
print(f"Trajectory-Punkte: {len(trajectory_data)}")
print(f"Fehlende Nüsse: {len(missing_nuts)}")
if missing_nuts:
    for i, nut in enumerate(missing_nuts):
        print(f"  Nuss {i+1}: Position {nut['center']}, Größe {int(nut['area'])}px")
print(f"{'='*50}\n")

# Anzeigen
cv2.imshow('Combined Analysis', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()