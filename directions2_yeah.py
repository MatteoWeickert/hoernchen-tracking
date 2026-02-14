import cv2
import numpy as np

# --- KONFIGURATION ---
VIDEO_PATH = '/Users/ankenienaber/Documents/Uni/Master/project/hoernchen-tracking/mp4_snippets/Squirrels_new_leaf2.mp4'
OUTPUT_IMAGE_PATH = 'movement_profile_speed.png'
THRESHOLD_VALUE = 20
RESIZE_FACTOR = 0.7
MIN_CONTOUR_AREA = 300
MAX_STEP_DISTANCE = 800

# --- VIDEO ÖFFNEN ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError(f"Error: Could not open video file at {VIDEO_PATH}")

# --- INITIALISIERUNG ---
prev_gray = None
frame_idx = 0

original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

resized_width = int(original_width * RESIZE_FACTOR)
resized_height = int(original_height * RESIZE_FACTOR)

# Liste für Positionen UND Frame-Nummern
trajectory_data = []  # (cx, cy, frame_number)
background_frame = None

print(f"Verarbeite Video mit {fps} FPS...")

# --- SCHLEIFE ÜBER ALLE FRAMES ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
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

cap.release()

# --- BILD ERSTELLEN ---
if background_frame is not None and len(trajectory_data) > 1:
    profile_image = background_frame.copy()
    
    # Berechne echte Geschwindigkeit (Pixel pro Frame)
    speeds = []
    valid_segments = []
    
    for i in range(len(trajectory_data) - 1):
        x1, y1, f1 = trajectory_data[i]
        x2, y2, f2 = trajectory_data[i+1]
        
        # Distanz
        dx = x2 - x1
        dy = y2 - y1
        distance = np.sqrt(dx**2 + dy**2)
        
        # Zeitdifferenz in Frames
        frame_diff = f2 - f1
        
        # Geschwindigkeit = Distanz / Zeit
        # (wenn frame_diff = 1, dann ist es die Distanz pro Frame)
        if frame_diff > 0:
            speed = distance / frame_diff
        else:
            speed = 0
        
        speeds.append(speed)
        
        # Filter: nur realistische Sprünge
        if distance <= MAX_STEP_DISTANCE:
            valid_segments.append(i)
    
    if valid_segments:
        # Nur gültige Geschwindigkeiten für Normalisierung
        valid_speeds = [speeds[i] for i in valid_segments]
        min_speed = min(valid_speeds)
        max_speed = max(valid_speeds)
        
        # Zeichne Segmente
        for i in valid_segments:
            # Normalisiere Geschwindigkeit auf 0-1
            if max_speed > min_speed:
                normalized = (speeds[i] - min_speed) / (max_speed - min_speed)
            else:
                normalized = 0.5
            
            # normalized: 0 = langsam, 1 = schnell
            # Wir wollen: langsam = rot, schnell = grün
            # Also invertieren
            progress = 1 - normalized  # 1 = langsam (rot), 0 = schnell (grün)
            
            # Farbverlauf: Rot (langsam) -> Gelb -> Grün (schnell)
            if progress > 0.5:
                # Rot zu Gelb
                b = 0
                g = int(255 * (1 - (progress - 0.5) * 2))
                r = 255
            else:
                # Gelb zu Grün
                b = 0
                g = 255
                r = int(255 * (progress * 2))
            
            color = (b, g, r)
            
            x1, y1, _ = trajectory_data[i]
            x2, y2, _ = trajectory_data[i+1]
            
            cv2.line(profile_image, (x1, y1), (x2, y2), 
                     color, thickness=3, lineType=cv2.LINE_AA)
    
    # Legende
    legend_y = resized_height - 60
    cv2.putText(profile_image, "Geschwindigkeit:", 
                (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    bar_width = 100
    for x in range(bar_width):
        progress = x / bar_width
        if progress < 0.5:
            b, g, r = 0, 255, int(255 * (progress * 2))
        else:
            b, g, r = 0, int(255 * (1 - (progress - 0.5) * 2)), 255
        cv2.line(profile_image, (10 + x, legend_y + 10), 
                 (10 + x, legend_y + 20), (b, g, r), 2)
    
    cv2.putText(profile_image, "schnell", 
                (10, legend_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(profile_image, "langsam", 
                (70, legend_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Statistiken
    if valid_segments:
        avg_speed = np.mean([speeds[i] for i in valid_segments])
        max_speed_display = max([speeds[i] for i in valid_segments])
        min_speed_display = min([speeds[i] for i in valid_segments])
    
    skipped_segments = len(speeds) - len(valid_segments)
    
    cv2.putText(profile_image, f"Punkte: {len(trajectory_data)}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(profile_image, f"Durchschn. Speed: {avg_speed:.1f}px/frame", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imwrite(OUTPUT_IMAGE_PATH, profile_image)
    
    print(f"\n{'='*50}")
    print(f"BEWEGUNGSPROFIL ERSTELLT (Geschwindigkeit):")
    print(f"{'='*50}")
    print(f"Bild gespeichert unter: {OUTPUT_IMAGE_PATH}")
    print(f"Pfad-Punkte: {len(trajectory_data)}")
    print(f"Gültige Segmente: {len(valid_segments)}/{len(speeds)}")
    print(f"Gefilterte: {skipped_segments}")
    print(f"\nGeschwindigkeiten:")
    print(f"  Min: {min_speed_display:.2f} px/frame")
    print(f"  Max: {max_speed_display:.2f} px/frame")
    print(f"  Durchschnitt: {avg_speed:.2f} px/frame")
    print(f"\nROT = Langsam")
    print(f"GRÜN = Schnell")
    print(f"{'='*50}\n")
    
    cv2.imshow('Movement Profile', profile_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Fehler: Keine Bewegung erkannt oder kein Hintergrund verfügbar.")