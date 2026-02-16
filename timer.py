import cv2
import numpy as np

# --- KONFIGURATION --- test
VIDEO_PATH = '/Users/ankenienaber/Documents/Uni/Master/project/hoernchen-tracking/mp4_snippets/Squirrels_new_leaf2.mp4'
THRESHOLD_VALUE = 20      # Schwellenwert für Bewegungserkennung
RESIZE_FACTOR = 0.7       
MIN_CONTOUR_AREA = 300    # Mindestgröße eines Objekts (in Pixeln) - feiner eingestellt!

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
delay = int(1000 / fps) if fps > 0 else 30

resized_width = int(original_width * RESIZE_FACTOR)
resized_height = int(original_height * RESIZE_FACTOR)

output_path = 'output_squirrel_tracking.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (resized_width, resized_height))

# --- ZEIT-TRACKING VARIABLEN ---
frames_with_movement = 0
total_frames = 0

# --- SCHLEIFE ÜBER ALLE FRAMES ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break

    frame_resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    
    # Gaußscher Blur um Rauschen zu reduzieren
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if prev_gray is not None:
        # Differenz berechnen
        diff = cv2.absdiff(gray, prev_gray)
        
        # Threshold anwenden
        _, mask = cv2.threshold(diff, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        
        # Morphologische Operationen um kleine Bewegungen (Blätter) zu entfernen
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Entfernt kleine weiße Flecken
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Füllt kleine Löcher
        
        # Dilatation um Objekte etwas zu vergrößern
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Finde Konturen (zusammenhängende Bewegungsobjekte)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtere nach Größe und zeichne Bounding Boxes
        frame_with_boxes = frame_resized.copy()
        
        squirrel_detected = False
        large_contours = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > MIN_CONTOUR_AREA:  # Nur große Objekte (Eichhörnchen)
                large_contours += 1
                squirrel_detected = True
                
                # Zeichne Bounding Box
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Zeige Größe an
                cv2.putText(frame_with_boxes, f"{int(area)}px", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Zähle Frame nur wenn großes Objekt erkannt wurde
        if squirrel_detected:
            frames_with_movement += 1
            status_text = f"EICHHOERNCHEN! ({large_contours} Objekt(e))"
            color = (0, 255, 0)
        else:
            status_text = f"Keine relevante Bewegung"
            color = (0, 0, 255)
        
        # Status-Text
        cv2.putText(frame_with_boxes, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Zeit-Info
        time_in_seconds = frames_with_movement / fps
        minutes = int(time_in_seconds // 60)
        seconds = int(time_in_seconds % 60)
        time_text = f"Zeit: {minutes:02d}:{seconds:02d}"
        cv2.putText(frame_with_boxes, time_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Nur das Original-Video mit Annotations speichern
        out.write(frame_with_boxes)
        cv2.imshow('Squirrel Tracking', frame_with_boxes)
    else:
        out.write(frame_resized)
        cv2.imshow('Squirrel Tracking', frame_resized)

    prev_gray = gray.copy()
    frame_idx += 1
    total_frames += 1

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        print("Video processing stopped by user.")
        break

# --- AUFRÄUMEN UND ERGEBNISSE ---
cap.release()
out.release()
cv2.destroyAllWindows()

total_time_seconds = frames_with_movement / fps
minutes = int(total_time_seconds // 60)
seconds = total_time_seconds % 60

print(f"\n{'='*50}")
print(f"ERGEBNISSE:")
print(f"{'='*50}")
print(f"Video erfolgreich gespeichert unter: {output_path}")
print(f"Verarbeitete Frames: {frame_idx}")
print(f"Frames mit Eichhörnchen: {frames_with_movement}")
print(f"Anteil: {frames_with_movement/total_frames*100:.1f}%")
print(f"\nZeit mit Eichhörnchen-Aktivität: {minutes} Minuten {seconds:.1f} Sekunden")
print(f"(= {total_time_seconds:.2f} Sekunden)")
print(f"{'='*50}\n")
