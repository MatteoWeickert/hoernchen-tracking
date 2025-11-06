import cv2
import numpy as np

# --- KONFIGURATION ---
VIDEO_PATH = 'C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Master\\Semester 1\\Study Project\\hoernchen-tracking\\squirrel_vid1_cutted.mp4'
THRESHOLD_VALUE = 10      # Schwellenwert für die Bewegungserkennung
RESIZE_FACTOR = 0.7       # EINHEITLICHER Faktor zur Verkleinerung (0.7 = 70%)

# --- VIDEO ÖFFNEN ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError(f"Error: Could not open video file at {VIDEO_PATH}")

# --- INITIALISIERUNG ---
prev_gray = None
frame_idx = 0

# Hole die Original-Dimensionen und FPS
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Verzögerung als Integer berechnen
# Wenn fps 0 ist (z.B. bei einem Fehler), setze einen Standardwert
delay = int(1000 / fps) if fps > 0 else 30

# Berechne die neuen Dimensionen basierend auf dem RESIZE_FACTOR
resized_width = int(original_width * RESIZE_FACTOR)
resized_height = int(original_height * RESIZE_FACTOR)

# Berechne die Dimensionen für den kombinierten Frame (zwei Bilder nebeneinander)
combined_width = resized_width * 2
combined_height = resized_height

# VideoWriter zum Speichern des Ausgabevideos mit den KORREKTEN Dimensionen
output_path = 'output_frame_differencing.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))

# --- SCHLEIFE ÜBER ALLE FRAMES ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break  # Ende des Videos

    # KORREKTUR: Frame direkt am Anfang auf die finale Größe skalieren
    frame_resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    # Konvertierung in Graustufen
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Sicherstellen, dass wir einen vorherigen Frame zum Vergleich haben
    if prev_gray is not None:
        # 1. Differenz zwischen aufeinanderfolgenden Frames berechnen
        diff = cv2.absdiff(gray, prev_gray)

        # 2. Threshold anwenden, um signifikante Änderungen zu isolieren
        _, mask = cv2.threshold(diff, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        
        # 3. Maske von 1-Kanal (Grau) in 3-Kanal (BGR) umwandeln für die Anzeige
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # 4. Skaliertes Originalbild und Maske horizontal zusammenfügen
        combined_frame = cv2.hconcat([frame_resized, mask_bgr])
    
        # Schreibe den Frame in die Ausgabedatei (Dimensionen stimmen jetzt überein)
        out.write(combined_frame)

        # 5. Das kombinierte Bild anzeigen
        cv2.imshow('Original vs. Change Detection', combined_frame)
    else:
        # Für den allerersten Frame, erstelle einen Platzhalter, um die Video-Dimensionen zu wahren
        # Erzeugt ein schwarzes Bild in der Größe der Maske
        black_mask = np.zeros_like(frame_resized)
        combined_frame_first = cv2.hconcat([frame_resized, black_mask])
        out.write(combined_frame_first)


    # Den aktuellen Graustufen-Frame für die nächste Iteration speichern
    prev_gray = gray.copy()
    frame_idx += 1

    # Warten auf Tastendruck (Abbruch mit 'q')
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        print("Video processing stopped by user.")
        break

# --- AUFRÄUMEN ---
print(f"Video erfolgreich gespeichert unter: {output_path}")
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Done – {frame_idx} Frames processed.")