import cv2
import numpy as np

# --- KONFIGURATION ---
VIDEO_PATH = '/Users/ankenienaber/Documents/Uni/Master/project/hoernchen-tracking/mp4_snippets/Squirrels_new_cups2.mp4'
video_id = 'newCups2'
THRESHOLD_VALUE = 10      # Schwellenwert für die allgemeine Bewegungserkennung
RESIZE_FACTOR = 0.2       # EINHEITLICHER Faktor zur Verkleinerung (0.2 = 20%)
# NEU: Schwellenwert für das Starten/Stoppen des Timers (bezogen auf die Originalauflösung)
# 1.5e6 entspricht 1.5 * 10^6, also 1.5 Millionen Pixel
ENTRY_EXIT_THRESHOLD_ORIGINAL = 1.5e6 
# NEU: Cooldown in Sekunden, um sofortiges Umkippen des Timers zu verhindern
COOLDOWN_SECONDS = 1.0 


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
delay = int(1000 / fps) if fps > 0 else 30

# Berechne die neuen Dimensionen basierend auf dem RESIZE_FACTOR
resized_width = int(original_width * RESIZE_FACTOR)
resized_height = int(original_height * RESIZE_FACTOR)

# Berechne die Dimensionen für den kombinierten Frame
combined_width = resized_width * 2
combined_height = resized_height

# VideoWriter zum Speichern des Ausgabevideos
output_path = f'output_frame_differencing_with_timer_{video_id}.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))


# --- TIMER-INITIALISIERUNG ---
timer_running = False
start_frame = 0
total_elapsed_frames = 0
cooldown_counter = 0 # Zähler in Frames
cooldown_frames_total = int(COOLDOWN_SECONDS * fps) if fps > 0 else 30

# Der Schwellenwert für die Pixelanzahl muss an die neue Größe angepasst werden,
# da sich die Gesamtpixelzahl quadratisch mit dem RESIZE_FACTOR ändert.
adjusted_threshold = ENTRY_EXIT_THRESHOLD_ORIGINAL * (RESIZE_FACTOR ** 2)
print(f"Angepasster Schwellenwert für Timer: {int(adjusted_threshold)} Pixel")


# --- SCHLEIFE ÜBER ALLE FRAMES ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break

    frame_resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    
    # Platzhalter für das kombinierte Bild
    combined_frame = None

    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        _, mask = cv2.threshold(diff, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        
        # --- NEU: TIMER-LOGIK ---
        changed_pixel_count = cv2.countNonZero(mask)

        # Cooldown herunterzählen
        if cooldown_counter > 0:
            cooldown_counter -= 1

        # Prüfen, ob der Schwellenwert überschritten wurde und der Cooldown abgelaufen ist
        if changed_pixel_count > adjusted_threshold and cooldown_counter == 0:
            if not timer_running:
                # EREIGNIS: Timer starten
                timer_running = True
                start_frame = frame_idx
                print(f"Frame {frame_idx}: Timer GESTARTET (Änderung: {changed_pixel_count} Pixel)")
            else:
                # EREIGNIS: Timer stoppen
                timer_running = False
                elapsed_this_period = frame_idx - start_frame
                total_elapsed_frames += elapsed_this_period
                duration_sec = elapsed_this_period / fps if fps > 0 else 0
                print(f"Frame {frame_idx}: Timer GESTOPPT (Änderung: {changed_pixel_count} Pixel). Dauer: {duration_sec:.2f}s")
            
            # Cooldown nach jedem Ereignis neu starten
            cooldown_counter = cooldown_frames_total
        
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined_frame = cv2.hconcat([frame_resized, mask_bgr])
    
    else:
        # Für den allerersten Frame einen schwarzen Platzhalter erstellen
        black_mask = np.zeros_like(frame_resized)
        combined_frame = cv2.hconcat([frame_resized, black_mask])

    # --- NEU: TIMER AUF BILD ZEICHNEN ---
    # Aktuell zu anzeigende Zeit berechnen
    current_total_frames = total_elapsed_frames
    if timer_running:
        current_total_frames += (frame_idx - start_frame)
    
    # Umrechnung in MM:SS Format
    total_seconds = current_total_frames / fps if fps > 0 else 0
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    time_string = f"Zeit in Box: {minutes:02d}:{seconds:02d}"

    # Text auf das kombinierte Bild schreiben
    cv2.putText(combined_frame, time_string, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Frame speichern und anzeigen
    out.write(combined_frame)
    cv2.imshow('Original vs. Change Detection', combined_frame)

    prev_gray = gray.copy()
    frame_idx += 1

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        print("Video processing stopped by user.")
        break

# --- AUFRÄUMEN ---
final_total_seconds = total_elapsed_frames / fps if fps > 0 else 0
print(f"Verarbeitung abgeschlossen. Geschätzte Gesamtzeit in der Box: {final_total_seconds:.2f} Sekunden.")
print(f"Video erfolgreich gespeichert unter: {output_path}")

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Done – {frame_idx} Frames processed.")