import cv2
import numpy as np
import sys

# --- Video laden ---
video_path = 'Hörnchen_Video1.mp4' 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    sys.exit("Fehler: Video nicht gefunden.")

# --- Interaktive Box-Auswahl ---
ret, first_frame = cap.read()
if not ret: sys.exit("Video leer.")

print("--- ANLEITUNG ---")
print("1. Zeichne BOX 1: Nur der EINGANG (Der 'Türrahmen'). Enter drücken.")
print("2. Zeichne BOX 2: Die gesamte Box (Der Innenbereich). Enter drücken.")

cv2.namedWindow("Setup", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Setup", 1280, 720)

# Box 1: Eingang
roi1 = cv2.selectROI("Setup", first_frame, showCrosshair=True)
# Box 2: Innenbereich (Tief)
roi2 = cv2.selectROI("Setup", first_frame, showCrosshair=True)
cv2.destroyWindow("Setup")

box1 = (int(roi1[0]), int(roi1[1]), int(roi1[0]+roi1[2]), int(roi1[1]+roi1[3]))
box2 = (int(roi2[0]), int(roi2[1]), int(roi2[0]+roi2[2]), int(roi2[1]+roi2[3]))

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# --- Settings ---
fgbg = cv2.createBackgroundSubtractorKNN(history=300, detectShadows=True)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
scale_percent = 60

# Hilfsfunktion zum Skalieren der Boxen
def get_scaled_box(original_box, scale):
    x, y, x2, y2 = original_box
    return (int(x*scale), int(y*scale), int(x2*scale), int(y2*scale))

original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
res_w = int(original_width * scale_percent / 100)
res_h = int(original_height * scale_percent / 100)
fps = int(cap.get(cv2.CAP_PROP_FPS))

# --- VideoWriter ---
combined_width = res_w * 2
combined_height = res_h
output_path = 'output_squirrel_box_detection.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))

#Timer Settings
COOLDOWN_SECONDS = 1.0 
timer_running = False
start_frame = 0
total_elapsed_frames = 0
cooldown_counter = 0 # Zähler in Frames
cooldown_frames_total = int(COOLDOWN_SECONDS * fps) if fps > 0 else 30
frame_idx = 0


while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame_idx += 1

    # Skalieren
    frame_small = cv2.resize(frame, (res_w, res_h))
    
    # Maske erstellen
    fgmask = fgbg.apply(frame_small)
    # Schatten entfernen (Schatten sind in KNN grau, wir wollen nur weiß/255)
    _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
    # Rauschen entfernen
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Boxen auf aktuelle Größe skalieren
    scale = scale_percent / 100
    b1 = get_scaled_box(box1, scale) # Eingang
    b2 = get_scaled_box(box2, scale) # Innen

    # ROI Slices erstellen
    roi_entrance = fgmask[b1[1]:b1[3], b1[0]:b1[2]] # von y1 bis y2 und  x1 bis x2
    roi_inside   = fgmask[b2[1]:b2[3], b2[0]:b2[2]]

    # Pixel zählen
    pixels_entrance = cv2.countNonZero(roi_entrance)
    pixels_inside   = cv2.countNonZero(roi_inside)
    pixels_not_entrance = pixels_inside - pixels_entrance

    # Status bestimmen
    status = "LEER"
    color = (200, 200, 200)

    THRESHOLD_PEEK = 80   # Ein paar Pixel am Eingang
    THRESHOLD_INSIDE = 1500 # Deutliche Bewegung im hinteren Bereich

    # Timer Cooldown herunterzählen
    if cooldown_counter > 0:
        cooldown_counter -= 1

    if pixels_not_entrance > THRESHOLD_INSIDE:
        status = "!!! VOLL DRIN !!!"
        color = (0, 0, 255) # Rot

        if not timer_running and cooldown_counter == 0:
            # EREIGNIS: Timer starten
            timer_running = True
            start_frame = frame_idx
            print(f"Frame {frame_idx}: Timer GESTARTET (Anzahl Pixel: {pixels_not_entrance})")

    elif pixels_entrance > THRESHOLD_PEEK:
        status = "Kopf rein (Peeking)"
        color = (0, 255, 255) # Gelb


    if pixels_not_entrance <= THRESHOLD_INSIDE and timer_running:
            # EREIGNIS: Timer stoppen
            timer_running = False
            elapsed_this_period = frame_idx - start_frame
            total_elapsed_frames += elapsed_this_period
            duration_sec = elapsed_this_period / fps if fps > 0 else 0
            print(f"Frame {frame_idx}: Timer PAUSIERT (Anzahl Pixel: {pixels_not_entrance}). Dauer: {duration_sec:.2f}s")
            
            # Cooldown nach jedem Ereignis neu starten
            cooldown_counter = cooldown_frames_total
    

    # --- Schwerpunkt berechnen (Der Rote Punkt) ---
    # Wir schauen uns das ganze Bild an
    M = cv2.moments(fgmask)
    if M["m00"] > 0: # 0. Moment = Fläche weißer Pixel
        cX = int(M["m10"] / M["m00"]) # 1. Moment (X-Richtung - wie weit ist rechts oder links ist die Masse verteilt)/ 0. Moment 
        cY = int(M["m01"] / M["m00"]) # 1. Moment (Y-Richtung) - wie weit oben/unten ist die Masse verteilt / 0. Moment
        # Zeichne den Schwerpunkt der Bewegung
        cv2.circle(frame_small, (cX, cY), 5, (0, 0, 255), -1)

    # Zeichnen
    cv2.rectangle(frame_small, (b1[0], b1[1]), (b1[2], b1[3]), (0, 255, 255), 2) # Gelbe Box (Eingang)
    cv2.rectangle(frame_small, (b2[0], b2[1]), (b2[2], b2[3]), (0, 0, 255), 2)   # Rote Box (Innen)
    
    cv2.putText(frame_small, f"Eingang: {pixels_entrance}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame_small, f"Innen:   {pixels_inside}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.putText(frame_small, status, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # --- ZEIT BERECHNUNG ---
    # Wir nehmen die bisher gespeicherte Zeit
    display_frames = total_elapsed_frames
    # Wenn der Timer gerade läuft, addieren wir die Zeit seit dem Start des aktuellen Intervalls dazu
    if timer_running:
        display_frames += (frame_idx - start_frame)
    
    total_seconds = display_frames / fps if fps > 0 else 0
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    # Zehntelsekunden
    fraction = int((total_seconds * 10) % 10)
    
    time_string = f"Zeit in Box: {minutes:02d}:{seconds:02d}.{fraction}"

    # Text auf das kombinierte Bild schreiben
    cv2.putText(frame_small, time_string, (20, res_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2, cv2.LINE_AA)

    # Maske bunt machen für Anzeige
    fgmask_color = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    combined = cv2.hconcat([frame_small, fgmask_color])

    out.write(combined)

    cv2.imshow("Analyse", combined)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"FERTIG! Gesamzeit in Box: {minutes:02d}:{seconds:02d}.{fraction}")