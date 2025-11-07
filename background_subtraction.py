import cv2
import numpy as np
import matplotlib.pyplot as plt

# Video-Datei laden
video_path = '.\\data\\Squirrels_new_cups1.mp4'
cap = cv2.VideoCapture(video_path)

# Kernel für die morphologische Operation
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# Hintergrundsubstraktionsmodell initialisieren
fgbg = cv2.createBackgroundSubtractorKNN(history=250, detectShadows=True)

# Prozentsatz des Bildes, das oben und unten ignoriert werden soll
crop_percent_top = 0.0
crop_percent_bottom = 0.0

# Skalierungsfaktor für das verkleinerte Bild
scale_percent = 60

# Hole die Original-Dimensionen und FPS
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

resized_width = int(original_width * scale_percent / 100)
resized_height = int(original_height * scale_percent / 100)

# Berechne Dimensionen des kombinierten Frames
combined_width = resized_width * 2
combined_height = resized_height

# VideoWriter initialisieren
output_path = 'output_background_subtraction.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))

# --- Listen für Histogramm ---
changed_pixels_per_frame = []
frame_numbers = []

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Bildhöhe und -breite
    height, width = frame.shape[:2]

    # Zuschneiden
    top_crop = int(height * crop_percent_top)
    bottom_crop = int(height * (1 - crop_percent_bottom))
    cropped_frame = frame[top_crop:bottom_crop, :]

    # Größe anpassen
    width_resized = int(cropped_frame.shape[1] * scale_percent / 100)
    height_resized = int(cropped_frame.shape[0] * scale_percent / 100)
    cropped_frame_resized = cv2.resize(cropped_frame, (width_resized, height_resized))

    # Hintergrundsubstraktion
    fgmask = fgbg.apply(cropped_frame_resized)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    # --- Anzahl veränderter Pixel berechnen ---
    changed_pixels = cv2.countNonZero(fgmask)
    changed_pixels_per_frame.append(changed_pixels)
    frame_numbers.append(frame_idx)
    frame_idx += 1

    # Maske für Anzeige vorbereiten
    fgmask_color = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    fgmask_resized = cv2.resize(fgmask_color, (width_resized, height_resized))

    # Konturen finden und einzeichnen
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if 100 < cv2.contourArea(contour):  # nur Konturen >100 Pixel Fläche
            cv2.drawContours(cropped_frame_resized, [contour], -1, (0, 0, 255), 2)

    # Frames nebeneinander kombinieren
    combined_frame = cv2.hconcat([cropped_frame_resized, fgmask_resized])
    out.write(combined_frame)

    # Anzeige (optional)
    cv2.imshow('Combined Frame', combined_frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

print(f"Video erfolgreich gespeichert unter: {output_path}")

# --- Ressourcen freigeben ---
cap.release()
out.release()
cv2.destroyAllWindows()

# --- Histogramm anzeigen ---
plt.figure(figsize=(10, 5))
plt.plot(frame_numbers, changed_pixels_per_frame, color='blue')
plt.title('Number of Changed Pixels per Frame')
plt.xlabel('Frame Number')
plt.ylabel('Changed Pixels')
plt.grid(True)
plt.tight_layout()
plt.show()
