import cv2
import numpy as np
import matplotlib.pyplot as plt

# Video-Datei laden
video_path = 'C:\\Users\\hemin\\sciebo\\Master_Geoinformatik\\GI_Master_1\\Squirrels\\mp4_snippets\\Squirrels_new_cups4.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError("Error: Could not open video.")

# Kernel für die morphologische Operation
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# Hintergrundsubtraktionsmodell initialisieren
fgbg = cv2.createBackgroundSubtractorKNN(history=500, detectShadows=True)

# Prozentsatz des Bildes, das oben und unten ignoriert werden soll
crop_percent_top = 0.0
crop_percent_bottom = 0.0

# Skalierungsfaktor für das verkleinerte Bild
scale_percent = 60  # Reduziere auf 60 % der Originalgröße

frame_idx = 0
changes = []  # Liste zum Speichern der Foreground-Pixel

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break  # Ende des Videos
    
    # Bildhöhe und -breite des Frames herausfinden
    height, width = frame.shape[:2]
    
    # Bereiche berechnen, die oben und unten abgeschnitten werden sollen
    top_crop = int(height * crop_percent_top)
    bottom_crop = int(height * (1 - crop_percent_bottom))
    
    # Frame auf den mittleren Bereich zuschneiden
    cropped_frame = frame[top_crop:bottom_crop, :]
    
    # Größe von cropped_frame verkleinern
    width_resized = int(cropped_frame.shape[1] * scale_percent / 100)
    height_resized = int(cropped_frame.shape[0] * scale_percent / 100)
    cropped_frame_resized = cv2.resize(cropped_frame, (width_resized, height_resized))
    
    # Hintergrundsubtraktion anwenden
    fgmask = fgbg.apply(cropped_frame_resized)
    
    # Rauschen mit morphologischen Operationen reduzieren
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    
    # Anzahl der Vordergrund-Pixel zählen
    count_t = np.sum(fgmask > 0)
    changes.append(count_t)
    
    # Optional: Ergebnis während Verarbeitung anzeigen
    fgmask_display = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    combined = cv2.hconcat([cropped_frame_resized, fgmask_display])
    cv2.imshow('Processing...', combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
print(f"Done – {frame_idx} Frames processed.")

# --- Glätten (Moving Average) ---
window_size = 10
kernel_smooth = np.ones(window_size) / window_size
smoothed_changes = np.convolve(changes, kernel_smooth, mode='valid')

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(changes, label='Raw Data', alpha=0.5, color='gray')
plt.plot(range(len(smoothed_changes)), smoothed_changes, color='red', 
         linewidth=2, label=f'Smoothed (Window={window_size})')
plt.title("Movement / Change Over Time (Background Subtraction)")
plt.xlabel("Frame Number")
plt.ylabel("Number of Foreground Pixels")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()