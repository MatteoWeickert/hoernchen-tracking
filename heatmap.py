import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Video laden ---
video_path = ".\\data\\Sam_3_b_w.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("Video konnte nicht geöffnet werden!")

# --- Zielordner für Frames ---
output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

# --- Background Subtractor ---
backSub = cv2.createBackgroundSubtractorKNN(history=250, detectShadows=True)

# --- Heatmap initialisieren ---
heatmap = None

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Ende des Videos

    # Frames kleiner machen, um RAM zu sparen
    frame = cv2.resize(frame, (640, 480))

    # Background Subtraction
    fgMask = backSub.apply(frame)

    # Heatmap initialisieren, sobald die Größe bekannt ist
    if heatmap is None:
        heatmap = np.zeros_like(fgMask, dtype=np.float32)

    # Bewegung aufsummieren
    heatmap += fgMask / 255  # Normierung: weiß = 1

    # Frame speichern (optional)
    frame_name = os.path.join(output_folder, f"frame_{frame_idx:04d}.png")
    cv2.imwrite(frame_name, frame)

    frame_idx += 1
    if frame_idx % 50 == 0:
        print(f"{frame_idx} Frames verarbeitet...")

cap.release()
print(f"Fertig! Insgesamt {frame_idx} Frames verarbeitet.")

# --- Heatmap visualisieren ---
plt.figure(figsize=(10,6))
plt.imshow(heatmap, cmap='hot')
plt.title("Heatmap Squirrel-movement")
plt.colorbar(label="movement-intensity")
plt.show()
