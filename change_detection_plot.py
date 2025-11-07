import cv2
import numpy as np
import matplotlib.pyplot as plt

# Video öffnen
source = cv2.VideoCapture('.\\data\\Squirrels_new_cups4.mp4')
if not source.isOpened():
    raise ValueError("Error: Could not open video.")

# Initialisierung
prev_gray = None
frame_idx = 0
changes = []  # Liste zum Speichern der count_t-Werte

# Schleife über alle Frames
while True:
    ret, frame = source.read()
    if not ret:
        break  # Ende des Videos

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:
        # Differenz zwischen aufeinanderfolgenden Frames
        diff = cv2.absdiff(gray, prev_gray)

        # Threshold anwenden
        tau = 10  # kleiner Wert = empfindlicher
        _, mask = cv2.threshold(diff, tau, 255, cv2.THRESH_BINARY)

        # Anzahl veränderter Pixel zählen
        count_t = np.sum(mask > 0)
        changes.append(count_t)

    prev_gray = gray.copy()
    frame_idx += 1

source.release()
print(f"Done – {frame_idx} Frames processed.")

# --- Glätten (Moving Average) ---
window_size = 10  # Anzahl der Frames für den Mittelwert
kernel = np.ones(window_size) / window_size
smoothed_changes = np.convolve(changes, kernel, mode='valid')

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(changes, label='raw data', alpha=0.5)
plt.plot(range(len(smoothed_changes)), smoothed_changes, color='red', label=f'Smoothed (window={window_size})')
plt.title("Movement / change over time")
plt.xlabel("frame number")
plt.ylabel("= number of changed pixel")
plt.legend()
plt.grid(True)
plt.show()