import cv2
import numpy as np
import matplotlib.pyplot as plt

# Video öffnen
source = cv2.VideoCapture('videos/20241008_TrepN_04_in (2) (1).MOV')
if not source.isOpened():
    raise ValueError("Error: Could not open video.")

# FPS (Frames pro Sekunde) des Videos abrufen
fps = source.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

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

# Zeitachse in Sekunden erstellen
# Wir erstellen ein Array von Frame-Nummern und teilen es durch die FPS
time_axis = np.arange(len(changes)) / fps

# --- Glätten (Moving Average) ---
window_size = 10  # Anzahl der Frames für den Mittelwert
kernel = np.ones(window_size) / window_size
smoothed_changes = np.convolve(changes, kernel, mode='valid')
smoothed_time_axis = np.arange(len(smoothed_changes)) / fps


# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(changes, label='raw data', alpha=0.5)
plt.plot(smoothed_time_axis, smoothed_changes, color='red', label=f'Geglättet (window={window_size})')
plt.title("Movement / change over time")
plt.xlabel("time in s")
plt.ylabel("= number of changed pixel")
plt.legend()
plt.grid(True)
plt.show()