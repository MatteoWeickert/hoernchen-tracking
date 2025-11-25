import cv2
import numpy as np
import matplotlib.pyplot as plt

# Video öffnen
source = cv2.VideoCapture('.\\data\\Squirrels_new_leaf1.mp4')
if not source.isOpened():
    raise ValueError("Error: Could not open video.")

fps = source.get(cv2.CAP_PROP_FPS)

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


# --- Schwellenwert wählen ---
# typischerweise z.B. Mittelwert + Faktor * Standardabweichung
threshold = np.mean(smoothed_changes) + 0.1 * np.std(smoothed_changes)

# Binäre Maske: 1 = Bewegung (Eichhörnchen da)
movement_mask = smoothed_changes > threshold

# --- Start- und Endpunkte finden ---
movement_regions = []
in_region = False
for i, active in enumerate(movement_mask):
    if active and not in_region:
        start = i
        in_region = True
    elif not active and in_region:
        end = i
        in_region = False
        movement_regions.append((start, end))
# falls am Ende noch aktiv:
if in_region:
    movement_regions.append((start, len(movement_mask)-1))

# --- Ergebnisse ausgeben ---
print("Estimated periods during which the squirrel is visible:")
for (start, end) in movement_regions:
    start_time = start / fps
    end_time = end / fps
    print(f"From frame {start} to {end}  →  approx {start_time:.2f}s to {end_time:.2f}s")

    # --- Gesamtdauer berechnen ---
total_time = sum((end - start) / fps for start, end in movement_regions)
print(f"\n total time the squirrel is visible:  {total_time:.2f} seconds")


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