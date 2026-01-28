import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.spatial.distance import cdist


SAM_VIDEO_PATH = "C:\\Users\\maweo\\Downloads\\sam_1617128279441201.mp4"

# Farbwerte für Rosa im HSV-Farbraum
PINK_HSV_TARGET = np.array([165, 150, 239])
THRESHOLD = 20 
LOWER_PINK = np.array([165 - 10, 100, 100])
UPPER_PINK = np.array([165 + 10, 255, 255])

def get_box_points(roi, density=1):
    """Erzeugt eine Liste von Punkten entlang der vier Kanten der Box."""
    x, y, w, h = roi
    pts = []
    # Oben und Unten
    for i in range(x, x + w, density):
        pts.append([i, y])
        pts.append([i, y + h])
    # Links und Rechts
    for j in range(y, y + h, density):
        pts.append([x, j])
        pts.append([x + w, j])
    return np.array(pts, dtype=np.float64)

# 1. Video laden
cap = cv2.VideoCapture(SAM_VIDEO_PATH)
ret, first_frame = cap.read()

# VideoWriter für Speicherung vorbereiten
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = "C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Master\\Semester 1\\Study Project\\hoernchen-tracking\\output_videos\\extract_mask_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 2. Box Auswahl
print("Box einzeichnen und ENTER drücken...")
box_roi = cv2.selectROI("Box-Auswahl", first_frame, fromCenter=False)
cv2.destroyWindow("Box-Auswahl")

# Punkte entlang der Box-Ränder generieren
box_points = get_box_points(box_roi, density=2)
if box_points.ndim != 2 or box_points.shape[0] == 0:
    print("Fehler beim Erzeugen der Box-Punkte. XB ist nicht 2D.")
    exit()

# 3. Plot-Setup
plt.ion()
fig, ax = plt.subplots()
x_data, y_data = [], []
line, = ax.plot([], [], 'm-', label="Minimal Distance (Squirrel to Box)")
ax.set_ylim(0, 500) # Je nach Videoauflösung anpassen
ax.legend()

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1
    
    # 4. Eichhörnchen finden
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_PINK, UPPER_PINK)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        best_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(best_cnt) > 150:
            # Alle Punkte der Eichhörnchen-Kontur extrahieren
            sq_points = best_cnt.reshape(-1, 2).astype(np.float64)
            
            # 5. Abstand zwischen ALLEN Eichhörnchen-Punkten und ALLEN Box-Punkten
            # Wir nutzen cdist für eine hocheffiziente Matrix-Berechnung
            if sq_points.shape[0] > 0 and box_points.shape[0] > 0:
                distances = cdist(sq_points, box_points, metric='euclidean')
                
                # Finde das Minimum in der Distanzmatrix
                min_dist = np.min(distances)
                idx_sq, idx_bx = np.unravel_index(np.argmin(distances), distances.shape)
                
                # Die zwei Punkte, die sich am nächsten sind
                p_sq = tuple(sq_points[idx_sq].astype(int))
                p_bx = tuple(box_points[idx_bx].astype(int))
                
                # Daten speichern
                x_data.append(frame_idx)
                y_data.append(min_dist)
                
                # Visualisierung
                cv2.line(frame, p_sq, p_bx, (0, 255, 255), 2) # Gelbe Verbindungslinie
                cv2.circle(frame, p_sq, 5, (0, 0, 255), -1)   # Punkt am Eichhörnchen
                cv2.circle(frame, p_bx, 5, (255, 0, 0), -1)   # Punkt an der Box
                
                # Graph aktualisieren
                line.set_data(x_data, y_data)
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.001)

    # Box zeichnen
    x, y, w, h = box_roi
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Frame speichern
    out.write(frame)
    
    cv2.imshow("Präzise Rand-Analyse", frame)
    if cv2.waitKey(80) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()