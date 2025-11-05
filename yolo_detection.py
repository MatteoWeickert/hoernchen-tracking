import cv2
from ultralytics import YOLO

# 1. Lade das vortrainierte YOLOv8-Modell
# 'yolov8n.pt' ist das kleinste und schnellste Modell
model = YOLO('yolov8n.pt')

# 2. Öffne deine Videodatei
video_path = 'squirrel_vid1_cutted.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Fehler: Video konnte nicht geöffnet werden.")
    exit()

# 3. Verarbeite das Video Frame für Frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 4. Führe die Objekterkennung auf dem Frame aus
    results = model(frame)

    # 5. Visualisiere die Ergebnisse
    # results enthält alle erkannten Objekte
    annotated_frame = results[0].plot()

    # Zeige den bearbeiteten Frame an
    cv2.imshow("YOLOv8n Squirrel Detection", annotated_frame)

    # Taste 'q', um Programm zu beenden
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Aufräumen
cap.release()
cv2.destroyAllWindows()