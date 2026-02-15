import cv2
from ultralytics import YOLO

# 1. Lade das vortrainierte YOLOv8-Modell
# 'yolov8n.pt' ist das kleinste und schnellste Modell
model = YOLO('C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Master\\Semester 1\\Study Project\\hoernchen-tracking\\runs\\detect\\train3\\weights\\best.pt')
model_andi = YOLO('C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Master\\Semester 1\\Study Project\\hoernchen-tracking\\runs\\detect\\best_andi.pt')

# 2. Öffne deine Videodatei
video_path = "C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Master\\Semester 1\\Study Project\\hoernchen-tracking\\videos\\20241015_TrepS_02_in (1)_clip-2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Fehler: Video konnte nicht geöffnet werden.")
    exit()

# Speichere Werte für die Videoeigenschaften
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# VideoWriter zum Speichern des Ausgabevideos
output_path = 'output_squirrel_detection_outside.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 3. Verarbeite das Video Frame für Frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 4. Führe die Objekterkennung auf dem Frame aus
    results = model(frame)
    results_andi = model_andi(frame)

    # 5. Visualisiere die Ergebnisse
    # results enthält alle erkannten Objekte
    annotated_frame_andi = results_andi[0].plot()
    out.write(annotated_frame_andi)

    # Zeige den bearbeiteten Frame an
    cv2.imshow("YOLOv11n Squirrel Detection", annotated_frame_andi)

    # Taste 'q', um Programm zu beenden
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Aufräumen
print(f"Video erfolgreich gespeichert unter: {output_path}")
cap.release()
out.release()
cv2.destroyAllWindows()