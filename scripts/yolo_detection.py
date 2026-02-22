import cv2
from ultralytics import YOLO

model = YOLO('C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Master\\Semester 1\\Study Project\\hoernchen-tracking\\runs\\detect\\train3\\weights\\best.pt')
model_andi = YOLO('C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Master\\Semester 1\\Study Project\\hoernchen-tracking\\runs\\detect\\best_andi.pt')

video_path = "C:\\Users\\maweo\\OneDrive - Universität Münster\\Dokumente\\Master\\Semester 1\\Study Project\\hoernchen-tracking\\videos\\20241015_TrepS_02_in (1)_clip-2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = 'output_squirrel_detection_outside.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    results_andi = model_andi(frame)

    annotated_frame_andi = results_andi[0].plot()
    out.write(annotated_frame_andi)

    cv2.imshow("YOLOv11n Squirrel Detection", annotated_frame_andi)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Video saved to: {output_path}")
cap.release()
out.release()
cv2.destroyAllWindows()