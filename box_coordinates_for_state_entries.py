import cv2

# Load video and read first frame
cap = cv2.VideoCapture('.\\data\\Squirrels_new_cups1.mp4')
ret, frame = cap.read()
cap.release()

# Resize for better display
frame_resized = cv2.resize(frame, (640, 360))  # e.g., to 640x360

# Interactively select ROI (Region of Interest)
x, y, w, h = cv2.selectROI("Select Box Entrance", frame_resized, False)
cv2.destroyAllWindows()

# Scale back to original size if necessary
scale_x = frame.shape[1] / frame_resized.shape[1]
scale_y = frame.shape[0] / frame_resized.shape[0]

x1 = int(x * scale_x)
y1 = int(y * scale_y)
x2 = int((x + w) * scale_x)
y2 = int((y + h) * scale_y)

print(f"x1={x1}, y1={y1}, x2={x2}, y2={y2}")
